"""CPU-only server tests. Do not run on the local workstation."""
import copy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.models.clip_preboq import PatchAdapter, PreBoQ, MeanAdversary, replay_backward
from src.semantic_region_protocol import state_hash


class TinyAggregation(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 12, 3, padding=1)
        self.linear = nn.Linear(12, 16)

    def forward(self, x):
        return F.normalize(self.linear(F.gelu(self.conv(x)).mean((2, 3))), dim=1), None


class TinyOfficial(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(3, 8, 1)
        self.aggregator = TinyAggregation()

    def forward(self, x): return self.aggregator(self.backbone(x))[0]


def example():
    torch.manual_seed(42)
    return PreBoQ(TinyOfficial(), channels=8, hidden=4), torch.randn(8, 3, 4, 5)


def test_zero_start_matches_official_and_raw_patch_means():
    m, x = example()
    with torch.no_grad():
        f = m.encode(x); z, means = m.describe(f)
        assert torch.equal(m.adapter(f), f)
        assert torch.allclose(z, m.visual(x), atol=1e-7, rtol=1e-6)
        assert torch.equal(means, f.mean((2, 3)))


def test_only_adapter_trains_and_rgb_inference_does_not_need_labels_or_clip():
    m, x = example(); m.train()
    assert not m.visual.training
    assert all(n.startswith('adapter.') for n, p in m.named_parameters() if p.requires_grad)
    assert m(x).shape == (8, 16)
    with pytest.raises(TypeError): m(x, x)
    assert not any('predictor' in n or 'teacher' in n for n in m.state_dict())


def test_inner_parameters_open_after_zero_start_without_changing_official():
    m, x = example(); before = state_hash(m.visual)
    opt = torch.optim.Adam(m.adapter.parameters(), lr=.01)
    for step in range(2):
        opt.zero_grad(); z = m(x); (z*torch.arange(16)).sum().backward()
        assert m.adapter.up.weight.grad.abs().sum() > 0
        if step: assert m.adapter.down.weight.grad.abs().sum() > 0
        opt.step()
    assert state_hash(m.visual) == before and all(p.grad is None for p in m.visual.parameters())


def test_adversary_has_the_correct_reversed_gradient():
    torch.manual_seed(2)
    head = MeanAdversary(torch.ones(8), hidden=12, targets=3)
    means = torch.randn(8, 8, requires_grad=True)
    labels = torch.arange(2).repeat_interleave(4); y = torch.randn(8, 3)
    params = list(head.parameters())
    a = torch.autograd.grad(F.mse_loss(head(means, labels), y), [means, *params])
    b = torch.autograd.grad(F.mse_loss(head(means, labels, True), y), [means, *params])
    assert torch.allclose(a[0], -b[0]) and a[0].abs().sum() > 0
    assert all(torch.equal(u, v) for u, v in zip(a[1:], b[1:]))


def test_adversary_update_is_detached_from_adapter():
    m, x = example(); head = MeanAdversary(torch.ones(8), targets=3)
    _, means = m.describe(m.encode(x)); labels = torch.arange(2).repeat_interleave(4)
    head(means.detach(), labels).square().mean().backward()
    assert all(p.grad is None for p in m.adapter.parameters())
    assert any(p.grad is not None for p in head.parameters())


@pytest.mark.parametrize('microbatch', [1, 2, 3, 8])
def test_gradient_cache_matches_full_batch_both_paths(microbatch):
    m, x = example()
    with torch.no_grad(): m.adapter.up.weight.normal_(std=.1)
    direct = copy.deepcopy(m); replay = copy.deepcopy(m)
    features = m.encode(x)
    torch.manual_seed(7)
    weights_z, weights_m = torch.randn(8, 16), torch.randn(8, 8)
    z, mean = direct.describe(features)
    ((z*weights_z).sum() + (mean*weights_m).sum()).backward()
    with torch.no_grad():
        zs, ms = zip(*(replay.describe(features[i:i+microbatch]) for i in range(0, 8, microbatch)))
    zr = torch.cat(zs).requires_grad_(); mr = torch.cat(ms).requires_grad_()
    ((zr*weights_z).sum() + (mr*weights_m).sum()).backward()
    replay_backward(replay, features, zr.grad, mr.grad, microbatch)
    for a, b in zip(direct.adapter.parameters(), replay.adapter.parameters()):
        assert torch.allclose(a.grad, b.grad, atol=2e-6, rtol=1e-4)


def test_nuisance_loss_connects_to_adapter_not_frozen_backbone():
    m, x = example(); head = MeanAdversary(torch.ones(8), targets=3).requires_grad_(False)
    _, mean = m.describe(m.encode(x)); labels = torch.arange(2).repeat_interleave(4)
    F.mse_loss(head(mean, labels, True), torch.randn(8, 3)).backward()
    assert m.adapter.up.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in m.visual.parameters())


def test_config_is_isolated_fixed_budget_and_low_memory():
    from scripts.clip_preboq_study import configuration
    c = configuration(Path(__file__).resolve().parents[1]/'config/clip_preboq.yaml')
    assert c['epochs']*c['train_places']//c['places_per_batch'] == 2560
    assert c['runs'] == 'logs/clip_preboq_v1' and c['cuda_memory_fraction'] <= .2
    assert c['hidden'] == 64 and c['predictor_hidden'] == 128


def test_artifact_corruption_is_not_silently_resumed(tmp_path):
    from scripts.clip_preboq_study import finish, verified
    from src.semantic_region_protocol import write
    write(tmp_path/'contract.json', dict(schema='clip_preboq_variation_v1'))
    write(tmp_path/'summary.json', dict(pass_check=True)); finish(tmp_path)
    assert verified(tmp_path)['schema'] == 'clip_preboq_variation_v1'
    write(tmp_path/'summary.json', {})
    with pytest.raises(ValueError): verified(tmp_path)


def test_training_and_eval_need_explicit_flags():
    script = Path(__file__).resolve().parents[1]/'scripts/clip_preboq_study.py'
    for command in ('smoke', 'train', 'eval'):
        r = subprocess.run([sys.executable, str(script), command], capture_output=True, text=True)
        assert r.returncode != 0
        assert ('--evaluate' if command == 'eval' else '--allow-training') in r.stderr
