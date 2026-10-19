"""Server CPU tests: paired initialization, locality, independent gradients/resume."""
import copy
import io
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.models.spatial_adapter_repeat import paired_models, adapter_inventory
from src.models.clip_preboq import replay_backward
from src.semantic_region_protocol import state_hash


class TinyAggregation(nn.Module):
    def __init__(self, channels=8):
        super().__init__(); self.conv = nn.Conv2d(channels, 12, 3, padding=1); self.linear = nn.Linear(12, 16)

    def forward(self, x): return F.normalize(self.linear(F.gelu(self.conv(x)).mean((2, 3))), dim=1), None


class TinyOfficial(nn.Module):
    def __init__(self, channels=8):
        super().__init__(); self.backbone = nn.Conv2d(3, channels, 1); self.aggregator = TinyAggregation(channels)

    def forward(self, x): return self.aggregator(self.backbone(x))[0]


def example(seed=42):
    torch.manual_seed(seed)
    # Production wrapper expects768 channels. Only the tests replace the adapter
    # channel count after constructing matched model objects.
    from src.models.clip_preboq import PatchAdapter
    models = paired_models(TinyOfficial(), hidden=4)
    models['spatial'].adapter = PatchAdapter(channels=8, hidden=4)
    models['channel'].adapter = copy.deepcopy(models['spatial'].adapter)
    old = models['channel'].adapter.spatial
    new = nn.Conv2d(4, 4, 1, groups=4)
    with torch.no_grad(): new.weight.copy_(old.weight[:, :, 1:2, 1:2]); new.bias.copy_(old.bias)
    models['channel'].adapter.spatial = new
    return models, torch.randn(8, 3, 5, 5)


def test_production_counts_and_only_kernel_removal():
    models = paired_models(TinyOfficial(768))
    info = adapter_inventory(models)
    assert info['spatial']['parameters'] == 99776
    assert info['channel']['parameters'] == 99264
    assert (99776-99264)/99776 < .0052
    assert info['spatial']['middle_kernel'] == [3, 3] and info['channel']['middle_kernel'] == [1, 1]
    assert info['spatial']['middle_groups'] == info['channel']['middle_groups'] == 64


def test_real_pair_common_weights_center_and_no_shared_student_parameters():
    from scripts.spatial_adapter_repeat import common_initial
    models = paired_models(TinyOfficial(768))
    assert common_initial(models)
    assert models['spatial'].visual is models['channel'].visual
    assert not ({id(p) for p in models['spatial'].adapter.parameters()} & {id(p) for p in models['channel'].adapter.parameters()})
    assert all(not p.requires_grad for p in models['spatial'].visual.parameters())


def test_both_zero_start_and_rgb_only():
    models, images = example()
    for model in models.values():
        model.train()
        assert not model.visual.training
        with torch.no_grad(): assert torch.allclose(model(images), model.visual(images), atol=1e-7)
        with pytest.raises(TypeError): model(images, images)
        assert all(n.startswith('adapter.') for n, p in model.named_parameters() if p.requires_grad)
        assert not any('predictor' in n or 'teacher' in n for n in model.state_dict())


def test_channel_is_pointwise_spatial_has_neighbor_receptive_field():
    models, _ = example()
    for arm, model in models.items():
        for p in model.adapter.parameters():
            with torch.no_grad(): p.fill_(.03)
        x = torch.zeros(1, 8, 5, 5, requires_grad=True)
        gradient = torch.autograd.grad(model.adapter(x)[0, 0, 2, 2], x)[0]
        neighbor = float(gradient[:, :, 1, 2].abs().sum())
        assert (neighbor > 0) if arm == 'spatial' else (neighbor == 0)


def test_pointwise_adapter_commutes_with_pixel_permutation():
    models, _ = example(); x = torch.randn(2, 8, 5, 5); order = torch.randperm(25)
    for arm, model in models.items():
        with torch.no_grad(): model.adapter.up.weight.normal_(std=.1)
        left = model.adapter(x.flatten(2)[:, :, order].reshape_as(x))
        right = model.adapter(x).flatten(2)[:, :, order].reshape_as(x)
        if arm == 'channel': assert torch.allclose(left, right, atol=1e-6)
        else: assert not torch.allclose(left, right, atol=1e-6)


def test_up_then_inner_gradients_and_frozen_original():
    models, x = example(); original = state_hash(models['spatial'].visual)
    for arm, m in models.items():
        opt = torch.optim.AdamW(m.adapter.parameters(), lr=.01, weight_decay=0.)
        for step in range(2):
            opt.zero_grad(); (m(x)*torch.arange(16)).sum().backward()
            assert m.adapter.up.weight.grad.abs().sum() > 0
            if step: assert m.adapter.down.weight.grad.abs().sum() > 0
            opt.step()
    assert state_hash(models['spatial'].visual) == original


@pytest.mark.parametrize('arm', ['spatial', 'channel'])
def test_descriptor_cache_matches_direct_gradient(arm):
    models, x = example(); m = models[arm]
    with torch.no_grad(): m.adapter.up.weight.normal_(std=.1)
    direct = copy.deepcopy(m); replay = copy.deepcopy(m); f = m.encode(x)
    weights = torch.randn(8, 16)
    (direct.describe(f)[0]*weights).sum().backward()
    with torch.no_grad(): z = torch.cat([replay.describe(f[i:i+2])[0] for i in range(0, 8, 2)])
    z.requires_grad_(); (z*weights).sum().backward()
    replay_backward(replay, f, z.grad, torch.zeros(8, 8), 2)
    for a, b in zip(direct.adapter.parameters(), replay.adapter.parameters()):
        assert torch.allclose(a.grad, b.grad, atol=2e-6, rtol=1e-4)


def test_updating_spatial_does_not_touch_channel_or_official():
    models, x = example(); before = {a: state_hash(m.adapter) for a, m in models.items()}
    official = state_hash(models['spatial'].visual)
    opt = torch.optim.AdamW(models['spatial'].adapter.parameters(), lr=.01)
    (models['spatial'](x)*torch.arange(16)).sum().backward(); opt.step()
    assert state_hash(models['channel'].adapter) == before['channel']
    assert all(p.grad is None for p in models['channel'].adapter.parameters())
    assert state_hash(models['spatial'].visual) == official


def test_paired_optimizer_checkpoint_resume_exact():
    models, x = example()
    opts = {a: torch.optim.AdamW(m.adapter.parameters(), lr=.01) for a, m in models.items()}
    weights = torch.randn(8, 16)

    def step(ms, os):
        for a, m in ms.items():
            os[a].zero_grad(); (m(x)*weights).sum().backward(); os[a].step()

    step(models, opts)
    stream = io.BytesIO()
    torch.save(dict(adapters={a: m.adapter.state_dict() for a, m in models.items()},
                    optimizers={a: o.state_dict() for a, o in opts.items()}), stream)
    step(models, opts)
    resumed, _ = example(); ropts = {a: torch.optim.AdamW(m.adapter.parameters(), lr=.01) for a, m in resumed.items()}
    stream.seek(0); value = torch.load(stream, weights_only=True)
    for a, m in resumed.items(): m.adapter.load_state_dict(value['adapters'][a]); ropts[a].load_state_dict(value['optimizers'][a])
    step(resumed, ropts)
    assert all(state_hash(models[a].adapter) == state_hash(resumed[a].adapter) for a in models)


def test_seed_shuffles_keep_same_places_and_are_paired(monkeypatch):
    from scripts import spatial_adapter_repeat as s
    base = dict(images=['same'], places=['same'], schedule=[['original']], train_rows=32)
    seen = []
    def get(c): seen.append(c['seed']); return base
    monkeypatch.setattr(s.key, 'open_plan', get)
    c = dict(seeds=[42, 43, 44], split_seed=42, train_places=8, epochs=2, places_per_batch=2)
    a, b = s.open_plan(c, 42), s.open_plan(c, 43)
    assert seen == [42, 42] and base['schedule'] == [['original']]
    assert a['images'] == b['images'] == base['images'] and a['schedule'] != b['schedule']
    assert s.open_plan(c, 43)['schedule'] == b['schedule']


def test_seed_statistics_do_not_hide_worse_seed():
    from scripts.eval_spatial_adapter_repeat import delta_summary
    r = delta_summary([.1, .2, -.1])
    assert r['positive_seeds'] == 2 and r['negative_seeds'] == 1 and not r['all_three_positive']
    assert r['mean_delta_r1_pp'] == pytest.approx(.2/3)
    with pytest.raises(ValueError): delta_summary([1., 2.])


def test_seals_refuse_modified_artifacts(tmp_path):
    from scripts.spatial_adapter_repeat import finish, verified
    from src.semantic_region_protocol import write
    write(tmp_path/'contract.json', dict(schema='spatial_adapter_repeat_v1')); write(tmp_path/'summary.json', dict(ok=True))
    finish(tmp_path); assert verified(tmp_path)['schema'] == 'spatial_adapter_repeat_v1'
    write(tmp_path/'summary.json', {})
    with pytest.raises(ValueError): verified(tmp_path)


def test_config_locked_budget_and_separate_outputs():
    from scripts.spatial_adapter_repeat import configuration
    c = configuration(Path(__file__).resolve().parents[1]/'config/spatial_adapter_repeat.yaml')
    assert c['seeds'] == [42, 43, 44] and c['epochs']*c['train_places']//c['places_per_batch'] == 2560
    assert c['runs'] == 'logs/spatial_adapter_repeat_v1' and c['cuda_memory_fraction'] <= .2
    assert not ({'clip_checkpoint', 'target_dim', 'adversarial_weight', 'predictor_lr'} & set(c))


def test_commands_require_explicit_update_eval_flags():
    script = Path(__file__).resolve().parents[1]/'scripts/spatial_adapter_repeat.py'
    for command in ('smoke', 'train', 'eval', 'aggregate'):
        r = subprocess.run([sys.executable, str(script), command], capture_output=True, text=True)
        assert r.returncode != 0
        assert ('--allow-training' if command in ('smoke', 'train') else '--evaluate') in r.stderr
