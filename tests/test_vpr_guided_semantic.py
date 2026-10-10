"""Run on training machine, not the development PC; no downloads required."""
import copy

import pytest
import torch
from torch import nn

from src.models.seg_aux import SegAuxVPR
from src.models.semantic_region_gate import SemanticRegionGate
from src.models.vpr_guided_attention import VPRGuidedAttention


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_channels = 4
        self.dino = nn.Module()
        self.dino.blocks = nn.ModuleList([nn.Conv2d(4, 4, 1) for _ in range(3)])

    def forward(self, x):
        for block in self.dino.blocks:
            x = block(x)
        return x


class Pool(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x):
        return nn.functional.normalize(self.proj(x.mean((2, 3))), dim=1)


def make_model(mode):
    config = {'seg_aux': dict(mode=mode, lr=2e-6, head_lr=1e-4,
        weight_decay=1e-3, lambda_seg=0.02, diagnostic_interval=100,
        min_confidence=0.5, unfrozen_blocks=2, vpr_guidance=True,
        attention_lr=1e-5, attention_hidden=4)}
    return SegAuxVPR(Backbone(), Pool(), SemanticRegionGate(4), nn.Identity(), config)


def test_attention_unit_start_and_spatial_output():
    attention = VPRGuidedAttention(4, 4)
    weights = attention(torch.randn(2, 4, 7, 9))
    assert weights.shape == (2, 1, 7, 9)
    assert torch.equal(weights, torch.ones_like(weights))


@pytest.mark.parametrize('mode', ['vpr_only', 'plain', 'guided'])
def test_exact_ru_start_optimizer_coverage_and_no_inference_segmentation(mode):
    model = make_model(mode).eval()
    images = torch.randn(2, 4, 5, 5)
    features = model.backbone(images)
    baseline = model.aggregator(model.semantic_region_gate(features)[0])
    torch.testing.assert_close(model(images), baseline, rtol=0, atol=0)
    ids = [id(p) for g in model._optimizer_param_groups() for p in g['params']]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    before = model(images).detach()
    with torch.no_grad():
        for p in model.seg_head.parameters():
            p.add_(100)
    torch.testing.assert_close(model(images), before, rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['plain', 'guided'])
def test_semantic_reaches_encoder_but_not_attention(mode):
    model = make_model(mode)
    with torch.no_grad():
        model.vpr_guidance.output.weight.normal_(0, 0.2)
    logits = model.semantic_logits(model.backbone(torch.randn(2, 4, 5, 5)))
    nn.functional.cross_entropy(logits, torch.zeros(2, 5, 5, dtype=torch.long)).backward()
    assert model.backbone.dino.blocks[-1].weight.grad.abs().sum() > 0
    assert model.backbone.dino.blocks[0].weight.grad is None
    assert all(p.grad is None for p in model.vpr_guidance.parameters())
    assert all(p.grad is None for p in model.semantic_region_gate.parameters())


@pytest.mark.parametrize('mode', ['vpr_only', 'plain', 'guided'])
def test_retrieval_gradient_trains_attention(mode):
    model = make_model(mode)
    descriptor = model(torch.randn(3, 4, 5, 5))
    descriptor[:, 0].sum().backward()
    assert model.vpr_guidance.output.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.seg_head.parameters())


def test_guided_uses_attention_plain_does_not():
    model = make_model('guided')
    features = torch.randn(2, 4, 5, 5)
    with torch.no_grad():
        model.vpr_guidance.output.bias.fill_(2)
    guided = model.semantic_logits(features)
    expected = model.seg_head(features * model.vpr_guidance(features).detach())
    torch.testing.assert_close(guided, expected)
    model.mode = 'plain'
    torch.testing.assert_close(model.semantic_logits(features), model.seg_head(features))
    assert not torch.allclose(guided, model.semantic_logits(features))


@pytest.mark.parametrize('mode', ['plain', 'guided'])
def test_prewarm_does_not_modify_retrieval_attention(tmp_path, mode):
    from types import SimpleNamespace
    from scripts.train_seg_aux import prewarm_head
    model = make_model(mode)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    batch = (torch.randn(2, 2, 4, 5, 5), torch.zeros(2, 2), {
        'query_semantic_labels': torch.zeros(2, 2, 5, 5, dtype=torch.long),
        'query_semantic_confidence': torch.ones(2, 2, 5, 5),
        'query_semantic_cache_indices': torch.arange(4).reshape(2, 2)})
    dm = SimpleNamespace(setup=lambda stage: None, train_dataloader=lambda: [batch])
    cfg = {'seed': 42, 'seg_aux': dict(model.hparams['seg_aux'], head_warmup_steps=2)}
    prewarm_head(model, dm, cfg, tmp_path, torch.device('cpu'), smoke=True)
    for n, p in model.named_parameters():
        if not n.startswith('seg_head.'):
            assert torch.equal(before[n], p)


def test_checkpoint_loader_retains_learned_attention_and_rejects_missing_weights(tmp_path, monkeypatch):
    from scripts import eval_condition_robustness as evaluation
    model = make_model('guided').eval()
    cfg = copy.deepcopy(dict(model.hparams))
    cfg['backbone'] = dict(module='tiny', **{'class': 'Backbone'}, params={})
    cfg['aggregator'] = dict(module='tiny', **{'class': 'Pool'}, params={})
    cfg['distillation'] = {'semantic_region': dict(enabled=True,
        apply_pretrained_gate=True, alpha=0.2, lambda_target=0)}
    monkeypatch.setattr(evaluation, 'get_instance',
                        lambda module, name, params: Backbone() if name == 'Backbone' else Pool())
    with torch.no_grad():
        model.vpr_guidance.output.weight.normal_(0, 0.2)
    path = tmp_path / 'model.ckpt'
    state = model.state_dict()
    torch.save(dict(hyper_parameters=cfg, state_dict=state), path)
    restored = evaluation.load_inference_model_from_ckpt(path, 'cpu')
    images = torch.randn(2, 4, 5, 5)
    torch.testing.assert_close(model(images), restored(images), rtol=0, atol=0)
    del state['vpr_guidance.output.weight']
    torch.save(dict(hyper_parameters=cfg, state_dict=state), path)
    with pytest.raises(RuntimeError):
        evaluation.load_inference_model_from_ckpt(path, 'cpu')
