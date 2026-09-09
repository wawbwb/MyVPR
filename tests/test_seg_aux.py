"""CPU unit tests for the training machine; no external weights/downloads."""
import torch
from torch import nn

from src.models.seg_aux import SegAuxVPR
from src.models.semantic_region_gate import SemanticRegionGate


class TinyBackbone(nn.Module):
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
        return torch.nn.functional.normalize(self.proj(x.mean((2, 3))), dim=1)


def make_model(mode='aligned'):
    config = {'seg_aux': dict(mode=mode, lr=2e-6, head_lr=1e-4,
                             weight_decay=1e-3, lambda_seg=0.02,
                             diagnostic_interval=100, min_confidence=0.5,
                             unfrozen_blocks=2)}
    return SegAuxVPR(TinyBackbone(), Pool(), SemanticRegionGate(4), nn.Identity(), config)


def test_semantic_gradient_reaches_shared_blocks_not_frozen_prefix():
    model = make_model()
    features, _ = model.features_and_descriptor(torch.randn(2, 4, 3, 3))
    logits = model.seg_head(features)
    loss, _ = model.target(logits, torch.zeros(2, 3, 3, dtype=torch.long),
                           torch.ones(2, 3, 3), torch.arange(2))
    loss.backward()
    assert model.backbone.dino.blocks[0].weight.grad is None
    assert model.backbone.dino.blocks[-1].weight.grad.abs().sum() > 0
    assert model.seg_head[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.semantic_region_gate.parameters())


def test_head_does_not_affect_inference_and_all_trainable_params_optimized():
    model = make_model()
    images = torch.randn(2, 4, 3, 3)
    before = model(images).detach()
    with torch.no_grad():
        for p in model.seg_head.parameters():
            p.add_(100)
    torch.testing.assert_close(before, model(images))
    optimized = [id(p) for g in model._optimizer_param_groups() for p in g['params']]
    assert len(optimized) == len(set(optimized))
    assert set(optimized) == {id(p) for p in model.parameters() if p.requires_grad}


def test_vpr_only_has_no_trainable_semantic_head():
    model = make_model('vpr_only')
    assert model.seg_weight == 0
    assert not any(p.requires_grad for p in model.seg_head.parameters())


def test_semantic_ramp_and_legacy_weight():
    model = make_model()
    assert model.semantic_weight_at_step(0) == 0.02
    model.hparams['seg_aux']['seg_ramp_steps'] = 1000
    assert abs(model.semantic_weight_at_step(0) - 0.00002) < 1e-12
    assert model.semantic_weight_at_step(499) == 0.01
    assert model.semantic_weight_at_step(999) == 0.02
    assert model.semantic_weight_at_step(2000) == 0.02
    model.seg_weight = 0.0
    assert model.semantic_weight_at_step(500) == 0.0


def test_head_prewarm_preserves_retrieval_weights(tmp_path):
    from types import SimpleNamespace
    from scripts.train_seg_aux import prewarm_head
    model = make_model()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    batch = (torch.randn(2, 2, 4, 3, 3), torch.zeros(2, 2), {
        'query_semantic_labels': torch.zeros(2, 2, 3, 3, dtype=torch.long),
        'query_semantic_confidence': torch.ones(2, 2, 3, 3),
        'query_semantic_cache_indices': torch.arange(4).reshape(2, 2)})
    dm = SimpleNamespace(setup=lambda stage: None, train_dataloader=lambda: [batch])
    cfg = {'seed': 42, 'seg_aux': dict(model.hparams['seg_aux'], head_warmup_steps=2)}
    prewarm_head(model, dm, cfg, tmp_path, torch.device('cpu'), smoke=True)
    for n, p in model.named_parameters():
        if not n.startswith('seg_head.'):
            assert torch.equal(before[n], p)
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters()
               if n.startswith('seg_head.'))


def test_empty_confidence_is_finite_and_differentiable():
    model = make_model()
    logits = torch.randn(2, 150, 3, 3, requires_grad=True)
    loss, _ = model.target(logits, torch.zeros(2, 3, 3, dtype=torch.long),
                           torch.zeros(2, 3, 3), torch.arange(2))
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(logits.grad).all()
