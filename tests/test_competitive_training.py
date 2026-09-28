from types import SimpleNamespace
import torch
from torch import nn
from src.models.aggregators.boq import BoQ
from src.models.competitive_query import install, MODES
from scripts.train_competitive_query import better


def test_dev_selection_includes_baseline_and_never_msls():
    baseline = dict(correct=100, margin_loss=.2)
    assert not better(dict(correct=99, margin_loss=.1), baseline)
    assert not better(dict(correct=100, margin_loss=.2), baseline)
    assert better(dict(correct=100, margin_loss=.1), baseline)


def test_three_arms_train_only_head_scalars_and_roundtrip():
    for mode in MODES:
        torch.manual_seed(2)
        agg = BoQ(in_channels=12, proj_channels=64, num_queries=4, num_layers=2, row_dim=2).eval()
        x = torch.randn(2, 12, 3, 3)
        with torch.no_grad(): expected = agg(x)[0]
        adapters = install(agg, mode)
        active = [p for p in agg.parameters() if p.requires_grad]
        assert sum(p.numel() for p in active) == sum(m.original.num_heads for m in adapters)
        frozen = {k: v.clone() for k, v in agg.state_dict().items() if not k.endswith('.strength')}
        assert torch.allclose(agg(x)[0], expected, atol=2e-6, rtol=0)
        opt = torch.optim.AdamW(active, lr=.01, weight_decay=0)
        for _ in range(2):
            opt.zero_grad()
            (agg(x)[0]*torch.randn_like(expected)).sum().backward()
            assert sum(p.grad.abs().sum() for p in active) > 0
            opt.step()
            for m in adapters: m.project_()
        assert all(torch.equal(v, agg.state_dict()[k]) for k, v in frozen.items())
        state = {k: v.clone() for k, v in agg.state_dict().items()}
        before = agg(x)[0].detach()
        agg.load_state_dict(state, strict=True)
        assert torch.equal(before, agg(x)[0])
