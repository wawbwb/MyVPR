import pytest
import torch
from torch import nn
from src.models.competitive_query import CompetitiveAttention, calibrate, redundancy_metrics, MODES


@pytest.mark.parametrize('mode', MODES)
def test_zero_start_exact_and_gradient(mode):
    torch.manual_seed(8)
    original = nn.MultiheadAttention(32, 4, batch_first=True).eval().requires_grad_(False)
    q, x = torch.randn(2, 5, 32), torch.randn(2, 13, 32)
    expected = original(q, x, x)[0].detach()
    module = CompetitiveAttention(original, mode)
    out, attention = module(q, x, x)
    assert torch.equal(out, expected)
    assert torch.allclose(attention.sum(-1), torch.ones(2, 5))
    (out * torch.randn_like(out)).sum().backward()
    assert module.strength.grad.abs().sum() > 0
    assert all(p.grad is None for p in original.parameters())


def test_projection_and_wrong_location_control():
    torch.manual_seed(11)
    logits = torch.randn(2, 3, 5, 13)
    strength = torch.ones(3)
    a = calibrate(logits, strength, 'competitive')
    b = calibrate(logits, strength, 'shuffled')
    assert not torch.allclose(a, b)
    assert torch.equal(b, calibrate(logits, strength, 'shuffled'))
    module = CompetitiveAttention(nn.MultiheadAttention(12, 3, batch_first=True), 'competitive')
    with torch.no_grad(): module.strength.copy_(torch.tensor([-1., .5, 9.]))
    module.project_()
    assert torch.equal(module.strength, torch.tensor([0., .5, 2.]))


def test_metric_extremes():
    repeated = torch.full((2, 3, 4, 4), .25)
    separate = torch.eye(4).expand(2, 3, -1, -1)
    slots = torch.eye(4).expand(2, -1, -1)
    r = redundancy_metrics(repeated, slots)
    s = redundancy_metrics(separate, slots)
    assert torch.allclose(r[:, 0], torch.ones(2))
    assert torch.allclose(s[:, 0], torch.zeros(2))
    assert torch.allclose(r[:, 1], torch.ones(2))
    assert torch.allclose(r[:, 2], torch.full((2,), 4.))
