import pytest
import torch
from torch import nn
from src.models.second_order_query import MODES, moments, compact_moment, install
from src.models.aggregators.boq import BoQ


def test_same_mean_different_covariance():
    a = torch.tensor([[[-1., 0.], [1., 0.], [0., -2.], [0., 2.]]])
    b = a.flip(-1)
    weights = torch.full((1, 1, 4), .25)
    assert torch.equal(moments(a, weights, 'mean_outer'), moments(b, weights, 'mean_outer'))
    assert not torch.allclose(compact_moment(moments(a, weights, 'query_cov')),
                              compact_moment(moments(b, weights, 'query_cov')))


def test_covariance_matches_explicit_centering_and_global_repeat():
    torch.manual_seed(3)
    x = torch.randn(2, 7, 4)
    w = torch.randn(2, 3, 7).softmax(-1)
    mean = w @ x
    centered = x[:, None] - mean[:, :, None]
    expected = torch.einsum('bqn,bqni,bqnj->bqij', w, centered, centered)
    assert torch.allclose(moments(x, w, 'query_cov'), expected, atol=1e-6)
    global_cov = moments(x, w, 'global_cov')
    assert torch.equal(global_cov[:, 0], global_cov[:, 1])
    assert not torch.allclose(expected[:, 0], expected[:, 1])


def test_compact_zero_finite_gradient():
    x = torch.zeros(2, 3, 16, 16, requires_grad=True)
    y = compact_moment(x)
    assert y.shape == (2, 3, 136)
    y.sum().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()


def test_controls_equal_parameters_and_initialization():
    states = []
    for mode in MODES:
        torch.manual_seed(8)
        agg = BoQ(in_channels=12, proj_channels=64, num_queries=4, num_layers=2, row_dim=2)
        install(agg, mode)
        states.append({k:p.detach().clone() for k,p in agg.named_parameters() if p.requires_grad})
    assert states[0].keys() == states[1].keys() == states[2].keys()
    assert all(torch.equal(states[0][k], s[k]) for s in states[1:] for k in states[0])


@pytest.mark.parametrize('mode', MODES)
def test_baseline_equal_frozen_and_two_step_gradients(mode):
    torch.manual_seed(4)
    agg = BoQ(in_channels=12, proj_channels=64, num_queries=4, num_layers=2, row_dim=2).eval().requires_grad_(False)
    x = torch.randn(2, 12, 3, 3)
    with torch.no_grad(): baseline = agg(x)[0]
    adapters = install(agg, mode)
    assert torch.equal(agg(x)[0], baseline)
    params = [p for p in agg.parameters() if p.requires_grad]
    frozen = {k:v.clone() for k,v in agg.state_dict().items() if k not in {n for n,p in agg.named_parameters() if p.requires_grad}}
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0)
    target = torch.randn_like(baseline)
    for _ in range(2):
        opt.zero_grad()
        (agg(x)[0]*target).sum().backward()
        opt.step()
    for m in adapters:
        assert m.projection.weight.grad.abs().sum() > 0
        assert m.output.weight.grad.abs().sum() > 0
        m.enabled = False
    assert torch.equal(agg(x)[0], baseline)
    assert all(torch.equal(v, agg.state_dict()[k]) for k,v in frozen.items())
