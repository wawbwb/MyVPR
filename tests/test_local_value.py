import copy
import numpy as np
import pytest
import torch
from torch import nn
from src.models.local_contrast_value import MODES, LocalValueAttention, neighborhood, install
from src.models.aggregators.boq import BoQ
from scripts.build_local_value_plan import distance, safe_pair, make_batches


def test_difference_removes_constant_and_boundary_artifacts():
    x = torch.ones(2, 5, 4, 4)
    w = torch.randn(8, 5)
    assert torch.count_nonzero(neighborhood(x, w, True)) == 0
    assert torch.allclose(neighborhood(x, w, False), w.sum(0)[None, :, None, None].expand_as(x), atol=1e-6)


def test_directional_neighbor_difference():
    x = torch.arange(9.).reshape(1, 1, 3, 3)
    w = torch.zeros(8, 1); w[0] = 1
    out = neighborhood(x, w, True)
    assert out[0, 0, 1, 1] == -4
    assert out[0, 0, 0, 0] == 0


@pytest.mark.parametrize('mode', MODES)
def test_perhead_value_formula_and_routing_unchanged(mode):
    torch.manual_seed(3)
    original = nn.MultiheadAttention(16, 4, batch_first=True, dropout=0).eval()
    module = LocalValueAttention(original, mode, rank=8, grid=3)
    q, x = torch.randn(2, 5, 16), torch.randn(2, 9, 16)
    base, attn = original(q, x, x)
    zero, actual_attn = module(q, x, x)
    assert torch.allclose(zero, base, atol=1e-7)
    assert torch.equal(actual_attn, attn)
    nn.init.normal_(module.output.weight, std=.03)
    u = module.projection(module.feature_norm(x))
    if mode == 'shuffled_contrast': u = u[:, module.permutation]
    local = neighborhood(u.transpose(1, 2).reshape(2, 8, 3, 3), module.kernel, mode != 'local_conv').flatten(2).transpose(1, 2)
    if mode == 'shuffled_contrast': local = local[:, module.inverse]
    dv = module.output(torch.nn.functional.gelu(local)).reshape(2, 9, 4, 4).transpose(1, 2)
    h, weights = original(q, x, x, average_attn_weights=False)
    expected = h+torch.nn.functional.linear((weights@dv).transpose(1, 2).reshape(2, 5, 16), original.out_proj.weight)
    actual, actual_attn = module(q, x, x)
    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.equal(actual_attn, attn)
    assert not torch.equal(actual, base)
    assert torch.equal(module.permutation[module.inverse], torch.arange(9))
    with pytest.raises(ValueError): module(q, x[:, :4], x[:, :4])


def test_equal_parameters_and_initialization():
    states = []
    for mode in MODES:
        torch.manual_seed(10)
        a = LocalValueAttention(nn.MultiheadAttention(16, 4, batch_first=True), mode, grid=3)
        states.append({k: p.detach().clone() for k, p in a.named_parameters() if p.requires_grad})
    assert states[0].keys() == states[1].keys() == states[2].keys()
    assert all(torch.equal(states[0][k], state[k]) for state in states[1:] for k in states[0])


@pytest.mark.parametrize('mode', MODES)
def test_two_updates_and_frozen_weights(mode):
    torch.manual_seed(4)
    agg = BoQ(in_channels=12, proj_channels=64, num_queries=4, num_layers=2, row_dim=2).eval().requires_grad_(False)
    x = torch.randn(2, 12, 3, 3)
    with torch.no_grad(): baseline = agg(x)[0]
    adapters = install(agg, mode, grid=3)
    assert torch.allclose(agg(x)[0], baseline, atol=1e-6)
    params = [p for p in agg.parameters() if p.requires_grad]
    names = {k for k,p in agg.named_parameters() if p.requires_grad}
    frozen = {k:v.clone() for k,v in agg.state_dict().items() if k not in names}
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0)
    target = torch.randn_like(baseline)
    for _ in range(2):
        opt.zero_grad(); (agg(x)[0]*target).sum().backward(); opt.step()
    for m in adapters:
        for p in (m.projection.weight, m.kernel, m.output.weight):
            assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        m.enabled = False
    assert torch.allclose(agg(x)[0], baseline, atol=1e-6)
    assert all(torch.equal(v, agg.state_dict()[k]) for k,v in frozen.items())


def test_geographic_filter_and_reproducible_hard_batches():
    centers = np.stack([np.arange(64)*.01, np.zeros(64)], axis=1)
    radii = np.zeros(64)
    pairs = [[i, i+1] for i in range(0, 32, 2)]
    assert distance([0, 0], [0, 0]) == 0
    assert 1100 < distance([0, 0], [.01, 0]) < 1120
    assert not safe_pair(0, 0, centers, radii)
    unsafe = radii.copy(); unsafe[0] = 1200
    assert not safe_pair(0, 1, centers, unsafe)
    batches, links = make_batches(pairs, centers, radii, 42, count=3)
    assert (batches, links) == make_batches(pairs, centers, radii, 42, count=3)
    for batch, planted in zip(batches, links):
        assert len(set(batch)) == 16 and len(planted) == 4
        assert all(i in batch and j in batch for i,j in planted)
        assert all(safe_pair(i,j,centers,radii) for n,i in enumerate(batch) for j in batch[n+1:])
    with pytest.raises(ValueError): make_batches([], centers, radii, 42, count=1)
