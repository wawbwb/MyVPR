from copy import deepcopy
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from src.models.scale_reliability_gate import ARMS, ScaleReliabilityGate, centered, cross_view_teacher
from src.models.scale_agreement_fusion import mix
from src.models.complementary_covariance import permute_positions
from scripts.scale_reliability_gate import configuration, VARIANTS, PRIMARY


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(43)
    torch.set_num_threads(2)


def sample(n=6):
    return [F.normalize(torch.randn(n, 60), dim=1) for _ in range(2)]


@pytest.mark.parametrize('arm', ARMS)
def test_zero_reproduces_frozen_scalar_with_matched_capacity(arm):
    low, high = sample()
    model = ScaleReliabilityGate(arm, .38, 12, 5)
    z, g = model(low, high, list(range(6)))
    assert sum(p.numel() for p in model.parameters()) == 128
    assert torch.equal(z, mix(low, high, torch.full((6, 5), .38), 12, 5))
    assert torch.equal(g, torch.full((6, 5), .38))


def test_bounds_centering_and_inference_batch_independence():
    low, high = sample()
    for arm in ARMS:
        m = ScaleReliabilityGate(arm, .38, 12, 5)
        torch.nn.init.normal_(m.up.weight, std=10)
        together, g = m(low, high, list(range(6)))
        separate = torch.cat([m(low[i:i+1], high[i:i+1], [i])[0] for i in range(6)])
        torch.testing.assert_close(together, separate, rtol=1e-5, atol=1e-6)
        assert g.abs().max() <= .500001
        if arm != 'quality_global':
            torch.testing.assert_close(g.mean(1), torch.full((6,), .38), rtol=0, atol=1e-7)
        if arm == 'quality_global':
            assert torch.equal(g, g[:, :1].expand_as(g))
        if arm == 'quality_static':
            assert torch.equal(g, g[:1].expand_as(g))


def test_same_weights_shuffle_preserves_actual_gate_value_multiset():
    low, high = sample()
    group = ScaleReliabilityGate('quality_group', .38, 12, 5)
    torch.nn.init.normal_(group.up.weight, std=2)
    shuffled = ScaleReliabilityGate('quality_shuffled', .38, 12, 5)
    shuffled.load_state_dict(group.state_dict())
    g = group.gates(low, high)
    p = permute_positions(g[:, :, None], list(range(6))).squeeze(-1)
    torch.testing.assert_close(p.sort(1).values, g.sort(1).values, rtol=0, atol=0)
    torch.testing.assert_close(p, shuffled.gates(low, high, list(range(6))))
    assert not torch.allclose(mix(low, high, g, 12, 5), mix(low, high, p, 12, 5))


def test_teacher_excludes_self_has_common_negative_and_correct_direction():
    # Two places, two views; high is invariant/discriminative, low conflicts.
    high = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    low = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
    labels = torch.tensor([0, 0, 1, 1])
    coordinates = torch.tensor([[0., 0.], [0., 0.], [0., 1.], [0., 1.]])
    low.requires_grad_()
    target, valid, info = cross_view_teacher(low, high, labels, coordinates, 2, 1)
    assert valid.all() and (target > .9).all() and not target.requires_grad
    assert info['positive_pairs'] == 4 and info['safe_negative_pairs'] == 8
    assert all(labels[i] != labels[j] for i, j in enumerate(info['selected_negative_rows']))
    target0, _, _ = cross_view_teacher(high, high, labels, coordinates, 2, 1)
    assert not target0.any()
    with pytest.raises(ValueError, match='No valid'):
        cross_view_teacher(low, high, torch.arange(4), coordinates, 2, 1)


def test_teacher_gps_filter_and_batch_permutation_equivariance():
    low, high = sample()
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    coords = torch.tensor([[0., 0.], [0., 0.], [0., .000001], [0., .000001], [1., 1.], [1., 1.]])
    target, valid, info = cross_view_teacher(low, high, labels, coords, 12, 5)
    assert info['selected_negative_rows'][:4] == [j for j in info['selected_negative_rows'][:4] if j in (4, 5)]
    assert info['safe_negative_pairs'] == 16
    order = torch.tensor([2, 5, 0, 4, 1, 3])
    other, v, _ = cross_view_teacher(low[order], high[order], labels[order], coords[order], 12, 5)
    torch.testing.assert_close(other, target[order])
    assert torch.equal(v, valid[order])


@pytest.mark.parametrize('arm', ARMS)
def test_quality_training_gradients_and_exact_optimizer_resume(arm):
    low, high = sample()
    target = torch.randn(6, 5).tanh()
    m = ScaleReliabilityGate(arm, .38, 12, 5)
    opt = torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=0)
    seen = {n: False for n, p in m.named_parameters()}

    def update(model, optimizer):
        optimizer.zero_grad()
        z, _ = model(low, high, list(range(6)))
        loss = (z-low).square().sum()+.1*F.mse_loss(model.scores(low, high, list(range(6))), target)
        loss.backward()
        optimizer.step()

    for _ in range(4):
        update(m, opt)
        for n, p in m.named_parameters():
            seen[n] |= p.grad is not None and bool((p.grad != 0).any())
    assert all(seen.values())
    other = deepcopy(m)
    other_opt = torch.optim.AdamW(other.parameters(), lr=.001, weight_decay=0)
    other_opt.load_state_dict(deepcopy(opt.state_dict()))
    update(m, opt)
    update(other, other_opt)
    for p, q in zip(m.parameters(), other.parameters()):
        assert torch.equal(p, q)


def test_fixed_design_and_zero_mean():
    c = configuration()
    assert c['seeds'] == [43, 44, 45] and c['pilot_steps'] == 1024
    assert c['quality_weight'] == c['teacher_temperature'] == .1
    assert len(VARIANTS) == 16 and len(PRIMARY) == 6
    x = torch.rand(6, 5)*2-1
    assert centered(x).abs().max() <= 1
    torch.testing.assert_close(centered(x).mean(1), torch.zeros(6), rtol=0, atol=1e-7)
