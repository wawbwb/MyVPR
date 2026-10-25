from copy import deepcopy
import pytest
import torch
from torch.nn import functional as F
from src.models.scale_agreement_fusion import ARMS, ScaleAgreementFusion, agreement_features, mix
from scripts.scale_agreement_fusion import configuration, VARIANTS, shard_write, shard_ready


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)
    torch.set_num_threads(2)


def sample(n=4):
    return [F.normalize(torch.randn(n, 60), dim=1) for _ in range(2)]


@pytest.mark.parametrize('arm', ARMS)
def test_zero_at_average_and_parameter_capacity(arm):
    low, high = sample()
    m = ScaleAgreementFusion(arm, channels=12, groups=5)
    z, gate = m(low, high, [1, 2, 3, 4])
    assert torch.equal(z, F.normalize(low+high, dim=1))
    assert not gate.any()
    assert sum(p.numel() for p in m.parameters()) == 128


def test_statistic_and_mixing_layout():
    low, high = sample()
    l, h = low.reshape(4, 12, 5), high.reshape(4, 12, 5)
    features = agreement_features(l, h)
    assert features.shape == (4, 5, 6)
    torch.testing.assert_close(features[:, :, -1], (low*high).sum(1, keepdim=True).expand(-1, 5))
    g = torch.zeros(4, 5)
    g[:, 2] = .5
    expected = l+h
    expected[:, :, 2] = .5*l[:, :, 2]+1.5*h[:, :, 2]
    torch.testing.assert_close(mix(low, high, g, 12, 5), F.normalize(expected.flatten(1), dim=1))
    with pytest.raises(ValueError):
        mix(low, high, g+1, 12, 5)
    with pytest.raises(ValueError):
        mix(low, -low, torch.zeros_like(g), 12, 5)


def test_gate_routing_batch_independence_and_fixed_control():
    low, high = sample()
    states = ScaleAgreementFusion('group_gate', 12, 5)
    torch.nn.init.normal_(states.up.weight, std=.7)
    all_models = {}
    for arm in ARMS:
        m = ScaleAgreementFusion(arm, 12, 5)
        m.load_state_dict(states.state_dict())
        all_models[arm] = m
    z, real = all_models['group_gate'](low, high, [1, 2, 3, 4])
    y, shuffled = all_models['shuffled_gate'](low, high, [1, 2, 3, 4])
    torch.testing.assert_close(real.sort(1).values, shuffled.sort(1).values)
    assert not torch.allclose(z, y)
    for m in all_models.values():
        together, gate = m(low, high, [1, 2, 3, 4])
        apart = torch.cat([m(low[i:i+1], high[i:i+1], [i+1])[0] for i in range(4)])
        torch.testing.assert_close(together, apart)
        assert gate.abs().max() <= .5
        torch.testing.assert_close(m(low, low, [1, 2, 3, 4])[0], low)
    static = all_models['static_gate'].gates(low, high)
    assert torch.equal(static, static[:1].expand_as(static))
    global_gate = all_models['global_gate'].gates(low, high)
    assert torch.equal(global_gate, global_gate[:, :1].expand_as(global_gate))


@pytest.mark.parametrize('arm', ARMS)
def test_gradients_and_optimizer_resume(arm):
    low, high = sample()
    target = F.normalize(torch.randn_like(low), dim=1)
    m = ScaleAgreementFusion(arm, 12, 5)
    opt = torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=0.)
    seen = {n: False for n, p in m.named_parameters()}

    def update(a, o):
        o.zero_grad()
        (a(low, high, [1, 2, 3, 4])[0]-target).square().sum().backward()
        o.step()

    for _ in range(4):
        update(m, opt)
        for n, p in m.named_parameters():
            seen[n] |= p.grad is not None and bool((p.grad != 0).any())
    assert all(seen.values())
    other = deepcopy(m)
    other_opt = torch.optim.AdamW(other.parameters(), lr=.001, weight_decay=0.)
    other_opt.load_state_dict(deepcopy(opt.state_dict()))
    update(m, opt)
    update(other, other_opt)
    for p, q in zip(m.parameters(), other.parameters()):
        assert torch.equal(p, q)


def test_cache_corruption_detection(tmp_path):
    import numpy as np
    shard_write(tmp_path, 0, high=np.zeros((2, 3), np.float32))
    assert shard_ready(tmp_path, 0) and not shard_ready(tmp_path, 1)
    with (tmp_path/'0000.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='changed'):
        shard_ready(tmp_path, 0)


def test_registered_protocol():
    c = configuration()
    assert c['pilot_steps'] == 1024 and c['smoke_steps'] == 8 and c['learning_rate'] == .001
    assert len(VARIANTS) == 8 and {'descriptor_average', 'train_mean_gate', 'group_gate'} <= set(VARIANTS)
