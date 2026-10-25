from copy import deepcopy
import hashlib
import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.models.cross_resolution_detail import (ARMS, HIGH_SIZE, DetailAdapter,
    CrossResolutionVPR, scale_difference, normalize_map, degraded_high)
from scripts.cross_resolution_detail import configuration, replay, ScaleImages, VARIANTS
from scripts.semantic_region_study import transform


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)
    torch.set_num_threads(2)


def test_area_difference_and_scale_contract():
    x, h = torch.randn(2, 12, 3, 4), torch.randn(2, 12, 6, 8)
    expected = normalize_map(F.avg_pool2d(h, 2))-normalize_map(x)
    torch.testing.assert_close(scale_difference(x, h), expected)
    with pytest.raises(ValueError):
        scale_difference(x, x)


@pytest.mark.parametrize('arm', ARMS)
def test_zero_start_real_bound_and_null_difference(arm):
    m = DetailAdapter(arm, channels=12, hidden=8)
    x, d, e = [torch.randn(3, 12, 3, 5) for _ in range(3)]
    assert torch.equal(m(x, d, e, [1, 2, 3])[0], x)
    nn.init.normal_(m.up.weight, std=10)
    y, s = m(x, d, e, [1, 2, 3])
    assert s['relative_delta'].max() <= .025001
    assert ((y-x).norm(dim=1)/x.norm(dim=1)).max() <= .025001
    z = torch.zeros_like(x)
    assert torch.equal(m(x, z, z, [1, 2, 3])[0], x)
    assert torch.equal(m(z, d, e, [1, 2, 3])[0], z)
    assert sum(p.numel() for p in DetailAdapter(arm).parameters()) == 49152


def test_shuffled_correspondence_not_distribution_and_batch_independence():
    a = DetailAdapter('aligned_detail', 12, 8)
    nn.init.normal_(a.up.weight, std=.1)
    b = DetailAdapter('shuffled_detail', 12, 8)
    b.load_state_dict(a.state_dict())
    x, d, e = [torch.randn(3, 12, 3, 5) for _ in range(3)]
    y, s = a(x, d, e, [1, 2, 3])
    v, t = b(x, d, e, [1, 2, 3])
    assert not torch.allclose(y, v)
    torch.testing.assert_close(s['difference_rms'], t['difference_rms'])
    torch.testing.assert_close(v, torch.cat([b(x[i:i+1], d[i:i+1], e[i:i+1], [i+1])[0] for i in range(3)]))
    c = DetailAdapter('degraded_detail', 12, 8)
    c.load_state_dict(a.state_dict())
    assert not torch.allclose(c(x, d, e, [1, 2, 3])[0], y)


class Anchor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(12, 16)

    def readout(self, x):
        return F.normalize(self.fc(x.mean((2, 3))), dim=1)


@pytest.mark.parametrize('arm', ARMS)
def test_replay_freezing_trainability_and_optimizer_resume(arm):
    a = CrossResolutionVPR(Anchor(), arm, channels=12, hidden=8)
    nn.init.normal_(a.adapter.up.weight, std=.05)
    b = deepcopy(a)
    x, d, e = [torch.randn(5, 12, 3, 5) for _ in range(3)]
    f = (x, torch.randn(5, 16), d, e, torch.randn(5, 16))
    z, _ = a.describe(f, [1, 2, 3, 4, 5])
    g = torch.randn_like(z)
    z.backward(g)
    replay(b, [tuple(t[i:i+2] for t in f) for i in range(0, 5, 2)], [1, 2, 3, 4, 5], g)
    for p, q in zip(a.adapter.parameters(), b.adapter.parameters()):
        assert p.grad is not None and p.grad.abs().max() > 0
        torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=1e-4)
    assert all(p.grad is None for p in a.anchor.parameters())
    m = DetailAdapter(arm, 12, 8)
    opt = torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=0)
    seen = {n: False for n, p in m.named_parameters()}

    def update(v, o):
        o.zero_grad()
        (v(x, d, e, [1, 2, 3, 4, 5])[0]-d).square().mean().backward()
        o.step()

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


def test_raw_multiscale_inputs_preserve_low_and_verify_bytes(tmp_path):
    data = np.random.default_rng(42).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    image_path = tmp_path/'image.png'
    Image.fromarray(data).save(image_path)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    dataset = ScaleImages(tmp_path, ['image.png'], [digest])
    lo, hi, i, h, native = dataset[0]
    assert native.tolist() == [640, 480] and i == 0 and h == digest
    assert torch.equal(lo, transform(322)(Image.fromarray(data)))
    assert torch.equal(hi, transform(518)(Image.fromarray(data)))
    fake = degraded_high(lo[None])[0]
    assert fake.shape == hi.shape and not torch.allclose(fake, hi)
    image_path.write_bytes(image_path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='changed'):
        dataset[0]


def test_locked_protocol_and_simple_alternatives():
    c = configuration()
    assert c['pilot_steps'] == 1024 and c['learning_rate'] == .0001 and c['keep_weight'] == 0
    assert c['microbatch'] == 2 and HIGH_SIZE == 518 and len(VARIANTS) == 6
    assert {'innovation1_518', 'descriptor_average', 'aligned_detail'} <= set(VARIANTS)
