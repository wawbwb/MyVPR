import pytest
import torch
from torch import nn
from src.discriminative_subspace import scatter, fit, shuffle_places, ratio, overlap, assess
from src.models.discriminative_subspace import SubspaceQKV


def test_scatter_decomposition():
    torch.manual_seed(1)
    x = torch.randn(20, 4, 8, dtype=torch.float64)
    sw,sb = scatter(x)
    flat = x.flatten(0,1); centered = flat-flat.mean(0)
    assert torch.allclose(sw+sb,centered.T@centered/len(flat))


def test_fisher_recovers_stable_place_axis_on_new_places():
    def generate(seed):
        torch.manual_seed(seed)
        means = torch.randn(100,1,1,dtype=torch.float64)*3
        return torch.cat((means+torch.randn(100,4,1,dtype=torch.float64)*.1, torch.randn(100,4,3,dtype=torch.float64)*5),dim=-1)
    x,dev = generate(2),generate(3)
    u,_ = fit(x,rank=1)
    p,_ = fit(x,rank=1,fisher=False)
    assert float(u[0].square()) > .95
    assert ratio(dev,u)['ratio'] > ratio(dev,p)['ratio']
    assert overlap(u,u) == pytest.approx(1)


def test_shuffle_preserves_values_and_sizes():
    x = torch.arange(64.).reshape(4,4,4)
    a = shuffle_places(x)
    assert a.shape == x.shape and torch.equal(a,shuffle_places(x))
    assert not torch.equal(a,x)
    assert torch.equal(a.flatten().sort().values,x.flatten().sort().values)


def test_gate_requires_both_controls_and_retained_signal():
    metrics = {'place_fisher':dict(ratio=2.,between=1.),'pca':dict(ratio=1.,between=2.),'shuffled_fisher':dict(ratio=1.,between=1.)}
    assert assess(metrics,.5)['pass_gate']
    assert not assess(metrics,.99)['pass_gate']
    metrics['place_fisher']['between']=.1
    assert not assess(metrics,.5)['pass_gate']


def test_qkv_zero_start_keys_fixed_and_gradients():
    torch.manual_seed(8)
    original=nn.Linear(8,24)
    u=torch.linalg.qr(torch.randn(8,3)).Q
    adapter=SubspaceQKV(original,u)
    x=torch.randn(2,5,8)
    expected=original(x).detach()
    assert torch.equal(adapter(x),expected)
    target=torch.randn_like(expected)
    (adapter(x)*target).sum().backward()
    assert adapter.query.weight.grad.abs().sum()>0
    assert adapter.value.weight.grad.abs().sum()>0
    assert original.weight.grad is None and adapter.basis.grad is None
    with torch.no_grad(): adapter.query.weight.add_(.01);adapter.value.weight.add_(.02)
    result=adapter(x)
    assert torch.equal(result[...,8:16],expected[...,8:16])
    assert not torch.equal(result[...,:8],expected[...,:8])
    adapter.enabled=False
    assert torch.equal(adapter(x),expected)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError): fit(torch.zeros(4,4,8),rank=2)
    with pytest.raises(ValueError): SubspaceQKV(nn.Linear(8,24),torch.ones(8,3))
