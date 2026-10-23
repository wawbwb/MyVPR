from copy import deepcopy
import pytest
import torch
from torch.nn import functional as F
from src.models.moment_channel_adapter import ARMS, MomentChannelAdapter, MomentChannelVPR, preserve_moments
from src.models.structural_channel_adapter import StructuralChannelAdapter
from scripts.moment_channel import replay, configuration, check_moments


@pytest.fixture(autouse=True)
def seed():
    torch.set_num_threads(2)
    torch.manual_seed(42)


@pytest.mark.parametrize('arm', ARMS)
def test_exact_zero_start(arm):
    x=torch.randn(3,12,5,5)
    y,s=MomentChannelAdapter(arm,channels=12,hidden=8)(x)
    assert torch.equal(x,y)
    assert not s['relative_delta'].any()


@pytest.mark.parametrize('arm', ARMS)
def test_nonzero_moments_and_batch_independence(arm):
    x=torch.randn(3,12,5,5)+2
    r=torch.randn_like(x)*.3
    y=preserve_moments(x,r,arm)
    assert not torch.equal(x,y)
    torch.testing.assert_close(y,torch.cat([preserve_moments(x[i:i+1],r[i:i+1],arm) for i in range(3)]))
    if arm in ('mean','mean_norm'):torch.testing.assert_close(y.mean(1),x.mean(1),rtol=1e-6,atol=1e-6)
    if arm in ('norm','mean_norm'):torch.testing.assert_close(y.norm(dim=1),x.norm(dim=1),rtol=1e-6,atol=1e-6)


def test_plain_reproduces_existing_initialization_and_forward():
    torch.manual_seed(42);a=MomentChannelAdapter('channel',channels=12,hidden=8)
    torch.manual_seed(42);b=StructuralChannelAdapter('channel',channels=12,hidden=8)
    assert list(a.state_dict())==list(b.state_dict())
    for n,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[n])
    torch.nn.init.normal_(a.up.weight,std=.02);b.load_state_dict(a.state_dict())
    x=torch.randn(3,12,5,5)
    assert torch.equal(a(x)[0],b(x)[0])


def test_factorial_capacity_and_initialization():
    states=[]
    for arm in ARMS:
        torch.manual_seed(42);m=MomentChannelAdapter(arm)
        assert sum(p.numel() for p in m.parameters())==99264
        states.append(m.state_dict())
    for state in states[1:]:
        for n,v in state.items():assert torch.equal(v,states[0][n])


@pytest.mark.parametrize('arm', ('norm','mean_norm'))
def test_degenerate_and_cancelling_tokens_finite(arm):
    for x in (torch.zeros(2,12,3,3),torch.ones(2,12,3,3),torch.randn(2,12,3,3)):
        r=(-x).requires_grad_()
        y=preserve_moments(x,r,arm)
        assert torch.equal(y,x)
        y.sum().backward()
        assert torch.isfinite(r.grad).all()


def test_per_token_not_spatial_centering():
    x=torch.randn(2,12,4,4)
    r=torch.randn(2,12,1,1).expand_as(x)
    y=preserve_moments(x,r,'mean')
    assert not torch.equal(y,x)
    torch.testing.assert_close(y.mean(1),x.mean(1),atol=2e-7,rtol=1e-5)


class TinyAnchor(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layer=torch.nn.Linear(12,16)
    def readout(self,x):return F.normalize(self.layer(x.mean((2,3))),dim=1)


@pytest.mark.parametrize('arm',ARMS)
def test_gradient_replay_matches_full_batch(arm):
    a=MomentChannelVPR(TinyAnchor(),arm,channels=12,hidden=8)
    torch.nn.init.normal_(a.adapter.up.weight,std=.02)
    b=deepcopy(a);f=(torch.randn(5,12,5,5),torch.randn(5,16))
    z,_=a.describe(f,None);g=torch.randn_like(z);z.backward(g)
    replay(b,[tuple(t[i:i+2] for t in f) for i in range(0,5,2)],[1,2,3,4,5],g)
    for p,q in zip(a.adapter.parameters(),b.adapter.parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0
        torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=3e-4)
    assert all(p.grad is None for p in a.anchor.parameters())
    a.train();assert not a.anchor.training


@pytest.mark.parametrize('arm',ARMS)
def test_trainability_from_zero_and_optimizer_resume(arm):
    m=MomentChannelAdapter(arm,channels=12,hidden=8)
    opt=torch.optim.AdamW(m.parameters(),lr=.001)
    x=torch.randn(3,12,5,5);target=torch.randn_like(x)
    seen={n:False for n,p in m.named_parameters()}
    def update(model,optimizer):
        optimizer.zero_grad();y,s=model(x);(y-target).square().mean().backward()
        check_moments(arm,float(s['mean_error']),float(s['norm_error']))
        optimizer.step()
    for _ in range(4):
        update(m,opt)
        for n,p in m.named_parameters():seen[n]|=p.grad is not None and bool(p.grad.abs().max()>0)
    assert all(seen.values())
    b=deepcopy(m);ob=torch.optim.AdamW(b.parameters(),lr=.001);ob.load_state_dict(deepcopy(opt.state_dict()))
    update(m,opt);update(b,ob)
    for p,q in zip(m.parameters(),b.parameters()):assert torch.equal(p,q)


def test_contract_and_guard():
    c=configuration()
    assert c['pilot_steps']==2560 and c['smoke_steps']==8 and c['cuda_memory_fraction']==.2
    with pytest.raises(ValueError):check_moments('mean_norm',1e-3,0)
    with pytest.raises(ValueError):check_moments('norm',0,float('nan'))
