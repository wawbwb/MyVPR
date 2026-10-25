from copy import deepcopy
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from src.models.spatial_resample import ARMS,SpatialResample,ResampleVPR,warp
from scripts.spatial_resample import replay,configuration


@pytest.fixture(autouse=True)
def seed():torch.manual_seed(42);torch.set_num_threads(2)


def test_warp_coordinate_direction_and_border():
    x=torch.arange(15,dtype=torch.float32).reshape(1,1,3,5);offset=torch.zeros(1,2,3,5)
    torch.testing.assert_close(warp(x,offset),x)
    offset[:,0]=1;y=warp(x,offset)
    torch.testing.assert_close(y[:,:,:,:-1],x[:,:,:,1:]);torch.testing.assert_close(y[:,:,:,-1],x[:,:,:,-1])


@pytest.mark.parametrize('arm',ARMS)
def test_zero_start_and_capacity(arm):
    m=SpatialResample(arm);x=torch.randn(2,768,3,5);y,s=m(x,[1,2])
    assert torch.equal(x,y) and not s['relative_delta'].any()
    assert sum(p.numel() for p in m.parameters())==24928


@pytest.mark.parametrize('arm',ARMS)
def test_real_bound_constant_preservation_and_finite_zero(arm):
    m=SpatialResample(arm,channels=12,hidden=8);nn.init.normal_(m.output.weight,std=10)
    x=torch.randn(2,12,3,5);y,s=m(x,[1,2])
    assert s['relative_delta'].max()<=.025001 and s['displacement_max']<=1
    assert ((y-x).norm(dim=1)/x.norm(dim=1)).max()<=.025001
    c=torch.randn(2,12,1,1).expand(2,12,3,5)
    torch.testing.assert_close(m(c,[1,2])[0],c,atol=2e-7,rtol=1e-6)
    assert torch.equal(m(torch.zeros_like(x),[1,2])[0],torch.zeros_like(x))


def test_shuffled_field_and_batch_independence():
    a=SpatialResample('local_shift',12,8);nn.init.normal_(a.output.weight,std=.2)
    b=SpatialResample('shuffled_shift',12,8);b.load_state_dict(a.state_dict());x=torch.randn(3,12,3,5)
    y,s=a(x,[1,2,3]);v,t=b(x,[1,2,3])
    assert not torch.allclose(y,v)
    torch.testing.assert_close(s['displacement_mean'],t['displacement_mean'])
    torch.testing.assert_close(v,torch.cat([b(x[i:i+1],[i+1])[0] for i in range(3)]))


class Anchor(nn.Module):
    def __init__(self):super().__init__();self.fc=nn.Linear(12,16)
    def readout(self,x):return F.normalize(self.fc(x.mean((2,3))),dim=1)


@pytest.mark.parametrize('arm',ARMS)
def test_gradient_replay_and_frozen_anchor(arm):
    a=ResampleVPR(Anchor(),arm,channels=12,hidden=8);nn.init.normal_(a.adapter.output.weight,std=.05)
    b=deepcopy(a);f=(torch.randn(5,12,3,5),torch.randn(5,16));z,_=a.describe(f,[1,2,3,4,5]);g=torch.randn_like(z);z.backward(g)
    replay(b,[tuple(t[i:i+2] for t in f) for i in range(0,5,2)],[1,2,3,4,5],g)
    for p,q in zip(a.adapter.parameters(),b.adapter.parameters()):
        assert p.grad is not None and p.grad.abs().max()>0
        torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=3e-4)
    assert all(p.grad is None for p in a.anchor.parameters())


@pytest.mark.parametrize('arm',ARMS)
def test_zero_start_trainability_and_resume(arm):
    m=SpatialResample(arm,channels=12,hidden=8);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=0)
    x=torch.randn(3,12,3,5);target=torch.randn_like(x);seen={n:False for n,p in m.named_parameters()}
    def update(a,o):
        o.zero_grad();(a(x,[1,2,3])[0]-target).square().mean().backward();o.step()
    for _ in range(4):
        update(m,opt)
        for n,p in m.named_parameters():seen[n]|=p.grad is not None and bool(p.grad.abs().max()>0)
    assert all(seen.values())
    b=deepcopy(m);ob=torch.optim.AdamW(b.parameters(),lr=.001,weight_decay=0);ob.load_state_dict(deepcopy(opt.state_dict()))
    update(m,opt);update(b,ob)
    for p,q in zip(m.parameters(),b.parameters()):assert torch.equal(p,q)


def test_fixed_budget():
    c=configuration();assert c['pilot_steps']==1024 and c['learning_rate']==.0003 and c['keep_weight']==0
