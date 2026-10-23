from copy import deepcopy
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from src.models.joint_local_refine import ARMS,BoundedRefine,JointLocalVPR
from scripts.joint_local_refine import replay,configuration,optimizer_groups


@pytest.fixture(autouse=True)
def seed():torch.manual_seed(42);torch.set_num_threads(2)


@pytest.mark.parametrize('permuted',[False,True])
def test_exact_start_and_true_amplitude_bound(permuted):
    m=BoundedRefine(channels=12,rank=8,permuted=permuted);x=torch.randn(3,12,3,3)
    y,r=m(x,[1,2,3]);assert torch.equal(x,y) and not r.any()
    nn.init.normal_(m.up.weight,std=100)
    y,r=m(x,[1,2,3]);assert r.max()<=.010001 and r.min()>0
    token=(y-x).norm(dim=1)/x.norm(dim=1).clamp_min(1e-8)
    assert token.max()<=.010001


def test_geometry_control_and_per_image_seeds():
    a=BoundedRefine(12,8);b=BoundedRefine(12,8,permuted=True);b.load_state_dict(a.state_dict())
    nn.init.normal_(a.up.weight,std=.1);b.up.load_state_dict(a.up.state_dict());x=torch.randn(3,12,3,3)
    y=a(x,[1,2,3])[0];v=b(x,[1,2,3])[0]
    assert not torch.allclose(y,v,atol=1e-7)
    torch.testing.assert_close(v,torch.cat([b(x[i:i+1],[i+1])[0] for i in range(3)]))


class Neck(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(768,768,1);self.aux=nn.Linear(768,16)
        self.last_raw=None;self.last_conditional_stats=None
    def forward(self,layers,auxiliary=False):
        x=layers[-1];r=.025*self.conv(x).tanh();return x+r,(),r


class Anchor(nn.Module):
    def __init__(self):super().__init__();self.neck=Neck()
    def readout(self,x):return F.normalize(x.mean((2,3)),dim=1)


def sample(anchor):
    x=torch.randn(3,768,3,3);layers=[x*.8,x*.9,x]
    with torch.no_grad():f=anchor.neck(layers)[0];z=anchor.readout(f)
    return x,z,layers,f


@pytest.mark.parametrize('arm',ARMS)
def test_baseline_copy_freezing_and_replay(arm):
    a=JointLocalVPR(Anchor(),arm);features=sample(a.anchor);z,_=a.describe(features,[1,2,3])
    assert torch.equal(z,features[1])
    b=deepcopy(a);g=torch.randn_like(z);z.backward(g)
    chunks=[(features[0][i:i+1],features[1][i:i+1],[x[i:i+1] for x in features[2]],features[3][i:i+1]) for i in range(3)]
    replay(b,chunks,[1,2,3],g)
    for p,q in zip(a.adapter.parameters(),b.adapter.parameters()):
        if p.requires_grad:torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=3e-4)
        else:assert p.grad is None
    assert all(p.grad is None for p in a.anchor.parameters())
    assert all(not p.requires_grad for p in a.adapter['neck'].aux.parameters())


@pytest.mark.parametrize('arm',ARMS)
def test_learning_rates_trainability_and_resume(arm):
    a=JointLocalVPR(Anchor(),arm);features=sample(a.anchor);c=configuration()
    opt=torch.optim.AdamW(optimizer_groups(a,c),weight_decay=0)
    assert {g['lr'] for g in opt.param_groups}==({1e-5} if arm=='neck_continue' else {1e-4} if arm=='frozen_refine' else {1e-5,1e-4})
    target=torch.randn(3,768);seen={n:False for n,p in a.adapter.named_parameters() if p.requires_grad}
    def update(m,o):
        o.zero_grad();z,_=m.describe(features,[1,2,3]);(z-target).square().mean().backward();o.step()
    for _ in range(4):
        update(a,opt)
        for n,p in a.adapter.named_parameters():
            if p.requires_grad:seen[n]|=p.grad is not None and bool(p.grad.abs().max()>0)
    assert all(seen.values()),seen
    b=deepcopy(a);ob=torch.optim.AdamW(optimizer_groups(b,c),weight_decay=0);ob.load_state_dict(deepcopy(opt.state_dict()))
    update(a,opt);update(b,ob)
    for p,q in zip(a.adapter.parameters(),b.adapter.parameters()):assert torch.equal(p,q)


def test_budget():
    c=configuration();assert c['pilot_steps']==1024 and c['keep_weight']==0 and c['neck_learning_rate']==1e-5
