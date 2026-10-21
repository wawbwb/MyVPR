from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from src.models.structural_channel_adapter import (ARMS,StructuralChannelAdapter,
    StructuralChannelVPR,conditioning_statistics,condition_vector,image_seeds)
from scripts.structural_channel import configuration,replay


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2);torch.manual_seed(42)


def test_constant_map_no_boundary_detail():
    x=torch.randn(2,8,1,1).expand(2,8,5,5)
    mean,rms,detail=conditioning_statistics(x)
    torch.testing.assert_close(detail,torch.full_like(detail,.001),atol=1e-7,rtol=0)
    torch.testing.assert_close(mean,x[:,:,0,0])


def test_detail_detects_layout_not_only_histogram():
    x=torch.zeros(1,4,6,6)
    x[:,:,:3]=1
    checker=torch.zeros_like(x)
    checker[:,:,::2,::2]=1;checker[:,:,1::2,1::2]=1
    a,b=conditioning_statistics(x),conditioning_statistics(checker)
    torch.testing.assert_close(a[0],b[0]);torch.testing.assert_close(a[1],b[1])
    assert not torch.allclose(a[2],b[2])


def test_condition_batch_independent_and_scramble_uses_same_marginals():
    x=torch.randn(3,8,5,5);seeds=image_seeds(['a'*64,'b'*64,'c'*64])
    z=condition_vector(x,'structure_gate',seeds)
    torch.testing.assert_close(z,torch.cat([condition_vector(x[i:i+1],'structure_gate',seeds[i:i+1]) for i in range(3)]))
    scrambled=condition_vector(x,'scrambled_gate',seeds)
    torch.testing.assert_close(z[:,:8],scrambled[:,:8])
    torch.testing.assert_close(z[:,8:].sort(1).values,scrambled[:,8:].sort(1).values)
    assert not torch.allclose(z,scrambled)


@pytest.mark.parametrize('arm',ARMS)
def test_exact_zero_start_and_bounded_gate(arm):
    m=StructuralChannelAdapter(arm,channels=12,hidden=8,rank=4)
    x=torch.randn(3,12,5,5)
    y,s=m(x,[1,2,3])
    assert torch.equal(x,y) and torch.count_nonzero(s['relative_delta'])==0
    assert .5<=s['gate_mean']<=1.5


def test_plain_channel_matches_ungated_formula():
    m=StructuralChannelAdapter('channel',channels=12,hidden=8,rank=4)
    torch.nn.init.normal_(m.up.weight,std=.01)
    x=torch.randn(2,12,5,5)
    y,_=m(x)
    expected=x+.1*m.up(F.gelu(m.middle(F.gelu(m.down(x)))))
    torch.testing.assert_close(y,expected,rtol=0,atol=0)


def test_matched_gated_parameters_and_shared_core_initialization():
    models={}
    for a in ARMS:
        torch.manual_seed(42);models[a]=StructuralChannelAdapter(a,channels=12,hidden=8,rank=4)
    for a,m in models.items():
        for part in ('down','middle','up'):
            for n,v in getattr(m,part).state_dict().items():assert torch.equal(v,getattr(models['channel'],part).state_dict()[n])
    assert len({sum(p.numel() for p in models[a].parameters()) for a in ARMS[1:]})==1
    assert sum(p.numel() for p in models['channel'].parameters())<sum(p.numel() for p in models['structure_gate'].parameters())


class TinyAnchor(torch.nn.Module):
    def __init__(self):
        super().__init__();self.readout_layer=torch.nn.Linear(12,16)
    def readout(self,x):return F.normalize(self.readout_layer(x.mean((2,3))),dim=1)


@pytest.mark.parametrize('arm',ARMS)
def test_replay_matches_full_batch_and_frozen_anchor(arm):
    a=StructuralChannelVPR(TinyAnchor(),arm,channels=12,hidden=8,rank=4)
    torch.nn.init.normal_(a.adapter.up.weight,std=.02)
    b=deepcopy(a)
    features=(torch.randn(5,12,5,5),torch.randn(5,16));seeds=[1,2,3,4,5]
    z,_=a.describe(features,seeds);g=torch.randn_like(z);z.backward(g)
    chunks=[tuple(t[i:i+2] for t in features) for i in range(0,5,2)]
    replay(b,chunks,seeds,g)
    for (n,p),(n2,q) in zip(a.adapter.named_parameters(),b.adapter.named_parameters()):
        assert n==n2 and p.grad is not None and torch.isfinite(p.grad).all() and (p.grad!=0).any()
        torch.testing.assert_close(p.grad,q.grad,rtol=3e-4,atol=2e-5)
    assert all(p.grad is None for p in a.anchor.parameters())
    a.train();assert not a.anchor.training


@pytest.mark.parametrize('arm',ARMS)
def test_actual_updates_reach_gate_without_zero_gradient_lock(arm):
    m=StructuralChannelAdapter(arm,channels=12,hidden=8,rank=4)
    opt=torch.optim.AdamW(m.parameters(),lr=.001)
    seen={n:False for n,p in m.named_parameters()}
    x=torch.randn(4,12,5,5);target=torch.randn_like(x)
    for _ in range(4):
        opt.zero_grad();y,_=m(x,[1,2,3,4]);(y-target).square().mean().backward()
        for n,p in m.named_parameters():seen[n]|=p.grad is not None and bool((p.grad!=0).any())
        opt.step()
    assert all(seen.values()),seen


def test_finite_resume_next_update():
    m=StructuralChannelAdapter('structure_gate',channels=12,hidden=8,rank=4)
    opt=torch.optim.AdamW(m.parameters(),lr=.001)
    x=torch.randn(3,12,5,5)
    def update(model,optimizer):
        optimizer.zero_grad();model(x,[1,2,3])[0].square().mean().backward();optimizer.step()
    update(m,opt)
    m2=deepcopy(m);opt2=torch.optim.AdamW(m2.parameters(),lr=.001);opt2.load_state_dict(deepcopy(opt.state_dict()))
    update(m,opt);update(m2,opt2)
    for p,q in zip(m.parameters(),m2.parameters()):assert torch.equal(p,q)


@pytest.mark.parametrize('bad',['arm','dim','seed'])
def test_invalid_inputs(bad):
    with pytest.raises(ValueError):
        if bad=='arm':StructuralChannelAdapter('bad')
        elif bad=='dim':StructuralChannelAdapter('channel',hidden=0)
        else:condition_vector(torch.randn(2,8,5,5),'scrambled_gate')


def test_registered_full_budget():
    c=configuration()
    assert c['pilot_steps']==2560 and c['epochs']==5 and c['learning_rate']==.0003
    assert c['adapter_gain']==.1 and c['keep_weight']==.1 and c['cuda_memory_fraction']==.2
