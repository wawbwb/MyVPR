from copy import deepcopy
import pytest
import torch
from torch.nn import functional as F
from src.models.spatial_adapter_repeat import paired_models
from src.models.original_channel_transfer import OriginalChannelVPR
from src.models.structural_channel_adapter import StructuralChannelAdapter
from scripts.original_channel_transfer import replay, configuration


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42);torch.set_num_threads(2)


class TinyAnchor(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layer=torch.nn.Linear(768,16)
    def readout(self,x):return F.normalize(self.layer(x.mean((2,3))),dim=1)


def original():
    return paired_models(torch.nn.Identity())['channel'].adapter


def test_original_kernel_and_zero_output():
    models=paired_models(torch.nn.Identity());a=models['channel'].adapter;s=models['spatial'].adapter
    assert torch.equal(a.spatial.weight,s.spatial.weight[:,:,1:2,1:2])
    assert torch.equal(a.spatial.bias,s.spatial.bias)
    x=torch.randn(2,768,3,3)
    assert torch.equal(a(x),x)
    assert sum(p.numel() for p in a.parameters())==99264


def test_direct_mapping_preserves_previous_formula():
    a=original();b=StructuralChannelAdapter('channel')
    torch.nn.init.normal_(b.up.weight,std=.01)
    a.down.load_state_dict(b.down.state_dict());a.spatial.load_state_dict(b.middle.state_dict());a.up.load_state_dict(b.up.state_dict())
    x=torch.randn(2,768,3,3)
    assert torch.equal(a(x),b(x)[0])


@pytest.mark.parametrize('warm',[False,True])
def test_replay_and_frozen_anchor(warm):
    adapter=original()
    if warm:torch.nn.init.normal_(adapter.up.weight,std=.01)
    a=OriginalChannelVPR(TinyAnchor(),adapter);b=deepcopy(a)
    f=(torch.randn(5,768,3,3),torch.randn(5,16));z,_=a.describe(f);g=torch.randn_like(z);z.backward(g)
    replay(b,[tuple(x[i:i+2] for x in f) for i in range(0,5,2)],list(range(5)),g)
    for p,q in zip(a.adapter.parameters(),b.adapter.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=3e-4)
    assert all(p.grad is None for p in a.anchor.parameters())
    a.train();assert not a.anchor.training


def test_weight_transfer_is_independent_and_exact():
    adapter=original();torch.nn.init.normal_(adapter.up.weight,std=.01)
    a=OriginalChannelVPR(TinyAnchor(),adapter);b=OriginalChannelVPR(a.anchor,adapter)
    x=torch.randn(2,768,3,3)
    assert torch.equal(a.adapter(x),adapter(x))
    with torch.no_grad():a.adapter.up.weight.add_(1)
    assert torch.equal(b.adapter(x),adapter(x))
    assert not torch.equal(a.adapter(x),adapter(x))


@pytest.mark.parametrize('warm',[False,True])
def test_all_parameter_gradients_and_resume(warm):
    a=original()
    if warm:torch.nn.init.normal_(a.up.weight,std=.01)
    opt=torch.optim.AdamW(a.parameters(),lr=.0003,weight_decay=0)
    x=torch.randn(2,768,3,3);target=torch.randn_like(x);seen={n:False for n,p in a.named_parameters()}
    def update(m,o):
        o.zero_grad();(m(x)-target).square().mean().backward();o.step()
    for _ in range(3):
        update(a,opt)
        for n,p in a.named_parameters():seen[n]|=bool(p.grad.abs().max()>0)
    assert all(seen.values())
    b=deepcopy(a);ob=torch.optim.AdamW(b.parameters(),lr=.0003,weight_decay=0);ob.load_state_dict(deepcopy(opt.state_dict()))
    update(a,opt);update(b,ob)
    for p,q in zip(a.parameters(),b.parameters()):assert torch.equal(p,q)


def test_budget():
    c=configuration();assert c['pilot_steps']==2560 and c['smoke_steps']==8 and c['adapter_gain']==.1
