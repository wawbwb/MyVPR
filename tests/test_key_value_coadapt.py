from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from src.models.key_value_coadapt import ARMS,KVResidual,KVCoAdaptVPR,prepare,manual_attention,heads
from src.models.official_boq import native_feature_layout
from scripts.key_value_coadapt import configuration,replay


@pytest.fixture(autouse=True)
def seed():torch.manual_seed(42);torch.set_num_threads(2)


class Block(nn.Module):
    def __init__(self):
        super().__init__();d=24
        self.encoder=nn.TransformerEncoderLayer(d,3,48,dropout=0,batch_first=True)
        self.queries=nn.Parameter(torch.randn(1,4,d));self.self_attn=nn.MultiheadAttention(d,3,batch_first=True)
        self.norm_q=nn.LayerNorm(d);self.cross_attn=nn.MultiheadAttention(d,3,batch_first=True);self.norm_out=nn.LayerNorm(d)


class Agg(nn.Module):
    def __init__(self):
        super().__init__();self.proj_c=nn.Conv2d(12,24,3,padding=1);self.norm_input=nn.LayerNorm(24)
        self.boqs=nn.ModuleList([Block(),Block()]);self.fc=nn.Linear(8,4)
    def forward(self,f):
        x=self.norm_input(self.proj_c(native_feature_layout(f)).flatten(2).transpose(1,2));outs=[]
        for b in self.boqs:
            x=b.encoder(x);q=b.queries.repeat(len(x),1,1);q=b.norm_q(q+b.self_attn(q,q,q)[0])
            outs.append(b.norm_out(b.cross_attn(q,x,x)[0]))
        return F.normalize(self.fc(torch.cat(outs,1).transpose(1,2)).flatten(1),dim=1)


class Anchor(nn.Module):
    def __init__(self):
        super().__init__();self.visual=nn.Module();self.visual.aggregator=Agg()
    def readout(self,f):return self.visual.aggregator(f)


@pytest.mark.parametrize('arm',ARMS)
def test_zero_exact_and_parameter_counts(arm):
    m=KVResidual(arm);x=torch.randn(2,9,384);k,v=m(x,(3,3),[1,2])
    assert not k.any() and not v.any()
    count={'plain_kv':36864,'value_only':25376,'local_kv':37664,'permuted_kv':37664}
    assert sum(p.numel() for p in m.parameters())==count[arm]


def test_manual_per_head_attention_matches_original_and_deltas():
    m=nn.MultiheadAttention(24,3,batch_first=True).eval();q=torch.randn(2,4,24);x=torch.randn(2,9,24)
    w,b=m.in_proj_weight,m.in_proj_bias
    pq=F.linear(q,w[:24],b[:24]);pk=F.linear(x,w[24:48],b[24:48]);pv=F.linear(x,w[48:],b[48:])
    y=manual_attention(heads(pq,3),heads(pk,3),heads(pv,3),m.out_proj.weight)+m.out_proj.bias
    torch.testing.assert_close(y,m(q,x,x)[0],atol=2e-7,rtol=2e-5)
    dk=torch.randn_like(pk)*.03;dv=torch.randn_like(pv)*.03
    y=manual_attention(heads(pq,3),heads(pk+dk,3),heads(pv+dv,3),m.out_proj.weight)
    ref=F.scaled_dot_product_attention(heads(pq,3),heads(pk+dk,3),heads(pv+dv,3)).transpose(1,2).reshape(2,4,24)
    torch.testing.assert_close(y,F.linear(ref,m.out_proj.weight),atol=2e-7,rtol=2e-5)


@pytest.mark.parametrize('arm',ARMS)
def test_full_zero_baseline_and_gradient_replay(arm):
    a=KVCoAdaptVPR(Anchor(),arm);f=torch.randn(3,12,3,3)
    with torch.no_grad():features=(f,a.anchor.readout(f),prepare(a.anchor.visual.aggregator,f))
    z,_=a.describe(features,[1,2,3]);torch.testing.assert_close(z,features[1],atol=1e-6,rtol=0)
    b=deepcopy(a);g=torch.randn_like(z);z.backward(g)
    chunks=[]
    for i in range(3):chunks.append((f[i:i+1],features[1][i:i+1],[tuple(x[i:i+1] for x in state) for state in features[2]]))
    replay(b,chunks,[1,2,3],g)
    for p,q in zip(a.adapter.parameters(),b.adapter.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=3e-4)
    assert all(p.grad is None for p in a.anchor.parameters())
    for layer in a.adapter:
        assert layer.uv.weight.grad.abs().max()>0
        if layer.uk is not None:assert layer.uk.weight.grad.abs().max()>0


def test_geometry_control_changes_output_and_is_batch_independent():
    a=KVResidual('local_kv',24);b=KVResidual('permuted_kv',24);b.load_state_dict(a.state_dict())
    for m in (a,b):nn.init.normal_(m.uv.weight,std=.01)
    b.uv.load_state_dict(a.uv.state_dict());x=torch.randn(2,9,24)
    ka,va=a(x,(3,3),[1,2]);kb,vb=b(x,(3,3),[1,2])
    assert torch.equal(ka,kb) and not torch.allclose(va,vb)
    torch.testing.assert_close(vb,torch.cat([b(x[i:i+1],(3,3),[i+1])[1] for i in range(2)]))


@pytest.mark.parametrize('arm',ARMS)
def test_all_parameters_reachable_and_resume(arm):
    a=KVCoAdaptVPR(Anchor(),arm);f=torch.randn(2,12,3,3)
    with torch.no_grad():features=(f,a.anchor.readout(f),prepare(a.anchor.visual.aggregator,f))
    opt=torch.optim.AdamW(a.adapter.parameters(),lr=.001,weight_decay=0);target=torch.randn(2,96)
    seen={n:False for n,p in a.adapter.named_parameters()}
    def update(m,o):
        o.zero_grad();z,_=m.describe(features,[1,2]);(z-target).square().mean().backward();o.step()
    for _ in range(4):
        update(a,opt)
        for n,p in a.adapter.named_parameters():seen[n]|=p.grad is not None and bool(p.grad.abs().max()>0)
    assert all(seen.values()),seen
    b=deepcopy(a);ob=torch.optim.AdamW(b.adapter.parameters(),lr=.001,weight_decay=0);ob.load_state_dict(deepcopy(opt.state_dict()))
    update(a,opt);update(b,ob)
    for p,q in zip(a.adapter.parameters(),b.adapter.parameters()):assert torch.equal(p,q)
    a.train();assert not a.anchor.training


def test_fixed_protocol():
    c=configuration();assert c['pilot_steps']==2560 and c['learning_rate']==.0001 and c['keep_weight']==0
