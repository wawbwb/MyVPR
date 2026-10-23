"""Frozen innovation-one with local/global residuals in projected attention K/V."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.models.structural_channel_adapter import StructuralChannelVPR, image_seeds

ARMS=('plain_kv','value_only','local_kv','permuted_kv')


def shuffled_local(h,conv,shape,seeds):
    if seeds is None or len(seeds)!=len(h):raise ValueError('One image seed required')
    result=[]
    for x,seed in zip(h,seeds):
        p=torch.randperm(len(x),generator=torch.Generator().manual_seed(int(seed))).to(x.device)
        y=conv(x[p].T.reshape(1,-1,*shape)).flatten(2).transpose(1,2)[0]
        result.append(y[p.argsort()])
    return torch.stack(result)


class KVResidual(nn.Module):
    def __init__(self,arm,dim=384):
        super().__init__()
        if arm not in ARMS:raise ValueError('Unknown arm')
        self.arm=arm
        if arm=='plain_kv':
            self.dk=nn.Linear(dim,24,bias=False);self.dv=nn.Linear(dim,24,bias=False)
            self.uk=nn.Linear(24,dim,bias=False);self.uv=nn.Linear(24,dim,bias=False)
        else:
            self.down=nn.Linear(dim,32,bias=False)
            self.local=nn.Conv2d(32,32,3,padding=1,groups=32,bias=False)
            self.gate=nn.Sequential(nn.Linear(32,8,bias=False),nn.GELU(),nn.Linear(8,32,bias=False),nn.Tanh())
            self.uk=None if arm=='value_only' else nn.Linear(32,dim,bias=False)
            self.uv=nn.Linear(32,dim,bias=False)
        if self.uk is not None:nn.init.zeros_(self.uk.weight)
        nn.init.zeros_(self.uv.weight)

    def forward(self,x,shape,seeds):
        if x.ndim!=3 or x.shape[1]!=math.prod(shape):raise ValueError('Invalid token grid')
        if self.arm=='plain_kv':return .1*self.uk(self.dk(x)),.1*self.uv(self.dv(x))
        h=self.down(F.layer_norm(x,(x.shape[-1],)));g=self.gate(h.mean(1))[:,None,:]
        if self.arm=='permuted_kv':local=shuffled_local(h,self.local,shape,seeds)
        else:local=self.local(h.transpose(1,2).reshape(len(h),32,*shape)).flatten(2).transpose(1,2)
        dv=.1*self.uv(local*h*g)
        dk=torch.zeros_like(x) if self.uk is None else .1*self.uk(h*(1+g))
        return dk,dv


def heads(x,n):return x.reshape(len(x),x.shape[1],n,-1).transpose(1,2)


def manual_attention(q,k,v,weight):
    a=(q@k.transpose(-1,-2)/math.sqrt(q.shape[-1])).softmax(-1)
    y=(a@v).transpose(1,2).reshape(len(q),q.shape[2],-1)
    return F.linear(y,weight,bias=None)


@torch.no_grad()
def prepare(aggregator,features):
    from src.models.official_boq import native_feature_layout
    x=aggregator.norm_input(aggregator.proj_c(native_feature_layout(features)).flatten(2).transpose(1,2))
    states=[]
    for block in aggregator.boqs:
        x=block.encoder(x);q=block.queries.repeat(len(x),1,1)
        q=block.norm_q(q+block.self_attn(q,q,q)[0]);m=block.cross_attn
        if not m.batch_first or m.dropout!=0 or m.bias_k is not None or m.bias_v is not None or m.add_zero_attn:
            raise ValueError('Unsupported historical attention')
        dim=x.shape[-1];w=m.in_proj_weight;b=m.in_proj_bias
        pq=F.linear(q,w[:dim],None if b is None else b[:dim])
        pk=F.linear(x,w[dim:2*dim],None if b is None else b[dim:2*dim])
        pv=F.linear(x,w[2*dim:],None if b is None else b[2*dim:])
        pq,pk,pv=[heads(v,m.num_heads) for v in (pq,pk,pv)]
        historical=m(q,x,x)[0]
        old=manual_attention(pq,pk+torch.zeros_like(pk),pv+torch.zeros_like(pv),m.out_proj.weight)
        states.append((x,pq,pk,pv,historical,old))
    return states


class KVCoAdaptVPR(StructuralChannelVPR):
    def __init__(self,anchor,arm):
        nn.Module.__init__(self);self.anchor=anchor.requires_grad_(False).eval()
        self.adapter=nn.ModuleList([KVResidual(arm,b.cross_attn.embed_dim) for b in anchor.visual.aggregator.boqs])

    @torch.no_grad()
    def encode(self,images):
        features,baseline=super().encode(images)
        return features,baseline,prepare(self.anchor.visual.aggregator,features)

    def describe(self,features,seeds):
        agg=self.anchor.visual.aggregator;outs=[];ratios=[];kr=[];vr=[]
        for block,adapter,state in zip(agg.boqs,self.adapter,features[2]):
            x,q,k,v,historical,old=state
            dk,dv=adapter(x,features[0].shape[-2:],seeds)
            m=block.cross_attn
            new=manual_attention(q,k+heads(dk,m.num_heads),v+heads(dv,m.num_heads),m.out_proj.weight)
            delta=new-old;out=historical+delta
            outs.append(block.norm_out(out))
            with torch.no_grad():
                ratios.append(delta.detach().flatten(1).norm(dim=1)/historical.flatten(1).norm(dim=1).clamp_min(1e-8))
                kr.append(dk.detach().norm()/k.norm().clamp_min(1e-8));vr.append(dv.detach().norm()/v.norm().clamp_min(1e-8))
        z=F.normalize(agg.fc(torch.cat(outs,1).transpose(1,2)).flatten(1),dim=1)
        return z,dict(relative_delta=torch.stack(ratios).mean(0),key_ratio=torch.stack(kr).mean(),value_ratio=torch.stack(vr).mean())
