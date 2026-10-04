"""Frozen BoQ with a final-slot relative-geometry relation residual."""
import torch
from torch import nn
from torch.nn import functional as F

MODES=('appearance','aligned','shuffled')


def geometry(attention, coordinates):
    mu=attention@coordinates
    spread=(attention@coordinates.square().sum(-1)-mu.square().sum(-1)).clamp_min(0)
    delta=mu[:,None,:,:]-mu[:,:,None,:]
    distance=delta.square().sum(-1,keepdim=True).sqrt()
    overlap=F.normalize(attention,dim=-1)@F.normalize(attention,dim=-1).transpose(1,2)
    q=attention.shape[1]
    return torch.cat((delta,distance,spread[:,:,None,None].expand(-1,-1,q,-1),
                      spread[:,None,:,None].expand(-1,q,-1,-1),overlap[...,None]),-1)


class QueryRelationAttention(nn.Module):
    def __init__(self, original, mode, queries, grid=20):
        super().__init__()
        if mode not in MODES or not original.batch_first or original.dropout!=0:
            raise ValueError('Expected plain frozen batch-first attention')
        self.original=original.requires_grad_(False)
        self.mode,self.enabled=mode,True
        self.norm=nn.LayerNorm(original.embed_dim,elementwise_affine=False)
        self.project=nn.Linear(original.embed_dim,16,bias=False)
        self.edge=nn.Sequential(nn.Linear(38,32),nn.GELU(),nn.Linear(32,32),nn.GELU())
        self.output=nn.Linear(32,original.embed_dim,bias=False)
        nn.init.zeros_(self.output.weight)
        yy,xx=torch.meshgrid(torch.linspace(-1,1,grid),torch.linspace(-1,1,grid),indexing='ij')
        self.register_buffer('coordinates',torch.stack((xx,yy),-1).reshape(-1,2))
        self.register_buffer('permutation',torch.randperm(queries,generator=torch.Generator().manual_seed(42040)))
        self.last_diagnostics={}

    def forward(self,query,key,value,**kwargs):
        if any(v is not None for v in kwargs.values()): raise ValueError('Masks/flags unsupported')
        historical,weights=self.original(query,key,value,average_attn_weights=False)
        a=weights.mean(1)
        if not self.enabled:return historical,a
        b,q,_=historical.shape
        if q!=len(self.permutation) or key.shape[1]!=len(self.coordinates) or q<2:
            raise ValueError('Unexpected query/grid shape')
        g=geometry(a,self.coordinates)
        if self.mode=='appearance':g=torch.zeros_like(g)
        elif self.mode=='shuffled':g=g[:,self.permutation][:,:,self.permutation]
        u=self.project(self.norm(historical))
        messages=[]
        # Chunk the source-query axis, preserving all Q*(Q-1) edges.
        for start in range(0,q,8):
            stop=min(start+8,q)
            edge=torch.cat((u[:,start:stop,None,:].expand(-1,-1,q,-1),
                            u[:,None,:,:].expand(-1,stop-start,-1,-1),g[:,start:stop]),-1)
            value=self.edge(edge)
            mask=(torch.arange(start,stop,device=u.device)[:,None]!=torch.arange(q,device=u.device)[None,:])
            messages.append((value*mask[None,:,:,None]).sum(2)/(q-1))
        residual=self.output(torch.cat(messages,1))
        with torch.no_grad():
            mu=a@self.coordinates
            self.last_diagnostics=dict(centroid_std=float(mu.std(1,unbiased=False).mean()),
                residual_rms=float(residual.square().mean().sqrt()),attention_entropy=float(-(a*a.clamp_min(1e-12).log()).sum(-1).mean()))
        return historical+residual,a


def install(aggregator,mode):
    if aggregator.semantic_num_classes is not None:raise ValueError('Plain RU aggregator required')
    aggregator.requires_grad_(False)
    block=aggregator.boqs[-1]
    adapter=QueryRelationAttention(block.cross_attn,mode,block.queries.shape[1])
    block.cross_attn=adapter
    return adapter
