"""ReID-inspired cross-depth neck, not a MambaVision/ReID reproduction.

Native DINO layers share a spatial grid: different depths, NOT image scales.
Selective scan reuses our reference O(L log L) implementation, not fused CUDA.
"""
import torch
from torch import nn
from torch.nn import functional as F
from src.models.scan_mamba import SelectiveMixer,scan_order
from src.slgd_layer_protocol import LayerCapture

MODES=('last_only','hierarchy','hierarchy_rank')


class SASF(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.paths=nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim,dim,k,padding=d*(k//2),dilation=d,groups=dim,bias=False),
            nn.GELU(),nn.Conv2d(dim,dim,1,bias=False)) for k,d in [(3,1),(5,2)]])
        self.weights=nn.Parameter(torch.zeros(2))

    def forward(self,x):
        weights=self.weights.softmax(0)
        return x+.1*sum(w*path(x) for w,path in zip(weights,self.paths))


class FusionStage(nn.Module):
    def __init__(self,dim,inner=32,state=4):
        super().__init__()
        self.shallow=nn.Sequential(nn.Conv2d(dim,dim,3,padding=1,groups=dim,bias=False),
                                   nn.GELU(),nn.Conv2d(dim,dim,1,bias=False))
        self.norm=nn.LayerNorm(2*dim)
        self.ssm=SelectiveMixer(2*dim,inner=inner,state=state,rank=4)
        # Noncausal local convolution, as suggested by the uploaded mixer design.
        self.ssm.conv=nn.Conv1d(inner,inner,3,padding=1,groups=inner)
        self.symmetric=nn.Sequential(nn.Conv2d(2*dim,2*dim,3,padding=1,groups=2*dim,bias=False),
                                     nn.GELU(),nn.Conv2d(2*dim,2*dim,1,bias=False))
        self.low_delta=nn.Linear(2*dim,dim,bias=False)
        self.high_delta=nn.Linear(2*dim,dim,bias=False)
        self.spatial=SASF(dim)

    def forward(self,low,high):
        low=low+.1*self.shallow(low)
        joint=torch.cat((low,high),1)
        b,_,h,w=joint.shape
        tokens=joint.flatten(2).transpose(1,2)
        x=self.norm(tokens)
        row=scan_order(h,w,False,x.device);col=scan_order(h,w,True,x.device)
        orders=(row,row.flip(0),col,col.flip(0))
        scanned=self.ssm(torch.cat([x[:,order] for order in orders],0)).chunk(4,0)
        update=sum(value[:,order.argsort()] for value,order in zip(scanned,orders))*.25
        symmetric=self.symmetric(joint).flatten(2).transpose(1,2)
        mixed=tokens+update+.1*symmetric
        dl=self.low_delta(mixed).transpose(1,2).reshape_as(low)
        dh=self.high_delta(mixed).transpose(1,2).reshape_as(high)
        return self.spatial(.5*(low+high)+.1*(dl+dh))


class HierarchicalNeck(nn.Module):
    def __init__(self,channels=768,dim=128,inner=32,state=4):
        super().__init__()
        self.norms=nn.ModuleList([nn.LayerNorm(channels) for _ in range(3)])
        self.projections=nn.ModuleList([nn.Linear(channels,dim,bias=False) for _ in range(3)])
        self.stages=nn.ModuleList([FusionStage(dim,inner,state) for _ in range(2)])
        self.aux=nn.ModuleList([nn.Linear(dim,256,bias=False) for _ in range(2)])
        self.up=nn.Conv2d(dim,channels,1,bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self,layers,auxiliary=False):
        aligned=[]
        for f,n,p in zip(layers,self.norms,self.projections):
            b,_,h,w=f.shape
            aligned.append(p(n(f.flatten(2).transpose(1,2))).transpose(1,2).reshape(b,-1,h,w))
        g=self.stages[0](aligned[0],aligned[1])
        states=[g]
        g=self.stages[1](g,aligned[2]);states.append(g)
        last=layers[-1]
        rms=last.square().mean(1,keepdim=True).sqrt().detach()
        residual=.1*rms*self.up(g).tanh()
        auxiliaries=tuple(F.normalize(head(x.mean((2,3))),dim=-1)
                          for head,x in zip(self.aux,states)) if auxiliary else ()
        return last+residual,auxiliaries,residual


class HCENVPR(nn.Module):
    def __init__(self,visual,mode,channels=768,dim=128,inner=32,state=4):
        super().__init__()
        if mode not in MODES:raise ValueError('Unknown HCEN mode')
        self.mode=mode
        self.visual=visual.requires_grad_(False).eval()
        self.neck=HierarchicalNeck(channels,dim,inner,state)

    def train(self,mode=True):
        super().train(mode);self.visual.eval();return self

    def features(self,images):
        capture=LayerCapture(self.visual.backbone,[8,10,12])
        try:
            with torch.no_grad():features,_=capture(images)
        finally:capture.close()
        return tuple(features[i] for i in (8,10,12))

    def readout(self,features):
        if self.visual.semantic_region_gate is not None:
            features=self.visual.semantic_region_gate(features)[0]
        value=self.visual.aggregator(features)
        return value[0] if isinstance(value,tuple) else value

    def aggregate(self,l8,l10,l12,auxiliary=False):
        layers=(l12,l12,l12) if self.mode=='last_only' else (l8,l10,l12)
        mixed,aux,residual=self.neck(layers,auxiliary)
        return (self.readout(mixed),*aux,residual)

    def forward(self,images):
        return self.aggregate(*self.features(images))[0]
