"""Bounded local/global refinement with optional joint neck adaptation."""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from src.models.structural_channel_adapter import StructuralChannelVPR,image_seeds

ARMS=('neck_continue','frozen_refine','joint_refine','joint_permuted')


class BoundedRefine(nn.Module):
    def __init__(self,channels=768,rank=64,cap=.01,permuted=False):
        super().__init__();self.cap=cap;self.permuted=permuted
        self.down=nn.Conv2d(channels,rank,1,bias=False)
        self.local=nn.Conv2d(rank,rank,3,padding=1,groups=rank,bias=False)
        self.gate=nn.Sequential(nn.Linear(rank,rank//4,bias=False),nn.GELU(),nn.Linear(rank//4,rank,bias=False),nn.Tanh())
        self.up=nn.Conv2d(rank,channels,1,bias=False);nn.init.zeros_(self.up.weight)

    def forward(self,x,seeds):
        h=self.down(F.layer_norm(x.permute(0,2,3,1),(x.shape[1],)).permute(0,3,1,2))
        local=[]
        if self.permuted:
            if seeds is None or len(seeds)!=len(x):raise ValueError('Content seeds required')
            for v,seed in zip(h,seeds):
                p=torch.randperm(v.shape[-1]*v.shape[-2],generator=torch.Generator().manual_seed(int(seed))).to(v.device)
                w=self.local(v.flatten(1)[:,p].reshape_as(v)[None])[0].flatten(1)[:,p.argsort()].reshape_as(v)
                local.append(w)
            local=torch.stack(local)
        else:local=self.local(h)
        g=self.gate(h.mean((2,3)))[:,:,None,None]
        raw=self.up(local*h*g)
        rms=x.detach().square().mean(1,keepdim=True).sqrt()
        delta=self.cap*rms*raw.tanh()
        relative=delta.detach().flatten(1).norm(dim=1)/x.detach().flatten(1).norm(dim=1).clamp_min(1e-8)
        if float(relative.max())>self.cap+1e-6:raise ValueError('Refine bound violated')
        return x+delta,relative


class JointLocalVPR(StructuralChannelVPR):
    def __init__(self,anchor,arm):
        nn.Module.__init__(self)
        if arm not in ARMS:raise ValueError('Unknown arm')
        self.arm=arm;self.anchor=anchor.requires_grad_(False).eval()
        neck=deepcopy(anchor.neck)
        neck.requires_grad_(arm!='frozen_refine');neck.aux.requires_grad_(False)
        self.adapter=nn.ModuleDict(dict(neck=neck))
        if arm!='neck_continue':self.adapter['refine']=BoundedRefine(permuted=arm=='joint_permuted')

    @torch.no_grad()
    def encode(self,images):
        from src.hcen_plain_protocol import canonical_features
        layers=canonical_features(self.anchor.features(images))
        enhanced,_,_=self.anchor.neck(layers,auxiliary=False)
        self.anchor.neck.last_raw=None;self.anchor.neck.last_conditional_stats=None
        return layers[-1],self.anchor.readout(enhanced),layers,enhanced

    def describe(self,features,seeds):
        f,_,_=self.adapter['neck'](features[2],auxiliary=False)
        self.adapter['neck'].last_raw=None;self.adapter['neck'].last_conditional_stats=None
        if 'refine' in self.adapter:y,relative=self.adapter['refine'](f,seeds)
        else:y=f;relative=torch.zeros(len(f),device=f.device)
        with torch.no_grad():
            drift=(f-features[3]).flatten(1).norm(dim=1)/features[3].flatten(1).norm(dim=1).clamp_min(1e-8)
        return self.anchor.readout(y),dict(relative_delta=relative,neck_feature_drift=drift.mean())
