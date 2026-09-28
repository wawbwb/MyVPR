"""Slot-conditioned depth aggregation. No semantic labels or pairwise decoder."""
import torch
from torch import nn
from torch.nn import functional as F

MODES=('baseline','last_repeat','equal_depth','adaptive_depth')


class SlotDepthRouter(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.score=nn.Sequential(nn.Linear(dim*3,64),nn.GELU(),nn.Linear(64,3))
        nn.init.zeros_(self.score[-1].weight);nn.init.zeros_(self.score[-1].bias)

    def forward(self,slots,equal=False):
        # B,Q,L,D -> one depth distribution per image and learned query slot.
        if slots.shape[2]!=3:raise ValueError('Expected three feature depths')
        if equal:weights=torch.ones_like(slots[...,0])/3
        else:weights=self.score(slots.flatten(2)).softmax(-1)
        return (slots*weights[...,None]).sum(2),weights


class DepthQueryVPR(nn.Module):
    def __init__(self,visual,mode):
        super().__init__()
        if mode not in MODES:raise ValueError('Unknown depth-query mode')
        if visual.spatial_attn_head is not None:raise ValueError('Unexpected extra attention head')
        self.backbone=visual.backbone;self.aggregator=visual.aggregator
        self.gate=visual.semantic_region_gate;self.mode=mode
        if len(self.backbone.dino.blocks)!=12:raise ValueError('Expected ViT-B/14 twelve blocks')
        if self.backbone.crop_semantic_film is not None or self.backbone.residual_clip_fusion is not None:
            raise ValueError('Expected plain RU checkpoint')
        if self.aggregator.semantic_num_classes is not None:raise ValueError('Semantic BoQ not supported')
        self.backbone.requires_grad_(False)
        if self.gate is not None:self.gate.requires_grad_(False)
        self.aggregator.requires_grad_(True)
        dim=self.aggregator.proj_c.out_channels;channels=self.backbone.out_channels
        self.align=nn.ModuleList([nn.LayerNorm(channels) for _ in range(3)])
        # Small, identical-across-arms random affine differences break copy symmetry.
        for layer in self.align:nn.init.normal_(layer.weight,mean=1.,std=.01)
        self.routers=nn.ModuleList([SlotDepthRouter(dim) for _ in self.aggregator.boqs])
        self.residuals=nn.ModuleList([nn.Linear(dim,dim,bias=False) for _ in self.aggregator.boqs])
        for layer in self.residuals:nn.init.zeros_(layer.weight)
        if mode=='baseline':
            self.align.requires_grad_(False);self.routers.requires_grad_(False);self.residuals.requires_grad_(False)
        if mode=='equal_depth':self.routers.requires_grad_(False)
        self.last_depth_weights=None

    def train(self,mode=True):
        super().train(mode);self.backbone.eval()
        if self.gate is not None:self.gate.eval()
        return self

    @torch.no_grad()
    def features(self,images):
        self.backbone.eval();b,_,h,w=images.shape
        if h%14 or w%14:raise ValueError('Image size must be divisible by 14')
        x=self.backbone.dino.prepare_tokens_with_masks(images);layers=[]
        for i,block in enumerate(self.backbone.dino.blocks,1):
            x=block(x)
            if i in (6,9,12):layers.append(x[:,1:].transpose(1,2).reshape(b,-1,h//14,w//14))
        if self.gate is not None:layers[-1]=self.gate(layers[-1])[0]
        return layers

    def aggregate(self,layers):
        agg=self.aggregator
        if self.mode=='baseline':return agg(layers[-1])[0]
        raw=agg.norm_input(agg.proj_c(layers[-1]).flatten(2).transpose(1,2))
        sources=[layers[-1]]*3 if self.mode=='last_repeat' else layers
        branches=[]
        for norm,source in zip(self.align,sources):
            aligned=norm(source.permute(0,2,3,1)).permute(0,3,1,2)
            branches.append(agg.norm_input(agg.proj_c(aligned).flatten(2).transpose(1,2)))
        outs=[];routes=[]
        for block,router,residual in zip(agg.boqs,self.routers,self.residuals):
            raw,base_slots,_=block(raw)
            slots=[]
            for j in range(3):
                branches[j],out,_=block(branches[j]);slots.append(out)
            fused,weights=router(torch.stack(slots,dim=2),equal=self.mode=='equal_depth')
            outs.append(base_slots+residual(fused));routes.append(weights.detach())
        self.last_depth_weights=torch.stack(routes,dim=1)
        return F.normalize(agg.fc(torch.cat(outs,dim=1).transpose(1,2)).flatten(1),dim=-1)

    def forward(self,images):return self.aggregate(self.features(images))
