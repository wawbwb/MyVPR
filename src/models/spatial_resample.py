"""Content-conditioned single-image resampling, not cross-image registration."""
import torch
from torch import nn
from torch.nn import functional as F
from src.models.structural_channel_adapter import StructuralChannelVPR,image_seeds

ARMS=('global_shift','local_shift','shuffled_shift')


def warp(features,offset):
    b,_,h,w=features.shape
    if min(h,w)<2 or offset.shape!=(b,2,h,w):raise ValueError('Expected BCHW and B2HW grid')
    y,x=torch.meshgrid(torch.linspace(-1,1,h,device=features.device,dtype=features.dtype),
        torch.linspace(-1,1,w,device=features.device,dtype=features.dtype),indexing='ij')
    grid=torch.stack((x,y),-1)[None].expand(b,-1,-1,-1)
    scale=features.new_tensor([2/(w-1),2/(h-1)])[None,None,None,:]
    return F.grid_sample(features,grid+offset.permute(0,2,3,1)*scale,
        mode='bilinear',padding_mode='border',align_corners=True)


class SpatialResample(nn.Module):
    def __init__(self,arm,channels=768,hidden=32,cap=.025):
        super().__init__()
        if arm not in ARMS or not 0<cap<1:raise ValueError('Invalid sampling adapter')
        self.arm,self.cap=arm,cap
        self.down=nn.Conv2d(channels,hidden,1,bias=False)
        self.local=nn.Conv2d(hidden,hidden,3,padding=1,groups=hidden,bias=False)
        self.output=nn.Conv2d(hidden,2,1,bias=False);nn.init.zeros_(self.output.weight)

    def forward(self,features,seeds=None):
        x=F.layer_norm(features.permute(0,2,3,1),(features.shape[1],)).permute(0,3,1,2)
        offset=self.output(F.gelu(self.local(F.gelu(self.down(x))))).tanh()
        if self.arm=='global_shift':offset=offset.mean((2,3),keepdim=True).expand_as(offset)
        elif self.arm=='shuffled_shift':
            if seeds is None or len(seeds)!=len(offset):raise ValueError('Image-content seeds required')
            values=[]
            for v,seed in zip(offset,seeds):
                p=torch.randperm(v.shape[-2]*v.shape[-1],generator=torch.Generator().manual_seed(int(seed))).to(v.device)
                values.append(v.flatten(1)[:,p].reshape_as(v))
            offset=torch.stack(values)
        original=warp(features,torch.zeros_like(offset));sampled=warp(features,offset)
        rms=features.detach().square().mean(1,keepdim=True).sqrt()
        delta=self.cap*rms*((sampled-original)/rms.clamp_min(1e-6)).tanh()
        relative=delta.detach().flatten(1).norm(dim=1)/features.detach().flatten(1).norm(dim=1).clamp_min(1e-8)
        if not torch.isfinite(delta).all() or float(relative.max())>self.cap+1e-6:raise ValueError('Sampling bound violated')
        return features+delta,dict(relative_delta=relative,
            displacement_mean=offset.detach().square().sum(1).sqrt().mean(),
            displacement_max=offset.detach().abs().max(),
            spatial_offset_std=offset.detach().flatten(2).std(2,unbiased=False).mean())


class ResampleVPR(StructuralChannelVPR):
    def __init__(self,anchor,arm,**kwargs):
        nn.Module.__init__(self);self.anchor=anchor.requires_grad_(False).eval()
        self.adapter=SpatialResample(arm,**kwargs)
