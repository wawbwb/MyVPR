"""Label-free field interventions for frozen sampling models."""
import torch
from torch.nn import functional as F
from src.models.spatial_resample import warp

VARIANTS=('innovation1','trained_global','constant_global','trained_local',
    'constant_local','local_mean','local_centered','local_centered_shuffled')


def field(adapter,features):
    x=F.layer_norm(features.permute(0,2,3,1),(features.shape[1],)).permute(0,3,1,2)
    out=adapter.output(F.gelu(adapter.local(F.gelu(adapter.down(x))))).tanh()
    if adapter.arm=='global_shift':out=out.mean((2,3),keepdim=True).expand_as(out)
    elif adapter.arm!='local_shift':raise ValueError('Only registered source branches')
    return out


def centered(offset):
    x=offset-offset.mean((2,3),keepdim=True)
    scale=x.abs().flatten(1).amax(1).clamp_min(1)[:,None,None,None]
    return x/scale


def shuffle(offset,seeds):
    if len(offset)!=len(seeds):raise ValueError('Image seeds differ')
    result=[]
    for x,seed in zip(offset,seeds):
        p=torch.randperm(x.shape[-2]*x.shape[-1],generator=torch.Generator().manual_seed(int(seed))).to(x.device)
        result.append(x.flatten(1)[:,p].reshape_as(x))
    return torch.stack(result)


def apply(features,offset):
    if offset.abs().max()>1.000001:raise ValueError('Offset exceeds one patch per axis')
    original=warp(features,torch.zeros_like(offset));sampled=warp(features,offset)
    rms=features.detach().square().mean(1,keepdim=True).sqrt()
    delta=.025*rms*((sampled-original)/rms.clamp_min(1e-6)).tanh()
    relative=delta.flatten(1).norm(dim=1)/features.flatten(1).norm(dim=1).clamp_min(1e-8)
    if not torch.isfinite(delta).all() or relative.max()>.025001:raise ValueError('Invalid residual')
    return features+delta,relative


def intervention_fields(global_field,local_field,constants,seeds):
    zero_mean=centered(local_field)
    return dict(trained_global=global_field,
        constant_global=constants['global'][None,:,None,None].expand_as(global_field),
        trained_local=local_field,
        constant_local=constants['local'][None,:,None,None].expand_as(local_field),
        local_mean=local_field.mean((2,3),keepdim=True).expand_as(local_field),
        local_centered=zero_mean,local_centered_shuffled=shuffle(zero_mean,seeds))
