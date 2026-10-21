"""Local contrast CONDITIONS a channel-only update; it never replaces values.

SE/GCT and conditional channel calibration are prior art, not claimed new here.
"""
import torch
from torch import nn
from torch.nn import functional as F

from src.models.complementary_covariance import image_seeds, permute_positions

ARMS = ('channel', 'global_gate', 'structure_gate', 'scrambled_gate')


def conditioning_statistics(hidden):
    if hidden.ndim != 4 or min(hidden.shape[-2:]) < 2:
        raise ValueError('Need BCHW spatial features')
    mean = hidden.mean((2, 3))
    rms = (hidden.square().mean((2, 3))+1e-6).sqrt()
    # Reflect-padding preserves constants, unlike zero-padding at boundaries.
    local = F.avg_pool2d(F.pad(hidden,(1,1,1,1),mode='reflect'),3,stride=1)
    detail = ((hidden-local).square().mean((2,3))+1e-6).sqrt()
    return mean, rms, detail


def condition_vector(hidden, arm, seeds=None):
    if arm not in ARMS[1:]: raise ValueError('Not a conditional arm')
    mean,rms,detail = conditioning_statistics(hidden)
    second = rms if arm == 'global_gate' else detail
    if arm == 'scrambled_gate':
        if seeds is None: raise ValueError('Content-addressed per-image seeds required')
        # B,C,1: shuffle channel identity, not spatial locations or other images.
        second = permute_positions(second.unsqueeze(-1),seeds).squeeze(-1)
    # Separate non-affine normalization avoids differences in statistic scale
    # trivially driving one condition to saturation. No labels/GPS in features.
    return torch.cat((F.layer_norm(mean,(mean.shape[1],)),F.layer_norm(second,(second.shape[1],))),1)


class StructuralChannelAdapter(nn.Module):
    def __init__(self, arm, channels=768, hidden=64, rank=16, gain=.1):
        super().__init__()
        if arm not in ARMS or min(channels,hidden,rank)<1 or not 0<gain<=1:
            raise ValueError('Invalid adapter settings')
        self.arm,self.gain=arm,float(gain)
        self.down=nn.Conv2d(channels,hidden,1)
        self.middle=nn.Conv2d(hidden,hidden,1,groups=hidden)
        self.up=nn.Conv2d(hidden,channels,1)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
        self.condition=(None if arm=='channel' else nn.Sequential(
            nn.Linear(2*hidden,rank),nn.GELU(),nn.Linear(rank,hidden)))

    def forward(self,features,seeds=None):
        hidden=F.gelu(self.down(features))
        if self.condition is None:
            gate=torch.ones((len(features),hidden.shape[1]),device=features.device,dtype=features.dtype)
        else:
            gate=1+.5*self.condition(condition_vector(hidden,self.arm,seeds)).tanh()
        raw=self.up(F.gelu(self.middle(hidden*gate[:,:,None,None])))
        delta=self.gain*raw
        diagnostics=dict(gate_mean=gate.detach().mean(),gate_std=gate.detach().std(unbiased=False),
            relative_delta=delta.detach().flatten(1).norm(dim=1)/features.detach().flatten(1).norm(dim=1).clamp_min(1e-8))
        return features+delta,diagnostics


class StructuralChannelVPR(nn.Module):
    def __init__(self,anchor,arm,**kwargs):
        super().__init__()
        self.anchor=anchor.requires_grad_(False).eval()
        self.adapter=StructuralChannelAdapter(arm,**kwargs)

    def train(self,mode=True):
        super().train(mode);self.anchor.eval();return self

    @torch.no_grad()
    def encode(self,images):
        from src.hcen_plain_protocol import canonical_features
        layers=canonical_features(self.anchor.features(images))
        enhanced,_,_=self.anchor.neck(layers,auxiliary=False)
        self.anchor.neck.last_raw=None;self.anchor.neck.last_conditional_stats=None
        return enhanced,self.anchor.readout(enhanced)

    def describe(self,features,seeds):
        adapted,stats=self.adapter(features[0],seeds)
        return self.anchor.readout(adapted),stats
