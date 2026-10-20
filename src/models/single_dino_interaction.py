"""One frozen DINO/innovation-1, two lightweight operators, one BoQ readout.

No second encoder, CNN weight loader, CLIP, SAM, or query-candidate interaction.
All arms have identical active parameters; only terminal token routing changes.
"""
import torch
from torch import nn
from torch.nn import functional as F

from src.models.scan_mamba import SelectiveMixer
from src.models.hetero_interleave import MapProjection, LocalGlobalRefinement, interleave, uninterleave

ARMS = ('separate', 'aligned', 'permuted')


def local_permutation(height, width, device):
    generator = torch.Generator(device='cpu').manual_seed(42+height*1009+width)
    return torch.randperm(height*width, generator=generator).to(device)


class SingleDinoInteraction(nn.Module):
    def __init__(self, arm, channels=768, dim=64, rank=16, rho=.025):
        super().__init__()
        if arm not in ARMS or not 0 < rho < 1 or rank < 1 or dim < rank:
            raise ValueError('Invalid single-DINO arm/dimensions')
        self.arm, self.rho = arm, rho
        self.local_projection = MapProjection(channels, dim)
        self.global_projection = MapProjection(channels, dim)
        self.local_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(), nn.Conv2d(dim, dim, 1))
        self.context_norm = nn.LayerNorm(dim)
        self.context_scan = SelectiveMixer(dim, inner=32, state=4, rank=4)
        self.exchange_norm = nn.LayerNorm(dim)
        self.exchange_scan = SelectiveMixer(dim, inner=32, state=4, rank=4)
        for scan in (self.context_scan, self.exchange_scan):
            nn.init.normal_(scan.out.weight, std=.02)
        self.down = nn.Conv2d(2*dim, rank, 1)
        self.dynamic = nn.Linear(2*dim, rank)
        self.low_rank_up = nn.Conv2d(rank, 2*dim, 1)
        self.fuse = nn.Conv2d(2*dim, dim, 1)
        self.refine = LocalGlobalRefinement(dim, rank)
        self.output = nn.Conv2d(dim, channels, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    @staticmethod
    def scan_tokens(tokens, norm, scan):
        n = norm(tokens)
        return tokens+.5*(scan(n)+scan(n.flip(1)).flip(1))

    def forward(self, anchor):
        if anchor.ndim != 4:
            raise ValueError('Need innovation-1 enhanced BCHW map')
        local = self.local_projection(anchor)
        local = local+.1*self.local_conv(local)
        global_map = self.global_projection(anchor)
        g = self.scan_tokens(global_map.flatten(2).transpose(1,2), self.context_norm, self.context_scan)
        context = g.transpose(1,2).reshape_as(global_map)
        l = local.flatten(2).transpose(1,2)
        if self.arm == 'separate':
            g = self.scan_tokens(g, self.exchange_norm, self.exchange_scan)
            l = self.scan_tokens(l, self.exchange_norm, self.exchange_scan)
        else:
            if self.arm == 'permuted':
                permutation = local_permutation(*anchor.shape[-2:], anchor.device)
                l = l[:,permutation]
            g,l = uninterleave(self.scan_tokens(interleave(g,l), self.exchange_norm, self.exchange_scan))
            if self.arm == 'permuted':
                l = l[:,permutation.argsort()]
        joint = torch.cat((g.transpose(1,2).reshape_as(local), l.transpose(1,2).reshape_as(local)), 1)
        gate = self.dynamic(torch.cat((context.mean((2,3)),local.mean((2,3))),1)).sigmoid()
        mixed = joint+self.low_rank_up(F.silu(self.down(joint))*gate[:,:,None,None])
        fused, refine_gate = self.refine(self.fuse(mixed))
        residual = self.rho*anchor.square().mean(1,keepdim=True).sqrt().detach()*self.output(fused).tanh()
        auxiliary = (F.normalize(context.mean((2,3)),dim=-1),F.normalize(local.mean((2,3)),dim=-1))
        stats = dict(relative_residual=residual.detach().flatten(1).norm(dim=1)/anchor.flatten(1).norm(dim=1).clamp_min(1e-8),
            dynamic_gate=gate.detach().mean(1),refine_gate=refine_gate.detach().mean((1,2,3)))
        return anchor+residual,auxiliary,stats


class SingleDinoVPR(nn.Module):
    def __init__(self, anchor, arm, **kwargs):
        super().__init__()
        self.anchor = anchor.requires_grad_(False).eval()
        self.fusion = SingleDinoInteraction(arm, **kwargs)

    def train(self,mode=True):
        super().train(mode);self.anchor.eval();return self

    @torch.no_grad()
    def encode(self,images):
        from src.hcen_plain_protocol import canonical_features
        layers = canonical_features(self.anchor.features(images))
        enhanced,_,_ = self.anchor.neck(layers,auxiliary=False)
        self.anchor.neck.last_raw = None
        self.anchor.neck.last_conditional_stats = None
        # The sealed reusable trainer has a six-slot feature-pack interface.
        # Alias the SAME three frozen maps, not another encoder/forward pass.
        # Only enhanced (slot2) is consumed by the new fusion operator.
        return layers[0],layers[1],enhanced,layers[0],layers[1],enhanced

    def describe(self,features):
        mixed,auxiliary,stats = self.fusion(features[2])
        return self.anchor.readout(mixed),*auxiliary,stats

    def forward(self,images):
        return self.describe(self.encode(images))[0]
