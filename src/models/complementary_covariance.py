"""One-image local/context cross-statistics; independent of the frozen BoQ.

Covariance pooling is prior art. This is a complementary-descriptor hypothesis,
not a claim of inventing bilinear pooling or heterogeneous feature interaction.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

ARMS = ('mean_outer', 'aligned_cov', 'permuted_cov')


def image_seeds(digests):
    """Content-addressed, batch/order independent; labels/GPS never enter."""
    if any(len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in digests):
        raise ValueError('Expected SHA256 image digests')
    return [int(h[:16], 16) % (2**63-1) for h in digests]


def permute_positions(tokens, seeds):
    if tokens.ndim != 3 or len(seeds) != len(tokens) or tokens.shape[1] < 2:
        raise ValueError('Expected BNC and one image seed per image')
    values = []
    for x, seed in zip(tokens, seeds):
        g = torch.Generator(device='cpu').manual_seed(int(seed))
        p = torch.randperm(len(x), generator=g)
        # Explicitly avoid a complete identity for tiny test grids.
        if torch.equal(p, torch.arange(len(x))):
            p = p.roll(1)
        values.append(x[p.to(tokens.device)])
    return torch.stack(values)


def cross_statistic(local, context, arm, seeds=None):
    if arm not in ARMS or local.shape != context.shape or local.ndim != 3 or local.shape[1] < 2:
        raise ValueError('Matched BNC maps with at least two positions required')
    if arm == 'mean_outer':
        return local.mean(1).unsqueeze(2) * context.mean(1).unsqueeze(1)
    local = local-local.mean(1, keepdim=True)
    context = context-context.mean(1, keepdim=True)
    if arm == 'permuted_cov':
        if seeds is None:
            raise ValueError('Per-image content seeds required, never a global permutation')
        context = permute_positions(context, seeds)
    return local.transpose(1, 2).bmm(context)/(local.shape[1]-1)


def combine_descriptors(base, complement, weight):
    if not math.isfinite(weight) or weight < 0 or base.ndim != 2 or complement.ndim != 2 or len(base) != len(complement):
        raise ValueError('Invalid descriptor fusion')
    # Both inputs are unit-normalized. Weight zero exactly preserves base block.
    return torch.cat((base, weight*complement), dim=1)/math.sqrt(1+weight*weight)


class ComplementaryCovariance(nn.Module):
    def __init__(self, arm, channels=768, dim=64, rank=32, descriptor_dim=512):
        super().__init__()
        if arm not in ARMS or min(channels, dim, rank, descriptor_dim) < 1 or rank > dim:
            raise ValueError('Invalid arm/dimensions')
        self.arm = arm
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.local_in = nn.Conv2d(channels, dim, 1, bias=False)
        self.context_in = nn.Conv2d(channels, dim, 1, bias=False)
        self.local3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.local5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
        self.local_mix = nn.Conv2d(2*dim, dim, 1, bias=False)
        # DINO tokens already have full-image context. No extra encoder/scan.
        self.context_mix = nn.Conv2d(dim, dim, 1, bias=False)
        self.dynamic = nn.Linear(2*dim, 2*rank)
        self.local_rank = nn.Conv2d(dim, rank, 1, bias=False)
        self.context_rank = nn.Conv2d(dim, rank, 1, bias=False)
        self.readout = nn.Linear(rank*rank, descriptor_dim, bias=False)

    def forward(self, features, seeds=None):
        if features.ndim != 4 or min(features.shape[-2:]) < 2:
            raise ValueError('Expected spatial BCHW map')
        x = self.norm(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        local, context = self.local_in(x), self.context_in(x)
        local = local + self.local_mix(torch.cat((F.gelu(self.local3(local)), F.gelu(self.local5(local))), 1))
        context = context + self.context_mix(F.gelu(context))
        gate = self.dynamic(torch.cat((local.mean((2, 3)), context.mean((2, 3))), 1)).sigmoid()
        gl, gc = gate.chunk(2, dim=1)
        local = self.local_rank(local).flatten(2).transpose(1, 2)*gl[:, None, :]
        context = self.context_rank(context).flatten(2).transpose(1, 2)*gc[:, None, :]
        statistic = cross_statistic(local, context, self.arm, seeds)
        # Smooth signed square root (finite derivative at zero), then L2.
        powered = statistic.flatten(1)/(statistic.flatten(1).abs()+1e-6).sqrt()
        pooled = F.normalize(powered, dim=1)
        descriptor = F.normalize(self.readout(pooled), dim=1)
        return descriptor


class ComplementaryVPR(nn.Module):
    def __init__(self, anchor, arm, **kwargs):
        super().__init__()
        self.anchor = anchor.requires_grad_(False).eval()
        self.head = ComplementaryCovariance(arm, **kwargs)

    def train(self, mode=True):
        super().train(mode)
        self.anchor.eval()
        return self

    @torch.no_grad()
    def encode(self, images):
        from src.hcen_plain_protocol import canonical_features
        layers = canonical_features(self.anchor.features(images))
        enhanced, _, _ = self.anchor.neck(layers, auxiliary=False)
        self.anchor.neck.last_raw = None
        self.anchor.neck.last_conditional_stats = None
        return enhanced, self.anchor.readout(enhanced)

    def describe(self, features, seeds, weight):
        h = self.head(features[0], seeds)
        return combine_descriptors(features[1], h, weight), h
