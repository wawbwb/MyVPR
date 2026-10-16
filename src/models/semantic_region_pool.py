"""Semantic x spatial soft pooling, with a teacher-free RGB inference path.

This is NOT SegVLAD or SCLIP reproduction. DINO features, not CLIP features,
are pooled. No class is discarded, and disconnected instances are not claimed
to be perfectly separated by the four fixed overlapping spatial bases.
"""
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.official_boq import native_feature_layout


ARMS = ('visual', 'aligned', 'shuffled')


def spatial_basis(height, width, device, dtype):
    y = (torch.arange(height, device=device, dtype=dtype) + .5) / height
    x = (torch.arange(width, device=device, dtype=dtype) + .5) / width
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    centers = ((.25, .25), (.25, .75), (.75, .25), (.75, .75))
    logits = torch.stack([-((yy-cy)**2 + (xx-cx)**2) / (2*.30**2)
                          for cy, cx in centers])
    return logits.flatten(1).softmax(dim=0)


def pooled_context(features, probabilities):
    """Each patch belongs softly to class x spatial slots; broadcast context.

    The assignment sums to one over slots, not over patches. Each slot's
    visual descriptor is then separately normalized by its own mass.
    """
    if features.ndim != 4 or probabilities.ndim != 4:
        raise ValueError('Expected BCHW features and class probabilities')
    b, c, h, w = features.shape
    if probabilities.shape[0] != b or probabilities.shape[-2:] != (h, w):
        raise ValueError('Feature/assignment geometry differs')
    basis = spatial_basis(h, w, features.device, features.dtype)
    weights = (probabilities.flatten(2)[:, :, None, :] * basis[None, None]).flatten(1, 2)
    tokens = features.flatten(2)
    mass = weights.sum(-1).clamp_min(1e-6)
    regions = torch.einsum('bsn,bcn->bsc', weights, tokens) / mass[:, :, None]
    context = torch.einsum('bsn,bsc->bcn', weights, regions).reshape(b, c, h, w)
    return context, mass


def permute_target(probabilities, keys, seed=42):
    """Fixed image-keyed derangement at TEACHER native grid, never batch keyed.

    Preserve each image's class masses and confidence distribution; destroy
    its pixel correspondence. All B/C/D runs consume the same original cache.
    """
    if probabilities.ndim != 4 or len(keys) != len(probabilities):
        raise ValueError('One stable image identity required per target')
    n = probabilities.shape[-2] * probabilities.shape[-1]
    if n < 2:
        raise ValueError('Need at least two target positions')
    result = []
    for row, key in zip(probabilities, keys):
        value = int(hashlib.sha256(f'srp:{seed}:{key}'.encode()).hexdigest()[:15], 16)
        generator = torch.Generator(device='cpu').manual_seed(value)
        perm = torch.randperm(n, generator=generator)
        while bool((perm == torch.arange(n)).any()):
            perm = torch.randperm(n, generator=generator)
        result.append(row.flatten(1)[:, perm.to(row.device)].reshape_as(row))
    return torch.stack(result)


def guidance_loss(logits, target):
    """Confidence-weighted KL, not feature/attention matching or dynamic masking."""
    target = F.interpolate(target.detach().float(), logits.shape[-2:],
                           mode='bilinear', align_corners=False).clamp_min(1e-7)
    target = target / target.sum(1, keepdim=True)
    entropy = -(target * target.log()).sum(1)
    confidence = (1 - entropy / math.log(target.shape[1])).clamp(0, 1).detach()
    kl = (target * (target.log() - logits.float().log_softmax(1))).sum(1)
    loss = (confidence * kl).sum() / confidence.sum().clamp_min(1e-6)
    return loss, confidence.mean().detach()


class SemanticSpatialPool(nn.Module):
    def __init__(self, channels=768, classes=18, hidden=128, rho=.10):
        super().__init__()
        if min(channels, hidden) < 1 or classes < 2 or not 0 < rho <= 1:
            raise ValueError('Invalid pooling dimensions or residual cap')
        self.norm = nn.LayerNorm(channels)
        self.router = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU(),
            nn.Conv2d(hidden, classes, 1))
        self.down = nn.Conv2d(channels, hidden, 1)
        self.up = nn.Conv2d(hidden, channels, 1, bias=False)
        nn.init.zeros_(self.up.weight)
        self.rho = float(rho)

    def forward(self, features):
        b, c, h, w = features.shape
        normalized = self.norm(features.flatten(2).transpose(1, 2))
        logits = self.router(normalized.transpose(1, 2).reshape(b, c, h, w))
        probabilities = logits.float().softmax(1)
        context, mass = pooled_context(features, probabilities)
        raw = self.up(F.gelu(self.down(context - features)))
        scale = features.detach().square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)
        delta = self.rho * scale * raw.tanh()
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1)
        diagnostics = dict(route_entropy=entropy.detach().mean(),
            min_slot_mass=mass.detach().min(), max_slot_mass=mass.detach().max(),
            delta_rms=delta.detach().square().mean().sqrt())
        return features + delta, logits, diagnostics


class SemanticRegionBoQ(nn.Module):
    """Register ONLY official student + pool. Never register a CLIP teacher."""
    def __init__(self, official, classes=18, hidden=128, rho=.10):
        super().__init__()
        self.visual = official
        self.visual.backbone.requires_grad_(False).eval()
        self.visual.aggregator.requires_grad_(True)
        self.pool = SemanticSpatialPool(classes=classes, hidden=hidden, rho=rho)

    def train(self, mode=True):
        super().train(mode)
        self.visual.backbone.eval()
        return self

    def encode(self, images):
        with torch.no_grad():
            return self.visual.backbone(images)

    def aggregate(self, features, auxiliary=False):
        output, logits, diagnostics = self.pool(features)
        descriptor = self.visual.aggregator(native_feature_layout(output))[0]
        if auxiliary:
            return descriptor, logits, diagnostics
        return descriptor

    def forward(self, images, auxiliary=False):
        return self.aggregate(self.encode(images), auxiliary)
