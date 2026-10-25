"""Frozen innovation-one with aligned cross-resolution difference injection.

Multi-resolution VPR is prior art. This is an exploratory residual mechanism,
not a claim that another image scale is a new contribution by itself.
"""
import torch
from torch import nn
from torch.nn import functional as F
from src.models.structural_channel_adapter import StructuralChannelVPR, image_seeds
from src.models.complementary_covariance import permute_positions

ARMS = ('degraded_detail', 'aligned_detail', 'shuffled_detail')
HIGH_SIZE = 518


def normalize_map(x):
    return F.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],)).permute(0, 3, 1, 2)


def scale_difference(low, high):
    if low.ndim != 4 or high.ndim != 4 or low.shape[:2] != high.shape[:2]:
        raise ValueError('Matched BCHW maps required')
    if any(a >= b for a, b in zip(low.shape[-2:], high.shape[-2:])):
        raise ValueError('Supplement must have a strictly finer grid')
    pooled = F.interpolate(high, size=low.shape[-2:], mode='area')
    return normalize_map(pooled) - normalize_map(low)


def degraded_high(images):
    if images.ndim != 4 or images.shape[-2:] != (322, 322):
        raise ValueError('Expected canonical normalized 322 images')
    # The normalized 322 tensor is the sole information source for this control.
    return F.interpolate(images, size=(HIGH_SIZE, HIGH_SIZE), mode='bicubic',
                         align_corners=False, antialias=True)


class DetailAdapter(nn.Module):
    def __init__(self, arm, channels=768, hidden=32, cap=.025):
        super().__init__()
        if arm not in ARMS or min(channels, hidden) < 1 or not 0 < cap < 1:
            raise ValueError('Invalid detail adapter')
        self.arm, self.cap = arm, cap
        self.down = nn.Conv2d(channels, hidden, 1, bias=False)
        self.up = nn.Conv2d(hidden, channels, 1, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, low, real_difference, degraded_difference, seeds=None):
        if low.shape != real_difference.shape or low.shape != degraded_difference.shape:
            raise ValueError('Difference grids must match the anchor')
        difference = degraded_difference if self.arm == 'degraded_detail' else real_difference
        if self.arm == 'shuffled_detail':
            if seeds is None:
                raise ValueError('Content-addressed image seeds required')
            difference = permute_positions(difference.flatten(2).transpose(1, 2), seeds)
            difference = difference.transpose(1, 2).reshape_as(low)
        raw = self.up(F.gelu(self.down(difference)))
        rms = low.detach().square().mean(1, keepdim=True).sqrt()
        delta = self.cap * rms * raw.tanh()
        relative = delta.detach().flatten(1).norm(dim=1) / low.detach().flatten(1).norm(dim=1).clamp_min(1e-8)
        if not torch.isfinite(delta).all() or float(relative.max()) > self.cap + 1e-6:
            raise ValueError('Detail residual bound violated')
        return low + delta, dict(relative_delta=relative,
            difference_rms=difference.detach().square().mean().sqrt(),
            saturation=(raw.detach().tanh().abs() >= .99).float().mean())


class CrossResolutionVPR(StructuralChannelVPR):
    def __init__(self, anchor, arm, **kwargs):
        nn.Module.__init__(self)
        self.anchor = anchor.requires_grad_(False).eval()
        self.adapter = DetailAdapter(arm, **kwargs)

    @torch.no_grad()
    def enhanced(self, images):
        from src.hcen_plain_protocol import canonical_features
        value, _, _ = self.anchor.neck(canonical_features(self.anchor.features(images)), auxiliary=False)
        self.anchor.neck.last_raw = None
        self.anchor.neck.last_conditional_stats = None
        return value

    @torch.no_grad()
    def encode(self, images, high_images):
        if high_images.shape != (len(images), 3, HIGH_SIZE, HIGH_SIZE):
            raise ValueError('Expected matched native-source 518 images')
        low, baseline = super().encode(images)
        high = self.enhanced(high_images)
        high_descriptor = self.anchor.readout(high)
        real = scale_difference(low, high)
        del high
        degraded = self.enhanced(degraded_high(images))
        difference = scale_difference(low, degraded)
        return low, baseline, real, difference, high_descriptor

    def describe(self, features, seeds):
        value, stats = self.adapter(features[0], features[2], features[3], seeds)
        return self.anchor.readout(value), stats
