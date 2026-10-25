"""Bounded group-wise mixing of two complete, aligned global descriptors.

Attention, scale weighting and descriptor fusion are prior art. Group indices
refer to BoQ's final FC output groups, not localized landmarks or query slots.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.models.complementary_covariance import image_seeds, permute_positions

ARMS = ('static_gate', 'global_gate', 'group_gate', 'shuffled_gate')


def agreement_features(low, high):
    """B,C,G -> B,G,6; all statistics are per image, with no gallery inputs."""
    if low.shape != high.shape or low.ndim != 3:
        raise ValueError('Matched BCG descriptors required')
    groups = low.shape[2]
    el, eh = low.square().sum(1), high.square().sum(1)
    cross = (low*high).sum(1)
    cosine = cross / (el*eh).sqrt().clamp_min(1e-8)
    balance = (eh-el)/(eh+el).clamp_min(1e-8)
    absolute = (high-low).abs().mean(1) / ((el+eh)/low.shape[1]).sqrt().clamp_min(1e-8)
    global_cosine = cross.sum(1, keepdim=True).expand(-1, groups)
    return torch.stack(((groups*el).clamp_min(1e-8).log(),
        (groups*eh).clamp_min(1e-8).log(), cosine, balance, absolute, global_cosine), -1)


def mix(low, high, gate, channels=384, groups=32):
    if low.shape != high.shape or low.ndim != 2 or low.shape[1] != channels*groups:
        raise ValueError('Matched flattened descriptors required')
    if gate.shape != (len(low), groups) or not torch.isfinite(gate).all() or float(gate.detach().abs().max()) > .500001:
        raise ValueError('Invalid mixing gate')
    l, h = low.reshape(-1, channels, groups), high.reshape(-1, channels, groups)
    # No extra normalization/projection of the existing branches. Zero gate
    # exactly uses the same sum and normalization as the previous mean baseline.
    raw = (l+h+gate[:, None, :]*(h-l)).flatten(1)
    if not torch.isfinite(raw).all() or bool((raw.norm(dim=1) < 1e-8).any()):
        raise ValueError('Degenerate fused descriptor')
    return F.normalize(raw, dim=1)


class ScaleAgreementFusion(nn.Module):
    def __init__(self, arm, channels=384, groups=32, hidden=16):
        super().__init__()
        if arm not in ARMS or min(channels, groups, hidden) < 1:
            raise ValueError('Invalid scale fusion')
        self.arm, self.channels, self.groups = arm, channels, groups
        self.down = nn.Linear(6, hidden)
        self.up = nn.Linear(hidden, 1, bias=False)
        nn.init.zeros_(self.up.weight)
        t = torch.arange(groups, dtype=torch.float32) * (2*math.pi/groups)
        self.register_buffer('static_code', torch.stack([f(k*t) for k in (1, 2, 3) for f in (torch.sin, torch.cos)], -1))

    def gates(self, low, high, seeds=None):
        if low.shape != high.shape or low.ndim != 2 or low.shape[1] != self.channels*self.groups:
            raise ValueError('Invalid two-scale descriptors')
        feature = agreement_features(low.reshape(-1, self.channels, self.groups), high.reshape(-1, self.channels, self.groups))
        if self.arm == 'static_gate':
            feature = self.static_code[None].expand(len(low), -1, -1)
        elif self.arm == 'global_gate':
            feature = feature.mean(1, keepdim=True).expand(-1, self.groups, -1)
        gate = .5*self.up(F.gelu(self.down(feature))).squeeze(-1).tanh()
        if self.arm == 'shuffled_gate':
            if seeds is None:
                raise ValueError('Image-content seeds required')
            gate = permute_positions(gate[:, :, None], seeds).squeeze(-1)
        return gate

    def forward(self, low, high, seeds=None):
        gate = self.gates(low, high, seeds)
        return mix(low, high, gate, self.channels, self.groups), gate
