"""Last-layer anchored, input-conditioned cross-depth residuals.

This is a VPR adaptation inspired by the two private ReID manuscripts, not
their reproduction. No extra encoder, Mamba, semantic model, or reranker.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.conv_residual_ablation import residual_from_raw


ARMS = ('a_continue', 'b_last_condition', 'c_cross_condition')


class ConditionalBranch(nn.Module):
    def __init__(self, dim, channels, rank):
        super().__init__()
        self.local = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.mix = nn.Conv2d(3 * dim, dim, 1)
        self.down = nn.Conv2d(dim, rank, 1)
        self.spatial_gate = nn.Conv2d(3 * dim, 1, 1)
        self.channel_gate = nn.Sequential(
            nn.Linear(2 * dim, rank), nn.SiLU(), nn.Linear(rank, rank))
        self.up = nn.Conv2d(rank, channels, 1, bias=False)
        # Only this projection is zero: do not also zero a multiplicative gate.
        nn.init.zeros_(self.up.weight)

    def forward(self, source, anchor):
        local = source + .1 * F.gelu(self.local(source))
        joint = torch.cat((local, anchor, local - anchor), dim=1)
        content = F.silu(self.down(F.gelu(self.mix(joint))))
        spatial = self.spatial_gate(joint).sigmoid()
        pooled = torch.cat((source.mean((2, 3)), anchor.mean((2, 3))), dim=1)
        channel = self.channel_gate(pooled).sigmoid()[:, :, None, None]
        update = self.up(content * spatial * channel)
        stats = dict(spatial_gate_mean=spatial.detach().mean((1, 2, 3)),
                     spatial_gate_std=spatial.detach().flatten(1).std(1, unbiased=False),
                     channel_gate_mean=channel.detach().mean((1, 2, 3)),
                     channel_gate_std=channel.detach().flatten(1).std(1, unbiased=False),
                     update_rms=update.detach().square().mean((1, 2, 3)).sqrt())
        return update, stats


class ConditionalComplement(nn.Module):
    def __init__(self, channels=768, dim=128, rank=32):
        super().__init__()
        if any(type(x) is not int or x <= 0 for x in (channels, dim, rank)):
            raise ValueError('Positive integer channel dimensions required')
        if rank > dim:
            raise ValueError('Rank cannot exceed bottleneck width')
        self.channels = channels
        self.norms = nn.ModuleList([nn.LayerNorm(channels) for _ in range(3)])
        self.projections = nn.ModuleList([
            nn.Linear(channels, dim, bias=False) for _ in range(3)])
        self.branches = nn.ModuleList([
            ConditionalBranch(dim, channels, rank) for _ in range(2)])

    def forward(self, layers):
        if len(layers) != 3 or any(x.ndim != 4 or x.shape != layers[-1].shape
                                  for x in layers):
            raise ValueError('Three matched BCHW feature maps required')
        if layers[-1].shape[1] != self.channels:
            raise ValueError('Feature channel width differs')
        aligned = []
        for feature, norm, project in zip(layers, self.norms, self.projections):
            b, _, h, w = feature.shape
            aligned.append(project(norm(feature.flatten(2).transpose(1, 2)))
                           .transpose(1, 2).reshape(b, -1, h, w))
        updates, stats = [], {}
        for index, branch in enumerate(self.branches):
            update, row = branch(aligned[index], aligned[-1])
            updates.append(update)
            stats.update({f'branch{index}_{key}': value for key, value in row.items()})
        return updates[0] + updates[1], stats


class AnchoredConditionalNeck(nn.Module):
    """Preserve old names and operation order, including the old scaled cap.

    The anchor ALWAYS sees (L12,L12,L12). B and C have identical modules;
    only their complement input routing differs. Independent projections mean
    duplicated inputs in B are not forced to give zero feature differences.
    """
    def __init__(self, original, arm, channels=768, dim=128, rank=32):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Unknown conditional-complement arm')
        scale = getattr(original, 'output_scale', None)
        if scale is None or not math.isfinite(scale) or scale != .25:
            raise ValueError('Expected the sealed rho=.025 scaled-RMS anchor')
        for name in ('norms', 'projections', 'stages', 'aux', 'up'):
            self.add_module(name, getattr(original, name))
        self.output_scale = scale
        self.arm = arm
        # Existing auxiliary heads stay registered for checkpoint provenance,
        # but are genuinely frozen, not zero-gradient optimizer filler.
        self.aux.requires_grad_(False)
        self.complement = (None if arm == 'a_continue' else
                           ConditionalComplement(channels, dim, rank))
        self.last_raw = None
        self.last_conditional_stats = None

    def forward(self, layers, auxiliary=False):
        if len(layers) != 3 or any(x.ndim != 4 or x.shape != layers[-1].shape
                                  for x in layers):
            raise ValueError('Three matched BCHW feature maps required')
        last = layers[-1]
        aligned = []
        for norm, project in zip(self.norms, self.projections):
            b, _, h, w = last.shape
            aligned.append(project(norm(last.flatten(2).transpose(1, 2)))
                           .transpose(1, 2).reshape(b, -1, h, w))
        first = self.stages[0](aligned[0], aligned[1])
        final = self.stages[1](first, aligned[2])
        anchor_raw = self.up(final)
        if self.complement is None:
            raw = anchor_raw
            stats = {}
        else:
            inputs = ((last, last, last) if self.arm == 'b_last_condition' else layers)
            addition, stats = self.complement(inputs)
            raw = anchor_raw + addition
            stats['addition_rms'] = addition.detach().square().mean((1, 2, 3)).sqrt()
        stats['anchor_raw_rms'] = anchor_raw.detach().square().mean((1, 2, 3)).sqrt()
        # Keep multiplication order exactly as ScaledRMSNeck. Writing
        # .025*rms*tanh(raw) instead would change FP32 rounding at zero-start.
        residual = self.output_scale * residual_from_raw(last, raw, 'rms_tanh')
        self.last_raw = raw.detach()
        self.last_conditional_stats = stats
        aux = tuple(F.normalize(head(x.mean((2, 3))), dim=-1)
                    for head, x in zip(self.aux, (first, final))) if auxiliary else ()
        return last + residual, aux, residual


def anchor_parameters(model):
    return {name: p for name, p in model.named_parameters()
            if p.requires_grad and name.startswith('neck.')
            and not name.startswith('neck.complement.')}


def conditional_gradient_status(neck):
    if neck.complement is None:
        return {}
    groups = {}
    for i in range(2):
        for part in ('local', 'mix', 'down', 'spatial_gate', 'channel_gate', 'up'):
            prefix = f'branches.{i}.{part}.'
            groups[f'{i}_{part}'] = any(
                p.grad is not None and bool((p.grad != 0).any())
                for name, p in neck.complement.named_parameters() if name.startswith(prefix))
    for i in range(3):
        groups[f'projection{i}'] = any(
            p.grad is not None and bool((p.grad != 0).any())
            for p in neck.complement.projections[i].parameters())
    return groups
