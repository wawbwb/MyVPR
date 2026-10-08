"""Frozen RU plus a bounded tangential spatial-SSM correction.

This is a research hypothesis, not a reproduction of MambaAdapt. The backbone,
RU gate and BoQ are frozen. Only the spatial mixer learns; frozen BoQ remains
differentiable with respect to its inputs on the correction path.
"""
import torch
from torch import nn
from torch.nn import functional as F
from src.models.scan_mamba import SpatialMixer

MODES = ('conv_preserved', 'mamba_plain', 'mamba_preserved')


def bounded_tangent(reference, proposal, cap=.1):
    """||delta|| < cap and delta perpendicular to the unit reference.

    Smooth shrinkage (no clamp dead zone). Thus cosine(output, reference) is
    at least 1/sqrt(1+cap**2), up to floating-point error. NOT a rank guarantee.
    """
    if not 0 < cap < 1:
        raise ValueError('Expected residual cap in (0,1)')
    delta = proposal - reference
    delta = delta - (delta*reference).sum(-1, keepdim=True)*reference
    delta = delta * (cap/(cap + delta.norm(dim=-1, keepdim=True)))
    return F.normalize(reference+delta, dim=-1), delta


def preservation_loss(descriptor, reference, temperature=.07):
    """Batch relation KL + descriptor anchoring; no labels on validation sets.

    Teacher and student use exactly the same augmented images. Diagonal pairs
    excluded. T^2 scaling is explicit; report KL and drift separately.
    """
    if len(descriptor) < 2 or temperature <= 0:
        raise ValueError('At least two descriptors and positive temperature required')
    reference = reference.detach()
    mask = torch.eye(len(descriptor), device=descriptor.device, dtype=torch.bool)
    teacher = ((reference@reference.T)/temperature).masked_fill(mask, -1e4)
    student = ((descriptor@descriptor.T)/temperature).masked_fill(mask, -1e4)
    kl = F.kl_div(F.log_softmax(student, -1), F.softmax(teacher, -1), reduction='batchmean')
    # Roundoff can yield an insignificantly negative KL; do not clamp its gradient.
    drift = .5*(descriptor-reference).square().sum(-1).mean()
    return temperature**2*kl, drift


class ProtectedMambaVPR(nn.Module):
    def __init__(self, visual, mode, cap=.1):
        super().__init__()
        from src.models.depth_query import DepthQueryVPR
        if mode not in MODES:
            raise ValueError('Unknown protected-Mamba mode')
        self.mode, self.cap = mode, cap
        self.base = DepthQueryVPR(visual, 'baseline').requires_grad_(False)
        self.mixer = SpatialMixer(self.base.aggregator.proj_c.out_channels,
                                  'conv' if mode == 'conv_preserved' else 'mamba')
        self.base.eval()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def features(self, images):
        return self.base.features(images)[-1]

    def readout(self, tokens):
        agg = self.base.aggregator
        outs = []
        for block in agg.boqs:
            tokens, slots, _ = block(tokens)
            outs.append(slots)
        return F.normalize(agg.fc(torch.cat(outs, 1).transpose(1, 2)).flatten(1), dim=-1)

    def aggregate(self, features, bypass=False):
        with torch.no_grad():
            agg = self.base.aggregator
            tokens = agg.norm_input(agg.proj_c(features).flatten(2).transpose(1, 2))
            reference = self.readout(tokens)
        if bypass:
            return reference, reference, torch.zeros_like(reference)
        row, col = self.mixer(tokens, *features.shape[-2:])
        proposal = self.readout(row)
        if self.mode != 'conv_preserved':
            proposal = F.normalize(proposal+self.readout(col), dim=-1)
        descriptor, delta = bounded_tangent(reference, proposal, self.cap)
        return descriptor, reference, delta

    def forward(self, images):
        return self.aggregate(self.features(images))[0]
