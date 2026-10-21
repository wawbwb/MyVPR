"""Orthogonal spatial-DC decomposition, fixed checkpoint, no learned weights."""
import torch

VARIANTS=('innovation1','aligned','centered','mean_only')


def decompose(anchor,adapted):
    if anchor.shape!=adapted.shape or anchor.ndim!=4:
        raise ValueError('Matched BCHW feature maps required')
    if not torch.isfinite(anchor).all() or not torch.isfinite(adapted).all():
        raise ValueError('Nonfinite feature maps')
    residual=adapted-anchor
    dc=residual.mean((2,3),keepdim=True)
    centered=residual-dc
    norm=anchor.flatten(1).norm(dim=1).clamp_min(1e-20)
    energy=residual.square().sum((1,2,3))
    dc_energy=dc.square().sum((1,2,3))*anchor.shape[-2]*anchor.shape[-1]
    nonzero=energy>1e-20
    statistics=dict(residual_relative=residual.flatten(1).norm(dim=1)/norm,
        centered_relative=centered.flatten(1).norm(dim=1)/norm,
        dc_relative=dc.expand_as(residual).flatten(1).norm(dim=1)/norm,
        dc_energy_fraction=torch.where(nonzero,dc_energy/energy.clamp_min(1e-20),torch.zeros_like(energy)),
        residual_nonzero=nonzero)
    return dict(innovation1=anchor,aligned=adapted,centered=anchor+centered,mean_only=anchor+dc),statistics
