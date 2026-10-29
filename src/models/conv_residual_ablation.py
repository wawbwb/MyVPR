"""Change only the output residual rule of the sealed no-SASF conv neck."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.hcen_followup import build_model as historical_model
from src.models.hcen_followup import module_gradient_status


ARMS = ('linear_residual', 'fixed_tanh', 'rms_tanh')
RHO = .1
PARAMETERS = 818112
DISABLED_AUXILIARY_PARAMETERS = 65536


def residual_from_raw(last, raw, arm, rho=RHO):
    """Same rho in every arm; RMS is detached, exactly as the historical neck."""
    if arm not in ARMS:
        raise ValueError('Unknown residual arm')
    if not math.isfinite(rho) or not 0 < rho < 1:
        raise ValueError('rho must be finite and between zero and one')
    if last.ndim != 4 or last.shape != raw.shape:
        raise ValueError('Residual inputs must have the same BCHW shape')
    if arm == 'linear_residual':
        return rho * raw
    if arm == 'fixed_tanh':
        return rho * raw.tanh()
    rms = last.square().mean(1, keepdim=True).sqrt().detach()
    return rho * rms * raw.tanh()


class ResidualRuleNeck(nn.Module):
    """Keep all registered names/weights and all fusion operations unchanged.

    last_raw is a transient detached diagnostic, not a parameter or buffer.
    Consumers read it outside activation checkpointing, so recomputation does
    not add duplicate observations to the experiment report.
    """
    def __init__(self, original, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Unknown residual arm')
        for name in ('norms', 'projections', 'stages', 'aux', 'up'):
            self.add_module(name, getattr(original, name))
        self.arm = arm
        self.last_raw = None

    def forward(self, layers, auxiliary=False):
        aligned = []
        for feature, norm, projection in zip(layers, self.norms, self.projections):
            b, _, h, w = feature.shape
            aligned.append(projection(norm(feature.flatten(2).transpose(1, 2)))
                           .transpose(1, 2).reshape(b, -1, h, w))
        first = self.stages[0](aligned[0], aligned[1])
        final = self.stages[1](first, aligned[2])
        raw = self.up(final)
        residual = residual_from_raw(layers[-1], raw, self.arm)
        self.last_raw = raw.detach()
        aux = tuple(F.normalize(head(x.mean((2, 3))), dim=-1)
                    for head, x in zip(self.aux, (first, final))) if auxiliary else ()
        return layers[-1] + residual, aux, residual


def build_model(visual, arm, initialization_seed, **kwargs):
    model = historical_model(visual, 'conv', initialization_seed, **kwargs)
    # No RNG draws, no new weights, no inherited selective scan or SASF path.
    model.neck = ResidualRuleNeck(model.neck, arm)
    model.residual_arm = arm
    return model


@torch.no_grad()
def image_diagnostics(last, raw, residual, descriptor, reference, arm, rho=RHO):
    """One row per image; distinguish absolute/relative and zero-energy cases."""
    if arm not in ARMS or last.shape != raw.shape or last.shape != residual.shape:
        raise ValueError('Invalid diagnostic shapes/arm')
    if descriptor.shape != reference.shape or descriptor.shape[0] != last.shape[0]:
        raise ValueError('Unmatched descriptors')
    if not all(bool(torch.isfinite(x).all()) for x in (last, raw, residual, descriptor, reference)):
        raise ValueError('Nonfinite residual or descriptor diagnostic')
    last_norm = last.flatten(1).norm(dim=1)
    residual_norm = residual.flatten(1).norm(dim=1)
    valid_image = last_norm > 0
    relative = residual_norm / torch.where(valid_image, last_norm, torch.ones_like(last_norm))
    last_patch = last.square().mean(1).sqrt()
    residual_patch = residual.square().mean(1).sqrt()
    valid_patch = last_patch > 0
    patch_ratio = residual_patch / torch.where(valid_patch, last_patch, torch.ones_like(last_patch))
    # NaN is only an internal missing-value mask. It is never written to JSON.
    masked_ratio = patch_ratio.flatten(1).masked_fill(~valid_patch.flatten(1), float('nan'))
    counts = valid_patch.flatten(1).sum(1)
    nonempty = counts > 0
    patch_mean = torch.where(valid_patch, patch_ratio, 0.).flatten(1).sum(1) / counts.clamp_min(1)
    patch_max = torch.where(valid_patch, patch_ratio, 0.).flatten(1).max(1).values
    patch_p95 = torch.nanquantile(masked_ratio, .95, dim=1)
    reference_zero_changed = (~valid_patch & (residual_patch > 0)).flatten(1).sum(1)
    potential_saturation = (raw.tanh().abs() >= .99).float().flatten(1).mean(1)
    result = dict(
        reference_l2=last_norm,
        residual_l2=residual_norm,
        image_relative_l2=relative,
        patch_relative_l2_mean=patch_mean,
        patch_relative_l2_p95=patch_p95,
        patch_relative_l2_max=patch_max,
        patch_relative_above_rho_count=(valid_patch & (patch_ratio > rho + 1e-6)).flatten(1).sum(1),
        valid_reference_patch_count=counts,
        zero_reference_patch_count=(~valid_patch).flatten(1).sum(1),
        zero_reference_nonzero_residual_patch_count=reference_zero_changed,
        zero_reference_image_nonzero_residual=(~valid_image & (residual_norm > 0)).long(),
        reference_rms_mean=last_patch.flatten(1).mean(1),
        raw_projection_rms=raw.square().flatten(1).mean(1).sqrt(),
        residual_absolute_rms=residual.square().flatten(1).mean(1).sqrt(),
        potential_tanh_saturation_fraction=potential_saturation,
        descriptor_cosine_distance=(1. - (descriptor * reference).sum(1)).clamp_min(0.),
        descriptor_l2_distance=(descriptor - reference).norm(dim=1),
    )
    if arm == 'fixed_tanh':
        violations = residual.abs() > rho + 1e-6
    elif arm == 'rms_tanh':
        violations = residual.abs() > rho * last_patch[:, None] + 1e-6
    else:
        violations = None
    result['element_bound_violation_fraction'] = (
        violations.float().flatten(1).mean(1) if violations is not None else None)
    rows = []
    cpu = {name: value.cpu().tolist() if value is not None else None for name, value in result.items()}
    image_validity = valid_image.cpu().tolist()
    patch_validity = nonempty.cpu().tolist()
    for index in range(last.shape[0]):
        row = {name: values[index] if values is not None else None for name, values in cpu.items()}
        if not image_validity[index]:
            row['image_relative_l2'] = None
        if not patch_validity[index]:
            for name in ('patch_relative_l2_mean', 'patch_relative_l2_p95', 'patch_relative_l2_max'):
                row[name] = None
        row['applied_tanh_saturation_fraction'] = None if arm == 'linear_residual' else row['potential_tanh_saturation_fraction']
        rows.append(row)
    if arm == 'rms_tanh':
        if any(row['patch_relative_above_rho_count'] or row['zero_reference_nonzero_residual_patch_count']
               or row['element_bound_violation_fraction'] > 0
               or row['image_relative_l2'] is not None and row['image_relative_l2'] > rho + 1e-6
               for row in rows):
            raise ValueError('RMS relative/element residual bound violated')
    elif arm == 'fixed_tanh' and any(row['element_bound_violation_fraction'] > 0 for row in rows):
        raise ValueError('Fixed-tanh absolute bound violated')
    return rows
