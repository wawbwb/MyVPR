"""Simplified depth/mixer controls; sealed 512-step implementations stay intact."""
from torch import nn
from src.models.hcen_ablation import build_model as build_original
from src.models.hcen_ablation import module_gradient_status as original_gradients

ARMS = ('scan', 'linear', 'conv', 'last_only')
COUNTS = dict(scan=817024, linear=594432, conv=818112, last_only=817024)


def build_model(visual, arm, initialization_seed, **kwargs):
    if arm not in ARMS:
        raise ValueError('Unknown simplified follow-up arm')
    variant = {'scan': 'no_sasf', 'last_only': 'no_sasf',
               'linear': 'linear', 'conv': 'conv'}[arm]
    model = build_original(visual, variant, initialization_seed, **kwargs)
    if arm == 'conv':
        for stage in model.neck.stages:
            stage.spatial = nn.Identity()
    # Same scan neck/weights/capacity; ONLY the input depth changes.
    if arm == 'last_only':
        model.mode = 'last_only'
    model.followup_arm = arm
    return model


def module_gradient_status(model):
    return original_gradients(model)
