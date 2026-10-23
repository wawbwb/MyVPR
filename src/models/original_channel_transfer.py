"""Reuse the archived channel adapter, without moment constraints or gates."""
from copy import deepcopy
import torch
from torch import nn
from src.models.structural_channel_adapter import StructuralChannelVPR, image_seeds

ARMS = ('direct', 'original_init', 'transferred')


class OriginalChannelVPR(StructuralChannelVPR):
    def __init__(self, anchor, adapter):
        nn.Module.__init__(self)
        self.anchor = anchor.requires_grad_(False).eval()
        self.adapter = deepcopy(adapter).requires_grad_(True)

    def describe(self, features, seeds=None):
        adapted = self.adapter(features[0])
        with torch.no_grad():
            x = features[0].detach()
            relative = (adapted.detach()-x).flatten(1).norm(dim=1)/x.flatten(1).norm(dim=1).clamp_min(1e-8)
        return self.anchor.readout(adapted), dict(relative_delta=relative)
