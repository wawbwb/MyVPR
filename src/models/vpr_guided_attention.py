"""Retrieval-trained spatial attention; not semantic-class suppression."""
import torch
from torch import nn


class VPRGuidedAttention(nn.Module):
    """Multi-receptive-field adaptation of SegVPR, on one DINO patch map.

    Zero output projection gives exactly unit weights at initialization.
    Segmentation must consume forward(features).detach(), so only retrieval
    trains this module. This is NOT an extraction of BoQ's query attention.
    """
    def __init__(self, channels, hidden=32):
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.GELU())
        self.scales = nn.ModuleList([
            nn.Sequential(nn.Conv2d(hidden, hidden, k, padding=k // 2), nn.GELU())
            for k in (3, 5, 7)
        ])
        self.output = nn.Conv2d(3 * hidden, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features):
        reduced = self.reduce(features)
        return 2 * torch.sigmoid(self.output(torch.cat([
            branch(reduced) for branch in self.scales
        ], dim=1)))
