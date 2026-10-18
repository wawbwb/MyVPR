"""RGB-only pre-BoQ adaptation; CLIP variation adversary is training-only."""
import torch
from torch import nn

from src.models.clip_nuisance import Regressor, reverse, within_place
from src.models.official_boq import native_feature_layout

ARMS = ('visual', 'aligned', 'shuffled')


class PatchAdapter(nn.Module):
    def __init__(self, channels=768, hidden=64, gain=.1):
        super().__init__()
        if channels < 1 or hidden < 1 or not 0 < gain <= 1:
            raise ValueError('Invalid adapter size/gain')
        self.gain = float(gain)
        self.down = nn.Conv2d(channels, hidden, 1)
        self.spatial = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.up = nn.Conv2d(hidden, channels, 1)
        self.activation = nn.GELU()
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features):
        delta = self.up(self.activation(self.spatial(self.activation(self.down(features)))))
        return features + self.gain * delta


class PreBoQ(nn.Module):
    def __init__(self, official, channels=768, hidden=64, gain=.1):
        super().__init__()
        self.visual = official.requires_grad_(False).eval()
        self.adapter = PatchAdapter(channels, hidden, gain)

    def train(self, mode=True):
        super().train(mode)
        self.visual.eval()
        return self

    @torch.no_grad()
    def encode(self, images):
        return self.visual.backbone(images)

    def describe(self, features):
        adapted = self.adapter(features)
        descriptor = self.visual.aggregator(native_feature_layout(adapted))[0]
        return descriptor, adapted.mean((2, 3))

    def forward(self, images):
        return self.describe(self.encode(images))[0]


class MeanAdversary(nn.Module):
    def __init__(self, scale, hidden=128, targets=32):
        super().__init__()
        scale = torch.as_tensor(scale, dtype=torch.float32)
        if scale.ndim != 1 or not torch.isfinite(scale).all() or bool((scale <= 0).any()):
            raise ValueError('Need positive training-only feature scales')
        self.register_buffer('scale', scale.clone())
        self.head = Regressor(len(scale), hidden, targets)

    def forward(self, means, labels, adversarial=False):
        value = within_place(means, labels) / self.scale
        return self.head(reverse(value) if adversarial else value)


def replay_backward(model, features, descriptor_gradient, mean_gradient, microbatch):
    """Exact chain rule for deterministic per-image models, bounded graph memory.

    Call after differentiating the full-batch loss w.r.t. detached descriptor
    and mean leaves. Recompute each microbatch, including BOTH output paths.
    No optimizer update may happen between the first pass and this replay.
    """
    if len(features) != len(descriptor_gradient) or len(features) != len(mean_gradient):
        raise ValueError('Gradient-cache populations differ')
    for start in range(0, len(features), microbatch):
        stop = start + microbatch
        descriptor, means = model.describe(features[start:stop])
        torch.autograd.backward((descriptor, means),
                                (descriptor_gradient[start:stop], mean_gradient[start:stop]))
