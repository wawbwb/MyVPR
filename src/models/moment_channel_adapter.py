"""Per-token channel moments, without spatial mixing or learned gates."""
import torch
from torch import nn
from torch.nn import functional as F
from src.models.structural_channel_adapter import StructuralChannelVPR, image_seeds

ARMS = ('channel', 'mean', 'norm', 'mean_norm')


def preserve_moments(features, residual, arm, eps=1e-6):
    if arm not in ARMS or features.ndim != 4 or features.shape != residual.shape:
        raise ValueError('Expected matched BCHW features and a registered arm')
    if arm == 'channel':
        return features + residual
    mean_preserved = arm in ('mean', 'mean_norm')
    base = features - features.mean(1, keepdim=True) if mean_preserved else features
    update = residual - residual.mean(1, keepdim=True) if mean_preserved else residual
    if arm == 'mean':
        return features + update
    radius = base.norm(dim=1, keepdim=True)
    moved = base + update
    moved_radius = moved.norm(dim=1, keepdim=True)
    # Difference form is exactly identity at zero initialization. Degenerate
    # tokens retain their original value, including an exact cancelling update.
    direction_change = moved / moved_radius.clamp_min(eps) - base / radius.clamp_min(eps)
    valid = (radius > eps) & (moved_radius > eps)
    return features + torch.where(valid, radius * direction_change, torch.zeros_like(base))


class MomentChannelAdapter(nn.Module):
    def __init__(self, arm, channels=768, hidden=64, gain=.1):
        super().__init__()
        if arm not in ARMS or min(channels, hidden) < 1 or not 0 < gain <= 1:
            raise ValueError('Invalid adapter settings')
        self.arm, self.gain = arm, float(gain)
        self.down = nn.Conv2d(channels, hidden, 1)
        self.middle = nn.Conv2d(hidden, hidden, 1, groups=hidden)
        self.up = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features, seeds=None):
        residual = self.gain * self.up(F.gelu(self.middle(F.gelu(self.down(features)))))
        adapted = preserve_moments(features, residual, self.arm)
        with torch.no_grad():
            x, y = features.detach(), adapted.detach()
            rms = x.square().mean(1).sqrt().clamp_min(1e-6)
            mean_error = ((y.mean(1)-x.mean(1)).abs()/rms).max()
            norm_error = ((y.norm(dim=1)-x.norm(dim=1)).abs()/x.norm(dim=1).clamp_min(1e-6)).max()
            stats = dict(mean_error=mean_error, norm_error=norm_error,
                relative_delta=(y-x).flatten(1).norm(dim=1)/x.flatten(1).norm(dim=1).clamp_min(1e-8))
        return adapted, stats


class MomentChannelVPR(StructuralChannelVPR):
    def __init__(self, anchor, arm, **kwargs):
        nn.Module.__init__(self)
        self.anchor = anchor.requires_grad_(False).eval()
        self.adapter = MomentChannelAdapter(arm, **kwargs)
