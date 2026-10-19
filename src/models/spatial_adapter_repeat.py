"""Paired spatial/channel adapters, with no CLIP supervision or adversary."""
import torch
from torch import nn

from src.models.clip_preboq import PreBoQ

ARMS = ('spatial', 'channel')


def paired_models(official, hidden=64, gain=.1):
    spatial = PreBoQ(official, hidden=hidden, gain=gain)
    channel = PreBoQ(official, hidden=hidden, gain=gain)
    channel.adapter.load_state_dict(spatial.adapter.state_dict(), strict=True)
    previous = channel.adapter.spatial
    replacement = nn.Conv2d(hidden, hidden, 1, groups=hidden,
                            device=previous.weight.device, dtype=previous.weight.dtype)
    with torch.no_grad():
        replacement.weight.copy_(previous.weight[:, :, 1:2, 1:2])
        replacement.bias.copy_(previous.bias)
    channel.adapter.spatial = replacement
    # Shared official weights are read-only; adapters and optimizers are separate.
    return dict(spatial=spatial, channel=channel)


def adapter_inventory(models):
    return {a: dict(parameters=sum(p.numel() for p in m.adapter.parameters()),
        middle_kernel=list(m.adapter.spatial.kernel_size),
        middle_groups=m.adapter.spatial.groups) for a, m in models.items()}
