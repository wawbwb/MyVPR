"""Frozen BoQ routing with head-wise local value adaptation."""
import math
import torch
from torch import nn
from torch.nn import functional as F

MODES = ('local_conv', 'local_contrast', 'shuffled_contrast')
OFFSETS = tuple((y, x) for y in range(3) for x in range(3) if (y, x) != (1, 1))


def neighborhood(x, weight, difference):
    """Eight neighbors, replicate boundary; same learned weights in all arms."""
    h, w = x.shape[-2:]
    padded = F.pad(x, (1, 1, 1, 1), mode='replicate')
    out = torch.zeros_like(x)
    for j, (y, z) in enumerate(OFFSETS):
        v = padded[:, :, y:y+h, z:z+w]
        out = out + weight[j][None, :, None, None] * (v-x if difference else v)
    return out


class LocalValueAttention(nn.Module):
    def __init__(self, original, mode, rank=32, grid=20):
        super().__init__()
        if mode not in MODES or not original.batch_first or original.dropout != 0:
            raise ValueError('Expected supported mode and plain batch-first attention')
        self.original = original.requires_grad_(False)
        dim = original.embed_dim
        self.feature_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.projection = nn.Linear(dim, rank, bias=False)
        self.kernel = nn.Parameter(torch.empty(8, rank))
        nn.init.normal_(self.kernel, std=1/math.sqrt(8))
        self.output = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.output.weight)
        gen = torch.Generator().manual_seed(42019)
        perm = torch.randperm(grid*grid, generator=gen)
        self.register_buffer('permutation', perm)
        self.register_buffer('inverse', perm.argsort())
        self.grid, self.mode, self.enabled = grid, mode, True
        self.last_residual_rms = None

    def forward(self, query, key, value, **kwargs):
        if any(v is not None for v in kwargs.values()):
            raise ValueError('Masks/extra flags not supported in this frozen-RU experiment')
        historical, attention = self.original(query, key, value, average_attn_weights=False)
        if not self.enabled:
            return historical, attention.mean(1)
        b, n, dim = value.shape
        if n != self.grid**2:
            raise ValueError('Patch grid differs from the registered experiment')
        u = self.projection(self.feature_norm(value))
        if self.mode == 'shuffled_contrast': u = u[:, self.permutation]
        u = u.transpose(1, 2).reshape(b, -1, self.grid, self.grid)
        local = neighborhood(u, self.kernel, self.mode != 'local_conv').flatten(2).transpose(1, 2)
        if self.mode == 'shuffled_contrast': local = local[:, self.inverse]
        # Add to already-projected per-head values, NOT to keys or query logits.
        dv = self.output(F.gelu(local))
        heads = self.original.num_heads
        dv = dv.reshape(b, n, heads, dim//heads).transpose(1, 2)
        delta = (attention @ dv).transpose(1, 2).reshape(b, query.shape[1], dim)
        delta = F.linear(delta, self.original.out_proj.weight, bias=None)
        self.last_residual_rms = delta.detach().square().mean().sqrt()
        return historical + delta, attention.mean(1)


def install(aggregator, mode, grid=20):
    if aggregator.semantic_num_classes is not None:
        raise ValueError('Expected plain RU BoQ')
    aggregator.requires_grad_(False)
    adapters = []
    for block in aggregator.boqs:
        block.cross_attn = LocalValueAttention(block.cross_attn, mode, grid=grid)
        adapters.append(block.cross_attn)
    return adapters
