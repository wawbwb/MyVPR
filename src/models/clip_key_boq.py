"""Add semantic Keys to frozen author BoQ; never replace its visual Values.

This is an isolated experimental wrapper, not a modification of innovation 1.
"""
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

ARMS = ('visual', 'aligned', 'shuffled')


def permute_patches(patches, names, seed=42):
    """Deterministic image-keyed derangement; preserve each token/channel vector."""
    if patches.ndim != 4 or len(names) != len(patches):
        raise ValueError('Expected BCHW features and one image name per row')
    n = patches.shape[-2] * patches.shape[-1]
    if n < 2:
        raise ValueError('Need at least two positions')
    parts = []
    for value, name in zip(patches, names):
        key = int.from_bytes(hashlib.sha256(f'clip-key:{seed}:{name}'.encode()).digest()[:8], 'little')
        generator = torch.Generator().manual_seed(key)
        order = torch.randperm(n, generator=generator)
        mapping = torch.empty(n, dtype=torch.long)
        mapping[order] = order.roll(1)
        parts.append(value.flatten(1)[:, mapping.to(value.device)].reshape_as(value))
    return torch.stack(parts)


class KeyRouter(nn.Module):
    def __init__(self, query_channels, num_queries, semantic_channels=512,
                 hidden=64, cap=2.0, temperature=5.0):
        super().__init__()
        if hidden < 1 or not math.isfinite(cap) or cap <= 0 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Invalid routing dimensions/scale')
        self.query = nn.Linear(query_channels, hidden, bias=False)
        self.key = nn.Linear(semantic_channels, hidden, bias=False)
        self.beta = nn.Parameter(torch.zeros(num_queries))
        self.cap, self.temperature = float(cap), float(temperature)

    def forward(self, queries, semantics):
        # Remove image-global CLIP bias before using continuous local evidence.
        semantics = semantics - semantics.mean(1, keepdim=True)
        semantics = F.layer_norm(semantics, (semantics.shape[-1],))
        q = F.normalize(self.query(queries), dim=-1)
        k = F.normalize(self.key(semantics), dim=-1)
        logits = (q @ k.transpose(1, 2)) * self.temperature
        logits = logits - logits.mean(-1, keepdim=True)
        return self.cap * self.beta.tanh()[None, :, None] * logits.tanh()


class ClipKeyBoQ(nn.Module):
    """RGB DINO features + aligned CLIP grid -> one normalized global vector.

    All original backbone/BoQ parameters and running modes remain frozen. The
    visual arm feeds a fixed random projection of DINO into the SAME router.
    The CLIP encoder is external: frozen, required at inference, not distilled.
    """
    def __init__(self, official, hidden=64, cap=2.0, temperature=5.0, seed=42):
        super().__init__()
        self.visual = official.requires_grad_(False).eval()
        agg = self.visual.aggregator
        self.routers = nn.ModuleList([
            KeyRouter(block.queries.shape[-1], block.queries.shape[1],
                      hidden=hidden, cap=cap, temperature=temperature)
            for block in agg.boqs])
        generator = torch.Generator().manual_seed(seed + 991)
        projection = torch.randn(agg.proj_c.in_channels, 512, generator=generator)
        self.register_buffer('visual_projection', F.normalize(projection, dim=0))

    def train(self, mode=True):
        super().train(mode)
        self.visual.eval()
        return self

    @torch.no_grad()
    def encode(self, images):
        return self.visual.backbone(images)

    def aggregate(self, features, clip_grid, arm='aligned', auxiliary=False):
        if arm not in ARMS:
            raise ValueError('Unknown arm')
        if features.ndim != 4 or not torch.isfinite(features).all():
            raise ValueError('Invalid visual features')
        b, _, h, w = features.shape
        if arm == 'visual':
            with torch.no_grad():
                semantics = features.flatten(2).transpose(1, 2) @ self.visual_projection
        else:
            if (clip_grid is None or clip_grid.ndim != 4 or clip_grid.shape[:2] != (b, 512)
                    or not torch.isfinite(clip_grid).all()):
                raise ValueError('Aligned/shuffled arm needs finite B512HW CLIP features')
            # Spatial shuffling is applied BEFORE interpolation by the caller.
            semantics = F.interpolate(clip_grid.detach().float(), size=(h, w),
                                      mode='bilinear', align_corners=False).flatten(2).transpose(1, 2)
        agg = self.visual.aggregator
        with torch.no_grad():
            x = agg.norm_input(agg.proj_c(features).flatten(2).permute(0, 2, 1))
        outs, stats = [], []
        for block, router in zip(agg.boqs, self.routers):
            with torch.no_grad():
                x = block.encoder(x)
                q = block.queries.repeat(b, 1, 1)
                q = block.norm_q(q + block.self_attn(q, q, q)[0])
            bias = router(q, semantics)
            mask = bias[:, None].expand(b, block.cross_attn.num_heads, q.size(1), x.size(1))
            mask = mask.reshape(b * block.cross_attn.num_heads, q.size(1), x.size(1))
            if not bool(torch.count_nonzero(router.beta.detach())):
                # CUDA MHA may choose a different kernel for an all-zero mask.
                # Preserve the EXACT unmasked forward, but use the mathematical
                # biased-attention derivative to open the zero-initialized gate.
                with torch.no_grad():
                    out = block.cross_attn(q, x, x)[0]
                if torch.is_grad_enabled():
                    guided = block.cross_attn(q, x, x, attn_mask=mask)[0]
                    out = out + (guided - guided.detach())
            else:
                out = block.cross_attn(q, x, x, attn_mask=mask)[0]
            outs.append(block.norm_out(out))
            stats.append(torch.stack((bias.detach().abs().mean(), bias.detach().abs().max(),
                                       router.beta.detach().tanh().abs().max())))
        value = agg.fc(torch.cat(outs, 1).permute(0, 2, 1)).flatten(1)
        value = F.normalize(value, dim=-1)
        return (value, torch.stack(stats).mean(0)) if auxiliary else value

    def forward(self, images, clip_grid=None, arm='aligned'):
        return self.aggregate(self.encode(images), clip_grid, arm)
