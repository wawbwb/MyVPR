"""Fixed-basis Q/V adaptation in the last two DINO blocks; frozen BoQ."""
import torch
from torch import nn


class SubspaceQKV(nn.Module):
    def __init__(self, original, basis):
        super().__init__()
        d, r = basis.shape
        if original.in_features != d or original.out_features != 3*d:
            raise ValueError('Unexpected combined QKV dimensions')
        if not torch.isfinite(basis).all() or not torch.allclose(basis.T@basis, torch.eye(r, device=basis.device, dtype=basis.dtype), atol=1e-5):
            raise ValueError('Basis must be finite and orthonormal')
        self.original = original.requires_grad_(False)
        self.register_buffer('basis', basis.detach().float().clone())
        self.query = nn.Linear(r, d, bias=False)
        self.value = nn.Linear(r, d, bias=False)
        nn.init.zeros_(self.query.weight); nn.init.zeros_(self.value.weight)
        self.enabled = True

    def forward(self, x):
        original = self.original(x)
        if not self.enabled: return original
        low = x@self.basis
        dq, dv = self.query(low), self.value(low)
        return original+torch.cat((dq, torch.zeros_like(dq), dv), dim=-1)


class SubspaceVPR(nn.Module):
    def __init__(self, visual, bases):
        super().__init__()
        visual.requires_grad_(False)
        self.backbone, self.aggregator = visual.backbone, visual.aggregator
        self.gate = visual.semantic_region_gate
        dino = self.backbone.dino
        if len(dino.blocks) != 12 or getattr(dino, 'num_register_tokens', 0) != 0:
            raise ValueError('Expected twelve-block non-register RU DINO')
        if visual.spatial_attn_head is not None or self.backbone.crop_semantic_film is not None or self.backbone.residual_clip_fusion is not None:
            raise ValueError('Expected original plain RU model')
        if self.aggregator.semantic_num_classes is not None:
            raise ValueError('Semantic aggregation not supported')
        for index in (10, 11):
            dino.blocks[index].attn.qkv = SubspaceQKV(dino.blocks[index].attn.qkv, bases[index-10])

    @property
    def adapters(self):
        return [self.backbone.dino.blocks[i].attn.qkv for i in (10, 11)]

    def forward(self, images):
        if self.training: raise ValueError('Use eval mode even when optimizing adapters')
        b, _, h, w = images.shape
        if h % 14 or w % 14: raise ValueError('Image grid must be divisible by14')
        dino = self.backbone.dino
        # no_grad, NOT inference_mode: suffix backward may need this tensor.
        with torch.no_grad():
            x = dino.prepare_tokens_with_masks(images)
            for block in dino.blocks[:10]: x = block(x)
        # Do NOT detach or wrap this suffix in no_grad during optimization.
        for block in dino.blocks[10:]: x = block(x)
        features = x[:, 1:].transpose(1, 2).reshape(b, -1, h//14, w//14)
        if self.gate is not None: features = self.gate(features)[0]
        return self.aggregator(features)[0]
