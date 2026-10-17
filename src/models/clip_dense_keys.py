"""Frozen continuous SCLIP-equation keys, BEFORE text/class softmax.

Minimal last-block CSA adaptation, not an exact SCLIP segmentation reproduction.
Native 224-square, 14x14 grid, 512 channels; inference retains this encoder.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.models.official_boq import file_sha
from src.models.sclip_region_teacher import CLIP_SHA256, correlative_attention


class DenseClipKeys(nn.Module):
    def __init__(self, checkpoint, device='cpu'):
        super().__init__()
        checkpoint = Path(checkpoint).expanduser()
        if not checkpoint.is_file() or file_sha(checkpoint) != CLIP_SHA256:
            raise ValueError('Need pinned local OpenAI ViT-B-16.pt; no downloads')
        from open_clip.openai import load_openai_model
        base = load_openai_model(str(checkpoint), device='cpu', precision='fp32')
        self.visual = base.visual.requires_grad_(False).eval().to(device)
        if (self.visual.positional_embedding.shape != (197, 768)
                or tuple(self.visual.conv1.kernel_size) != (16, 16)):
            raise ValueError('Expected OpenAI ViT-B/16 at native 224')

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images):
        if tuple(images.shape[1:]) != (3, 224, 224):
            raise ValueError('Need CLIP-normalized B3x224x224 RGB')
        v = self.visual
        x = v.conv1(images.float()).flatten(2).transpose(1, 2)
        cls = v.class_embedding.reshape(1, 1, -1).expand(len(x), 1, -1)
        x = v.ln_pre(torch.cat((cls, x), 1) + v.positional_embedding)
        for block in v.transformer.resblocks[:-1]:
            x = block(x) if block.attn.batch_first else block(x.transpose(0, 1)).transpose(0, 1)
        block = v.transformer.resblocks[-1]
        x = x + block.ls_1(correlative_attention(block.attn, block.ln_1(x)))
        x = x + block.ls_2(block.mlp(block.ln_2(x)))
        x = F.normalize((v.ln_post(x) @ v.proj)[:, 1:].float(), dim=-1)
        return x.transpose(1, 2).reshape(len(x), 512, 14, 14)
