"""Offline OpenAI ViT-B/16 teacher with SCLIP's last-block CSA equation.

Reference: Wang et al., ECCV 2024, https://arxiv.org/abs/2312.01597
Equation: [softmax(QQ^T/sqrt(d)) + softmax(KK^T/sqrt(d))] V.
Own minimal OpenCLIP adapter, NOT an exact official segmentation reproduction:
224-square input, fixed prompts, no sliding windows, PAMR, or MMSegmentation.
No external code is downloaded/imported, and no base weights are downloaded.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.models.official_boq import file_sha


CLIP_SHA256 = '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
CLASSES = ('building', 'window', 'door', 'wall', 'roof', 'road', 'sidewalk',
           'traffic sign', 'street light', 'tree', 'grass', 'sky', 'water',
           'bridge', 'fence', 'vehicle', 'person', 'other scene content')
TEMPLATES = ('a photo of a {}.', 'a street scene with a {}.', 'a close-up photo of a {}.')


def correlative_attention(attention, tokens):
    """Input/output BNC, independent of MultiheadAttention.batch_first."""
    b, n, c = tokens.shape
    h = attention.num_heads
    q, k, v = F.linear(tokens, attention.in_proj_weight,
                        attention.in_proj_bias).chunk(3, dim=-1)
    q, k, v = [x.reshape(b, n, h, c//h).transpose(1, 2) for x in (q, k, v)]
    weight = ((q @ q.transpose(-1, -2)) * (c//h)**-.5).softmax(-1)
    weight = weight + ((k @ k.transpose(-1, -2)) * (c//h)**-.5).softmax(-1)
    output = (weight @ v).transpose(1, 2).reshape(b, n, c)
    return attention.out_proj(output)


class SCLIPRegionTeacher(nn.Module):
    def __init__(self, checkpoint, device='cpu'):
        super().__init__()
        import open_clip
        checkpoint = Path(checkpoint).expanduser()
        if not checkpoint.is_file() or file_sha(checkpoint) != CLIP_SHA256:
            raise ValueError('Need pinned existing OpenAI ViT-B-16.pt; no download fallback')
        # OpenAI checkpoints are JIT archives, not OpenCLIP state dictionaries.
        # load_openai_model accepts a local file and constructs QuickGELU correctly.
        from open_clip.openai import load_openai_model
        self.model = load_openai_model(str(checkpoint), device=device, precision='fp32')
        self.model.requires_grad_(False).eval()
        visual = self.model.visual
        if (tuple(visual.conv1.kernel_size) != (16, 16)
                or len(visual.transformer.resblocks) != 12
                or visual.positional_embedding.shape[0] != 197):
            raise ValueError('Teacher supports pinned ViT-B/16 at 224 only')
        tokenizer = open_clip.get_tokenizer('ViT-B-16')
        text = []
        with torch.no_grad():
            for label in CLASSES:
                value = self.model.encode_text(tokenizer(
                    [template.format(label) for template in TEMPLATES]).to(device))
                text.append(F.normalize(F.normalize(value.float(), dim=-1).mean(0), dim=0))
        self.register_buffer('text', torch.stack(text))

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images):
        if tuple(images.shape[1:]) != (3, 224, 224):
            raise ValueError('Teacher needs CLIP-normalized 224-square RGB')
        visual = self.model.visual
        x = visual.conv1(images.float()).flatten(2).transpose(1, 2)
        cls = visual.class_embedding.reshape(1, 1, -1).expand(len(x), 1, -1)
        x = visual.ln_pre(torch.cat((cls, x), dim=1) + visual.positional_embedding)
        blocks = visual.transformer.resblocks
        for block in blocks[:-1]:
            if block.attn.batch_first:
                x = block(x)
            else:
                x = block(x.transpose(0, 1)).transpose(0, 1)
        block = blocks[-1]
        x = x + block.ls_1(correlative_attention(block.attn, block.ln_1(x)))
        x = x + block.ls_2(block.mlp(block.ln_2(x)))
        patches = F.normalize((visual.ln_post(x) @ visual.proj)[:, 1:].float(), dim=-1)
        probabilities = (40. * (patches @ self.text.T)).softmax(-1)
        return probabilities.transpose(1, 2).reshape(-1, len(CLASSES), 14, 14)
