"""Frozen-BoQ attention calibration; no depth fusion or semantic inputs."""
import math
import torch
from torch import nn
from torch.nn import functional as F

MODES = ('temperature', 'competitive', 'shuffled')


def calibrate(logits, strength, mode):
    if mode not in MODES:
        raise ValueError(mode)
    if logits.shape[-2] < 2:
        raise ValueError('Competition needs at least two queries')
    a = logits.softmax(-1)
    scale = strength[None, :, None, None]
    if mode == 'temperature':
        # Signed log inverse-temperature, initialized at zero.
        return (logits * scale.exp()).softmax(-1)
    crowd = (a.sum(-2, keepdim=True) - a) / (a.shape[-2] - 1)
    if mode == 'shuffled':
        # Fixed seed, common spatial permutation for all images/heads/queries.
        gen = torch.Generator(device='cpu').manual_seed(42017)
        order = torch.randperm(a.shape[-1], generator=gen).to(a.device)
        crowd = crowd[..., order]
    return (logits - scale * torch.log1p(a.shape[-1] * crowd)).softmax(-1)


@torch.no_grad()
def redundancy_metrics(attention, slots):
    """Per-image statistics; overlap averaged within heads, not across heads."""
    q = attention.shape[-2]
    normalized = F.normalize(attention.float(), dim=-1)
    gram = normalized @ normalized.transpose(-1, -2)
    overlap = (gram.sum((-1, -2)) - gram.diagonal(dim1=-2, dim2=-1).sum(-1)) / (q * (q - 1))
    entropy = -(attention.float() * attention.float().clamp_min(1e-30).log()).sum(-1)
    singular = torch.linalg.svdvals(slots.float())
    probs = singular / singular.sum(-1, keepdim=True).clamp_min(1e-12)
    rank = (-(probs * probs.clamp_min(1e-30).log()).sum(-1)).exp()
    return torch.stack((overlap.mean(1), entropy.mean((1, 2)) / math.log(attention.shape[-1]), rank), -1)


class CompetitiveAttention(nn.Module):
    """Drop-in for plain BoQ cross attention, with a frozen historical anchor.

    Optimizer must call project_() after every step for nonnegative competition.
    Projected optimization leaves an ordinary nonzero derivative at strength=0.
    """
    def __init__(self, original, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        if (not original.batch_first or not original._qkv_same_embed_dim
                or original.bias_k is not None or original.bias_v is not None
                or original.add_zero_attn or original.dropout != 0):
            raise ValueError('Only plain dropout-free batch-first BoQ MHA supported')
        self.original = original.requires_grad_(False)
        self.strength = nn.Parameter(torch.zeros(original.num_heads))
        self.mode = mode
        self.collect = False
        self.last_metrics = None

    @torch.no_grad()
    def project_(self):
        self.strength.clamp_(-2 if self.mode == 'temperature' else 0, 2)

    def forward(self, query, key, value, **kwargs):
        if any(v is not None for v in kwargs.values()):
            raise ValueError('Attention masks/extra flags unsupported in this screen')
        m = self.original
        b, q, d = query.shape
        n = key.shape[1]
        heads = m.num_heads
        wq, wk, wv = m.in_proj_weight.chunk(3)
        biases = (None, None, None) if m.in_proj_bias is None else m.in_proj_bias.chunk(3)
        def split(x):
            return x.reshape(b, -1, heads, d // heads).transpose(1, 2)
        queries = split(F.linear(query, wq, biases[0]))
        keys = split(F.linear(key, wk, biases[1]))
        values = split(F.linear(value, wv, biases[2]))
        logits = (queries / math.sqrt(d // heads)) @ keys.transpose(-1, -2)
        original_attention = logits.softmax(-1)
        attention = calibrate(logits, self.strength, self.mode)
        def pool(weights):
            x = (weights @ values).transpose(1, 2).reshape(b, q, d)
            return m.out_proj(x)
        # Identical original-attention computations give EXACT zero correction
        # at initialization, while gradients reach strength immediately.
        historical, _ = m(query, key, value)
        correction = pool(attention) - pool(original_attention)
        if not bool(torch.count_nonzero(self.strength.detach())):
            # Autograd/non-autograd Linear paths may round differently even
            # with identical weights. Remove only the numerical offset at the
            # mathematical zero point; keep the complete correction derivative.
            correction = correction - correction.detach()
        out = historical + correction
        if self.collect:
            self.last_metrics = redundancy_metrics(attention.detach(), out.detach()).cpu()
        return out, attention.mean(1)


def install(aggregator, mode):
    if aggregator.semantic_num_classes is not None:
        raise ValueError('Expected historical plain BoQ')
    aggregator.requires_grad_(False)
    modules = []
    for block in aggregator.boqs:
        block.cross_attn = CompetitiveAttention(block.cross_attn, mode)
        modules.append(block.cross_attn)
    return modules
