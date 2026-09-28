"""Frozen BoQ with query-conditioned low-dimensional second-order residuals."""
import math
import torch
from torch import nn
from torch.nn import functional as F

MODES = ('mean_outer', 'global_cov', 'query_cov')


def moments(features, attention, mode):
    """B,N,R features, B,Q,N weights -> B,Q,R,R population moment.

    Global covariance is repeated, not separately learned per query. Mean outer
    uses only the first moment. No effective-sample correction or spatial claim.
    """
    if mode not in MODES: raise ValueError(mode)
    if mode == 'global_cov': attention = torch.full_like(attention, 1 / features.shape[1])
    mean = attention @ features
    outer = mean.unsqueeze(-1) * mean.unsqueeze(-2)
    if mode == 'mean_outer': return outer
    products = (features.unsqueeze(-1) * features.unsqueeze(-2)).flatten(-2)
    second = (attention @ products).reshape(*mean.shape[:-1], features.shape[-1], features.shape[-1])
    covariance = second - outer
    return (covariance + covariance.transpose(-1, -2)) * .5


def compact_moment(matrix):
    """Upper triangle, Frobenius-compatible weights, smooth power then L2.

    eps constants fixed across arms. This is NOT matrix-power normalization.
    """
    indices = torch.triu_indices(matrix.shape[-1], matrix.shape[-1], device=matrix.device)
    x = matrix[..., indices[0], indices[1]]
    scale = torch.where(indices[0] == indices[1], 1., math.sqrt(2.)).to(x.dtype)
    x = x * scale
    x = x / (x.abs() + 1e-4).sqrt()
    return F.normalize(x, dim=-1, eps=1e-6)


class SecondOrderAttention(nn.Module):
    def __init__(self, original, mode, rank=16, hidden=64):
        super().__init__()
        if mode not in MODES: raise ValueError(mode)
        if not original.batch_first or original.dropout != 0: raise ValueError('Expected plain BoQ attention')
        self.original = original.requires_grad_(False)
        dim = original.embed_dim
        self.feature_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.projection = nn.Linear(dim, rank, bias=False)
        self.hidden = nn.Sequential(nn.Linear(rank*(rank+1)//2, hidden), nn.GELU())
        self.output = nn.Linear(hidden, dim, bias=False)
        nn.init.zeros_(self.output.weight)
        self.mode = mode
        self.enabled = True
        self.last_residual_rms = None

    def forward(self, query, key, value, **kwargs):
        if any(v is not None for v in kwargs.values()): raise ValueError('Extra attention flags not supported')
        historical, attention = self.original(query, key, value)
        if not self.enabled: return historical, attention
        # Average the original heads; the historical per-head value path is untouched.
        features = self.projection(self.feature_norm(key))
        moment = moments(features, attention, self.mode)
        residual = self.output(self.hidden(compact_moment(moment)))
        self.last_residual_rms = residual.detach().square().mean().sqrt()
        return historical + residual, attention


def install(aggregator, mode):
    if aggregator.semantic_num_classes is not None: raise ValueError('Expected plain RU BoQ')
    aggregator.requires_grad_(False)
    adapters = []
    for block in aggregator.boqs:
        block.cross_attn = SecondOrderAttention(block.cross_attn, mode)
        adapters.append(block.cross_attn)
    return adapters
