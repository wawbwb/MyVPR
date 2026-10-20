"""Potentially-dynamic SAM-fragment prior, before fixed neighbour expansion.

No motion inference, no learned retrieval parameters, no changes to old methods.
CLIP softmax values below are prompt-relative scores, NOT calibrated confidence.
"""
import hashlib

import numpy as np
import torch
from torch.nn import functional as F

from src.region_vlad import neighbour_union


CLASSES = (
    'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian',
    'building facade', 'road surface', 'sidewalk', 'wall', 'tree', 'sky',
    'bridge', 'street sign', 'fence', 'street lamp',
    'car hood seen through a windshield', 'vehicle dashboard',
    'windshield wipers', 'unrecognizable object',
)
DYNAMIC_COUNT = 6
TEMPLATES = ('a photo of a {}.', 'a street scene containing a {}.')
ARMS = ('sam', 'center_only', 'pre', 'shuffled', 'uniform_mass')


def fragment_risk(cosines, scale, confidence=.60, margin=.02):
    """Require box AND masked-box views to agree on a dynamic top category.

    Gate with top-class softmax and raw cosine margin to ALL nondynamic
    competitors, including ego-vehicle parts. Uncertain/empty -> no intervention.
    """
    s = np.asarray(cosines, dtype=np.float64)
    if (s.ndim != 3 or s.shape[1:] != (2, len(CLASSES))
            or not np.isfinite(s).all() or not np.isfinite(scale)
            or scale <= 0 or not 0 <= confidence <= 1 or margin < 0):
        raise ValueError('Invalid fragment cosine scores or thresholds')
    if len(s) == 0:
        return np.empty(0, np.float32)
    logits = (s-s.max(-1, keepdims=True))*scale
    p = np.exp(logits)
    p /= p.sum(-1, keepdims=True)
    top = s.argmax(-1)
    raw_margin = s[:, :, :DYNAMIC_COUNT].max(-1)-s[:, :, DYNAMIC_COUNT:].max(-1)
    valid = ((top[:, 0] == top[:, 1]) & (top < DYNAMIC_COUNT).all(1)
             & (p.max(-1) >= confidence).all(1) & (raw_margin >= margin).all(1))
    return np.where(valid, p.max(-1).min(1), 0).astype(np.float32)


def pixel_risk(masks, risks):
    """Dynamic evidence wins overlaps: a building mask cannot dilute a car mask.

    Uncovered pixels remain weight 1. This conservative overlap choice can also
    propagate classification mistakes; save crops/scores for an independent audit.
    """
    m = np.asarray(masks, dtype=bool)
    r = np.asarray(risks, dtype=np.float32)
    if (m.ndim != 3 or r.shape != (len(m),) or not np.isfinite(r).all()
            or (r < 0).any() or (r > 1).any()):
        raise ValueError('Invalid masks/risks')
    return (m*r[:, None, None]).max(0) if len(m) else np.zeros(m.shape[1:], np.float32)


def interventions(masks, risks, image_key, strength=.5, order=1, grid=20):
    """Keep topology/count/order fixed; reweight contributions of raw fragments.

    All raw masks can contribute risk, including those overlapping a larger
    segment. No semantic mask deletion, no whole-SuperSegment deletion.
    """
    m = np.asarray(masks, bool)
    r = np.asarray(risks, np.float32)
    if not 0 <= strength < 1 or m.ndim != 3 or not len(m):
        raise ValueError('Need nonempty masks and bounded nonzero-floor strength')
    h, w = m.shape[1:]
    if h % grid or w % grid:
        raise ValueError('Mask/token geometry mismatch')
    # Validate risk before permutation / centre control, too.
    risk = pixel_risk(m, r)
    merged, fallback = neighbour_union(m, order)
    seed = int.from_bytes(hashlib.sha256(str(image_key).encode()).digest()[:8], 'little')
    shuffled = r[np.random.default_rng(seed).permutation(len(r))]
    maps = {'sam': np.zeros_like(risk), 'center_only': np.zeros_like(risk),
            'pre': risk, 'shuffled': pixel_risk(m, shuffled),
            'uniform_mass': np.full_like(risk, risk.mean())}
    # Pixel union avoids counting an overlap several times. Weighted pixel
    # coverage is averaged into each token, rather than ANY-pixel membership.
    base = merged.reshape(len(m), grid, h//grid, grid, w//grid).mean((2, 4)).reshape(len(m), -1)
    result = {}
    for arm in ARMS:
        weighted = merged*(1-strength*maps[arm][None])
        membership = weighted.reshape(len(m), grid, h//grid, grid, w//grid).mean((2, 4)).reshape(len(m), -1)
        reliability = weighted.sum((1, 2))/merged.sum((1, 2))
        if arm == 'center_only':
            reliability = 1-strength*r
        result[arm] = (membership.astype(np.float32), reliability.astype(np.float32))
    return result, {'delaunay_identity_fallback': bool(fallback),
        'dynamic_fragments': int((r > 0).sum()), 'pixel_risk_mean': float(risk.mean()),
        'shuffled_pixel_risk_mean': float(maps['shuffled'].mean()),
        'shuffled_assignment_changed': int((r != shuffled).sum()),
        'region_count': len(m), 'token_coverage_mean': float(base.mean())}


def weighted_vlad(tokens, centers, membership):
    """Soft membership multiplies residuals BEFORE intra-cluster normalization."""
    x = F.normalize(tokens.float(), dim=1)
    c = centers.to(x)
    m = torch.as_tensor(membership, device=x.device, dtype=x.dtype)
    if (x.ndim != 2 or c.ndim != 2 or x.shape[1] != c.shape[1]
            or m.ndim != 2 or m.shape[1] != len(x)
            or not torch.isfinite(m).all() or (m < 0).any() or (m.sum(1) <= 0).any()
            or not torch.isfinite(x).all() or not torch.isfinite(c).all()):
        raise ValueError('Invalid weighted VLAD inputs')
    labels = (x @ F.normalize(c, dim=1).T).argmax(1)
    residual = x-c[labels]
    parts = [F.normalize(m[:, labels == k] @ residual[labels == k], dim=1)
             for k in range(len(c))]
    z = F.normalize(torch.stack(parts, 1).flatten(1), dim=1)
    if not torch.isfinite(z).all() or (z.norm(dim=1) < 1e-8).any():
        raise ValueError('Degenerate weighted region VLAD')
    return z


def pair_score(q, d, qr, dr):
    """Symmetric region Chamfer; reliability retained OUTSIDE normalization.

    Cosine is clipped at zero. Fixed denominator prevents low reliability from
    disappearing through another weighted-mean normalization. No geometry/GT.
    """
    q, d = np.asarray(q, np.float32), np.asarray(d, np.float32)
    qr, dr = np.asarray(qr), np.asarray(dr)
    if not len(q) or not len(d):
        return 0.
    if qr.shape != (len(q),) or dr.shape != (len(d),):
        raise ValueError('Reliability shape mismatch')
    if not all(np.isfinite(a).all() for a in (q, d, qr, dr)) or any((a < 0).any() or (a > 1).any() for a in (qr, dr)):
        raise ValueError('Invalid region score inputs')
    s = np.maximum(q @ d.T, 0)*np.sqrt(qr[:, None]*dr[None, :])
    return float((s.max(1).mean()+s.max(0).mean())/2)


def fixed_ranking(global_scores, region_scores, alpha=.02):
    g, r = np.asarray(global_scores), np.asarray(region_scores)
    if g.shape != r.shape or not np.isfinite(g).all() or not np.isfinite(r).all():
        raise ValueError('Invalid scores')
    # Stable ties retain original RU order; no parameter search on retrieval GT.
    return np.argsort(-(g+alpha*r), kind='stable')
