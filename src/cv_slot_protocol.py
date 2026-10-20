"""Frozen correspondence preflight. No student model or retrieval training.

Candidate support is DINO-only; row permutation preserves each row's CLIP
weight multiset, NOT the entropy of the final joint distribution.
"""
import hashlib

import numpy as np
import torch
from torch.nn import functional as F

ARMS = ('dino', 'clip', 'shuffled')


def balanced_places(plan, split, count, seed):
    groups = {}
    for i, p in enumerate(plan['places']):
        if p['split'] == split:
            groups.setdefault(p['city'], []).append(i)
    if count < len(groups) or sum(map(len, groups.values())) < count:
        raise ValueError('Insufficient places for city-balanced sampling')
    def order(i):
        return hashlib.sha256(f"{seed}:{plan['places'][i]['place']}".encode()).digest()
    groups = {city: sorted(ids, key=order) for city, ids in sorted(groups.items())}
    result, cursor = [], 0
    while len(result) < count:
        for ids in groups.values():
            if cursor < len(ids):
                result.append(ids[cursor])
                if len(result) == count:
                    break
        cursor += 1
    return result


@torch.no_grad()
def correspondence(dino_a, dino_b, clip_a, clip_b, *, topk=8,
                   dino_temperature=.1, clip_temperature=.2, seed=42):
    if (dino_a.ndim != 2 or dino_b.ndim != 2 or clip_a.ndim != 2 or clip_b.ndim != 2
            or len(dino_a) != len(clip_a) or len(dino_b) != len(clip_b)
            or dino_a.shape[1] != dino_b.shape[1] or clip_a.shape[1] != clip_b.shape[1]
            or not 1 <= topk <= min(len(dino_a), len(dino_b))
            or not np.isfinite([dino_temperature, clip_temperature]).all()
            or min(dino_temperature, clip_temperature) <= 0):
        raise ValueError('Invalid correspondence shapes/temperatures/topk')
    values = (dino_a, dino_b, clip_a, clip_b)
    if any(not torch.isfinite(x).all() for x in values):
        raise ValueError('Nonfinite teacher features')
    da, db, ca, cb = [F.normalize(x.float(), dim=-1) for x in values]
    sim = da @ db.T
    logits = sim / dino_temperature
    # LoFTR-inspired confidence, not LoFTR features or verified geometry.
    confidence = logits.softmax(-1) * logits.softmax(-2)
    destinations = confidence.topk(topk, dim=-1).indices
    reverse = confidence.topk(topk, dim=0).indices
    reverse_support = torch.zeros_like(confidence, dtype=torch.bool)
    reverse_support.scatter_(0, reverse, True)
    support = reverse_support.gather(1, destinations)
    base = confidence.gather(1, destinations)
    c = (ca @ cb.T).gather(1, destinations)
    weight = ((c - c.max(-1, keepdim=True).values) / clip_temperature).exp()
    # Permute only within valid candidate edges. Do not move weight to an
    # invalid edge, which would alter the supported weight multiset.
    generator = torch.Generator(device='cpu').manual_seed(seed)
    random = torch.rand(weight.shape, generator=generator).to(weight.device)
    order = random.masked_fill(~support, 2.).argsort(-1)
    rank = (support.long().cumsum(-1)-1).clamp_min(0)
    indices = order.gather(1, rank)
    permuted = torch.where(support, weight.gather(1, indices), weight)
    def normalize(v):
        v = v * support
        return v / v.sum(-1, keepdim=True).clamp_min(1e-30)
    distributions = dict(dino=normalize(base), clip=normalize(base*weight),
                         shuffled=normalize(base*permuted))
    return dict(destinations=destinations, support=support, distributions=distributions,
                similarities=sim.gather(1, destinations),
                clip_weights=weight, shuffled_weights=permuted)


def crop_targets(grid=14, shift=2, reverse=False):
    if not 0 < shift < grid:
        raise ValueError('Invalid crop shift')
    r, c = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing='ij')
    if reverse:
        valid = (r < grid-shift) & (c < grid-shift)
        target = (r+shift)*grid + c+shift
    else:
        valid = (r >= shift) & (c >= shift)
        target = (r-shift)*grid + c-shift
    return valid.flatten(), target.flatten()


def geometry_metrics(graph, valid, targets):
    valid, targets = valid.to(graph['destinations'].device), targets.to(graph['destinations'].device)
    match = graph['destinations'].eq(targets[:, None]) & graph['support']
    result = dict(candidate_gt_coverage=float(match[valid].any(-1).float().mean()),
                  nonempty_rows=float(graph['support'][valid].any(-1).float().mean()))
    for arm, probability in graph['distributions'].items():
        correct = (probability * match).sum(-1)
        prediction = graph['destinations'].gather(1, probability.argmax(-1, keepdim=True)).flatten()
        nonempty = graph['support'].any(-1)
        result[arm] = dict(gt_probability=float(correct[valid].mean()),
            accuracy=float((prediction.eq(targets) & nonempty)[valid].float().mean()))
    return result


def graph_score(graph, arm):
    # Do NOT judge CLIP quality by its own cosine score. This heuristic uses
    # expected DINO similarity under each distribution, on fixed DINO support.
    valid = graph['support'].any(-1)
    if not valid.any():
        return 0.
    row = (graph['distributions'][arm] * graph['similarities']).sum(-1)
    return float(row[valid].mean())


def paired_interval(candidate, reference, *, seed=42, repeats=2000):
    delta = np.asarray(candidate, float) - np.asarray(reference, float)
    if delta.ndim != 1 or len(delta) < 2 or not np.isfinite(delta).all():
        raise ValueError('Need finite paired image/place observations')
    indices = np.random.default_rng(seed).integers(0, len(delta), (repeats, len(delta)))
    lo, hi = np.quantile(delta[indices].mean(1), [.025, .975])
    return dict(n=len(delta), mean=float(delta.mean()), ci95=[float(lo), float(hi)])


def decide(geometry, real, thresholds):
    """Fixed before extraction. CIs resample places, never individual patches."""
    comparisons = {arm: paired_interval([x['clip']['gt_probability'] for x in geometry],
        [x[arm]['gt_probability'] for x in geometry]) for arm in ('dino', 'shuffled')}
    means = {arm: float(np.mean([x[arm]['accuracy'] for x in geometry])) for arm in ARMS}
    real_means = {arm: float(np.mean([x[arm]['pairwise_win'] for x in real])) if real else None
                  for arm in ARMS}
    checks = dict(
        enough_geometry=len(geometry) >= thresholds['min_geometry_places'],
        enough_real_pairs=len(real) >= thresholds['min_real_places'],
        geometry_beats_dino=comparisons['dino']['mean'] >= thresholds['min_probability_gain']
            and comparisons['dino']['ci95'][0] > 0,
        geometry_beats_shuffled=comparisons['shuffled']['mean'] >= thresholds['min_probability_gain']
            and comparisons['shuffled']['ci95'][0] > 0,
        geometry_accuracy_noninferior=means['clip'] >= means['dino']-thresholds['accuracy_tolerance'],
        real_pair_noninferior=bool(real) and real_means['clip'] >= real_means['dino']-thresholds['real_pair_tolerance'],
        real_pair_beats_shuffled=bool(real) and real_means['clip'] > real_means['shuffled'])
    return dict(verdict='PASS' if all(checks.values()) else 'STOP_TEACHER_NOT_SUPPORTED',
        checks=checks, geometry_probability_paired=comparisons,
        geometry_accuracy=means, real_pairwise_win=real_means,
        scope='Teacher feasibility on reused GSV holdout; NOT a VPR improvement or a universal CLIP verdict')
