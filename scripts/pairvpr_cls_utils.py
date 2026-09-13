"""CPU-only geometry, fixed case selection and descriptive statistics."""
import hashlib
import numpy as np


def remap_fraction(fraction):
    f = np.asarray(fraction, dtype=np.float64)
    if f.shape != (20, 20) or not np.isfinite(f).all() or (f < 0).any() or (f > 1).any():
        raise ValueError('Expected finite 20x20 coverage in [0,1]')
    old = np.linspace(0, 1, 21)
    new = np.linspace(0, 1, 24)
    weights = np.maximum(0, np.minimum(new[1:, None], old[None, 1:])
                         - np.maximum(new[:-1, None], old[None, :-1])) * 23
    return np.clip(weights @ f @ weights.T, 0, 1).astype(np.float32)


def shuffled(fraction, identity, seed):
    digest = hashlib.sha256(f'{seed}:{identity}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'big'))
    return rng.permutation(fraction.reshape(-1)).reshape(fraction.shape)


def selected_pairs(refined, gt):
    cases, excluded = [], []
    for qi, (row, positives) in enumerate(zip(refined, gt)):
        pos = set(map(int, positives))
        yes = [int(d) for d in row if int(d) in pos]
        no = [int(d) for d in row if int(d) not in pos]
        if not yes or not no:
            excluded.append({'query': qi, 'reason': 'no_positive' if not yes else 'no_negative'})
            continue
        cases.append({'query': qi, 'positive': yes[0], 'negative': no[0],
                      'group': 'success' if int(row[0]) in pos else 'reachable_error'})
    return cases, excluded


def summarize(records):
    result = {'warning': 'Fixed GT-selected pairs; no recall estimate, no seed independence claim.',
              'groups': {}}
    for group in ['reachable_error', 'success']:
        rows = [r for r in records if r['group'] == group]
        if not rows:
            continue
        base = np.array([r['margins']['original'] for r in rows])
        stats = {}
        for name in rows[0]['margins']:
            values = np.array([r['margins'][name] for r in rows])
            delta = values-base
            stats[name] = {'mean_margin_delta': float(delta.mean()),
                           'median_margin_delta': float(np.median(delta)),
                           'improved': int((delta > 1e-4).sum()),
                           'worsened': int((delta < -1e-4).sum()),
                           'negative_to_positive': int(((base < -1e-4)&(values > 1e-4)).sum()),
                           'positive_to_negative': int(((base > 1e-4)&(values < -1e-4)).sum())}
        aligned = np.array([r['margins']['aligned'] for r in rows])
        random = np.array([[v for k,v in r['margins'].items() if k.startswith('shuffle_')]
                           for r in rows])
        result['groups'][group] = {'n': len(rows), 'variants': stats,
            'aligned_minus_shuffle_mean': float((aligned-random.mean(1)).mean()),
            'aligned_above_all_shuffles': int((aligned > random.max(1)+1e-4).sum()),
            'aligned_below_all_shuffles': int((aligned < random.min(1)-1e-4).sum())}
    return result
