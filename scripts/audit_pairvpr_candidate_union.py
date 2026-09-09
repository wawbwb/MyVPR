"""Offline candidate reachability only: Pair100 + RU20 versus Pair120."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--msls-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Choose a fresh output directory')
    audit, root = args.audit, args.msls_path
    summary = json.loads((audit / 'summary.json').read_text(encoding='utf8'))
    done = json.loads((audit / 'completed.json').read_text(encoding='utf8'))
    provenance = json.loads((audit / 'provenance.json').read_text(encoding='utf8'))
    require(summary['complete'] and done['complete'], 'Incomplete source audit')
    require(sha(audit / 'summary.json') == done['summary_sha256'], 'Summary changed')
    require(sha(audit / 'predictions.npz') == summary['predictions_sha256'], 'Predictions changed')
    for name, expected in provenance['index_sha256'].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()), 'Unsafe index path')
        require(sha(path) == expected, f'Index mismatch: {name}')
    gt = np.load(root / 'msls_val_gt_25m.npy', allow_pickle=True)
    with np.load(audit / 'predictions.npz') as data:
        g, r, u = [data[k].copy() for k in ('global_ids', 'refined_ids', 'ru_ids')]
    n, ndb = summary['queries'], summary['references']
    for arr in (g, r, u):
        require(arr.ndim == 2 and len(arr) == n and np.issubdtype(arr.dtype, np.integer), 'Invalid ranking shape/dtype')
        require((arr >= 0).all() and (arr < ndb).all(), 'Invalid candidate ID')
        require(all(len(np.unique(row)) == len(row) for row in arr), 'Duplicate ranking IDs')
    require(g.shape[1] >= 120 and u.shape[1] >= 20 and r.shape[1] == 100, 'Requires saved Pair top120, RU top20, refined top100')
    require(len(gt) == n, 'GT length mismatch')
    for positives in gt:
        require(len(positives) > 0 and np.issubdtype(np.asarray(positives).dtype, np.integer)
                and (np.asarray(positives) >= 0).all() and (np.asarray(positives) < ndb).all(), 'Invalid GT')
    require(all(set(a[:100]) == set(b) for a, b in zip(g, r)), 'Refined candidates changed')
    hits = lambda ids: np.array([np.isin(row, p).any() for row, p in zip(ids, gt)])
    base_hit = hits(g[:, :100])
    refined_hit = hits(r[:, :1])
    require(int(base_hit.sum()) == summary['top100_oracle'], 'Oracle reproduction failed')
    require(int(refined_hit.sum()) == summary['refined_correct'], 'Refined recall reproduction failed')
    require(int(hits(u[:, :1]).sum()) == summary['ru_correct'], 'RU recall reproduction failed')
    unions, padded, added = [], [], []
    # Lists depend ONLY on saved rankings, never GT.
    for grow, urow in zip(g, u):
        combined = list(dict.fromkeys(map(int, np.concatenate((grow[:100], urow[:20])))))
        unions.append(combined)
        added.append(len(combined) - 100)
        equal_budget = list(dict.fromkeys(combined + list(map(int, grow[:120]))))[:120]
        require(len(equal_budget) == 120, 'Failed equal-budget construction')
        padded.append(equal_budget)
    padded = np.asarray(padded, dtype=np.int64)
    variants = {'pair100': base_hit, 'pair120': hits(g[:, :120]),
                'pair100_union_ru20': hits(unions), 'union_padded120': hits(padded)}
    details = []
    for i in range(n):
        details.append({'query': i, 'refined_correct': bool(refined_hit[i]),
                        'added_ru_candidates': added[i],
                        **{name: bool(value[i]) for name, value in variants.items()}})
    measurements = {}
    for name, value in variants.items():
        measurements[name] = {'oracle_count': int(value.sum()), 'oracle_recall': float(value.mean()),
                              'newly_reachable_vs_pair100': np.flatnonzero(value & ~base_hit).tolist(),
                              'still_unreachable': np.flatnonzero(~value).tolist()}
    equal, deeper = variants['union_padded120'], variants['pair120']
    raw_gain = int((variants['pair100_union_ru20'] & ~base_hit).sum())
    report = {'complete': True, 'queries': n, 'variants': measurements,
              'equal_budget_comparison': {
                  'union_only_reachable': np.flatnonzero(equal & ~deeper).tolist(),
                  'pair120_only_reachable': np.flatnonzero(deeper & ~equal).tolist(),
                  'net_oracle_count': int(equal.sum()) - int(deeper.sum())},
              'extra_ru_candidates': {'min': min(added), 'mean': float(np.mean(added)), 'max': max(added), 'total': sum(added)},
              'candidate_capacity_gain_pp': raw_gain / n * 100,
              'decision': ('STOP_CANDIDATE_EXPANSION_NO_NEW_GT' if raw_gain == 0 else
                           'CANDIDATE_GAIN_ONLY_REQUIRES_SCORING_AND_REGRESSION_CHECK'),
              'warning': 'Oracle reachability is NOT achieved R@1. No model reranking, fusion weights or semantic claims. MSLS is exploratory.'}
    args.output.mkdir(parents=True)
    save(args.output / 'summary.json', report)
    save(args.output / 'per_query.json', details)
    np.savez_compressed(args.output / 'candidates.npz', pair120=g[:, :120], union_padded120=padded)
    save(args.output / 'completed.json', {'complete': True, 'script_sha256': sha(Path(__file__)),
         'source_summary_sha256': sha(audit / 'summary.json'), 'numpy': np.__version__,
         'source_predictions_sha256': sha(audit / 'predictions.npz'),
         'output_sha256': {name: sha(args.output / name) for name in ('summary.json', 'per_query.json', 'candidates.npz')}})
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('Wrote:', args.output.resolve())


if __name__ == '__main__':
    main()
