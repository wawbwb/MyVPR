"""CPU-only post-hoc diagnosis; no extraction, threshold search, or cache mutation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.region_pair_lite import conservative_order


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def analyze(candidates, ru, regional, supports, gt):
    rows = []
    for q, candidate in enumerate(candidates):
        positive = np.isin(candidate, np.asarray(gt[q]).reshape(-1))
        pos = np.flatnonzero(positive)
        neg = np.flatnonzero(~positive)
        best = int(np.argmax(regional[q]))
        selected = int(conservative_order(ru[q], regional[q], supports[q])[0])
        bp = int(pos[np.argmax(regional[q, pos])]) if len(pos) else None
        bn = int(neg[np.argmax(regional[q, neg])]) if len(neg) else None
        eligible = ((ru[q, 0]-ru[q] <= .02) & (supports[q] >= 3)
                    & (regional[q]-regional[q, 0] >= .05))
        eligible[0] = False
        blockers = []
        if bp is not None:
            if ru[q, 0]-ru[q, bp] > .02: blockers.append('ru_gap_above_0.02')
            if supports[q, bp] < 3: blockers.append('support_below_3')
            if regional[q, bp]-regional[q, 0] < .05: blockers.append('advantage_below_0.05')
            if bp != best: blockers.append('not_global_region_argmax_including_ties')
        rows.append({
            'query_index': q, 'ru_correct': bool(positive[0]),
            'reachable': bool(len(pos)), 'reranked_correct': bool(positive[selected]),
            'selected_position': selected, 'region_argmax_position': best,
            'best_positive_position': bp, 'best_negative_position': bn,
            'positive_beats_ru_top1_region': bool(bp is not None and regional[q, bp] > regional[q, 0]),
            'positive_beats_all_negatives_region': bool(bp is not None and (bn is None or regional[q, bp] > regional[q, bn])),
            'region_argmax_correct': bool(positive[best]),
            'eligible_positive_exists': bool((eligible & positive).any()),
            'best_positive_blockers': blockers,
            'candidates': [dict(db_index=int(d), positive=bool(positive[j]),
                ru_score=float(ru[q, j]), region_score=float(regional[q, j]),
                support=int(supports[q, j]), ru_gap=float(ru[q, 0]-ru[q, j]),
                region_advantage=float(regional[q, j]-regional[q, 0]),
                passes_fixed_gate=bool(eligible[j])) for j, d in enumerate(candidate)]})
    reachable_errors = [r for r in rows if not r['ru_correct'] and r['reachable']]
    correction = [r['query_index'] for r in rows if not r['ru_correct'] and r['reranked_correct']]
    regression = [r['query_index'] for r in rows if r['ru_correct'] and not r['reranked_correct']]
    report = {'queries': len(rows), 'baseline_correct': sum(r['ru_correct'] for r in rows),
        'correct': sum(r['reranked_correct'] for r in rows), 'corrections': correction,
        'regressions': regression, 'top1_changed': sum(r['selected_position'] != 0 for r in rows),
        'reachable_ru_errors': len(reachable_errors),
        'unreachable_ru_errors': sum(not r['ru_correct'] and not r['reachable'] for r in rows),
        'top20_oracle_correct': sum(r['reachable'] for r in rows),
        'reachable_error_diagnostics': {key: sum(r[key] for r in reachable_errors) for key in (
            'positive_beats_ru_top1_region', 'positive_beats_all_negatives_region',
            'region_argmax_correct', 'eligible_positive_exists')},
        'best_positive_blockers_on_reachable_errors': {key: sum(key in r['best_positive_blockers'] for r in reachable_errors)
            for key in ('ru_gap_above_0.02', 'support_below_3', 'advantage_below_0.05', 'not_global_region_argmax_including_ties')},
        'region_argmax_only_diagnostic': {
            'correct': sum(r['region_argmax_correct'] for r in rows),
            'corrections': sum(not r['ru_correct'] and r['region_argmax_correct'] for r in rows),
            'regressions': sum(r['ru_correct'] and not r['region_argmax_correct'] for r in rows)},
        'eligible_positive_blocked_by_global_argmax_query_ids': [r['query_index'] for r in reachable_errors
            if r['eligible_positive_exists'] and not r['reranked_correct']]}
    return report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('doc/region_pair_lite_v1'))
    parser.add_argument('--source', type=Path, default=Path('doc/visual_pair_msls_hard_mix_v2'))
    parser.add_argument('--dataset-root', type=Path, default=Path('datasets/msls-val'))
    parser.add_argument('--output', type=Path, default=Path('doc/region_pair_lite_diagnostic_v1'))
    a = parser.parse_args()
    require(not a.output.exists(), 'Choose a new output directory; no overwrites')
    contract = json.loads((a.run/'contract.json').read_text())
    complete = json.loads((a.run/'completed.json').read_text())
    require(complete.get('complete') and complete.get('contract_sha256') == sha(a.run/'contract.json'), 'Incomplete run or contract mismatch')
    require(contract['code']['src/region_pair_lite.py'] == sha(ROOT/'src/region_pair_lite.py'), 'Scoring code changed')
    verified = {}
    for name in ('per_query.npz', 'summary.json'):
        verified[str(a.source/name)] = sha(a.source/name)
        require(verified[str(a.source/name)] == contract['inputs'][name], 'Source changed: '+name)
    for name, digest in contract['split'].items():
        verified[str(a.dataset_root/name)] = sha(a.dataset_root/name)
        require(verified[str(a.dataset_root/name)] == digest, 'Split changed: '+name)
    with np.load(a.source/'per_query.npz', allow_pickle=False) as z:
        candidates, ru = z['candidates'], z['ru_scores']
    require(candidates.shape == (740, 20) and ru.shape == candidates.shape, 'Unexpected candidate shapes')
    require(np.issubdtype(candidates.dtype, np.integer) and (candidates >= 0).all() and (candidates < 18871).all(), 'Invalid DB indices')
    require(np.isfinite(ru).all() and (np.diff(ru, axis=1) <= 1e-6).all(), 'Invalid RU scores/order')
    require(all(len(set(row)) == 20 for row in candidates), 'Duplicate candidate')
    gt = np.load(a.dataset_root/'msls_val_gt_25m.npy', allow_pickle=True)
    require(len(gt) == 740, 'Wrong GT size')
    saved = json.loads((a.run/'summary.json').read_text())
    summaries, details, cases = {}, {}, {}
    with np.load(a.run/'region_scores.npz', allow_pickle=False) as scores, np.load(a.run/'predictions.npz', allow_pickle=False) as predictions:
        require(np.array_equal(predictions['ru'], candidates), 'RU predictions mismatch')
        for mode in ('sam', 'grid', 'shifted'):
            region, support = scores[mode], scores[mode+'_supports']
            require(region.shape == ru.shape and support.shape == ru.shape, 'Invalid score shapes')
            require(np.isfinite(region).all() and np.isfinite(support).all() and (support >= 0).all()
                    and (support <= 16).all() and (support == np.floor(support)).all(), 'Invalid scores/supports')
            expected = np.stack([row[conservative_order(ru[q], region[q], support[q])] for q, row in enumerate(candidates)])
            require(np.array_equal(expected, predictions[mode]), 'Saved ranking not reproduced: '+mode)
            summary, rows = analyze(candidates, ru, region, support, gt)
            require(summary['baseline_correct'] == 675, 'RU baseline not reproduced')
            for key in ('correct', 'corrections', 'regressions', 'top1_changed'):
                require(summary[key] == saved[mode][key], 'Saved summary mismatch: '+mode+'/'+key)
            summaries[mode], details[mode] = summary, rows
            cases[mode] = [r for r in rows if r['ru_correct'] != r['reranked_correct']]
    for name in ('contract.json', 'completed.json', 'summary.json', 'predictions.npz', 'region_scores.npz'):
        verified[str(a.run/name)] = sha(a.run/name)
    a.output.mkdir(parents=True)
    scope = ('Post-hoc diagnosis on MSLS-val, not an independent validation or threshold search. '
             'Positive selection uses GT only for diagnosis. Blocker counts overlap. '
             'Region argmax is an unsafe diagnostic, not a proposed new method. '
             'No shard, local.npy, descriptor, GPU or model reads required.')
    for name, value in (('summary.json', {'scope': scope, 'variants': summaries}),
                        ('per_query.json', details), ('changed_correctness_cases.json', cases),
                        ('provenance.json', {'verified_sha256': verified, 'script_sha256': sha(__file__)})):
        (a.output/name).write_text(json.dumps(value, indent=2), encoding='utf8')
    (a.output/'completed.json').write_text(json.dumps({'complete': True}), encoding='utf8')
    print(json.dumps({'scope': scope, 'variants': summaries}, indent=2))
    print('Results written to:', a.output)


if __name__ == '__main__':
    main()
