"""CPU-only frozen GSV supervision audit. No extraction, training or cache writes."""
import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.candidate_set_screen import read, write, sha, verify, load_npz, complete
from scripts.candidate_set_pitts import validate_row
from src.candidate_set_utils import summary


def distribution(values):
    x = np.asarray(values, dtype=np.float64)
    if not len(x): return {'n': 0, 'mean': None, 'quantiles': None}
    return {'n': len(x), 'mean': float(x.mean()),
            'quantiles': dict(zip(['min', 'p01', 'p05', 'p10', 'p50', 'p90', 'p95', 'p99', 'max'],
                                  np.quantile(x, [0, .01, .05, .1, .5, .9, .95, .99, 1]).tolist()))}


def logsumexp(x):
    x = np.asarray(x, dtype=np.float64)
    m = float(x.max())
    return m + float(np.log(np.exp(x-m).sum()))


def inspect_query(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if scores.ndim != 1 or scores.shape != labels.shape or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Invalid scores/labels')
    order = np.argsort(-scores, kind='stable')
    positive = np.flatnonzero(labels); negative = np.flatnonzero(~labels)
    valid = bool(len(positive) and len(negative))
    correct = bool(labels[order[0]])
    category = ('unreachable' if not len(positive) else 'all_positive' if not len(negative)
                else 'reachable_correct' if correct else 'reachable_error')
    margin = float(scores[positive].max()-scores[negative].max()) if valid else None
    # Stable log(1 + exp(log Z_negative - log Z_positive)). No cancellation.
    loss = float(np.logaddexp(0., logsumexp(scores[negative])-logsumexp(scores[positive]))) if valid else None
    ranks = np.flatnonzero(labels[order])
    return {'category': category, 'valid_for_training': valid, 'correct': correct,
            'positive_count': int(labels.sum()), 'best_positive_rank': int(ranks[0]+1) if len(ranks) else None,
            'positive_minus_negative_margin': margin, 'frozen_list_loss': loss,
            'negative_softmax_mass': float(-np.expm1(-loss)) if valid else None,
            'residual_range_only_fixable': bool(category == 'reachable_error' and margin > -8),
            'residual_range_blocked': bool(category == 'reachable_error' and margin <= -8)}


def aggregate(rows):
    valid = [r for r in rows if r['valid_for_training']]
    errors = [r for r in rows if r['category'] == 'reachable_error']
    ranked = sorted(valid, key=lambda r: (-r['frozen_list_loss'], r['query_index']))
    total = sum(r['frozen_list_loss'] for r in valid)
    concentration = {}
    for fraction in (.01, .05, .1):
        chosen = ranked[:math.ceil(len(valid)*fraction)]
        concentration[str(fraction)] = {
            'n': len(chosen), 'query_ids': [r['query_index'] for r in chosen],
            'loss_fraction': sum(r['frozen_list_loss'] for r in chosen)/total if total else None,
            'distinct_places': len({r['label'] for r in chosen}),
            'city_counts': dict(Counter(r['city'] for r in chosen))}
    losses = np.asarray([r['frozen_list_loss'] for r in valid])
    return {'queries': len(rows), 'categories': dict(Counter(r['category'] for r in rows)),
            'valid_queries': len(valid), 'reachable_errors': len(errors),
            'error_places': len({r['label'] for r in errors}),
            'error_city_counts': dict(Counter(r['city'] for r in errors)),
            'range_only_fixable_errors': sum(r['residual_range_only_fixable'] for r in rows),
            'range_blocked_errors': sum(r['residual_range_blocked'] for r in rows),
            'positive_count_all': distribution([r['positive_count'] for r in rows]),
            'best_positive_rank_reachable': distribution([r['best_positive_rank'] for r in rows if r['best_positive_rank'] is not None]),
            'margin_valid': distribution([r['positive_minus_negative_margin'] for r in valid]),
            'margin_errors': distribution([r['positive_minus_negative_margin'] for r in errors]),
            'frozen_list_loss_valid': distribution(losses),
            'negative_softmax_mass_valid': distribution([r['negative_softmax_mass'] for r in valid]),
            'loss_effective_query_count': float(losses.sum()**2/(losses@losses)) if len(losses) and losses@losses > 0 else None,
            'loss_concentration_valid_denominator': concentration,
            'cities': {city: {'queries': sum(r['city'] == city for r in rows),
                'reachable_errors': sum(r['city'] == city for r in errors),
                'loss_sum': sum(r['frozen_list_loss'] for r in valid if r['city'] == city)} for city in sorted({r['city'] for r in rows})}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=Path('.cache/candidate_set_v1/train'))
    p.add_argument('--plan', type=Path, default=Path('doc/candidate_set_plan_v1'))
    p.add_argument('--output', type=Path, default=Path('doc/candidate_training_audit_v1'))
    a = p.parse_args()
    if a.output.exists(): p.error('Choose a new output; existing reports are never overwritten')
    print('Verifying original plan and completed cache hashes (CPU / disk I/O)...', flush=True)
    c, plan = verify(a.cache), verify(a.plan)
    if c['split'] != 'train' or c['plan_sha256'] != sha(a.plan/'completed.json') or c['topk'] != 20:
        raise ValueError('Wrong cache/plan identity')
    part = plan['plan']['train']; queries, db = part['queries'], part['database']
    if c['queries'] != len(queries) or c['ndb'] != len(db): raise ValueError('Cache dimensions disagree with plan')
    if len({r['path'] for r in queries+db}) != len(queries)+len(db): raise ValueError('Duplicate images in plan')
    if {(r['city'], r['panoid']) for r in queries} & {(r['city'], r['panoid']) for r in db}:
        raise ValueError('Query/database panorama overlap')
    positives = {}
    for i, record in enumerate(db): positives.setdefault(record['label'], []).append(i)
    rows, bases, globals_, labels = [], [], [], []
    for i, query in enumerate(queries):
        row = load_npz(a.cache/'pairs'/f'{i:06d}.npz')
        validate_row(row, positives.get(query['label'], []), len(db))
        detail = inspect_query(row['base'], row['labels'])
        detail.update(query_index=i, path=query['path'], city=query['city'], label=query['label'],
                      panoid=query['panoid'], date=query['date'],
                      candidates=row['candidates'].tolist(), labels=row['labels'].tolist(),
                      pair_scores=row['base'].tolist(), global_scores=row['global_scores'].tolist())
        rows.append(detail); bases.append(row['base']); globals_.append(row['global_scores']); labels.append(row['labels'])
        if (i+1) % 128 == 0: print(f'Validated pair rows {i+1}/{len(queries)}', flush=True)
    bases, globals_, labels = map(np.stack, (bases, globals_, labels))
    pair_result = summary(bases, labels, bases)
    global_result = summary(globals_, labels, bases)
    if pair_result != read(a.cache/'baseline.json') or global_result != read(a.cache/'global_baseline.json'):
        raise ValueError('Frozen baseline summaries do not reproduce')
    result = aggregate(rows)
    result.update(frozen_pair=pair_result, frozen_global=global_result,
        scope='GSV place-ID training supervision, not benchmark generalization. Full cached top20 queries included. '
              'Pair recall is within global top20. Loss is BEFORE any head update, float64 CPU; '
              'negative mass is a score-gradient proxy, not parameter-gradient norm. '
              'Range test assumes independent ideal residuals strictly inside (-4,4), not learnability. '
              'Loss concentration does not prove overfitting. No automatic admission threshold or new training.',
        notes='No source images, models, GPU, feature extraction or cache edits. All manifest-listed cache files hashed; '
              'pair rows and sidecars validated against plan labels. Original image content not re-read.')
    a.output.mkdir(parents=True)
    write(a.output/'summary.json', result); write(a.output/'per_query.json', rows)
    write(a.output/'provenance.json', {'cache_completed_sha256': sha(a.cache/'completed.json'),
        'plan_completed_sha256': sha(a.plan/'completed.json'), 'cache_contract': c,
        'audit_code_sha256': hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()})
    complete(a.output)
    print('Frozen global / pair correct:', global_result['correct'], pair_result['correct'])
    print('Query categories:', result['categories'])
    print('Range-only fixable / blocked errors:', result['range_only_fixable_errors'], result['range_blocked_errors'])
    print('Frozen valid loss:', result['frozen_list_loss_valid'])
    print('Report:', a.output)


if __name__ == '__main__': main()
