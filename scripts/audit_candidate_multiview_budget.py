"""CPU-only equal-candidate-budget audit from completed multiview rankings."""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import candidate_multiview_screen as m
from scripts.candidate_set_screen import read, write, sha, complete, load_npz

POLICY = dict(fixed_k=44, fixed_k_reason='Rounded train union mean 44.1622, not selected on dev',
    matched_k='Per query M = original multiview union size; same M for full-image prefix',
    scope='Post-screen exploratory candidate coverage audit, not actual reranked R1; dev already inspected',
    compute='CPU only; reuse saved full80 order, no descriptors, image encoding or pair scoring',
    caveat='Equal candidate count is not equal total latency: crops cost encoding; topM is a diagnostic control requiring observed M')


def source_codes():
    return {**m.e.codes(), 'scripts/candidate_multiview_screen.py': hashlib.sha256(
        Path(m.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()}


def budget_sets(saved, ndb):
    arrays = {}
    for name, shape in [('full20', (20,)), ('full80', (80,)), ('crop_top20', (3, 20))]:
        x = np.asarray(saved[name])
        if x.shape != shape or x.dtype.kind not in 'iu' or (x < 0).any() or (x >= ndb).any():
            raise ValueError('Invalid candidate array: '+name)
        for row in x.reshape(-1, shape[-1]):
            if len(set(row.tolist())) != len(row): raise ValueError('Duplicate within ranking')
        arrays[name] = x
    union = np.asarray(saved['union'])
    if union.ndim != 1 or union.dtype.kind not in 'iu' or not 20 <= len(union) <= 80:
        raise ValueError('Invalid union')
    expected = list(dict.fromkeys(np.concatenate([arrays['full20'], arrays['crop_top20'].ravel()]).tolist()))
    if union.tolist() != expected: raise ValueError('Union does not reproduce stable dedup')
    if not np.array_equal(arrays['full20'], arrays['full80'][:20]): raise ValueError('Full20 prefix mismatch')
    return dict(full20=arrays['full20'], full44=arrays['full80'][:44],
        fullM=arrays['full80'][:len(union)], union=union, full80=arrays['full80'])


def query_record(qi, sets, positives, frozen_correct):
    positives = set(positives)
    return dict(query_index=qi, frozen_correct=bool(frozen_correct),
        hit={k: bool(set(v.tolist()) & positives) for k, v in sets.items()},
        count={k: len(v) for k, v in sets.items()})


def report(rows):
    n = len(rows)
    variants = {}
    for name in ('full20', 'full44', 'fullM', 'union', 'full80'):
        hit = sum(r['hit'][name] for r in rows)
        variants[name] = dict(queries=n, reachable=hit, coverage=hit/n if n else None,
            newly_reachable_ids=[r['query_index'] for r in rows if r['hit'][name] and not r['hit']['full20']],
            candidate_count=sum(r['count'][name] for r in rows),
            mean_candidates=sum(r['count'][name] for r in rows)/n if n else None,
            additional_pairs=sum(r['count'][name]-20 for r in rows))
    comparisons = {}
    for name in ('full44', 'fullM', 'full80'):
        gained = [r['query_index'] for r in rows if r['hit']['union'] and not r['hit'][name]]
        missed = [r['query_index'] for r in rows if r['hit'][name] and not r['hit']['union']]
        comparisons['union_vs_'+name] = dict(union_only_ids=gained, full_only_ids=missed,
            net_reachable=len(gained)-len(missed), delta_coverage_pp=100*(len(gained)-len(missed))/n if n else None)
    return dict(variants=variants, comparisons=comparisons, actual_pair_scoring=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-source', type=Path, default=Path('doc/candidate_multiview_train_v1'))
    p.add_argument('--dev-source', type=Path, default=Path('doc/candidate_multiview_dev_v1'))
    p.add_argument('--plan', type=Path, default=Path('doc/candidate_hard_plan_v1'))
    p.add_argument('--cache-root', type=Path, default=Path('.cache/candidate_hard_v1'))
    p.add_argument('--output', type=Path, default=Path('doc/candidate_multiview_budget_v1'))
    a = p.parse_args()
    if a.output.exists(): p.error('Choose a new output; reports are never overwritten')
    print('CPU/disk only: verifying original plan, complete caches and multiview outputs...', flush=True)
    plan = m.e.verify(a.plan, m.e.h.codes()); m.e.h.ensure_disjoint(plan['plan'])
    results = {}; records = {}; source_hashes = {}; identities = []
    for split, source in [('train', a.train_source), ('dev', a.dev_source)]:
        c = m.e.verify(source, source_codes())
        cache = a.cache_root/split; cc = m.e.verify(cache, m.e.h.codes())
        part = plan['plan'][split]; nd = len(part['database']); nq = len(part['queries'])
        if (c['policy'] != m.POLICY or c['split'] != split or cc['split'] != split or cc['topk'] != 20
            or c['cache_sha256'] != sha(cache/'completed.json') or c['official'] != cc['official']
            or c['plan_sha256'] != sha(a.plan/'completed.json') or cc['plan_sha256'] != c['plan_sha256']
            or cc['ndb'] != nd or cc['queries'] != nq): raise ValueError('Source identity mismatch')
        if read(source/'query_mapping.json') != part['queries']: raise ValueError('Query mapping changed')
        places = {}
        for di, d in enumerate(part['database']): places.setdefault(d['label'], []).append(di)
        rows = []; old_rows = []
        for qi, q in enumerate(part['queries']):
            saved = load_npz(source/'queries'/f'{qi:06d}.npz')
            sets = budget_sets(saved, nd)
            old = load_npz(cache/'pairs'/f'{qi:06d}.npz')
            positives = places[q['label']]
            m.e.h.validate_row(old, positives, nd)
            if not np.array_equal(old['candidates'], sets['full20']): raise ValueError('Original candidate mismatch')
            correct = old['labels'][np.argmax(old['base'])]
            rows.append(query_record(qi, sets, positives, correct))
            old_rows.append(m.outcome(qi, sets, positives, correct))
            if (qi+1) % 512 == 0 or qi+1 == nq: print(f'{split}: {qi+1}/{nq}', flush=True)
        if old_rows != read(source/'per_query.json') or m.summarize(old_rows) != read(source/'summary.json'):
            raise ValueError('Original multiview outcomes do not reproduce')
        if any(r['count']['fullM'] != r['count']['union'] for r in rows): raise ValueError('Budget mismatch')
        results[split] = report(rows); records[split] = rows
        source_hashes[split] = sha(source/'completed.json'); identities.append(c['official'])
        print(split, results[split]['comparisons'], flush=True)
    if identities[0] != identities[1]: raise ValueError('Train/dev model mismatch')
    a.output.mkdir(parents=True)
    write(a.output/'contract.json', dict(policy=POLICY, source_sha256=source_hashes,
        plan_sha256=sha(a.plan/'completed.json'), official=identities[0], code={**source_codes(),
            'scripts/audit_candidate_multiview_budget.py': hashlib.sha256(
                Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()}))
    write(a.output/'summary.json', results); write(a.output/'per_query.json', records); complete(a.output)
    print('Complete. Coverage only, not reranked accuracy:', a.output)


if __name__ == '__main__': main()
