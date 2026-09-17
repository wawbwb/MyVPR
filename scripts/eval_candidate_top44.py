"""Frozen full-image top20 vs top44 paired evaluation; no training."""
import argparse
from collections import OrderedDict
import hashlib
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_candidate_multiview_budget as b
from scripts.candidate_set_screen import read, write, sha, complete, load_npz, npz

POLICY = dict(split='dev', topk=44, precision='FP32, no AMP/TF32',
    scoring='sum of both directional Pair-VPR scores, same as frozen cache',
    tie_break='stable original global candidate order',
    reproduction='recompute cached pair winner for EVERY query; atol=1e-4 rtol=1e-4; stop on mismatch',
    scope='Exploratory repeatedly inspected GSV dev, not independent test; all queries, no GT selection')


def validate_result(row, ids, old, positives):
    if not np.array_equal(row['candidates'], ids): raise ValueError('Candidate mismatch')
    if row['scores'].shape != (44,) or not np.isfinite(row['scores']).all(): raise ValueError('Invalid scores')
    if not np.array_equal(row['scores'][:20], old['base']): raise ValueError('Old scores modified')
    if row['labels'].dtype != np.bool_ or not np.array_equal(row['labels'], np.isin(ids, positives)):
        raise ValueError('GT mismatch')
    j = int(np.argmax(old['base']))
    if row['reproduced_score'].shape != () or not np.isfinite(row['reproduced_score']):
        raise ValueError('Invalid reproduction score')
    if not np.isclose(row['reproduced_score'], old['base'][j], atol=1e-4, rtol=1e-4):
        raise ValueError('Frozen winner pair does not reproduce')


def paired_record(qi, row):
    scores, labels, ids = row['scores'], row['labels'], row['candidates']
    old = int(np.argmax(scores[:20])); new = int(np.argmax(scores))
    before, after = bool(labels[old]), bool(labels[new])
    return dict(query_index=qi, old_correct=before, new_correct=after,
        old_id=int(ids[old]), new_id=int(ids[new]), new_global_rank=new+1,
        correction=not before and after, regression=before and not after,
        newly_reachable=not bool(labels[:20].any()) and bool(labels.any()),
        correction_from_added_candidate=not before and after and new >= 20,
        regression_from_added_candidate=before and not after and new >= 20)


def summarize(rows):
    records = [paired_record(i, row) for i, row in enumerate(rows)]
    n = len(rows)
    out = dict(queries=n, corrections=[r['query_index'] for r in records if r['correction']],
        regressions=[r['query_index'] for r in records if r['regression']],
        top1_changed=sum(r['old_id'] != r['new_id'] for r in records),
        newly_reachable_ids=[r['query_index'] for r in records if r['newly_reachable']],
        realized_newly_reachable_corrections=[r['query_index'] for r in records if r['newly_reachable'] and r['correction']])
    for name, k in [('top20', 20), ('top44', 44)]:
        hit = []
        for row in rows:
            order = np.argsort(-row['scores'][:k], kind='stable')
            hit.append(row['labels'][:k][order])
        y = np.stack(hit)
        out[name] = dict(correct=int(y[:, 0].sum()), reachable=int(y.any(1).sum()),
            recall={str(r): float(y[:, :r].any(1).mean()) for r in (1, 5, 10, 20)})
    out['net'] = len(out['corrections'])-len(out['regressions'])
    out['delta_r1_pp'] = 100*out['net']/n
    return out, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('doc/candidate_multiview_dev_v1'))
    p.add_argument('--plan', type=Path, default=Path('doc/candidate_hard_plan_v1'))
    p.add_argument('--cache-root', type=Path, default=Path('.cache/candidate_hard_v1'))
    p.add_argument('--gsv-root', type=Path, default=Path('datasets/gsv_cities'))
    p.add_argument('--official-repo', type=Path, default=Path('/home/wt/workspace/Pair-VPR-official'))
    p.add_argument('--audit', type=Path, default=Path('doc/pairvpr_official_paired_audit_v1'))
    p.add_argument('--output', type=Path, default=Path('doc/candidate_top44_eval_v1'))
    p.add_argument('--resume', action='store_true')
    a = p.parse_args(); e = b.m.e
    print('Verifying completed sources and frozen identities...', flush=True)
    plan = e.verify(a.plan, e.h.codes()); e.h.ensure_disjoint(plan['plan'])
    c = e.verify(a.source, b.source_codes()); cache = a.cache_root/'dev'; cc = e.verify(cache, e.h.codes())
    part = plan['plan']['dev']; db, queries = part['database'], part['queries']; nd, nq = len(db), len(queries)
    if (c['split'] != 'dev' or cc['split'] != 'dev' or cc['topk'] != 20 or c['policy'] != b.m.POLICY
        or c['cache_sha256'] != sha(cache/'completed.json') or c['official'] != cc['official']
        or c['plan_sha256'] != sha(a.plan/'completed.json') or cc['plan_sha256'] != c['plan_sha256']
        or cc['ndb'] != nd or cc['queries'] != nq or read(a.source/'query_mapping.json') != queries):
        raise ValueError('Source identity mismatch')
    code = {**b.source_codes(), 'scripts/audit_candidate_multiview_budget.py': hashlib.sha256(
        Path(b.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        'scripts/eval_candidate_top44.py': hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()}
    contract = dict(code=code, policy=POLICY, source_sha256=sha(a.source/'completed.json'),
        cache_sha256=sha(cache/'completed.json'), official=c['official'], plan_sha256=sha(a.plan/'completed.json'))
    if a.output.exists():
        if not a.resume or read(a.output/'contract.json') != contract: raise ValueError('Use fresh output or identical --resume')
        if (a.output/'completed.json').exists(): e.verify(a.output, code); print('Already complete'); return
    places = {}
    for di, r in enumerate(db): places.setdefault(r['label'], []).append(di)
    inputs = []; old_rows = []
    for qi, q in enumerate(queries):
        sets = b.budget_sets(load_npz(a.source/'queries'/f'{qi:06d}.npz'), nd)
        old = load_npz(cache/'pairs'/f'{qi:06d}.npz'); e.h.validate_row(old, places[q['label']], nd)
        if not np.array_equal(sets['full20'], old['candidates']): raise ValueError('Old candidates changed')
        inputs.append(sets['full44']); old_rows.append(old)
    hashes = []
    for start in range(0, nd+nq, 128):
        hs = load_npz(cache/'global'/f'{start:07d}.npz')['hashes']
        if len(hs) != min(128, nd+nq-start): raise ValueError('Image hash count mismatch')
        hashes.extend(hs.tolist())
    from tqdm import tqdm
    records = db+queries
    for i in tqdm(sorted(set(np.concatenate(inputs).tolist()) | set(range(nd, nd+nq))), desc='Verify source images'):
        path = (a.gsv_root/records[i]['path']).resolve()
        if not path.is_relative_to(a.gsv_root.resolve()) or sha(path) != hashes[i]: raise ValueError('Source image changed')
    import torch
    model, identity = e.h.original.official(a)
    if identity != c['official']: raise ValueError('Frozen model identity mismatch')
    if not a.output.exists():
        a.output.mkdir(parents=True); write(a.output/'contract.json', contract); (a.output/'pairs').mkdir()
    memo = OrderedDict()
    def dense(i):
        if i in memo: memo.move_to_end(i); return memo[i].cuda()
        x, _ = e.h.original.image(a.gsv_root, records[i], hashes[i])
        f, _ = model(x[None].cuda(), None, 'global'); memo[i] = f.cpu()
        if len(memo) > 64: memo.popitem(last=False)
        return f
    def pair(qf, di):
        df = dense(di)
        return float((model(qf, df, 'pairvpr')+model(df, qf, 'pairvpr')).item())
    rows = []; fresh = 0; torch.cuda.synchronize(); begin = time.perf_counter()
    with torch.inference_mode():
        for qi in tqdm(range(nq), desc='Frozen top44 paired eval'):
            ids, old = inputs[qi], old_rows[qi]; file = a.output/'pairs'/f'{qi:06d}.npz'
            if file.exists() and file.with_suffix('.sha.json').exists():
                row = load_npz(file)
            else:
                qf = dense(nd+qi); reproduced = pair(qf, int(ids[np.argmax(old['base'])]))
                if not np.isclose(reproduced, old['base'].max(), atol=1e-4, rtol=1e-4):
                    raise ValueError(f'Query {qi}: frozen winner reproduction mismatch {reproduced} vs {old["base"].max()}')
                added = np.asarray([pair(qf, int(di)) for di in ids[20:]], dtype=np.float32)
                row = dict(candidates=ids, scores=np.concatenate([old['base'], added]),
                    labels=np.isin(ids, places[queries[qi]['label']]), reproduced_score=np.asarray(reproduced))
                validate_result(row, ids, old, places[queries[qi]['label']]); npz(file, **row); fresh += 1
            validate_result(row, ids, old, places[queries[qi]['label']]); rows.append(row)
    torch.cuda.synchronize(); elapsed = time.perf_counter()-begin
    result, outcomes = summarize(rows)
    baseline = read(cache/'baseline.json')
    if result['top20']['correct'] != baseline['correct'] or result['top20']['recall'] != baseline['recall']:
        raise ValueError('Original baseline summary mismatch')
    expected_coverage = sum(np.isin(ids, places[q['label']]).any() for ids, q in zip(inputs, queries))
    if result['top44']['reachable'] != expected_coverage: raise ValueError('Coverage mismatch')
    write(a.output/'summary.json', result); write(a.output/'per_query.json', outcomes)
    write(a.output/'query_mapping.json', queries)
    write(a.output/'timing.json', dict(new_queries_this_invocation=fresh, loop_seconds=elapsed,
        new_pairs=fresh*24, reproduction_pairs=fresh, directional_forwards=fresh*50,
        note='Includes dense re-encoding, cache I/O and one reproduction pair/query; excludes startup/hash checks and prior interrupted invocations. Not pure pair latency.'))
    complete(a.output); print(result); print('Complete:', a.output)


if __name__ == '__main__': main()
