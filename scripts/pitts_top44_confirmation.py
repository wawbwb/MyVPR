"""Frozen complete Pitts top20/top44 extension. Existing 1024 are reused, not retuned."""
import argparse
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import sys
import time
import uuid
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import eval_candidate_top44_pitts as previous
from scripts.candidate_set_screen import read, write, sha, complete, load_npz, npz
from src.top44_confirmation import partition, paired_statistics

POLICY = dict(topk=[20, 44], model='frozen official Pair-VPR', precision='FP32; TF32 off; no AMP',
    scoring='both directional scores summed, same 44 scores sliced for top20',
    tie_break='stable global DB order', primary='delta R@1 on remaining 6584 queries',
    secondary='old1024 and all7608 descriptive; R5/10/20, corrections/regressions, coverage and cost',
    inference='95% panorama-cluster percentile bootstrap, 10000 resamples seed42; exact query-IID McNemar auxiliary only',
    interpretation='positive delta and cluster CI lower>0 supports this sample; positive with CI crossing0 inconclusive; negative observed degradation',
    scope='Pitts30k-val has historical development exposure; complement is NOT independent test',
    timing='per-query decode/encode, first20 and next24 end-to-end pair sections; cache reuse varies; not pure deployment latency',
    no_tuning=True, no_accuracy_based_early_stop=True)


def codes():
    result = {**previous.cp.codes(), **previous.common.b.source_codes()}
    for module in (previous, previous.common, previous.common.b):
        path = Path(module.__file__)
        result[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    for name in ('src/top44_confirmation.py', 'scripts/pitts_top44_confirmation.py'):
        result[name] = hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    return result


def previous_codes():
    result = {**previous.cp.codes(), **previous.common.b.source_codes()}
    for module in (previous, previous.common, previous.common.b):
        path = Path(module.__file__)
        result[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    return result


@contextmanager
def lock(root):
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root/'worker.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally: fcntl.flock(handle, fcntl.LOCK_UN)


def load_contract(a):
    contract = read(a.work/'contract.json')
    if contract['code'] != codes() or contract['policy'] != POLICY: raise ValueError('Locked code/policy changed')
    if sha(a.work/'plan.json') != contract['plan_sha256']: raise ValueError('Locked plan changed')
    return contract, read(a.work/'plan.json')


def verify_sources(a, contract):
    for root, name in ((a.cache, 'old_cache'), (a.previous_run, 'old_top44')):
        if sha(root/'completed.json') != contract[name+'_sha256']: raise ValueError('Source completion changed')
    old_data, old_c = previous.cp.load_cache(a.cache, 'pitts')
    c = previous.common.b.m.e.verify(a.previous_run, previous_codes())
    if c['cache_sha256'] != contract['old_cache_sha256'] or c['official'] != old_c['official']:
        raise ValueError('Prior top44 and cache differ')
    if c['official'] != contract['official']: raise ValueError('Model identity changed')
    return old_data, old_c


def prepare(a):
    if (a.work/'contract.json').exists():
        c, _ = load_contract(a); verify_sources(a, c); print('Existing locked plan verified'); return
    data, old_c = previous.cp.load_cache(a.cache, 'pitts')
    prior = previous.common.b.m.e.verify(a.previous_run, previous_codes())
    if prior['cache_sha256'] != sha(a.cache/'completed.json') or prior['official'] != old_c['official']:
        raise ValueError('Prior result identity differs')
    full = previous.admission.load_index(a.dataset, 7608)
    rows = partition(full, old_c['index'])
    if len(rows) != 7608 or len(full['database']) != 10000 or sum(r['prior_query'] for r in rows) != 1024:
        raise ValueError('Expected original Pitts 7608 / 10000 / 1024')
    # Lock image bytes for remaining queries before any new scores are computed.
    from tqdm import tqdm
    for r in tqdm(rows, desc='Lock query image hashes'):
        r['image_sha256'] = sha(a.dataset/r['path'])
    original_hashes = []
    for start in range(0, 11024, 128):
        original_hashes.extend(load_npz(a.cache/'global'/f'{start:07d}.npz')['hashes'].tolist())
    old_hashes = dict(zip(old_c['index']['query_indices'], original_hashes[10000:]))
    if len(original_hashes) != 11024 or any(r['prior_query'] and r['image_sha256'] != old_hashes[r['query_index']] for r in rows):
        raise ValueError('Previously evaluated query image bytes changed')
    old_groups = {r['group'] for r in rows if r['prior_query']}
    new_groups = {r['group'] for r in rows if not r['prior_query']}
    for r in rows: r['shares_panorama_with_old_subset'] = r['group'] in old_groups
    plan = dict(database=full['database'], queries=rows, index_sha256=full['index_sha256'],
        old_query_ids=old_c['index']['query_indices'],
        grouping=dict(old_groups=len(old_groups), remaining_groups=len(new_groups), shared_groups=len(old_groups & new_groups),
            remaining_queries_from_new_panoramas=sum(not r['prior_query'] and not r['shares_panorama_with_old_subset'] for r in rows)))
    a.work.mkdir(parents=True, exist_ok=True); write(a.work/'plan.json', plan)
    write(a.work/'contract.json', dict(code=codes(), policy=POLICY, plan_sha256=sha(a.work/'plan.json'),
        old_cache_sha256=sha(a.cache/'completed.json'), old_top44_sha256=sha(a.previous_run/'completed.json'), official=old_c['official']))
    print('Locked plan:', plan['grouping']); print('Remaining queries: 6584; 289696 pairs / 579392 directional forwards; no score-based stopping')


def validate_new(row, record, nd):
    ids = row['candidates']
    if ids.shape != (44,) or ids.dtype.kind not in 'iu' or len(set(ids.tolist())) != 44 or (ids < 0).any() or (ids >= nd).any():
        raise ValueError('Invalid candidate IDs')
    if row['scores'].shape != (44,) or not np.isfinite(row['scores']).all(): raise ValueError('Invalid paired scores')
    if row['labels'].dtype != np.bool_ or not np.array_equal(row['labels'], np.isin(ids, record['gt'])): raise ValueError('GT mismatch')
    if row['global_scores'].shape != (44,) or not np.isfinite(row['global_scores']).all() or (np.diff(row['global_scores']) > 0).any():
        raise ValueError('Global ranking invalid')
    if row['query_index'].shape != () or int(row['query_index']) != record['query_index']: raise ValueError('Query ID changed')
    if str(row['image_sha256']) != record['image_sha256']: raise ValueError('Query bytes changed')
    if row['seconds'].shape != (3,) or not np.isfinite(row['seconds']).all() or (row['seconds'] < 0).any(): raise ValueError('Invalid timing')


def run(a):
    c, plan = load_contract(a); old_data, old_c = verify_sources(a, c)
    for name, digest in plan['index_sha256'].items():
        if sha(a.dataset/name) != digest: raise ValueError('Pitts index changed')
    from tqdm import tqdm
    nd = len(plan['database']); remaining = [r for r in plan['queries'] if not r['prior_query']]
    pairdir = a.work/'pairs'; pairdir.mkdir(exist_ok=True)
    pending = []
    for r in tqdm(remaining, desc='Verify/resume remaining query shards'):
        path = a.dataset/r['path']
        if sha(path) != r['image_sha256']: raise ValueError('Input query changed')
        file = pairdir/f'{r["query_index"]:06d}.npz'
        if file.exists() and file.with_suffix('.sha.json').exists(): validate_new(load_npz(file), r, nd)
        else: pending.append(r)
    if not pending: print('Remaining cache complete; ready for report'); return
    vectors = []; hashes = []
    for start in range(0, nd+1024, 128):
        z = load_npz(a.cache/'global'/f'{start:07d}.npz')
        count = min(128, nd+1024-start)
        if z['vectors'].shape != (count, 512) or len(z['hashes']) != count: raise ValueError('Invalid original globals')
        vectors.append(z['vectors']); hashes.extend(z['hashes'].tolist())
    vectors = np.concatenate(vectors)
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-4): raise ValueError('Invalid norms')
    for i, path in enumerate(tqdm(plan['database'], desc='Verify full database image hashes')):
        if sha(a.dataset/path) != hashes[i]: raise ValueError('Reference image changed')
    import torch
    a.gsv_root = a.dataset
    model, identity = previous.admission.official(a)
    if identity != c['official']: raise ValueError('Frozen model mismatch')
    memo = OrderedDict()
    def dense(i):
        if i in memo: memo.move_to_end(i); return memo[i].cuda()
        x, _ = previous.admission.image(a.dataset, {'path': plan['database'][i]}, hashes[i])
        f, _ = model(x[None].cuda(), None, 'global'); memo[i] = f.cpu()
        if len(memo) > 64: memo.popitem(last=False)
        return f
    def pair(q, di):
        d = dense(int(di)); return float((model(q, d, 'pairvpr')+model(d, q, 'pairvpr')).item())
    db = torch.from_numpy(vectors[:nd]).cuda()
    # Deterministic old-query sanity check; labels never select this query.
    with torch.inference_mode():
        x, _ = previous.admission.image(a.dataset, {'path': old_c['index']['queries'][0]}, hashes[nd])
        f, z = model(x[None].cuda(), None, 'global')
        if not np.allclose(z.cpu().numpy()[0], vectors[nd], atol=2e-5, rtol=2e-4): raise ValueError('Global encoder mismatch')
        j = int(np.argmax(old_data['base'][0])); v = pair(f, old_data['candidates'][0, j])
        if not np.isclose(v, old_data['base'][0, j], atol=1e-4, rtol=1e-4): raise ValueError('Pair scorer mismatch')
    session = uuid.uuid4().hex
    work = pending if a.max_new_queries is None else pending[:a.max_new_queries]
    progress = tqdm(work, desc='Pitts remaining: frozen top44 (no interim accuracy)')
    started = time.perf_counter(); timings = []
    with torch.inference_mode():
        for r in progress:
            torch.cuda.synchronize(); begin = time.perf_counter()
            x, _ = previous.admission.image(a.dataset, {'path': r['path']}, r['image_sha256'])
            q, z = model(x[None].cuda(), None, 'global')
            gs = (z[0]@db.T).cpu().numpy(); ids = np.argsort(-gs, kind='stable')[:44]
            torch.cuda.synchronize(); enc_end = time.perf_counter()
            first = [pair(q, di) for di in ids[:20]]
            torch.cuda.synchronize(); twenty_end = time.perf_counter()
            extra = [pair(q, di) for di in ids[20:]]
            torch.cuda.synchronize(); end = time.perf_counter()
            row = dict(query_index=np.asarray(r['query_index']), candidates=ids, global_scores=gs[ids],
                scores=np.asarray(first+extra, np.float32), labels=np.isin(ids, r['gt']),
                image_sha256=np.asarray(r['image_sha256']), session=np.asarray(session),
                seconds=np.asarray([enc_end-begin, twenty_end-enc_end, end-twenty_end]))
            validate_new(row, r, nd); npz(pairdir/f'{r["query_index"]:06d}.npz', **row)
            timings.append(float(row['seconds'].sum()))
            write(a.work/'progress.json', dict(completed_remaining=len(remaining)-len(pending)+len(timings),
                total_remaining=6584, phase='scoring', session=session, last_query_id=r['query_index'],
                mean_seconds_per_query_this_session=float(np.mean(timings))))
    write(a.work/f'timing_{session}.json', dict(new_queries=len(work), loop_seconds=time.perf_counter()-started,
        median_seconds=float(np.median(timings)), p95_seconds=float(np.quantile(timings, .95)),
        estimated_remaining_hours=max(0, len(pending)-len(work))*float(np.mean(timings))/3600,
        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        note='Observed LRU reuse; includes reference dense encoding. Throughput estimate, not fixed deployment benchmark.'))
    print('Saved', len(work), 'queries; median seconds:', float(np.median(timings)), '; interim accuracy intentionally not reported')


def report(a):
    c, plan = load_contract(a); old_data, _ = verify_sources(a, c)
    if a.output.exists():
        previous.common.b.m.e.verify(a.output, codes())
        oc = read(a.output/'contract.json')
        if oc['work_contract_sha256'] != sha(a.work/'contract.json'): raise ValueError('Wrong report')
        print('Verified completed report'); return
    old_lookup = {qid: i for i, qid in enumerate(plan['old_query_ids'])}
    all_rows = []; records = []; times = []; shard_hashes = {}
    from tqdm import tqdm
    for r in tqdm(plan['queries'], desc='Verify complete paired outcomes'):
        qi = r['query_index']
        if r['prior_query']:
            j = old_lookup[qi]; file = a.previous_run/'pairs'/f'{j:06d}.npz'; row = load_npz(file)
            previous.common.validate_result(row, row['candidates'], {'base': old_data['base'][j]}, r['gt'])
            if not np.array_equal(row['candidates'][:20], old_data['candidates'][j]): raise ValueError('Old candidate prefix changed')
        else:
            file = a.work/'pairs'/f'{qi:06d}.npz'; row = load_npz(file); validate_new(row, r, len(plan['database']))
            times.append(row['seconds']); shard_hashes[f'pairs/{qi:06d}.npz'] = sha(file)
        item = previous.common.paired_record(qi, row)
        item.update(group=r['group'], prior_query=r['prior_query'], shares_panorama_with_old_subset=r['shares_panorama_with_old_subset'])
        records.append(item); all_rows.append(row)
    groups = dict(remaining=[i for i, r in enumerate(records) if not r['prior_query']],
        prior=[i for i, r in enumerate(records) if r['prior_query']], all=list(range(len(records))))
    summaries = {}
    for name, indices in groups.items():
        rr = [records[i] for i in indices]; stats = paired_statistics(rr)
        recall, _ = previous.common.summarize([all_rows[i] for i in indices])
        stats['top20_metrics'] = recall['top20']; stats['top44_metrics'] = recall['top44']
        stats['realized_newly_reachable_corrections'] = [r['query_index'] for r in rr if r['newly_reachable'] and r['correction']]
        summaries[name] = stats
    old_summary = read(a.previous_run/'summary.json')
    if (summaries['prior']['top20_correct'] != old_summary['top20']['correct']
        or summaries['prior']['top44_correct'] != old_summary['top44']['correct']
        or set(summaries['prior']['corrections']) != set(old_summary['corrections'])
        or set(summaries['prior']['regressions']) != set(old_summary['regressions'])): raise ValueError('Prior report not reproduced')
    a.output.mkdir(parents=True)
    write(a.output/'contract.json', dict(code=codes(), policy=POLICY, work_contract_sha256=sha(a.work/'contract.json'),
        plan_sha256=c['plan_sha256'], old_top44_sha256=c['old_top44_sha256']))
    write(a.output/'summary.json', summaries); write(a.output/'per_query.json', records)
    write(a.output/'grouping.json', plan['grouping']); write(a.output/'new_shard_hashes.json', shard_hashes)
    times = np.asarray(times); costs = {}
    for name, v in [('query_encoding_and_retrieval', times[:, 0]), ('first20_with_dense', times[:, 1]),
                    ('next24_with_dense', times[:, 2]), ('total44', times.sum(1))]:
        costs[name] = dict(total_seconds=float(v.sum()), mean_seconds=float(v.mean()),
            median_seconds=float(np.median(v)), p95_seconds=float(np.quantile(v, .95)))
    write(a.output/'timing.json', dict(remaining_query_costs=costs,
        caveat='First20 and next24 share a sequential LRU cache. Not independent cold/warm runs; excludes startup, validation and per-row output writes. No deployment speedup claim.'))
    complete(a.output); write(a.work/'progress.json', dict(phase='complete', report=str(a.output), completed_remaining=6584))
    print('Report complete:', a.output)
    for name, stats in summaries.items(): print(name, stats['net'], stats['delta_r1_pp'], stats['panorama_cluster_bootstrap_95ci_pp'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'run', 'report', 'status'])
    p.add_argument('--cache', type=Path, default=Path('.cache/pitts_pair_admission_v1'))
    p.add_argument('--previous-run', type=Path, default=Path('doc/candidate_top44_pitts_v1'))
    p.add_argument('--dataset', type=Path, default=Path('datasets/pitts30k-val'))
    p.add_argument('--official-repo', type=Path, default=Path('/home/wt/workspace/Pair-VPR-official'))
    p.add_argument('--audit', type=Path, default=Path('doc/pairvpr_official_paired_audit_v1'))
    p.add_argument('--work', type=Path, default=Path('.cache/pitts_top44_confirmation_v1'))
    p.add_argument('--output', type=Path, default=Path('doc/pitts_top44_confirmation_v1'))
    p.add_argument('--max-new-queries', type=int)
    a = p.parse_args()
    if a.max_new_queries is not None and (a.stage != 'run' or a.max_new_queries < 1): p.error('Invalid benchmark limit')
    if a.stage == 'status':
        print(read(a.work/'progress.json') if (a.work/'progress.json').exists() else 'Not started'); return
    with lock(a.work): globals()[a.stage](a)


if __name__ == '__main__': main()
