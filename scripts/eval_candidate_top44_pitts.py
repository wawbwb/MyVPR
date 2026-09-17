"""Fixed top20/top44 cross-dataset replication on previously used Pitts dev."""
import argparse
from collections import OrderedDict
import hashlib
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import eval_candidate_top44 as common
from scripts import candidate_set_pitts as cp
from scripts import pitts_pair_admission as admission
from scripts.candidate_set_screen import read, write, sha, complete, load_npz, npz

SCOPE = 'Cross-dataset exploratory replication on previously used 1024-query Pitts30k-val dev; NOT independent test or full Pitts evaluation'


def ranking(scores, old_ids):
    scores = np.asarray(scores)
    if scores.ndim != 1 or len(scores) < 44 or not np.isfinite(scores).all(): raise ValueError('Invalid global scores')
    ids = np.argsort(-scores, kind='stable')[:44]
    if not np.array_equal(ids[:20], old_ids): raise ValueError('Original Pitts top20 not reproduced')
    return ids


def remap_report(result, records, qids):
    qids = list(qids)
    if len(qids) != len(records) or len(set(qids)) != len(qids): raise ValueError('Invalid original query mapping')
    output = dict(result)
    for key in ('corrections', 'regressions', 'newly_reachable_ids', 'realized_newly_reachable_corrections'):
        output[key] = [int(qids[i]) for i in result[key]]
    mapped = []
    for i, r in enumerate(records):
        if r['query_index'] != i: raise ValueError('Unexpected subset query order')
        mapped.append(dict(r, subset_index=i, query_index=int(qids[i])))
    output['query_id_space'] = 'Original Pitts30k-val query index, NOT 1024-subset position'
    output['scope'] = SCOPE
    return output, mapped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=Path('.cache/pitts_pair_admission_v1'))
    p.add_argument('--dataset', type=Path, default=Path('datasets/pitts30k-val'))
    p.add_argument('--official-repo', type=Path, default=Path('/home/wt/workspace/Pair-VPR-official'))
    p.add_argument('--audit', type=Path, default=Path('doc/pairvpr_official_paired_audit_v1'))
    p.add_argument('--output', type=Path, default=Path('doc/candidate_top44_pitts_v1'))
    p.add_argument('--resume', action='store_true')
    a = p.parse_args(); a.gsv_root = a.dataset
    print('Verify original fixed Pitts cache, GT and source indices; no new query selection...', flush=True)
    data, c = cp.load_cache(a.cache, 'pitts')
    index = admission.load_index(a.dataset, 1024)
    if c['index'] != index: raise ValueError('Pitts indices/GT changed')
    nd, nq = len(index['database']), len(index['queries'])
    code = {**cp.codes(), **common.b.source_codes()}
    for module in (common, common.b):
        path = Path(module.__file__)
        code[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    code['scripts/eval_candidate_top44_pitts.py'] = hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    contract = dict(code=code, policy=dict(common.POLICY, split='pitts_fixed_dev', scope=SCOPE),
        cache_sha256=sha(a.cache/'completed.json'), official=c['official'], index=index)
    if a.output.exists():
        if not a.resume or read(a.output/'contract.json') != contract: raise ValueError('Use new output or identical --resume')
        if (a.output/'completed.json').exists():
            common.b.m.e.verify(a.output, code); print('Already complete and verified'); return
    vectors = []; hashes = []
    for start in range(0, nd+nq, 128):
        z = load_npz(a.cache/'global'/f'{start:07d}.npz'); size = min(128, nd+nq-start)
        if z['vectors'].shape != (size, 512) or len(z['hashes']) != size: raise ValueError('Invalid global shard')
        vectors.append(z['vectors']); hashes.extend(z['hashes'].tolist())
    vectors = np.concatenate(vectors)
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-4):
        raise ValueError('Invalid cached global descriptors')
    from tqdm import tqdm
    records = [{'path': name} for name in index['database']+index['queries']]
    for i, r in enumerate(tqdm(records, desc='Verify Pitts images')):
        path = (a.dataset/r['path']).resolve()
        if not path.is_relative_to(a.dataset.resolve()) or sha(path) != hashes[i]: raise ValueError('Source image changed')
    import torch
    model, identity = admission.official(a)
    if identity != c['official']: raise ValueError('Frozen model mismatch')
    db = torch.from_numpy(vectors[:nd]).cuda()
    inputs = []
    with torch.inference_mode():
        for qi in tqdm(range(nq), desc='Reproduce full-image retrieval'):
            scores = (torch.from_numpy(vectors[nd+qi]).cuda()@db.T).cpu().numpy()
            inputs.append(ranking(scores, data['candidates'][qi]))
    del db
    if not a.output.exists():
        a.output.mkdir(parents=True); write(a.output/'contract.json', contract); (a.output/'pairs').mkdir()
    memo = OrderedDict()
    def dense(i):
        if i in memo: memo.move_to_end(i); return memo[i].cuda()
        x, _ = admission.image(a.dataset, records[i], hashes[i])
        f, _ = model(x[None].cuda(), None, 'global'); memo[i] = f.cpu()
        if len(memo) > 64: memo.popitem(last=False)
        return f
    def pair(qf, di):
        df = dense(di)
        return float((model(qf, df, 'pairvpr')+model(df, qf, 'pairvpr')).item())
    rows = []; fresh = 0; torch.cuda.synchronize(); begin = time.perf_counter()
    with torch.inference_mode():
        for qi in tqdm(range(nq), desc='Pitts frozen top44 paired eval'):
            ids = inputs[qi]; old = {'base': data['base'][qi]}; gt = index['gt'][qi]
            file = a.output/'pairs'/f'{qi:06d}.npz'
            if file.exists() and file.with_suffix('.sha.json').exists(): row = load_npz(file)
            else:
                qf = dense(nd+qi); reproduced = pair(qf, int(ids[np.argmax(old['base'])]))
                if not np.isclose(reproduced, old['base'].max(), atol=1e-4, rtol=1e-4):
                    raise ValueError(f'Query subset {qi}: old winner score mismatch')
                added = np.asarray([pair(qf, int(di)) for di in ids[20:]], dtype=np.float32)
                row = dict(candidates=ids, scores=np.concatenate([old['base'], added]), labels=np.isin(ids, gt),
                    reproduced_score=np.asarray(reproduced))
                common.validate_result(row, ids, old, gt); npz(file, **row); fresh += 1
            common.validate_result(row, ids, old, gt); rows.append(row)
    torch.cuda.synchronize(); elapsed = time.perf_counter()-begin
    result, outcomes = common.summarize(rows)
    baseline = read(a.cache/'report'/'decision.json')['pair']
    if result['top20']['correct'] != baseline['correct'] or result['top20']['recall'] != baseline['recall']:
        raise ValueError('Original Pitts baseline not reproduced')
    result, outcomes = remap_report(result, outcomes, index['query_indices'])
    write(a.output/'summary.json', result); write(a.output/'per_query.json', outcomes)
    write(a.output/'query_mapping.json', [dict(subset_index=i, query_index=int(qid), path=index['queries'][i])
        for i, qid in enumerate(index['query_indices'])])
    write(a.output/'timing.json', dict(new_queries_this_invocation=fresh, loop_seconds=elapsed,
        new_pairs=fresh*24, reproduction_pairs=fresh, directional_forwards=fresh*50,
        note='Final invocation only; includes dense encoding, I/O, winner checks; excludes startup and global ranking'))
    complete(a.output); print(result); print('Cross-dataset dev replication complete:', a.output)


if __name__ == '__main__': main()
