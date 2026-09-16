"""CPU-only reference-reference graph feasibility and exact pair budget.

No new image extraction, no Pair-VPR calls, no training, no retrieval claims.
Edges depend ONLY on cached visual descriptors and database IDs, never GT.
"""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import candidate_hard_exploratory as e
from scripts.candidate_set_screen import read, write, sha, complete, load_npz, npz

POLICY = dict(topk=20, reciprocal_neighbors=2, max_edges_per_query=10,
    selection='Mutual top2 cosine neighbors within cached global top20; take highest cosine 10 undirected edges',
    tie_break='ascending database ID, not candidate-list position',
    inference_labels=False, pair_verification='NOT RUN',
    scope='GSV adaptation train/holdout audit only. Holdout labels are diagnostics, not edge selection or threshold tuning.')


def graph_edges(ids, vectors, neighbors=2, cap=10):
    """Return unique ID pairs and cosines, including pre-cap graph for audit."""
    ids, z = np.asarray(ids), np.asarray(vectors, dtype=np.float64)
    n = len(ids)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(set(ids.tolist())) != n or (ids < 0).any():
        raise ValueError('Expected unique nonnegative integer DB IDs')
    if z.ndim != 2 or len(z) != n or not np.isfinite(z).all(): raise ValueError('Invalid descriptors')
    if neighbors < 1 or cap < 1: raise ValueError('Invalid graph budget')
    if n < 2: return [], []
    norms = np.linalg.norm(z, axis=1)
    if not np.allclose(norms, 1, atol=2e-4): raise ValueError('Unnormalized descriptors')
    z = z/norms[:, None]; sim = np.clip(z@z.T, -1, 1)
    nearest = []
    for i in range(n):
        others = np.flatnonzero(np.arange(n) != i)
        ordered = others[np.lexsort((ids[others], -sim[i, others]))]
        nearest.append(set(ordered[:min(neighbors, n-1)].tolist()))
    edges = []
    for i in range(n):
        for j in nearest[i]:
            if i < j and i in nearest[j]:
                left, right = sorted((int(ids[i]), int(ids[j])))
                edges.append((left, right, float(sim[i, j])))
    edges.sort(key=lambda edge: (-edge[2], edge[0], edge[1]))
    return edges[:cap], edges


def edge_diagnostics(edges, db_labels, query_label):
    """GT applied AFTER fixed visual edge construction, for audit only."""
    out = Counter(same_place=0, different_place=0, positive_positive=0,
                  positive_negative=0, wrong_same_place=0, wrong_different_place=0)
    for i, j, _ in edges:
        left, right = db_labels[i], db_labels[j]
        same = left == right
        out['same_place' if same else 'different_place'] += 1
        lp, rp = left == query_label, right == query_label
        if lp and rp: out['positive_positive'] += 1
        elif lp or rp: out['positive_negative'] += 1
        else: out['wrong_same_place' if same else 'wrong_different_place'] += 1
    return dict(out)


def compact_plan(query_edges):
    pairs = sorted({(i, j) for edges in query_edges for i, j, _ in edges})
    index = {edge: i for i, edge in enumerate(pairs)}
    offsets = [0]; references = []; cosines = []
    for edges in query_edges:
        references.extend(index[(i, j)] for i, j, _ in edges)
        cosines.extend(v for _, _, v in edges); offsets.append(len(references))
    return dict(pairs=np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
        offsets=np.asarray(offsets, dtype=np.int64),
        edge_indices=np.asarray(references, dtype=np.int64),
        global_cosines=np.asarray(cosines, dtype=np.float32))


def coverage(rows):
    n = len(rows)
    two = [r for r in rows if r['positive_candidates'] >= 2]
    pp = sum(r['selected']['positive_positive'] > 0 for r in rows)
    pre = sum(r['pre_cap']['positive_positive'] > 0 for r in rows)
    totals = {k: sum(r['selected'][k] for r in rows) for k in edge_diagnostics([], [], '')}
    count = sum(r['selected_edge_count'] for r in rows)
    return dict(queries=n, queries_with_two_positive_candidates=len(two),
        queries_with_positive_edge_before_cap=pre, queries_with_positive_edge=pp,
        positive_edge_coverage_of_two_positive_queries=pp/len(two) if two else None,
        positive_edge_lost_to_cap=pre-pp,
        queries_with_wrong_same_place_edge=sum(r['selected']['wrong_same_place'] > 0 for r in rows),
        queries_with_both_positive_and_wrong_same_place_edges=sum(
            r['selected']['positive_positive'] > 0 and r['selected']['wrong_same_place'] > 0 for r in rows),
        queries_with_no_edge=sum(r['selected_edge_count'] == 0 for r in rows),
        edge_occurrences=count, edge_gt_counts=totals,
        same_place_fraction=totals['same_place']/count if count else None,
        note='Positive-edge availability is only a structural opportunity, not predicted corrections. Wrong-place clusters can be internally consistent.')


def audit_split(a, split, plan):
    cache = a.cache_root/split
    c = e.verify(cache, e.h.codes())
    part = plan['plan'][split]; queries, database = part['queries'], part['database']
    if (c['plan_sha256'] != sha(a.plan/'completed.json') or c['split'] != split or c['topk'] != 20
        or c['queries'] != len(queries) or c['ndb'] != len(database)):
        raise ValueError('Cache/plan dimensions or identity mismatch')
    labels = [r['label'] for r in database]; places = {}
    for i, label in enumerate(labels): places.setdefault(label, []).append(i)
    rows = []; query_edges = []; seen_vectors = {}
    for qi, q in enumerate(queries):
        row = load_npz(cache/'pairs'/f'{qi:06d}.npz')
        e.h.validate_row(row, places[q['label']], len(database))
        # Every reusable reference pair must denote the same descriptor/image
        # irrespective of which query first encountered it.
        for di, vector in zip(row['candidates'], row['db_vectors']):
            key = int(di)
            if key in seen_vectors and not np.array_equal(seen_vectors[key], vector):
                raise ValueError('Database descriptor inconsistent across query shards')
            seen_vectors.setdefault(key, vector.copy())
        selected, pre = graph_edges(row['candidates'], row['db_vectors'])
        diag = e.h.inspect_query(row['base'], row['labels'])
        rows.append(dict(query_index=qi, path=q['path'], city=q['city'], label=q['label'],
            category=diag['category'], range_fixable=diag['residual_range_only_fixable'],
            positive_candidates=int(row['labels'].sum()),
            selected_edge_count=len(selected), pre_cap_edge_count=len(pre),
            selected=edge_diagnostics(selected, labels, q['label']),
            pre_cap=edge_diagnostics(pre, labels, q['label'])))
        query_edges.append(selected)
        if (qi+1) % 256 == 0 or qi+1 == len(queries):
            print(f'{split}: audited {qi+1}/{len(queries)} cached query rows (CPU)', flush=True)
    packed = compact_plan(query_edges)
    n = len(packed['pairs']); occurrences = len(packed['edge_indices'])
    unique_same = sum(labels[int(i)] == labels[int(j)] for i, j in packed['pairs'])
    bytes_plan = sum(v.nbytes for v in packed.values())
    # Future proposal only: two float32 directional scores + bool completion.
    # Not an end-to-end disk/RAM/runtime estimate.
    budget = dict(unique_undirected_reference_pairs=n, edge_occurrences=occurrences,
        deduplicated_occurrences_saved=occurrences-n, directional_pair_forwards=2*n,
        unique_reference_images=len(set(packed['pairs'].ravel().tolist())),
        packed_plan_array_bytes=bytes_plan, proposed_score_and_done_array_bytes=n*9,
        proposed_plan_plus_score_array_MiB=(bytes_plan+n*9)/2**20,
        excludes='Numpy headers, provenance, checkpoints, dense recomputation/cache and RAM. No extraction/runtime estimate; benchmark before approval.',
        gpu_started=False)
    subsets = dict(all=rows,
        reachable_errors=[r for r in rows if r['category'] == 'reachable_error'],
        range_fixable_errors=[r for r in rows if r['range_fixable']],
        baseline_correct=[r for r in rows if r['category'] == 'reachable_correct'],
        unreachable=[r for r in rows if r['category'] == 'unreachable'])
    out = dict(coverage={name: coverage(subset) for name, subset in subsets.items()}, budget=budget,
        unique_pair_gt=dict(same_place=unique_same, different_place=n-unique_same,
            same_place_fraction=unique_same/n if n else None),
        references_per_place=dict(Counter(str(len(v)) for v in places.values())))
    return c, out, rows, packed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, default=Path('doc/candidate_hard_plan_v1'))
    p.add_argument('--cache-root', type=Path, default=Path('.cache/candidate_hard_v1'))
    p.add_argument('--output', type=Path, default=Path('doc/candidate_reference_graph_audit_v1'))
    a = p.parse_args()
    if a.output.exists(): p.error('Choose a new output; reports are never overwritten')
    print('CPU / disk audit only. Verifying original plan and cache hashes...', flush=True)
    plan = e.verify(a.plan, e.h.codes()); e.h.ensure_disjoint(plan['plan'])
    results = {}; records = {}; arrays = {}; identities = {}
    for split in ('train', 'dev'):
        c, results[split], records[split], arrays[split] = audit_split(a, split, plan)
        identities[split] = c
    if identities['train']['official'] != identities['dev']['official']:
        raise ValueError('Train/dev frozen model identity mismatch')
    a.output.mkdir(parents=True)
    for split in ('train', 'dev'):
        npz(a.output/f'{split}_pair_plan.npz', **arrays[split])
    write(a.output/'summary.json', results); write(a.output/'per_query.json', records)
    write(a.output/'reference_mapping.json', {s: plan['plan'][s]['database'] for s in ('train', 'dev')})
    write(a.output/'contract.json', dict(code={**e.codes(), 'scripts/audit_candidate_reference_graph.py':
        hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()}, policy=POLICY,
        plan_sha256=sha(a.plan/'completed.json'), official=identities['train']['official'],
        cache_sha256={s: sha(a.cache_root/s/'completed.json') for s in ('train', 'dev')},
        caution='GSV place-ID edge labels are audit-only, not visual-overlap truth. Two references/place is an artificial sampling constraint. No inference graph uses GT.'))
    complete(a.output)
    for split in ('train', 'dev'):
        v = results[split]
        print(split, 'pair budget:', v['budget'], flush=True)
        print(split, 'reachable-error coverage:', v['coverage']['reachable_errors'], flush=True)
    print('AUDIT COMPLETE, not a retrieval success/failure. No GPU verification launched. Download:', a.output)


if __name__ == '__main__': main()
