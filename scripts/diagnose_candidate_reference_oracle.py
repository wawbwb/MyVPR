"""CPU-only, label-assisted graph diagnostic. NOT deployable retrieval."""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_candidate_reference_graph as g
from scripts.candidate_set_screen import read, write, sha, complete, load_npz, npz

POLICY = dict(alpha=0.5, temperature=1.0, support='maximum neighbor query probability',
    formula='softmax(base)[i] + 0.5 * max_neighbor softmax(base)[j]; empty support=0',
    selection='No tuning, no training; same fixed rule for visual and label-filtered graph',
    tie_break='original candidate order, matching frozen baseline',
    scope='EXPLORATORY LABEL-ASSISTED DIAGNOSTIC; not deployable, not a method upper bound; dev is repeatedly inspected')


def score_graph(base, ids, edges, db_labels=None):
    """No query GT accepted. db_labels filters edges ONLY in oracle mode."""
    base = np.asarray(base, dtype=np.float64)
    ids = np.asarray(ids)
    if base.ndim != 1 or len(base) == 0 or ids.shape != base.shape or not np.isfinite(base).all():
        raise ValueError('Invalid scores/IDs')
    if ids.dtype.kind not in 'iu' or (ids < 0).any() or len(set(ids.tolist())) != len(ids):
        raise ValueError('Invalid candidate IDs')
    index = {int(di): i for i, di in enumerate(ids)}
    prob = np.exp(base-base.max()); prob /= prob.sum()
    support = np.zeros_like(prob); retained = []; seen = set()
    for left, right, _ in edges:
        if left not in index or right not in index or left == right:
            raise ValueError('Invalid edge endpoints')
        key = tuple(sorted((left, right)))
        if key in seen: raise ValueError('Duplicate edge')
        seen.add(key)
        if db_labels is not None and db_labels[left] != db_labels[right]: continue
        i, j = index[left], index[right]
        support[i] = max(support[i], prob[j]); support[j] = max(support[j], prob[i])
        retained.append(key)
    return prob + POLICY['alpha']*support, retained


def summarize(records):
    result = {}
    for variant in ('frozen', 'visual_graph', 'label_filtered_graph'):
        fixes = [r['query_index'] for r in records if not r['frozen_correct'] and r[variant+'_correct']]
        losses = [r['query_index'] for r in records if r['frozen_correct'] and not r[variant+'_correct']]
        n = len(records); correct = sum(r[variant+'_correct'] for r in records)
        result[variant] = dict(queries=n, correct=correct, r1=correct/n if n else None,
            corrections=fixes, regressions=losses, net=len(fixes)-len(losses),
            top1_changed=sum(r[variant+'_id'] != r['frozen_id'] for r in records))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, default=Path('doc/candidate_reference_graph_audit_v1'))
    p.add_argument('--plan', type=Path, default=Path('doc/candidate_hard_plan_v1'))
    p.add_argument('--cache-root', type=Path, default=Path('.cache/candidate_hard_v1'))
    p.add_argument('--output', type=Path, default=Path('doc/candidate_reference_oracle_v1'))
    a = p.parse_args()
    if a.output.exists(): p.error('Choose a new output directory; reports are never overwritten')
    graph_code = {**g.e.codes(), 'scripts/audit_candidate_reference_graph.py':
        hashlib.sha256(Path(g.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()}
    print('CPU only: verifying graph audit, original plan and complete caches...', flush=True)
    contract = g.e.verify(a.audit, graph_code)
    plan = g.e.verify(a.plan, g.e.h.codes()); g.e.h.ensure_disjoint(plan['plan'])
    if contract['policy'] != g.POLICY or contract['plan_sha256'] != sha(a.plan/'completed.json'):
        raise ValueError('Graph policy/plan mismatch')
    all_records = {}; results = {}; predictions = {}
    for split in ('train', 'dev'):
        if contract['cache_sha256'][split] != sha(a.cache_root/split/'completed.json'):
            raise ValueError('Cache completion identity mismatch')
        identity, _, audited, packed = g.audit_split(a, split, plan)
        if identity['official'] != contract['official']: raise ValueError('Frozen model mismatch')
        saved = load_npz(a.audit/f'{split}_pair_plan.npz')
        if set(saved) != set(packed) or any(not np.array_equal(saved[k], packed[k]) for k in packed):
            raise ValueError('Graph plan does not reproduce')
        if audited != read(a.audit/'per_query.json')[split]: raise ValueError('Graph diagnostics changed')
        labels = [r['label'] for r in plan['plan'][split]['database']]
        records = []; score_rows = []; base_rows = []; truth_rows = []; id_rows = []
        for qi, info in enumerate(audited):
            row = load_npz(a.cache_root/split/'pairs'/f'{qi:06d}.npz')
            start, end = packed['offsets'][qi:qi+2]
            edges = [(int(l), int(r), 0.) for l, r in packed['pairs'][packed['edge_indices'][start:end]]]
            visual, _ = score_graph(row['base'], row['candidates'], edges)
            oracle, _ = score_graph(row['base'], row['candidates'], edges, labels)
            record = dict(query_index=qi, category=info['category'], path=info['path'],
                positive_support=info['selected']['positive_positive'] > 0,
                wrong_support=info['selected']['wrong_same_place'] > 0)
            for name, scores in [('frozen', row['base']), ('visual_graph', visual), ('label_filtered_graph', oracle)]:
                winner = int(np.argmax(scores))
                record[name+'_id'] = int(row['candidates'][winner])
                record[name+'_correct'] = bool(row['labels'][winner])
            records.append(record); score_rows.append(np.stack([visual, oracle]))
            base_rows.append(row['base']); truth_rows.append(row['labels']); id_rows.append(row['candidates'])
        groups = {'all': records}
        for category in ('reachable_error', 'reachable_correct', 'unreachable'):
            groups[category] = [r for r in records if r['category'] == category]
        for positive in (False, True):
            for wrong in (False, True):
                groups[f'errors_positive_support_{positive}_wrong_support_{wrong}'] = [r for r in records
                    if r['category'] == 'reachable_error' and r['positive_support'] == positive and r['wrong_support'] == wrong]
        results[split] = {name: summarize(rows) for name, rows in groups.items()}
        all_records[split] = records
        predictions[split] = dict(scores=np.stack(score_rows), base=np.stack(base_rows),
            labels=np.stack(truth_rows), candidates=np.stack(id_rows))
        print(split, results[split]['all'], flush=True)
    a.output.mkdir(parents=True)
    write(a.output/'contract.json', dict(policy=POLICY, graph_audit_sha256=sha(a.audit/'completed.json'),
        code={**graph_code, 'scripts/diagnose_candidate_reference_oracle.py': hashlib.sha256(
            Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()},
        score_axis=['visual_graph', 'label_filtered_graph'], gpu_started=False))
    write(a.output/'summary.json', results); write(a.output/'per_query.json', all_records)
    for split, arrays in predictions.items(): npz(a.output/f'{split}_predictions.npz', **arrays)
    complete(a.output)
    print('Diagnostic complete. GT-filtered scores cannot be used as deployable results:', a.output)


if __name__ == '__main__': main()
