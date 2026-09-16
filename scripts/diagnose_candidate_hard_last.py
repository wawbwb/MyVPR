"""Read-only final-epoch margin diagnostics, NOT selected epoch-zero scores."""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import candidate_hard_exploratory as e
from scripts.candidate_set_screen import read, write, sha, complete, npz
from scripts.audit_candidate_training import distribution

EPS = 1e-5  # Numerical deadband only, not a significance threshold.


def diagnose_rows(values):
    s, b, y = (np.asarray(values[k]) for k in ('scores', 'base', 'labels'))
    if s.shape != b.shape or y.shape != s.shape or y.dtype != np.bool_:
        raise ValueError('Invalid score/label shapes')
    if not np.isfinite(s).all() or not np.isfinite(b).all(): raise ValueError('Nonfinite scores')
    delta = s.astype(np.float64)-b.astype(np.float64)
    if np.abs(delta).max() > 4.0001: raise ValueError('Residual bound violated')
    rows = []
    for j, qi in enumerate(values['query_indices']):
        pos = np.flatnonzero(y[j]); neg = np.flatnonzero(~y[j])
        old_top, new_top = int(np.argmax(b[j])), int(np.argmax(s[j]))
        row = dict(query_index=int(qi), baseline_correct=bool(y[j, old_top]),
            last_correct=bool(y[j, new_top]), top1_changed=old_top != new_top,
            corrected=bool(y[j, new_top] and not y[j, old_top]),
            regressed=bool(y[j, old_top] and not y[j, new_top]),
            residual_mean=float(delta[j].mean()), residual_std=float(delta[j].std()),
            residual_range=float(np.ptp(delta[j])),
            saturation_fraction=float((np.abs(delta[j]) >= 3.96).mean()),
            margin_before=None, margin_after=None, margin_change=None,
            fixed_pair_delta_advantage=None)
        if len(pos) and len(neg):
            p = int(pos[np.argmax(b[j, pos])]); n = int(neg[np.argmax(b[j, neg])])
            before = float(b[j, p])-float(b[j, n])
            after = float(s[j, pos].max())-float(s[j, neg].max())
            row.update(margin_before=before, margin_after=after, margin_change=after-before,
                fixed_pair_delta_advantage=float(delta[j, p]-delta[j, n]),
                baseline_positive_position=p, baseline_negative_position=n,
                positive_saturation=bool(abs(delta[j, p]) >= 3.96),
                negative_saturation=bool(abs(delta[j, n]) >= 3.96))
        rows.append(row)
    return rows, delta


def group_stats(rows, groups):
    result = {}
    for name, ids in groups.items():
        chosen = [r for r in rows if r['query_index'] in set(ids)]
        valid = [r for r in chosen if r['margin_change'] is not None]
        changes = [r['margin_change'] for r in valid]
        result[name] = dict(queries=len(chosen), margin_queries=len(valid),
            corrected=[r['query_index'] for r in chosen if r['corrected']],
            regressed=[r['query_index'] for r in chosen if r['regressed']],
            improved=sum(v > EPS for v in changes), worsened=sum(v < -EPS for v in changes),
            unchanged=sum(abs(v) <= EPS for v in changes),
            margin_before=distribution([r['margin_before'] for r in valid]),
            margin_after=distribution([r['margin_after'] for r in valid]),
            margin_change=distribution(changes),
            fixed_pair_delta_advantage=distribution([r['fixed_pair_delta_advantage'] for r in valid]),
            residual_std=distribution([r['residual_std'] for r in chosen]),
            residual_range=distribution([r['residual_range'] for r in chosen]),
            saturation_fraction=distribution([r['saturation_fraction'] for r in chosen]))
    return result


def correlation(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12: return None
    return float(np.corrcoef(a, b)[0, 1])


def compare(left, right):
    # left/right are raw deltas. Center within query so shared score offsets
    # cannot masquerade as agreement in ranking-relevant modifications.
    lc, rc = left-left.mean(1, keepdims=True), right-right.mean(1, keepdims=True)
    active = (np.abs(lc) > EPS) & (np.abs(rc) > EPS)
    return dict(raw_delta_correlation=correlation(left, right),
        centered_delta_correlation=correlation(lc, rc),
        centered_mae=float(np.abs(lc-rc).mean()),
        centered_active_entries=int(active.sum()),
        centered_sign_agreement=float((np.sign(lc[active]) == np.sign(rc[active])).mean()) if active.any() else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=Path('logs/candidate_hard_exploratory_v1'))
    p.add_argument('--plan', type=Path, default=Path('doc/candidate_hard_plan_v1'))
    p.add_argument('--mining', type=Path, default=Path('doc/candidate_hard_mining_v1'))
    p.add_argument('--cache-root', type=Path, default=Path('.cache/candidate_hard_v1'))
    p.add_argument('--output', type=Path, default=Path('doc/candidate_hard_last_diagnostic_v1'))
    a = p.parse_args()
    if a.output.exists(): p.error('Use a new output directory; existing reports are never overwritten')
    import torch
    from src.models.candidate_set import CandidateSet
    e.h.original.cuda(); e.h.original.seed()
    run = e.verify(a.run, e.codes())
    if run['mining_sha256'] != sha(a.mining/'completed.json') or not run['exploratory']:
        raise ValueError('Wrong exploratory run/mining identity')
    plan, mining_rows, selection, _ = e.validate_inputs(a)
    chosen = selection['selected']
    indices = dict(train=chosen, dev=list(range(len(mining_rows['dev']))))
    groups = dict(train=e.groups_for([r for r in mining_rows['train'] if r['query_index'] in set(chosen)], selection),
                  dev=e.groups_for(mining_rows['dev']))
    summaries = {}; all_rows = {}; all_deltas = {}; all_values = {}; checkpoints = {}
    for mode in e.h.MODES:
        path = a.run/f'{mode}_last.pt'
        if sha(path) != read(path.with_suffix('.sha.json'))['sha256']: raise ValueError('Changed last checkpoint')
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['contract'] != run or state['epoch'] != run['policy']['epochs']:
            raise ValueError('Expected completed final epoch, not best/partial checkpoint')
        model = CandidateSet(mode).cuda(); model.load_state_dict(state['model'], strict=True)
        checkpoints[mode] = dict(sha256=sha(path), epoch=state['epoch'], selected_epoch=state['best']['epoch'])
        last_history = state['history'][-1]
        if last_history != read(a.run/f'{mode}.json')['history'][-1]: raise ValueError('History/checkpoint mismatch')
        summaries[mode] = {}; all_rows[mode] = {}; all_deltas[mode] = {}; all_values[mode] = {}
        for split in ('train', 'dev'):
            print(f'Evaluate LAST epoch {state["epoch"]}: {mode}/{split}', flush=True)
            _, values = e.h.evaluate(model, a.cache_root/split, indices[split])
            if e.group_report(values, groups[split]) != last_history[split]:
                raise ValueError('Final checkpoint does not reproduce logged outcomes')
            rows, delta = diagnose_rows(values)
            mapping = plan['plan'][split]['queries']
            for r in rows:
                rec = mapping[r['query_index']]
                r.update(city=rec['city'], label=rec['label'], path=rec['path'])
            extra = dict(groups[split])
            extra['remaining_range_fixable_errors'] = [r['query_index'] for r in rows
                if r['query_index'] in groups[split]['range_fixable_errors'] and not r['last_correct']]
            summaries[mode][split] = group_stats(rows, extra)
            all_rows[mode][split] = rows; all_deltas[mode][split] = delta; all_values[mode][split] = values
            focus = summaries[mode][split]['range_fixable_errors']
            print('Range-fixable margin improved/worsened:', focus['improved'], focus['worsened'],
                  'corrections:', len(focus['corrected']), flush=True)
        del model, state
    paired = {}
    for split in ('train', 'dev'):
        paired[split] = {}
        for name, members in groups[split].items():
            mask = np.isin(indices[split], members)
            if not mask.any(): continue
            result = compare(all_deltas['independent'][split][mask], all_deltas['set'][split][mask])
            ir, sr = all_rows['independent'][split], all_rows['set'][split]
            valid = [i for i in np.flatnonzero(mask) if ir[i]['margin_change'] is not None]
            result['set_minus_independent_margin_change'] = distribution([
                sr[i]['margin_change']-ir[i]['margin_change'] for i in valid])
            result['set_vs_independent_outcomes'] = e.group_report(
                {**all_values['set'][split], 'base': all_values['independent'][split]['scores']}, {name: members})[name]
            paired[split][name] = result
    a.output.mkdir(parents=True)
    write(a.output/'summary.json', summaries); write(a.output/'per_query.json', all_rows)
    write(a.output/'paired.json', paired)
    write(a.output/'contract.json', dict(code={**e.codes(), 'scripts/diagnose_candidate_hard_last.py':
        hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()},
        run_sha256=sha(a.run/'completed.json'), mining_sha256=sha(a.mining/'completed.json'), checkpoints=checkpoints,
        epsilon=EPS, saturation_definition='abs(score-base)/4 >= 0.99; measured effective residual, not raw head logit',
        scope='Post-hoc final-epoch diagnosis; GSV holdout previously used for selection. No automatic pass or extension decision.'))
    for mode in e.h.MODES:
        for split in ('train', 'dev'): npz(a.output/f'{mode}_{split}_last.npz', **all_values[mode][split])
    complete(a.output); print('Diagnostic complete:', a.output)


if __name__ == '__main__': main()
