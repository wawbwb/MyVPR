"""Exploratory independent-head ablation: list loss vs margin preservation.

No old source/cache changes; frozen-correct membership uses TRAIN labels only.
"""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import candidate_hard_exploratory as e
from scripts.candidate_set_screen import read, write, sha, complete, npz, load_npz
from scripts.diagnose_candidate_hard_last import diagnose_rows, group_stats

ARMS = ('original', 'preserve')
POLICY = dict(e.h.POLICY, architecture='independent', arms=list(ARMS),
    preservation_weight=1.0, preservation_tolerance=0.0,
    preservation='mean over ALL batch queries of frozen-correct * relu(detached frozen margin - current margin); multi-positive max margin',
    selection='Full GSV holdout R1; earliest ties including epoch0. Selected and final epochs both reported.',
    scope='Post-hoc exploratory objective ablation, not an independent benchmark test; no Pitts access.')


def codes():
    return {**e.codes(), **{name: hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        for name in ('scripts/candidate_margin_preserve.py', 'scripts/diagnose_candidate_hard_last.py')}}


def objective(scores, base, labels, weight):
    """Detached reference; no penalty on frozen errors or unreachable queries."""
    import torch
    import torch.nn.functional as F
    valid = labels.any(-1) & (~labels).any(-1)
    if not bool(valid.all()): raise ValueError('Training batch must have positive and negative candidates')
    pos = scores.masked_fill(~labels, float('-inf'))
    neg = scores.masked_fill(labels, float('-inf'))
    retrieval = F.softplus(neg.logsumexp(-1)-pos.logsumexp(-1)).mean()
    frozen = base.detach()
    correct = labels.gather(1, frozen.argmax(-1, keepdim=True)).squeeze(1)
    frozen_margin = (frozen.masked_fill(~labels, float('-inf')).max(-1).values
                     - frozen.masked_fill(labels, float('-inf')).max(-1).values)
    margin = pos.max(-1).values-neg.max(-1).values
    per_query = F.relu(frozen_margin-margin)*correct.to(scores.dtype)
    penalty = per_query.mean()
    return retrieval+weight*penalty, retrieval, penalty, correct.sum()


def step(model, optimizer, d, epoch, offset, weight):
    import torch
    x, s, z, y = [d[k] for k in ('evidence', 'base', 'db_vectors', 'labels')]
    scores = model(x, s, z)
    extra = np.random.default_rng(100000*epoch+offset+42).integers(0, 20, size=5)
    ix = torch.tensor(list(range(20))+extra.tolist(), device=x.device)
    augmented = model(x[:, ix], s[:, ix], z[:, ix])[:, :20]
    total1, retrieval1, preserve1, n = objective(scores, s, y, weight)
    total2, retrieval2, preserve2, _ = objective(augmented, s, y, weight)
    total = .5*(total1+total2)
    if not torch.isfinite(total): raise ValueError('Nonfinite training objective')
    optimizer.zero_grad(set_to_none=True); total.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
    optimizer.step()
    return dict(total=float(total.detach()), retrieval=float(.5*(retrieval1+retrieval2).detach()),
                preservation=float(.5*(preserve1+preserve2).detach()), frozen_correct=int(n), grad_norm=float(norm))


def evaluate(model, a, indices, groups):
    results = {}; arrays = {}; per_query = {}
    for split in ('train', 'dev'):
        _, v = e.h.evaluate(model, a.cache_root/split, indices[split])
        rows, _ = diagnose_rows(v)
        results[split] = dict(outcomes=e.group_report(v, groups[split]), margins=group_stats(rows, groups[split]))
        arrays[split] = v; per_query[split] = rows
    return results, arrays, per_query


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, default=Path('doc/candidate_hard_plan_v1'))
    p.add_argument('--mining', type=Path, default=Path('doc/candidate_hard_mining_v1'))
    p.add_argument('--cache-root', type=Path, default=Path('.cache/candidate_hard_v1'))
    p.add_argument('--output', type=Path, default=Path('logs/candidate_margin_preserve_v1'))
    p.add_argument('--acknowledge-exploratory', action='store_true', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke-test', action='store_true')
    a = p.parse_args()
    import torch
    from tqdm import tqdm
    from src.models.candidate_set import CandidateSet
    e.h.original.cuda()
    plan, rows, selected, gate = e.validate_inputs(a)
    indices = dict(train=selected['selected'], dev=list(range(len(rows['dev']))))
    groups = dict(train=e.groups_for([r for r in rows['train'] if r['query_index'] in set(indices['train'])], selected),
                  dev=e.groups_for(rows['dev']))
    if a.smoke_test:
        smoke_ids = selected['errors'][:2]+selected['near_correct'][:2]+selected['ordinary_correct'][:2]
        d = e.h.batch(a.cache_root/'train', smoke_ids)
        for arm in ARMS:
            e.h.original.seed(); model = CandidateSet('independent').cuda()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.001)
            with torch.no_grad():
                out = model(d['evidence'], d['base'], d['db_vectors'])
                if not torch.equal(out, d['base']): raise ValueError('Zero-start mismatch')
            stats = step(model, opt, d, 0, 0, float(arm == 'preserve'))
            if stats['grad_norm'] <= 0 or stats['frozen_correct'] != 4: raise ValueError('Invalid smoke update')
            with torch.no_grad():
                if torch.equal(model(d['evidence'], d['base'], d['db_vectors']), d['base']):
                    raise ValueError('Smoke head did not update')
            print('SMOKE', arm, stats)
        print('SMOKE PASS; no run or checkpoint written'); return
    contract = dict(code=codes(), policy=POLICY, exploratory=True, original_gate=gate,
        mining_sha256=sha(a.mining/'completed.json'),
        reason='User-authorized single-objective ablation after final-epoch margin diagnostics. Not a passed original gate.')
    if a.output.exists():
        if not a.resume or read(a.output/'contract.json') != contract: raise ValueError('Use a new output or matching --resume')
        if (a.output/'completed.json').exists():
            e.verify(a.output, codes()); print('Already complete'); return
    else:
        a.output.mkdir(parents=True); write(a.output/'contract.json', contract)
    write(a.output/'original_gate.json', gate); write(a.output/'groups.json', groups)
    final_summaries = {}
    for arm in ARMS:
        e.h.original.seed(); model = CandidateSet('independent').cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.001)
        last_path = a.output/f'{arm}_last.pt'; best_path = a.output/f'{arm}_best.pt'
        history = []; start_epoch = 0
        if last_path.exists():
            if sha(last_path) != read(last_path.with_suffix('.sha.json'))['sha256']: raise ValueError('Damaged checkpoint')
            state = torch.load(last_path, map_location='cpu', weights_only=True)
            if state['contract'] != contract or state['arm'] != arm: raise ValueError('Checkpoint contract mismatch')
            model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
            start_epoch, history, best = state['epoch'], state['history'], state['best']
            best_state = state['best_model']; e.h.save_checkpoint(best_path, best_state)
            torch.set_rng_state(state['rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        else:
            stats, values, _ = evaluate(model, a, indices, groups)
            if any(not np.array_equal(v['scores'], v['base']) for v in values.values()): raise ValueError('Zero-start mismatch')
            best = dict(epoch=0, correct=stats['dev']['outcomes']['all']['correct'])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            e.h.save_checkpoint(best_path, best_state); history.append(dict(epoch=0, evaluation=stats))
        for epoch in range(start_epoch, 5):
            model.train(); order = np.random.default_rng(42+epoch).permutation(indices['train']); batches = []
            progress = tqdm(range(0, len(order), 8), desc=f'{arm} epoch {epoch+1}/5')
            for offset in progress:
                d = e.h.batch(a.cache_root/'train', order[offset:offset+8])
                loss = step(model, opt, d, epoch, offset, float(arm == 'preserve'))
                loss['queries'] = len(order[offset:offset+8]); batches.append(loss)
                progress.set_postfix(loss=loss['total'], keep=loss['preservation'])
            stats, _, _ = evaluate(model, a, indices, groups)
            history.append(dict(epoch=epoch+1, batches=batches, evaluation=stats))
            if stats['dev']['outcomes']['all']['correct'] > best['correct']:
                best = dict(epoch=epoch+1, correct=stats['dev']['outcomes']['all']['correct'])
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                e.h.save_checkpoint(best_path, best_state)
            e.h.save_checkpoint(last_path, dict(contract=contract, arm=arm, model=model.state_dict(), optimizer=opt.state_dict(),
                epoch=epoch+1, history=history, best=best, best_model=best_state,
                rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
            write(a.output/f'{arm}_history.json', history)
            dv = stats['dev']['outcomes']['all']; tr = stats['train']['outcomes']
            print(arm, 'train fixes:', len(tr['range_fixable_errors']['corrected']),
                'near regressions:', len(tr['near_correct']['regressed']),
                'dev fixes/regressions:', len(dv['corrected']), len(dv['regressed']), flush=True)
        # Export final AND selected states automatically. Best may be epoch0.
        arm_summary = dict(best=best, history=history)
        for phase in ('last', 'selected'):
            if phase == 'selected': model.load_state_dict(best_state)
            stats, values, per_query = evaluate(model, a, indices, groups)
            expected = history[-1 if phase == 'last' else best['epoch']]['evaluation']
            if stats != expected: raise ValueError('Checkpoint does not reproduce epoch diagnostics')
            arm_summary[phase] = stats
            for split in ('train', 'dev'): npz(a.output/f'{arm}_{phase}_{split}.npz', **values[split])
            write(a.output/f'{arm}_{phase}_per_query.json', per_query)
        write(a.output/f'{arm}.json', arm_summary); final_summaries[arm] = arm_summary
    comparisons = {}
    for phase in ('last', 'selected'):
        comparisons[phase] = {}
        for split in ('train', 'dev'):
            original, preserve = [load_npz(a.output/f'{arm}_{phase}_{split}.npz') for arm in ARMS]
            if not np.array_equal(original['query_indices'], preserve['query_indices']) or not np.array_equal(original['labels'], preserve['labels']):
                raise ValueError('Unpaired outputs')
            comparisons[phase][split] = e.group_report({**preserve, 'base': original['scores']}, groups[split])
    write(a.output/'preserve_vs_original.json', comparisons)
    write(a.output/'query_mapping.json', {s: plan['plan'][s]['queries'] for s in ('train', 'dev')})
    write(a.output/'summary.json', {arm: {k: v for k, v in data.items() if k != 'history'} for arm, data in final_summaries.items()})
    report = a.output/'report'; report.mkdir(exist_ok=True)
    for file in a.output.iterdir():
        if file.is_file() and ((file.suffix == '.json' and not file.name.endswith('.sha.json') and file.name != 'completed.json') or file.suffix == '.npz'):
            (report/file.name).write_bytes(file.read_bytes())
    complete(report); complete(a.output)
    print('EXPLORATORY ABLATION COMPLETE. Download:', report)


if __name__ == '__main__': main()
