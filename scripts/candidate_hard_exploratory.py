"""Explicit post-gate exploratory run; never modifies the failed mining report."""
import argparse
from pathlib import Path
import sys
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import candidate_hard_screen as h
from scripts.candidate_set_screen import read, write, sha, complete, load_npz, npz
from src.candidate_set_utils import summary


def codes():
    return {**h.codes(), 'scripts/candidate_hard_exploratory.py': hashlib.sha256(
        Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()}


def verify(path, expected):
    done = read(path/'completed.json')
    if not done['complete']: raise ValueError('Incomplete artifact: '+str(path))
    for name, digest in done['files'].items():
        target = (path/name).resolve()
        if not target.is_relative_to(path.resolve()) or sha(target) != digest:
            raise ValueError('Changed artifact: '+str(target))
    c = read(path/'contract.json')
    if c['code'] != expected: raise ValueError('Code identity mismatch: '+str(path))
    return c


def group_report(values, groups):
    """Return source query IDs, never confusing subset positions with IDs."""
    ids = np.asarray(values['query_indices'])
    if len(set(ids.tolist())) != len(ids): raise ValueError('Duplicate query IDs')
    result = {}
    for name, members in groups.items():
        if not set(members).issubset(set(ids.tolist())): raise ValueError('Unknown group query')
        mask = np.isin(ids, members); group_ids = ids[mask]
        if not mask.any():
            result[name] = {'queries': 0, 'correct': 0, 'corrected': [], 'regressed': [], 'net': 0}
            continue
        r = summary(values['scores'][mask], values['labels'][mask], values['base'][mask])
        r['corrected'] = group_ids[r['corrected']].tolist()
        r['regressed'] = group_ids[r['regressed']].tolist()
        r['query_indices'] = group_ids.tolist()
        r['net'] = len(r['corrected'])-len(r['regressed'])
        result[name] = r
    return result


def groups_for(rows, selection=None):
    groups = {'all': [r['query_index'] for r in rows]}
    for name in ('reachable_error', 'reachable_correct', 'unreachable'):
        groups[name] = [r['query_index'] for r in rows if r['category'] == name]
    groups['range_fixable_errors'] = [r['query_index'] for r in rows if r['residual_range_only_fixable']]
    groups['range_blocked_errors'] = [r['query_index'] for r in rows if r['residual_range_blocked']]
    groups['correct_margin_below_8'] = [r['query_index'] for r in rows
        if r['category'] == 'reachable_correct' and r['positive_minus_negative_margin'] < 8]
    if selection is not None:
        groups.update({k: selection[k] for k in ('errors', 'near_correct', 'ordinary_correct')})
    return groups


def validate_inputs(a):
    print('Verifying immutable mining/plan/cache hashes; no image extraction...', flush=True)
    expected = h.codes()
    report = verify(a.mining, expected); plan = verify(a.plan, expected)
    gate = read(a.mining/'gate.json')
    if report['policy'] != h.POLICY or report['plan_sha256'] != sha(a.plan/'completed.json'):
        raise ValueError('Wrong mining policy or plan')
    # This is a narrowly scoped exception for a reviewed count-only failure,
    # not a generic --ignore-checks flag.
    wanted = dict(train_error_places=False, dev_error_places=True,
                  train_error_cities=True, three_nonempty_equal_groups=True)
    if gate['passed'] or gate['checks'] != wanted:
        raise ValueError('Expected reviewed count-only gate failure')
    h.ensure_disjoint(plan['plan'])
    rows = read(a.mining/'per_query.json'); selection = read(a.mining/'selection.json')
    if selection != h.select_training(rows['train']): raise ValueError('Training selection changed')
    counts = (len(selection['errors']), len(selection['near_correct']), len(selection['ordinary_correct']))
    if counts != (35, 35, 35): raise ValueError('This exploratory protocol is registered for the reviewed 105-query sample')
    for split in ('train', 'dev'):
        c = verify(a.cache_root/split, expected)
        if (sha(a.cache_root/split/'completed.json') != report['cache_hashes'][split]
            or c['official'] != report['official'] or c['split'] != split
            or c['plan_sha256'] != report['plan_sha256'] or c['topk'] != 20):
            raise ValueError('Cache identity mismatch')
        part = plan['plan'][split]
        if len(rows[split]) != len(part['queries']) or c['queries'] != len(rows[split]) or c['ndb'] != len(part['database']):
            raise ValueError('Query/database count mismatch')
        positives = {}
        for j, r in enumerate(part['database']): positives.setdefault(r['label'], []).append(j)
        for i, (record, logged) in enumerate(zip(part['queries'], rows[split])):
            row = load_npz(a.cache_root/split/'pairs'/f'{i:06d}.npz')
            h.validate_row(row, positives[record['label']], c['ndb'])
            reconstructed = dict(h.inspect_query(row['base'], row['labels']), query_index=i,
                                 label=record['label'], city=record['city'], path=record['path'])
            if reconstructed != logged: raise ValueError('Mining rows do not reproduce from cache')
    return plan, rows, selection, gate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--acknowledge-failed-gate', action='store_true', required=True)
    p.add_argument('--plan', type=Path, default=Path('doc/candidate_hard_plan_v1'))
    p.add_argument('--mining', type=Path, default=Path('doc/candidate_hard_mining_v1'))
    p.add_argument('--cache-root', type=Path, default=Path('.cache/candidate_hard_v1'))
    p.add_argument('--output', type=Path, default=Path('logs/candidate_hard_exploratory_v1'))
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    import torch
    from tqdm import tqdm
    from src.models.candidate_set import CandidateSet
    h.original.cuda()
    plan, rows, selected, gate = validate_inputs(a)
    contract = dict(code=codes(), policy=h.POLICY, exploratory=True, original_gate=gate,
        mining_sha256=sha(a.mining/'completed.json'),
        reason='Post-audit user-authorized 35-error pilot; original >=40 gate remains failed. No confirmatory claim.',
        scope='GSV adaptation holdout used for epoch selection, not an independent test; no Pitts access.')
    if a.output.exists():
        if not a.resume or read(a.output/'contract.json') != contract: raise ValueError('Use a fresh output or matching --resume')
        if (a.output/'completed.json').exists():
            verify(a.output, codes()); print('Already complete'); return
    else:
        a.output.mkdir(parents=True); write(a.output/'contract.json', contract)
    write(a.output/'original_gate.json', gate)
    chosen = selected['selected']; chosen_set = set(chosen)
    groups = dict(train=groups_for([r for r in rows['train'] if r['query_index'] in chosen_set], selected),
                  dev=groups_for(rows['dev']))
    write(a.output/'groups.json', groups)
    dev_ids = list(range(len(rows['dev'])))
    for mode in h.MODES:
        h.original.seed(); model = CandidateSet(mode).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.001)
        last = a.output/f'{mode}_last.pt'; best_path = a.output/f'{mode}_best.pt'
        history = []; start_epoch = 0
        if last.exists():
            if sha(last) != read(last.with_suffix('.sha.json'))['sha256']: raise ValueError('Damaged epoch snapshot')
            state = torch.load(last, map_location='cpu', weights_only=True)
            if state['contract'] != contract: raise ValueError('Checkpoint protocol mismatch')
            model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
            start_epoch, history, best = state['epoch'], state['history'], state['best']
            best_state = state['best_model']; h.save_checkpoint(best_path, best_state)
            torch.set_rng_state(state['rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        else:
            dv, values = h.evaluate(model, a.cache_root/'dev', dev_ids)
            tr, training = h.evaluate(model, a.cache_root/'train', chosen)
            if not np.array_equal(values['scores'], values['base']) or not np.array_equal(training['scores'], training['base']):
                raise ValueError('Zero-start mismatch')
            best = dict(epoch=0, dev=dv)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            h.save_checkpoint(best_path, best_state)
            history.append(dict(epoch=0, train=group_report(training, groups['train']), dev=group_report(values, groups['dev'])))
        for epoch in range(start_epoch, 5):
            model.train(); order = np.random.default_rng(42+epoch).permutation(chosen); losses = []
            progress = tqdm(range(0, len(order), 8), desc=f'EXPLORATORY {mode} {epoch+1}/5')
            for offset in progress:
                d = h.batch(a.cache_root/'train', order[offset:offset+8])
                x, s, z, y = [d[k] for k in ('evidence', 'base', 'db_vectors', 'labels')]
                scores = model(x, s, z)
                extra = np.random.default_rng(100000*epoch+offset+42).integers(0, 20, size=5)
                ix = torch.tensor(list(range(20))+extra.tolist(), device='cuda')
                augmented = model(x[:, ix], s[:, ix], z[:, ix])[:, :20]
                def loss(v):
                    pos = v.masked_fill(~y, float('-inf')).logsumexp(-1)
                    neg = v.masked_fill(y, float('-inf')).logsumexp(-1)
                    return torch.nn.functional.softplus(neg-pos).mean()
                total = .5*(loss(scores)+loss(augmented))
                if not torch.isfinite(total): raise ValueError('Nonfinite loss')
                opt.zero_grad(set_to_none=True); total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True); opt.step()
                losses.append(float(total.detach())); progress.set_postfix(loss=losses[-1])
            dv, values = h.evaluate(model, a.cache_root/'dev', dev_ids)
            tr, training = h.evaluate(model, a.cache_root/'train', chosen)
            entry = dict(epoch=epoch+1, loss_mean=float(np.mean(losses)),
                train=group_report(training, groups['train']), dev=group_report(values, groups['dev']))
            history.append(entry)
            if dv['correct'] > best['dev']['correct']:
                best = dict(epoch=epoch+1, dev=dv)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                h.save_checkpoint(best_path, best_state)
            h.save_checkpoint(last, dict(contract=contract, model=model.state_dict(), optimizer=opt.state_dict(),
                epoch=epoch+1, history=history, best=best, best_model=best_state,
                rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
            write(a.output/f'{mode}_history.json', history)
            print(mode, 'train range-fixable corrected:', len(entry['train']['range_fixable_errors']['corrected']),
                  'train near-correct regressions:', len(entry['train']['near_correct']['regressed']),
                  'holdout corrections/regressions:', len(dv['corrected']), len(dv['regressed']), flush=True)
        model.load_state_dict(best_state)
        dv, values = h.evaluate(model, a.cache_root/'dev', dev_ids)
        _, training = h.evaluate(model, a.cache_root/'train', chosen)
        if dv != best['dev']: raise ValueError('Best checkpoint does not reproduce')
        npz(a.output/f'{mode}_holdout.npz', **values)
        npz(a.output/f'{mode}_train.npz', **training)
        write(a.output/f'{mode}.json', dict(best=best, history=history,
            selected_train=group_report(training, groups['train']), selected_dev=group_report(values, groups['dev'])))
    comparison = {}
    for split, suffix in [('dev', 'holdout'), ('train', 'train')]:
        left, right = [load_npz(a.output/f'{m}_{suffix}.npz') for m in ('set', 'independent')]
        if not np.array_equal(left['query_indices'], right['query_indices']) or not np.array_equal(left['labels'], right['labels']):
            raise ValueError('Unpaired comparisons')
        comparison[split] = group_report({**left, 'base': right['scores']}, groups[split])
    write(a.output/'set_vs_independent.json', comparison)
    write(a.output/'query_mapping.json', {s: plan['plan'][s]['queries'] for s in ('train', 'dev')})
    # Lightweight report has its own complete manifest; no checkpoints required
    # when downloading this report directory to the local machine.
    report_dir = a.output/'report'; report_dir.mkdir(exist_ok=True)
    for file in a.output.iterdir():
        if file.is_file() and (file.suffix == '.json' and not file.name.endswith('.sha.json') and file.name != 'completed.json' or file.suffix == '.npz'):
            (report_dir/file.name).write_bytes(file.read_bytes())
    complete(report_dir); complete(a.output)
    print('EXPLORATORY COMPLETE. Download:', report_dir)


if __name__ == '__main__': main()
