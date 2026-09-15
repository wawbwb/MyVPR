"""GSV place-disjoint hard-supervision pilot; two frozen-evidence heads only.

The original cache implementation is reused with an extended code identity in
this process. Original source files and completed v1 caches remain unchanged.
"""
import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import scripts.candidate_set_screen as original
from scripts.candidate_set_screen import read, write, sha, complete, load_npz, npz
from scripts.candidate_set_pitts import validate_row
from scripts.audit_candidate_training import inspect_query, aggregate
from src.candidate_set_utils import make_plan, choose_views, stable_key, summary

LEGACY_CODES = original.codes
MODES = ('independent', 'set')
POLICY = dict(min_train_error_places=40, min_dev_error_places=10,
              min_train_error_cities=4, epochs=5, seed=42, batch_size=8,
              lr=1e-4, weight_decay=.001, residual_bound=4,
              selection='GSV full holdout R1, earliest tie, including epoch zero',
              train_sampling='all reachable errors + equal-count lowest-margin correct + equal-count ordinary correct; no replacement',
              duplicate_count=5)


def codes():
    names = ['scripts/candidate_hard_screen.py', 'scripts/candidate_set_pitts.py',
             'scripts/audit_candidate_training.py']
    return {**LEGACY_CODES(), **{n: hashlib.sha256(
        (ROOT/n).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for n in names}}


def split_plan(groups, train_queries, eval_queries, train_places):
    # First exclude the original city holdouts, then partition eligible places
    # BEFORE extraction/mining. No query, panorama or candidate crosses splits.
    pool = make_plan(groups, 1024, 512, 1024)
    excluded = {pool['split_info']['dev_city'], pool['split_info']['test_city']}
    records = {'database': [], 'queries': []}
    used_panos = set(); skipped_overlap = 0
    for label in sorted(groups, key=stable_key):
        if groups[label][0]['city'] in excluded: continue
        chosen = choose_views(groups[label])
        if chosen is None: continue
        q, refs = chosen
        keys = {(r['city'], r['panoid']) for r in [q]+refs}
        if keys & used_panos:
            skipped_overlap += 1; continue
        used_panos.update(keys)
        records['queries'].append(dict(q, label=label))
        records['database'].extend(dict(r, label=label) for r in refs)
        if len(records['queries']) == train_places: break
    if len(records['queries']) != train_places:
        raise ValueError('Insufficient panorama-disjoint places for registered pool size')
    labels = sorted({r['label'] for r in records['database']},
                    key=lambda x: stable_key('hard-holdout-42:'+x))
    if train_queries + eval_queries != train_places:
        raise ValueError('train_places must equal train_queries + eval_queries')
    holdout = set(labels[:eval_queries]); train = set(labels[eval_queries:])
    result = {}
    for split, chosen in [('train', train), ('dev', holdout)]:
        result[split] = {key: [r for r in records[key] if r['label'] in chosen]
                         for key in ('database', 'queries')}
        if len(result[split]['queries']) != len(chosen):
            raise ValueError('Every selected place must have one distinct query')
    result['test'] = {'database': [], 'queries': []}
    result['split_info'] = dict(original_city_holdouts=pool['split_info'],
        policy=POLICY, train_places=len(train), holdout_places=len(holdout),
        skipped_panorama_overlap_places=skipped_overlap,
        scope='GSV place-disjoint adaptation holdout, not unseen pretrained data or independent benchmark. Pitts not accessed.',
        budgets={'pair_queries': train_places, 'directional_pair_forwards': train_places*40,
                 'pair_arrays_approx_GiB': train_places*20*(1536+512)*4/2**30,
                 'note': 'FP32 compact CLS only; native dense features kept in bounded RAM memo, not saved.'})
    ensure_disjoint(result)
    return result


def ensure_disjoint(plan):
    tr, dv = plan['train'], plan['dev']
    for key in ('label', 'path'):
        left = {r[key] for r in tr['database']+tr['queries']}
        right = {r[key] for r in dv['database']+dv['queries']}
        if left & right: raise ValueError('Train/holdout overlap: '+key)
    panos = []
    for part in (tr, dv):
        q = {(r['city'], r['panoid']) for r in part['queries']}
        d = {(r['city'], r['panoid']) for r in part['database']}
        if q & d: raise ValueError('Query/reference panorama overlap')
        panos.append(q | d)
    if panos[0] & panos[1]: raise ValueError('Train/holdout panorama overlap')


def scan(cache, plan, split):
    c = original.verify(cache)
    if c['split'] != split or c['plan_sha256'] != sha(plan/'completed.json') or c['topk'] != 20:
        raise ValueError('Cache/plan/split mismatch')
    part = original.verify(plan)['plan'][split]
    if c['queries'] != len(part['queries']) or c['ndb'] != len(part['database']):
        raise ValueError('Cache dimensions mismatch')
    places = {}
    for i, r in enumerate(part['database']): places.setdefault(r['label'], []).append(i)
    rows = []
    for i, r in enumerate(part['queries']):
        row = load_npz(cache/'pairs'/f'{i:06d}.npz')
        validate_row(row, places[r['label']], c['ndb'])
        rows.append(dict(inspect_query(row['base'], row['labels']), query_index=i,
                         label=r['label'], city=r['city'], path=r['path']))
    return rows, c


def select_training(rows):
    errors = [r for r in rows if r['category'] == 'reachable_error']
    correct = sorted([r for r in rows if r['category'] == 'reachable_correct'],
                     key=lambda r: (r['positive_minus_negative_margin'], r['query_index']))
    n = len(errors)
    near = correct[:n]
    ordinary = sorted(correct[n:], key=lambda r: stable_key('ordinary-42:'+r['path']))[:n]
    chosen = errors+near+ordinary
    return {'errors': [r['query_index'] for r in errors],
            'near_correct': [r['query_index'] for r in near],
            'ordinary_correct': [r['query_index'] for r in ordinary],
            'selected': [r['query_index'] for r in chosen]}


def mine(a):
    if a.output.exists(): raise ValueError('Use a new mining report directory')
    plan = original.verify(a.plan); ensure_disjoint(plan['plan'])
    rows, tc = scan(a.cache_root/'train', a.plan, 'train')
    dev, dc = scan(a.cache_root/'dev', a.plan, 'dev')
    if tc['official'] != dc['official']: raise ValueError('Frozen model identity mismatch')
    selected = select_training(rows)
    tr_stats, dv_stats = aggregate(rows), aggregate(dev)
    checks = dict(train_error_places=tr_stats['error_places'] >= POLICY['min_train_error_places'],
                  dev_error_places=dv_stats['error_places'] >= POLICY['min_dev_error_places'],
                  train_error_cities=len(tr_stats['error_city_counts']) >= POLICY['min_train_error_cities'],
                  three_nonempty_equal_groups=bool(selected['errors']) and
                  len(selected['errors']) == len(selected['near_correct']) == len(selected['ordinary_correct']))
    a.output.mkdir(parents=True)
    write(a.output/'contract.json', dict(code=codes(), policy=POLICY,
        plan_sha256=sha(a.plan/'completed.json'), official=tc['official'],
        cache_hashes={s: sha(a.cache_root/s/'completed.json') for s in ('train', 'dev')}))
    write(a.output/'selection.json', selected)
    write(a.output/'per_query.json', {'train': rows, 'dev': dev})
    write(a.output/'summary.json', {'train': tr_stats, 'dev': dv_stats,
        'selected_train': aggregate([rows[i] for i in selected['selected']])})
    write(a.output/'gate.json', dict(passed=all(checks.values()), checks=checks,
        note='Practical pilot sufficiency checks, not significance. Frozen before mining; do not lower to pass. No training started.'))
    complete(a.output)
    print('Error places train / holdout:', tr_stats['error_places'], dv_stats['error_places'])
    print('Selected:', {k: len(v) for k, v in selected.items()})
    print('Gate:', checks, flush=True)
    if not all(checks.values()): raise SystemExit(3)


def batch(cache, indices):
    import torch
    rows = [load_npz(cache/'pairs'/f'{int(i):06d}.npz') for i in indices]
    return {k: torch.from_numpy(np.stack([r[k] for r in rows])).cuda()
            for k in ('evidence', 'base', 'db_vectors', 'labels')}


def evaluate(model, cache, indices):
    import torch
    outputs, labels, bases = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 16):
            d = batch(cache, indices[start:start+16])
            s = model(d['evidence'], d['base'], d['db_vectors'])
            if not torch.isfinite(s).all() or (s-d['base']).abs().max() > 4.0001:
                raise ValueError('Invalid residual output')
            outputs.append(s.cpu().numpy()); labels.append(d['labels'].cpu().numpy()); bases.append(d['base'].cpu().numpy())
    out, y, base = map(np.concatenate, (outputs, labels, bases))
    return summary(out, y, base), dict(scores=out, labels=y, base=base, query_indices=np.asarray(indices))


def save_checkpoint(path, value):
    import torch
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp); tmp.replace(path)
    write(path.with_suffix('.sha.json'), {'sha256': sha(path)})


def train(a):
    import torch
    from tqdm import tqdm
    from src.models.candidate_set import CandidateSet
    original.cuda()
    report = original.verify(a.mining)
    if not read(a.mining/'gate.json')['passed']: raise ValueError('Mining gate failed; inspect report before spending on training')
    if report['policy'] != POLICY or report['plan_sha256'] != sha(a.plan/'completed.json'):
        raise ValueError('Mining policy/plan changed')
    plan = original.verify(a.plan); ensure_disjoint(plan['plan'])
    for split in ('train', 'dev'):
        original.verify(a.cache_root/split)
        if sha(a.cache_root/split/'completed.json') != report['cache_hashes'][split]:
            raise ValueError('Cache changed after mining')
    contract = dict(code=codes(), policy=POLICY, mining_sha256=sha(a.mining/'completed.json'),
                    scope=plan['plan']['split_info'])
    if a.output.exists():
        if not a.resume or read(a.output/'contract.json') != contract:
            raise ValueError('Existing run requires --resume with identical contract')
        if (a.output/'completed.json').exists():
            original.verify(a.output); print('Already complete'); return
    else:
        a.output.mkdir(parents=True); write(a.output/'contract.json', contract)
    chosen = read(a.mining/'selection.json')['selected']
    dev_indices = list(range(len(plan['plan']['dev']['queries'])))
    for mode in MODES:
        original.seed(); model = CandidateSet(mode).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=POLICY['lr'], weight_decay=POLICY['weight_decay'])
        last = a.output/f'{mode}_last.pt'; best_path = a.output/f'{mode}_best.pt'
        start_epoch = 0; history = []
        if last.exists():
            if sha(last) != read(last.with_suffix('.sha.json'))['sha256']: raise ValueError('Damaged last checkpoint')
            state = torch.load(last, map_location='cpu', weights_only=True)
            model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
            start_epoch, history, best = state['epoch'], state['history'], state['best']
            # The atomic epoch snapshot owns the selected model too. A kill
            # between writing best.pt and last.pt must not prevent recovery.
            best_state = state['best_model']
            save_checkpoint(best_path, best_state)
            torch.set_rng_state(state['rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        else:
            baseline, values = evaluate(model, a.cache_root/'dev', dev_indices)
            if not np.array_equal(values['scores'], values['base']): raise ValueError('Zero-start differs from frozen Pair')
            best = dict(epoch=0, dev=baseline)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_checkpoint(best_path, best_state)
        for epoch in range(start_epoch, POLICY['epochs']):
            model.train(); order = np.random.default_rng(42+epoch).permutation(chosen); losses = []
            progress = tqdm(range(0, len(order), 8), desc=f'{mode} epoch {epoch+1}/5')
            for offset in progress:
                d = batch(a.cache_root/'train', order[offset:offset+8])
                x, s, z, y = [d[k] for k in ('evidence', 'base', 'db_vectors', 'labels')]
                score = model(x, s, z)
                extra = np.random.default_rng(100000*epoch+offset+42).integers(0, 20, size=5)
                ix = torch.tensor(list(range(20))+extra.tolist(), device='cuda')
                aug = model(x[:, ix], s[:, ix], z[:, ix])[:, :20]
                # Stable multi-positive likelihood; no float32 cancellation near zero.
                def loss(v):
                    pos = v.masked_fill(~y, float('-inf')).logsumexp(-1)
                    neg = v.masked_fill(y, float('-inf')).logsumexp(-1)
                    return torch.nn.functional.softplus(neg-pos).mean()
                total = .5*(loss(score)+loss(aug))
                if not torch.isfinite(total): raise ValueError('Nonfinite loss')
                opt.zero_grad(set_to_none=True); total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True); opt.step()
                losses.append(float(total.detach())); progress.set_postfix(loss=losses[-1])
            dev_result, _ = evaluate(model, a.cache_root/'dev', dev_indices)
            train_result, _ = evaluate(model, a.cache_root/'train', chosen)
            history.append(dict(epoch=epoch+1, loss_mean=float(np.mean(losses)), train=train_result, dev=dev_result))
            if dev_result['correct'] > best['dev']['correct']:
                best = dict(epoch=epoch+1, dev=dev_result)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                save_checkpoint(best_path, best_state)
            save_checkpoint(last, dict(model=model.state_dict(), optimizer=opt.state_dict(), epoch=epoch+1,
                history=history, best=best, best_model=best_state, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
            print(mode, 'epoch', epoch+1, 'train', train_result['correct'], 'dev', dev_result['correct'], flush=True)
        model.load_state_dict(torch.load(best_path, map_location='cpu', weights_only=True))
        result, values = evaluate(model, a.cache_root/'dev', dev_indices)
        if result != best['dev']: raise ValueError('Selected model does not reproduce holdout result')
        npz(a.output/f'{mode}_holdout.npz', **values)
        write(a.output/f'{mode}.json', dict(best=best, history=history))
    left, right = [load_npz(a.output/f'{m}_holdout.npz') for m in ('set', 'independent')]
    write(a.output/'comparison.json', summary(left['scores'], left['labels'], right['scores']))
    write(a.output/'query_mapping.json', plan['plan']['dev']['queries'])
    complete(a.output); print('Complete:', a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'cache', 'mine', 'train'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', type=Path, default=Path('doc/candidate_hard_plan_v1'))
    p.add_argument('--cache-root', type=Path, default=Path('.cache/candidate_hard_v1'))
    p.add_argument('--mining', type=Path, default=Path('doc/candidate_hard_mining_v1'))
    p.add_argument('--split', choices=['train', 'dev'], default='train')
    p.add_argument('--gsv-root', type=Path, default=Path('datasets/gsv_cities'))
    p.add_argument('--official-repo', type=Path, default=Path('/home/wt/workspace/Pair-VPR-official'))
    p.add_argument('--audit', type=Path, default=Path('doc/pairvpr_official_paired_audit_v1'))
    p.add_argument('--train-queries', type=int, default=8192)
    p.add_argument('--eval-queries', type=int, default=2048)
    p.add_argument('--train-places', type=int, default=10240)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    original.codes = codes
    if a.stage == 'prepare':
        original.make_plan = split_plan; original.prepare(a)
    elif a.stage == 'cache': original.cache(a)
    else: globals()[a.stage](a)


if __name__ == '__main__': main()
