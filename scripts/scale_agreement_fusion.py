"""Frozen two-scale descriptor cache and finite128-parameter fusion study."""
import argparse
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from scripts import single_dino_interaction as old
from scripts import relation_metric as rm
from scripts.cross_resolution_detail import ScaleImages, ScaleSteps
from scripts.spatial_resample import mined_loss
from scripts.single_dino_supervision import check_gpu_capacity
from src.models.scale_agreement_fusion import ARMS, ScaleAgreementFusion, mix, image_seeds
from src.semantic_region_protocol import (read, write, sha, state_hash, owned_run, seed_all,
    save_npz, save_torch, rng_state, restore_rng)

SCHEMA = 'scale_agreement_fusion_v1'
NEW_FILES = ('src/models/scale_agreement_fusion.py', 'scripts/scale_agreement_fusion.py',
    'scripts/launch_scale_agreement_fusion.py', 'config/scale_agreement_fusion.json',
    'tests/test_scale_agreement_fusion.py', 'doc/SCALE_AGREEMENT_FUSION_V1.md')
SOURCES = (*NEW_FILES, 'scripts/relation_metric.py', 'scripts/cross_resolution_detail.py',
    'scripts/spatial_resample.py', 'scripts/eval_clip_key_study.py', 'scripts/semantic_region_study.py',
    'src/models/complementary_covariance.py', 'src/models/official_boq.py')
LOW_CACHE = ROOT/'.cache/relation_metric_v1/train'
ANCHOR_STATE = '85efdb822f8a805e5ba69dca0a27521c09954ce9cdee18e1c8cbc30d8212eee6'
TAGS = ('gsv-adaptation-holdout', 'msls-val', 'pitts30k-val')
VARIANTS = ('innovation1', 'innovation1_518', 'descriptor_average', *ARMS, 'train_mean_gate')


def configuration():
    extra = read(ROOT/'config/scale_agreement_fusion.json')
    if extra != dict(schema=SCHEMA, runs='logs/scale_agreement_fusion_v1',
            fusion_cache='.cache/scale_agreement_fusion_v1', learning_rate=.001,
            pilot_steps=1024, smoke_steps=8, keep_weight=0.):
        raise ValueError('Registered fusion configuration changed')
    return dict(old.configuration(), **extra)


def identity(c):
    v = old.identity(c)
    v.update(schema=SCHEMA, new_sources=old.key.source_hashes(SOURCES),
        low_cache_seal_sha256=sha(LOW_CACHE/'completed.json'),
        low_cache_contract_sha256=sha(LOW_CACHE/'contract.json'),
        baseline_reports={t: sha(ROOT/'logs/cross_resolution_detail_v1/eval'/(t+'.json')) for t in TAGS},
        mechanism='Complete two-scale descriptors, zero-at-mean group agreement mixing',
        scope='Fixed1024step single-seed exposed development study; not novelty or stable-gain proof')
    return v


def finish(out):
    files = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*'))
        if p.is_file() and p.name not in ('.lock', 'completed.json') and not p.name.endswith('.tmp')}
    write(out/'completed.json', dict(complete=True, schema=SCHEMA, files=files))


def verified(out):
    seal = read(out/'completed.json')
    if not seal.get('complete') or seal.get('schema') != SCHEMA:
        raise ValueError('Incomplete fusion artifact')
    for name, digest in seal['files'].items():
        p = (out/name).resolve()
        if not p.is_relative_to(out.resolve()) or sha(p) != digest:
            raise ValueError('Changed artifact: '+name)
    return read(out/'contract.json')


def rows_for(plan, c):
    rows = [r for p in plan['places'][:c['train_places']] for r in p['rows']]
    if len(rows) != 32768 or len(set(rows)) != len(rows) or any(plan['images'][r]['split'] != 'train' for r in rows):
        raise ValueError('Wrong training population')
    return rows


def low_bank(plan, c):
    contract = rm.verified(LOW_CACHE)
    dep = contract['dependency']
    if (dep['plan_sha256'] != sha(old.key.path(c, 'plan')/'plan.json') or
            dep['anchor_checkpoint_sha256'] != sha(old.key.path(c, 'anchor_run')/'last.pt') or
            read(LOW_CACHE/'extraction_checks.json')['anchor_sha256'] != ANCHOR_STATE):
        raise ValueError('Low-cache source differs from frozen innovation one')
    rows = rows_for(plan, c)
    bank = np.empty((len(rows), 12288), np.float32)
    for i in range(len(rows)//64):
        with np.load(LOW_CACHE/f'{i:04d}.npz', allow_pickle=False) as d:
            if d['row_ids'].tolist() != rows[i*64:(i+1)*64]:
                raise ValueError('Low-cache row order differs')
            rm.validate_vectors(d['descriptors'], 64)
            bank[i*64:(i+1)*64] = d['descriptors']
    return bank


def models(c):
    result = {}
    for a in ARMS:
        seed_all(c['seed'])
        result[a] = ScaleAgreementFusion(a).to(c['device'])
    if len({state_hash(m) for m in result.values()}) != 1:
        raise ValueError('Fusion initial states differ')
    return result


def frozen_anchor(c):
    anchor = old.make_anchor(c).requires_grad_(False).eval()
    if state_hash(anchor) != ANCHOR_STATE:
        raise ValueError('Wrong frozen innovation-one state')
    return anchor


def descriptors(anchor, pixels, c):
    return torch.cat([rm.encode(anchor, x.to(c['device'])) for x in pixels.split(2)])


def shard_ready(out, i):
    target, ready = out/f'{i:04d}.npz', out/f'{i:04d}.ready.json'
    if not ready.exists():
        return False
    if sha(target) != read(ready)['sha256']:
        raise ValueError('Cache shard changed')
    return True


def shard_write(out, i, **data):
    save_npz(out/f'{i:04d}.npz', **data)
    write(out/f'{i:04d}.ready.json', dict(sha256=sha(out/f'{i:04d}.npz')))


def cache_train(c, args):
    from scripts.semantic_region_study import ImageRows
    plan = old.key.open_plan(c)
    rows = rows_for(plan, c)
    out = old.key.path(c, 'fusion_cache')/'train_high'
    contract = dict(stage='cache_train', **identity(c))
    with owned_run(out, contract, args.resume):
        if (out/'completed.json').exists():
            verified(out)
            return
        anchor = frozen_anchor(c)
        todo = [i for i in range(len(rows)//64) if not shard_ready(out, i)]
        ds = ImageRows(old.key.path(c, 'dataset_root'), plan['images'], 518)
        loader = DataLoader(Subset(ds, [r for i in todo for r in rows[i*64:(i+1)*64]]),
            batch_size=64, num_workers=c['workers'], shuffle=False)
        with torch.inference_mode():
            for i, (images, ids) in zip(todo, loader):
                if ids.tolist() != rows[i*64:(i+1)*64]:
                    raise ValueError('High-cache row order changed')
                high = descriptors(anchor, images, c).cpu().numpy()
                rm.validate_vectors(high, len(images))
                shard_write(out, i, high=high, indices=ids.numpy())
                write(out/'progress.json', dict(shards=i+1, total=len(rows)//64, images=(i+1)*64))
                print('HIGH train', i+1, '/', len(rows)//64, flush=True)
        if state_hash(anchor) != ANCHOR_STATE:
            raise ValueError('Frozen encoder changed')
        write(out/'summary.json', dict(images=len(rows), descriptor_dimensions=12288,
            anchor_unchanged=True, train_only=True))
        finish(out)


def high_bank(plan, c):
    out = old.key.path(c, 'fusion_cache')/'train_high'
    if verified(out) != dict(stage='cache_train', **identity(c)):
        raise ValueError('High cache identity differs')
    rows = rows_for(plan, c)
    high = np.empty((len(rows), 12288), np.float32)
    for i in range(len(rows)//64):
        with np.load(out/f'{i:04d}.npz', allow_pickle=False) as d:
            if d['indices'].tolist() != rows[i*64:(i+1)*64]:
                raise ValueError('Training high row mismatch')
            rm.validate_vectors(d['high'], 64)
            high[i*64:(i+1)*64] = d['high']
    return high


def train(c, args):
    from src.losses.vpr_losses import VPRLossFunction
    plan = old.key.open_plan(c)
    runs = old.key.path(c, 'runs')
    stage = args.command
    contract = dict(stage=stage, **identity(c))
    if stage == 'train':
        smoke = verified(runs/'smoke')
        if ({k: v for k, v in smoke.items() if k != 'stage'} !=
                {k: v for k, v in contract.items() if k != 'stage'} or
                not all(read(runs/'smoke/summary.json')['checks'].values())):
            raise ValueError('Same-version smoke required')
    out = runs/stage
    with owned_run(out, contract, args.resume):
        if (out/'completed.json').exists():
            verified(out)
            return
        low = low_bank(plan, c)
        rows = rows_for(plan, c)
        mapping = {r: i for i, r in enumerate(rows)}
        high = high_bank(plan, c) if stage == 'train' else None
        anchor = frozen_anchor(c) if stage == 'smoke' else None
        heads = models(c)
        initial = {a: state_hash(m) for a, m in heads.items()}
        opts = {a: torch.optim.AdamW(m.parameters(), lr=c['learning_rate'], weight_decay=0.) for a, m in heads.items()}
        write(out/'initial.json', dict(heads=initial, parameters={a: sum(p.numel() for p in m.parameters()) for a, m in heads.items()}))
        state = dict(step=0, history=[], zero_errors=None, low_sentinel_error=None,
            contract_sha256=sha(out/'contract.json'), gradients={a: {n: False for n, p in m.named_parameters()} for a, m in heads.items()})

        def save():
            save_torch(out/'last.pt', dict(heads={a: m.state_dict() for a, m in heads.items()},
                optimizers={a: o.state_dict() for a, o in opts.items()}, state=state, rng=rng_state()))

        if (out/'last.pt').exists():
            saved = torch.load(out/'last.pt', map_location='cpu', weights_only=True)
            if saved['state']['contract_sha256'] != state['contract_sha256']:
                raise ValueError('Resume contract changed')
            for a, m in heads.items():
                m.load_state_dict(saved['heads'][a], strict=True)
                opts[a].load_state_dict(saved['optimizers'][a])
            state = saved['state']
            restore_rng(saved['rng'])
        else:
            seed_all(c['seed'])
            save()
        total = c['smoke_steps'] if stage == 'smoke' else c['pilot_steps']
        if not 0 <= state['step'] <= total:
            raise ValueError('Bad cursor')
        lossfn = VPRLossFunction()
        for step in range(state['step'], total):
            started = time.monotonic()
            places = plan['schedule'][step]
            ids = [r for p in places for r in plan['places'][p]['rows']]
            positions = [mapping[r] for r in ids]
            l = torch.tensor(low[positions], device=c['device'])
            if stage == 'smoke':
                lo, hi, _, _, row_ids = ScaleSteps(c, plan, [places])[0]
                if row_ids.tolist() != ids:
                    raise ValueError('Smoke rows differ')
                with torch.no_grad():
                    direct = descriptors(anchor, lo, c)
                    h = descriptors(anchor, hi, c)
                    error = float((direct-l).abs().max())
                if error > 1e-6:
                    raise ValueError('Reused low descriptors differ')
                state['low_sentinel_error'] = max(state['low_sentinel_error'] or 0., error)
            else:
                h = torch.tensor(high[positions], device=c['device'])
            labels = torch.tensor(places, device=c['device']).repeat_interleave(4)
            coords = torch.tensor([plan['images'][r]['coordinates'] for r in ids], dtype=torch.float64)
            seeds = image_seeds([plan['images'][r]['image_sha256'] for r in ids])
            mean = F.normalize(l+h, dim=1)
            if state['zero_errors'] is None:
                with torch.no_grad():
                    state['zero_errors'] = {a: float((m(l, h, seeds)[0]-mean).abs().max()) for a, m in heads.items()}
            record = {}
            for a, m in heads.items():
                opts[a].zero_grad(set_to_none=True)
                z, g = m(l, h, seeds)
                loss, mining = mined_loss(lossfn, z, labels, coords)
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite fusion loss')
                loss.backward()
                for name, p in m.named_parameters():
                    state['gradients'][a][name] |= p.grad is not None and bool((p.grad != 0).any())
                norm = torch.nn.utils.clip_grad_norm_(m.parameters(), c['gradient_clip'], error_if_nonfinite=True)
                opts[a].step()
                record[a] = dict(main_loss=float(loss.detach()), active=float(loss.detach()) > 0,
                    gradient_norm=float(norm), gate_mean=float(g.detach().mean()), gate_max=float(g.detach().abs().max()),
                    within_image_gate_std=float(g.detach().std(1, unbiased=False).mean()),
                    between_image_gate_std=float(g.detach().std(0, unbiased=False).mean()),
                    descriptor_l2_from_average=float((z.detach()-mean).norm(dim=1).mean()), mining=mining)
            state['step'] = step+1
            record.update(step=step+1, seconds=time.monotonic()-started, peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
            state['history'].append(record)
            write(out/'progress.json', dict(total=total, **record))
            if state['step'] % c['checkpoint_every'] == 0:
                save()
        save()
        if stage == 'train':
            sums, squares, within = np.zeros(32), np.zeros(32), 0.
            with torch.no_grad():
                for i in range(0, len(low), 64):
                    g = heads['group_gate'].gates(torch.tensor(low[i:i+64], device=c['device']),
                        torch.tensor(high[i:i+64], device=c['device'])).cpu().numpy().astype(np.float64)
                    sums += g.sum(0)
                    squares += (g*g).sum(0)
                    within += g.std(1).sum()
            mean_gate = (sums/len(low)).astype(np.float32)
            save_npz(out/'train_mean_gate.npz', mean=mean_gate,
                between_image_std=np.sqrt(np.maximum(squares/len(low)-(sums/len(low))**2, 0)))
            write(out/'calibration.json', dict(images=len(low), train_only=True,
                mean_gate=mean_gate.tolist(), within_image_std=within/len(low),
                source_head_sha256=state_hash(heads['group_gate'])))
        checks = dict(complete_steps=state['step'] == total, zero_start=max(state['zero_errors'].values()) <= 1e-7,
            all_heads_updated=all(state_hash(m) != initial[a] for a, m in heads.items()),
            all_parameters_gradient_seen=all(all(v.values()) for v in state['gradients'].values()),
            anchor_unchanged=anchor is None or state_hash(anchor) == ANCHOR_STATE,
            protected_sources_unchanged=old.key.source_hashes(old.key.PROTECTED) == contract['protected_sources'])
        write(out/'summary.json', dict(checks=checks, steps=state['step'], zero_errors=state['zero_errors'],
            low_sentinel_error=state['low_sentinel_error'], final_head_sha256={a: state_hash(m) for a, m in heads.items()},
            parameters={a: sum(p.numel() for p in m.parameters()) for a, m in heads.items()},
            effective_steps={a: sum(r[a]['active'] for r in state['history']) for a in ARMS},
            encoder_loaded=anchor is not None, verdict='SMOKE_DISCARD' if stage == 'smoke' else 'TRAINED_NOT_EVALUATED'))
        if not all(checks.values()):
            raise ValueError('Mechanical checks failed: '+str(checks))
        finish(out)


def populations(c):
    from scripts.eval_clip_key_study import populations as get
    return get(c, old.key.open_plan(c))


def population_contract(d):
    from scripts import eval_clip_key_study as metrics
    return dict(references=d['references'], queries=d['queries'], order=metrics.order_digest(d['names']), gt=metrics.gt_digest(d['ground_truth']))


def cache_eval(c, args):
    anchor = None
    for d in populations(c):
        out = old.key.path(c, 'fusion_cache')/('eval_'+d['tag'])
        contract = dict(stage='cache_eval', **identity(c), dataset=population_contract(d))
        with owned_run(out, contract, args.resume):
            if (out/'completed.json').exists():
                verified(out)
                continue
            if anchor is None:
                anchor = frozen_anchor(c)
            n = len(d['names'])
            todo = [i for i in range((n+63)//64) if not shard_ready(out, i)]
            ds = ScaleImages(d['root'], d['names'], d['hashes'])
            loader = DataLoader(Subset(ds, [j for i in todo for j in range(i*64, min(n, (i+1)*64))]),
                batch_size=64, num_workers=c['workers'], shuffle=False)
            with torch.inference_mode():
                for i, (lo, hi, ids, hashes, _) in zip(todo, loader):
                    if ids.tolist() != list(range(i*64, min(n, (i+1)*64))):
                        raise ValueError('Evaluation cache order differs')
                    l, h = descriptors(anchor, lo, c).cpu().numpy(), descriptors(anchor, hi, c).cpu().numpy()
                    rm.validate_vectors(l, len(lo))
                    rm.validate_vectors(h, len(hi))
                    shard_write(out, i, low=l, high=h, indices=ids.numpy(), hashes=np.asarray(hashes))
                    write(out/'progress.json', dict(images=int(ids[-1])+1, total=n))
                    print('EVAL cache', d['tag'], int(ids[-1])+1, '/', n, flush=True)
            if state_hash(anchor) != ANCHOR_STATE:
                raise ValueError('Frozen encoder changed')
            write(out/'summary.json', dict(images=n, anchor_unchanged=True))
            finish(out)


def evaluate(c, args):
    from scripts import eval_clip_key_study as metrics
    run = old.key.path(c, 'runs')/'train'
    if verified(run) != dict(stage='train', **identity(c)) or not all(read(run/'summary.json')['checks'].values()):
        raise ValueError('Training incomplete')
    sets = populations(c)
    contract = dict(stage='eval', **identity(c), checkpoint_sha256=sha(run/'last.pt'),
        cache_seals={d['tag']: sha(old.key.path(c, 'fusion_cache')/('eval_'+d['tag'])/'completed.json') for d in sets})
    out = old.key.path(c, 'runs')/'eval'
    with owned_run(out, contract, args.resume):
        if (out/'completed.json').exists():
            verified(out)
            return
        heads = models(c)
        saved = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
        if saved['state']['step'] != 1024 or saved['state']['contract_sha256'] != sha(run/'contract.json'):
            raise ValueError('Not fixed-final checkpoint')
        for a, m in heads.items():
            m.load_state_dict(saved['heads'][a], strict=True)
            if state_hash(m) != read(run/'summary.json')['final_head_sha256'][a]:
                raise ValueError('Head changed')
            m.requires_grad_(False).eval()
        with np.load(run/'train_mean_gate.npz') as d:
            mean_gate = torch.tensor(d['mean'], device=c['device'])[None]
        results = {}
        for d in sets:
            target, ready = out/(d['tag']+'.json'), out/(d['tag']+'.ready.json')
            if ready.exists():
                if sha(target) != read(ready)['sha256']:
                    raise ValueError('Report changed')
                results[d['tag']] = read(target)['comparison']
                continue
            cache = old.key.path(c, 'fusion_cache')/('eval_'+d['tag'])
            if verified(cache) != dict(stage='cache_eval', **identity(c), dataset=population_contract(d)):
                raise ValueError('Evaluation cache differs')
            n = len(d['names'])
            low, high = np.empty((n, 12288), np.float32), np.empty((n, 12288), np.float32)
            hashes = []
            for i in range((n+63)//64):
                with np.load(cache/f'{i:04d}.npz', allow_pickle=False) as shard:
                    if shard['indices'].tolist() != list(range(i*64, min(n, (i+1)*64))):
                        raise ValueError('Cache rows differ')
                    low[i*64:(i+1)*64], high[i*64:(i+1)*64] = shard['low'], shard['high']
                    hashes.extend(shard['hashes'].tolist())
            reports, diagnostics = {}, {}
            with torch.inference_mode():
                for a in VARIANTS:
                    values = low if a == 'innovation1' else high if a == 'innovation1_518' else np.empty_like(low)
                    stats = dict(images=0, gate_sum=0., max_gate=0., within_std_sum=0., l2_from_mean_sum=0.)
                    if a not in ('innovation1', 'innovation1_518'):
                        for i in range(0, n, 64):
                            l, h = torch.tensor(low[i:i+64], device=c['device']), torch.tensor(high[i:i+64], device=c['device'])
                            reference = F.normalize(l+h, dim=1)
                            if a == 'descriptor_average':
                                z, g = reference, torch.zeros((len(l), 32), device=c['device'])
                            elif a == 'train_mean_gate':
                                g = mean_gate.expand(len(l), -1)
                                z = mix(l, h, g)
                            else:
                                z, g = heads[a](l, h, image_seeds(hashes[i:i+64]))
                            rm.validate_vectors(z.cpu().numpy(), len(l))
                            values[i:i+64] = z.cpu().numpy()
                            stats['images'] += len(l)
                            stats['gate_sum'] += float(g.sum())
                            stats['max_gate'] = max(stats['max_gate'], float(g.abs().max()))
                            stats['within_std_sum'] += float(g.std(1, unbiased=False).sum())
                            stats['l2_from_mean_sum'] += float((z-reference).norm(dim=1).sum())
                    db = torch.tensor(values[:d['references']], device=c['device'])
                    ranks = np.concatenate([(torch.tensor(q, device=c['device'])@db.T).topk(20, dim=1).indices.cpu().numpy()
                        for q in np.array_split(values[d['references']:], max(1, (d['queries']+15)//16))])
                    reports[a] = metrics.retrieval_report(ranks, d['ground_truth'], d['references'])
                    diagnostics[a] = stats
                    del db, values
            prior_path = ROOT/'logs/cross_resolution_detail_v1/eval'/(d['tag']+'.json')
            if sha(prior_path) != contract['baseline_reports'][d['tag']]:
                raise ValueError('Prior baseline report changed')
            prior = read(prior_path)['variants']
            for a in ('innovation1', 'innovation1_518', 'descriptor_average'):
                if reports[a] != prior[a]:
                    raise ValueError('Full previous baseline Top20 not reproduced: '+a)
            comparison = dict(correct={a: r['correct'] for a, r in reports.items()},
                vs_average={a: metrics.paired_report(reports['descriptor_average'], r) for a, r in reports.items() if a != 'descriptor_average'},
                candidate_vs_others={a: metrics.paired_report(r, reports['group_gate']) for a, r in reports.items() if a != 'group_gate'})
            write(target, dict(dataset=population_contract(d), variants=reports, comparison=comparison, fusion_diagnostics=diagnostics,
                image_content_digest=metrics.order_digest([p+':'+h for p, h in zip(d['names'], hashes)])))
            write(ready, dict(sha256=sha(target)))
            results[d['tag']] = comparison
            print(d['tag'], comparison['correct'], flush=True)
            del low, high
        checks = dict(checkpoint_unchanged=sha(run/'last.pt') == contract['checkpoint_sha256'],
            sources_unchanged=old.key.source_hashes(old.key.PROTECTED) == contract['protected_sources'])
        signal = (all(v['net'] > 0 for v in results['pitts30k-val']['candidate_vs_others'].values()) and
            all(v['net'] >= 0 for v in results['msls-val']['candidate_vs_others'].values()) and
            results['gsv-adaptation-holdout']['vs_average']['group_gate']['net'] >= 0)
        write(out/'summary.json', dict(checks=checks, results=results, stable_gain_proven=False, blind_test_used=False,
            verdict='EXPLORATORY_SCALE_AGREEMENT_SIGNAL' if signal else 'NO_MATCHED_SCALE_AGREEMENT_SIGNAL'))
        if not all(checks.values()):
            raise ValueError('Integrity check failed')
        finish(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('plan', 'smoke', 'cache_train', 'train', 'cache_eval', 'eval'))
    p.add_argument('--allow-training', action='store_true')
    p.add_argument('--allow-shared-gpu', action='store_true')
    p.add_argument('--resume', action='store_true')
    args, c = p.parse_args(), configuration()
    if args.command in ('smoke', 'train') and not args.allow_training:
        p.error('Explicit training consent required')
    if args.command == 'plan':
        plan = old.key.open_plan(c)
        low_bank(plan, c)
        print(dict(identity=identity(c), training_started=False))
        return
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        p.error('Physical GPU1 only')
    import fcntl
    import subprocess
    with (ROOT/'logs/query_relation_screen_v1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        text = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=memory.total,memory.used,memory.free',
            '--format=csv,noheader,nounits'], text=True, timeout=20)
        check_gpu_capacity(*map(int, text.strip().split(',')), args.allow_shared_gpu, c['cuda_memory_fraction'])
        old.key.gpu(c)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if args.command in ('smoke', 'train'):
            train(c, args)
        else:
            {'cache_train': cache_train, 'cache_eval': cache_eval, 'eval': evaluate}[args.command](c, args)


if __name__ == '__main__':
    main()
