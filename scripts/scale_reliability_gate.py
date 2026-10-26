"""Finite cross-view reliability gate study on immutable cached descriptors."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import os
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from scripts import scale_agreement_repeat as prior
from scripts import scale_agreement_fusion as source
from scripts.summarize_scale_agreement_repeat import json_ready
from src.models.scale_reliability_gate import ARMS, ScaleReliabilityGate, cross_view_teacher
from src.models.scale_agreement_fusion import mix, image_seeds
from src.models.complementary_covariance import permute_positions
from src.scale_repeat_statistics import contrast, positive_components
from src.semantic_region_protocol import (read, write, sha, owned_run, state_hash, seed_all,
    save_torch, save_npz, rng_state, restore_rng)

SCHEMA = 'scale_reliability_gate_v1'
FILES = ('scripts/scale_reliability_gate.py', 'scripts/launch_scale_reliability_gate.py',
    'src/models/scale_reliability_gate.py', 'config/scale_reliability_gate.json',
    'tests/test_scale_reliability_gate.py', 'doc/SCALE_RELIABILITY_GATE_V1.md')
EXTRA = ('frozen_scalar', *ARMS, 'same_head_shuffled', 'same_head_train_mean')
VARIANTS = (*source.VARIANTS, *EXTRA)
PRIMARY = ('frozen_scalar', 'quality_global', 'quality_static', 'quality_shuffled', 'same_head_shuffled', 'same_head_train_mean')


def configuration():
    c = read(ROOT/'config/scale_reliability_gate.json')
    expected = dict(schema=SCHEMA, runs='logs/scale_reliability_gate_v1', seeds=[43, 44, 45],
        pilot_steps=1024, smoke_steps=8, learning_rate=.001, quality_weight=.1,
        teacher_temperature=.1, bootstrap_repetitions=2000, bootstrap_seed=20261026)
    if c != expected:
        raise ValueError('Fixed reliability policy changed')
    return c


def root(c):
    return ROOT/c['runs']


def base_value(seed):
    with np.load(prior.root(prior.configuration())/f'seed{seed}/train/train_mean_gate.npz') as d:
        return float(d['mean'].astype(np.float64).mean())


def identity(c, stage, seed=None):
    result = dict(source.identity(source.configuration()), schema=SCHEMA, stage=stage,
        reliability_policy=c, reliability_sources=source.old.key.source_hashes((*FILES,
            'scripts/scale_agreement_repeat.py', 'scripts/summarize_scale_agreement_repeat.py', 'src/scale_repeat_statistics.py')),
        historical_repeat_seals={str(s): {t: sha(prior.root(prior.configuration())/f'seed{s}'/t/'completed.json') for t in ('smoke', 'train', 'eval')} for s in c['seeds']},
        shared_cache_seals={t: sha(ROOT/'.cache/scale_agreement_fusion_v1'/t/'completed.json') for t in ('train_high', *('eval_'+t for t in source.TAGS))},
        scope='Train-only leave-self-out group reliability, image-only inference. Exposed development study; no new encoder or blind test access.')
    if seed is not None:
        import hashlib
        import json
        result.update(seed=seed, frozen_scalar=base_value(seed),
            schedule_sha256=hashlib.sha256(json.dumps(prior.training_schedule(seed), separators=(',', ':')).encode()).hexdigest())
    return result


def finish(out):
    files = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*'))
        if p.is_file() and p.name not in ('.lock', 'completed.json') and not p.name.endswith('.tmp')}
    write(out/'completed.json', dict(complete=True, schema=SCHEMA, files=files))


def verified(out):
    seal = read(out/'completed.json')
    if not seal.get('complete') or seal.get('schema') != SCHEMA:
        raise ValueError('Incomplete reliability artifact')
    for name, digest in seal['files'].items():
        p = (out/name).resolve()
        if not p.is_relative_to(out.resolve()) or sha(p) != digest:
            raise ValueError('Changed reliability artifact: '+name)
    return read(out/'contract.json')


def preflight(c, args):
    checks = {}
    for s in c['seeds']:
        for t in ('smoke', 'train', 'eval'):
            p = prior.root(prior.configuration())/f'seed{s}'/t
            prior.verified(p)
            checks[f'historical_{s}_{t}'] = all(read(p/'summary.json')['checks'].values())
    for name in ('train_high', *('eval_'+t for t in source.TAGS)):
        source.verified(ROOT/'.cache/scale_agreement_fusion_v1'/name)
        checks['cache_'+name] = True
    source.rm.verified(source.LOW_CACHE)
    with owned_run(root(c)/'preflight', identity(c, 'preflight'), args.resume) as out:
        if (out/'completed.json').exists():
            verified(out)
            return
        write(out/'summary.json', dict(checks=checks, frozen_scalars={str(s): base_value(s) for s in c['seeds']},
            encoder_loaded=False, training_started=False))
        if not all(checks.values()):
            raise ValueError('Prior checks failed')
        finish(out)


def models(seed, base):
    heads = {}
    for a in ARMS:
        seed_all(seed)
        heads[a] = ScaleReliabilityGate(a, base).to('cuda:0')
    if len({state_hash(m) for m in heads.values()}) != 1:
        raise ValueError('Matched initial states differ')
    return heads


def train_seed(c, args, seed, low, high, plan, base):
    from src.losses.vpr_losses import VPRLossFunction
    stage, b = args.command, base_value(seed)
    out = root(c)/f'seed{seed}'/stage
    contract = identity(c, stage, seed)
    if stage == 'train':
        if (verified(out.parent/'smoke') != identity(c, 'smoke', seed) or
                not all(read(out.parent/'smoke/summary.json')['checks'].values())):
            raise ValueError('Same-version smoke required')
    with owned_run(out, contract, args.resume):
        if (out/'completed.json').exists():
            verified(out)
            return
        heads = models(seed, b)
        initial = {a: state_hash(m) for a, m in heads.items()}
        write(out/'initial.json', dict(heads=initial, base=b, parameters={a: sum(p.numel() for p in m.parameters()) for a, m in heads.items()}))
        opts = {a: torch.optim.AdamW(m.parameters(), lr=c['learning_rate'], weight_decay=0.) for a, m in heads.items()}
        state = dict(step=0, history=[], zero_errors=None, contract_sha256=sha(out/'contract.json'),
            gradients={a: {n: False for n, p in m.named_parameters()} for a, m in heads.items()})

        def save():
            save_torch(out/'last.pt', dict(heads={a: m.state_dict() for a, m in heads.items()},
                optimizers={a: o.state_dict() for a, o in opts.items()}, state=state, rng=rng_state()))

        if (out/'last.pt').exists():
            payload = torch.load(out/'last.pt', map_location='cpu', weights_only=True)
            if payload['state']['contract_sha256'] != state['contract_sha256']:
                raise ValueError('Resume identity differs')
            for a, m in heads.items():
                m.load_state_dict(payload['heads'][a], strict=True)
                opts[a].load_state_dict(payload['optimizers'][a])
            state = payload['state']
            restore_rng(payload['rng'])
        else:
            seed_all(seed)
            save()
        rows = source.rows_for(plan, base)
        mapping = {r: i for i, r in enumerate(rows)}
        order = prior.training_schedule(seed)
        total = c['smoke_steps'] if stage == 'smoke' else c['pilot_steps']
        if not 0 <= state['step'] <= total:
            raise ValueError('Invalid cursor')
        lossfn = VPRLossFunction()
        for step in range(state['step'], total):
            started = time.monotonic()
            places = order[step]
            ids = [r for p in places for r in plan['places'][p]['rows']]
            positions = [mapping[r] for r in ids]
            l, h = torch.tensor(low[positions], device='cuda:0'), torch.tensor(high[positions], device='cuda:0')
            labels = torch.tensor(places, device='cuda:0').repeat_interleave(4)
            coords = torch.tensor([plan['images'][r]['coordinates'] for r in ids], dtype=torch.float64)
            seeds = image_seeds([plan['images'][r]['image_sha256'] for r in ids])
            target, valid, teacher = cross_view_teacher(l, h, labels, coords, temperature=c['teacher_temperature'])
            frozen = mix(l, h, torch.full((len(l), 32), b, device=l.device))
            if state['zero_errors'] is None:
                with torch.no_grad():
                    state['zero_errors'] = {a: float((m(l, h, seeds)[0]-frozen).abs().max()) for a, m in heads.items()}
            record = {}
            for a, m in heads.items():
                opts[a].zero_grad(set_to_none=True)
                z, g = m(l, h, seeds)
                main, mining = source.mined_loss(lossfn, z, labels, coords)
                score = m.scores(l, h, seeds)
                quality = F.mse_loss(score[valid], target[valid])
                loss = main+(0 if a == 'ms_group' else c['quality_weight'])*quality
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite reliability loss')
                loss.backward()
                for n, p in m.named_parameters():
                    state['gradients'][a][n] |= p.grad is not None and bool((p.grad != 0).any())
                norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True)
                opts[a].step()
                record[a] = dict(main_loss=float(main.detach()), quality_loss=float(quality.detach()), total_loss=float(loss.detach()),
                    active_ms=float(main.detach()) > 0, gradient_norm=float(norm), gate_mean=float(g.detach().mean()), gate_max=float(g.detach().abs().max()),
                    within_image_std=float(g.detach().std(1, unbiased=False).mean()),
                    mean_preservation_error=float((g.detach().mean(1)-b).abs().max()) if a != 'quality_global' else None,
                    l2_from_frozen=float((z.detach()-frozen).norm(dim=1).mean()), mining=mining)
            state['step'] = step+1
            record.update(step=step+1, seconds=time.monotonic()-started, teacher=teacher)
            state['history'].append(record)
            write(out/'progress.json', dict(total=total, **record))
            if state['step'] % 32 == 0:
                save()
            if state['step'] % 128 == 0:
                print('TRAIN', seed, stage, state['step'], '/', total, record['quality_group'], flush=True)
        save()
        if stage == 'train':
            sums, squares = np.zeros(32), np.zeros(32)
            with torch.no_grad():
                for i in range(0, len(low), 64):
                    g = heads['quality_group'].gates(torch.tensor(low[i:i+64], device='cuda:0'), torch.tensor(high[i:i+64], device='cuda:0')).cpu().numpy().astype(np.float64)
                    sums += g.sum(0)
                    squares += (g*g).sum(0)
            save_npz(out/'train_mean_gate.npz', mean=(sums/len(low)).astype(np.float32),
                between_image_std=np.sqrt(np.maximum(squares/len(low)-(sums/len(low))**2, 0)))
            write(out/'calibration.json', dict(images=len(low), train_only=True, mean_gate=(sums/len(low)).tolist(),
                source_head_sha256=state_hash(heads['quality_group'])))
        checks = dict(complete=state['step'] == total, zero_start=max(state['zero_errors'].values()) <= 1e-7,
            all_heads_updated=all(state_hash(m) != initial[a] for a, m in heads.items()),
            all_parameters_gradient_seen=all(all(v.values()) for v in state['gradients'].values()),
            centered_mean_preserved=max(r[a]['mean_preservation_error'] for r in state['history'] for a in ARMS if a != 'quality_global') <= 1e-7,
            gate_bound=max(r[a]['gate_max'] for r in state['history'] for a in ARMS) <= .500001,
            protected_sources_unchanged=source.old.key.source_hashes(source.old.key.PROTECTED) == contract['protected_sources'])
        write(out/'summary.json', dict(checks=checks, seed=seed, steps=state['step'], zero_errors=state['zero_errors'],
            final_head_sha256={a: state_hash(m) for a, m in heads.items()}, effective_ms_steps={a: sum(r[a]['active_ms'] for r in state['history']) for a in ARMS},
            encoder_loaded=False, images_reencoded=0, verdict='SMOKE_DISCARD' if stage == 'smoke' else 'TRAINED_NOT_EVALUATED'))
        if not all(checks.values()):
            raise ValueError('Mechanical checks failed')
        finish(out)


def train_all(c, args):
    if verified(root(c)/'preflight') != identity(c, 'preflight'):
        raise ValueError('Same-version preflight required')
    base = source.configuration()
    plan = source.old.key.open_plan(base)
    low, high = source.low_bank(plan, base), source.high_bank(plan, base)
    for seed in c['seeds']:
        write(root(c)/(args.command+'_progress.json'), dict(seed=seed, seeds=c['seeds']))
        train_seed(c, args, seed, low, high, plan, base)


def evaluate(c, args):
    from scripts import eval_clip_key_study as metrics
    active = {}
    with ExitStack() as stack:
        for seed in c['seeds']:
            train = root(c)/f'seed{seed}/train'
            if verified(train) != identity(c, 'train', seed) or not all(read(train/'summary.json')['checks'].values()):
                raise ValueError('Training not complete')
            out = root(c)/f'seed{seed}/eval'
            contract = dict(identity(c, 'eval', seed), checkpoint_sha256=sha(train/'last.pt'))
            stack.enter_context(owned_run(out, contract, args.resume))
            if (out/'completed.json').exists():
                verified(out)
                continue
            heads = models(seed, base_value(seed))
            saved = torch.load(train/'last.pt', map_location='cpu', weights_only=True)
            if saved['state']['step'] != c['pilot_steps'] or saved['state']['contract_sha256'] != sha(train/'contract.json'):
                raise ValueError('Not fixed-final checkpoint')
            for a, m in heads.items():
                m.load_state_dict(saved['heads'][a], strict=True)
                if state_hash(m) != read(train/'summary.json')['final_head_sha256'][a]:
                    raise ValueError('Head state changed')
                m.requires_grad_(False).eval()
            with np.load(train/'train_mean_gate.npz') as z:
                mean = torch.tensor(z['mean'], device='cuda:0')[None]
            active[seed] = dict(out=out, train=train, heads=heads, mean=mean, contract=contract, results={})
        for d in source.populations(source.configuration()) if active else []:
            cache = ROOT/'.cache/scale_agreement_fusion_v1'/('eval_'+d['tag'])
            source.verified(cache)
            n = len(d['names'])
            low, high = np.empty((n, 12288), np.float32), np.empty((n, 12288), np.float32)
            hashes = []
            for i in range((n+63)//64):
                with np.load(cache/f'{i:04d}.npz') as shard:
                    if shard['indices'].tolist() != list(range(i*64, min(n, (i+1)*64))):
                        raise ValueError('Cache order differs')
                    low[i*64:(i+1)*64], high[i*64:(i+1)*64] = shard['low'], shard['high']
                    hashes.extend(shard['hashes'].tolist())
            with torch.inference_mode():
                fixed = {}
                for a in ('innovation1', 'innovation1_518', 'descriptor_average'):
                    values = low if a == 'innovation1' else high if a == 'innovation1_518' else np.empty_like(low)
                    if a == 'descriptor_average':
                        for i in range(0, n, 64):
                            values[i:i+64] = F.normalize(torch.tensor(low[i:i+64], device='cuda:0')+torch.tensor(high[i:i+64], device='cuda:0'), dim=1).cpu().numpy()
                    fixed[a] = prior.ranking_report(values, d)
                    del values
                for seed, state in active.items():
                    out = state['out']
                    target, ready = out/(d['tag']+'.json'), out/(d['tag']+'.ready.json')
                    if ready.exists():
                        if sha(target) != read(ready)['sha256']:
                            raise ValueError('Report changed')
                        state['results'][d['tag']] = read(target)['comparison']
                        continue
                    reports = read(prior.root(prior.configuration())/f'seed{seed}/eval'/(d['tag']+'.json'))['variants']
                    if any(fixed[a] != reports[a] for a in fixed):
                        raise ValueError('Full fixed baseline Top20 differs')
                    diagnostics = {}
                    for a in EXTRA:
                        values = np.empty_like(low)
                        moments = []
                        for i in range(0, n, 64):
                            l, h = torch.tensor(low[i:i+64], device='cuda:0'), torch.tensor(high[i:i+64], device='cuda:0')
                            seeds = image_seeds(hashes[i:i+64])
                            if a == 'frozen_scalar':
                                g = state['heads']['quality_group'].base.expand(len(l), 32)
                            elif a == 'same_head_train_mean':
                                g = state['mean'].expand(len(l), -1)
                            elif a == 'same_head_shuffled':
                                g = state['heads']['quality_group'].gates(l, h)
                                g = permute_positions(g[:, :, None], seeds).squeeze(-1)
                            else:
                                g = state['heads'][a].gates(l, h, seeds)
                            z = mix(l, h, g)
                            source.rm.validate_vectors(z.cpu().numpy(), len(l))
                            values[i:i+64] = z.cpu().numpy()
                            moments.append(torch.stack((g.mean(1), g.std(1, unbiased=False)), 1).cpu().numpy())
                        reports[a] = prior.ranking_report(values, d)
                        moment = np.concatenate(moments)
                        diagnostics[a] = dict(mean_gate=float(moment[:, 0].mean()), between_image_mean_std=float(moment[:, 0].std()),
                            within_image_std=float(moment[:, 1].mean()))
                        del values
                    pairs = {a: metrics.paired_report(v, reports['quality_group']) for a, v in reports.items() if a != 'quality_group'}
                    comparison = dict(correct={a: v['correct'] for a, v in reports.items()}, candidate_vs_others=pairs)
                    write(target, dict(dataset=source.population_contract(d), variants=reports, comparison=comparison, gate_diagnostics=diagnostics))
                    write(ready, dict(sha256=sha(target)))
                    state['results'][d['tag']] = comparison
                    print('EVAL', seed, d['tag'], comparison['correct'], flush=True)
            del low, high
        for seed, state in active.items():
            checks = dict(checkpoint_unchanged=sha(state['train']/'last.pt') == state['contract']['checkpoint_sha256'],
                protected_sources_unchanged=source.old.key.source_hashes(source.old.key.PROTECTED) == state['contract']['protected_sources'])
            write(state['out']/'summary.json', dict(checks=checks, results=state['results'], seed=seed, verdict='RELIABILITY_EVALUATION_COMPLETE'))
            if not all(checks.values()):
                raise ValueError('Evaluation integrity failed')
            finish(state['out'])


def aggregate(c, args):
    with owned_run(root(c)/'aggregate', identity(c, 'aggregate'), args.resume) as out:
        if (out/'completed.json').exists():
            verified(out)
            return
        for seed in c['seeds']:
            verified(root(c)/f'seed{seed}/eval')
        results = {}
        for d in source.populations(source.configuration()):
            reports = {s: read(root(c)/f'seed{s}/eval'/(d['tag']+'.json'))['variants'] for s in c['seeds']}
            candidate = [reports[s]['quality_group']['outcomes'] for s in c['seeds']]
            clusters = positive_components(d['ground_truth'])
            comparisons = {a: contrast(candidate, [reports[s][a]['outcomes'] for s in c['seeds']], clusters,
                c['bootstrap_repetitions'], c['bootstrap_seed']) for a in VARIANTS if a != 'quality_group'}
            counts = {a: [reports[s][a]['correct'] for s in c['seeds']] for a in VARIANTS}
            results[d['tag']] = dict(seed_correct=counts, contrasts=comparisons, dependency_components=len(np.unique(clusters)))
        p = results['pitts30k-val']['contrasts']
        signal = (all(p[a]['mean_net_queries'] > 0 for a in PRIMARY) and
            all(p[a]['positive_seeds'] >= 2 for a in ('quality_global', 'same_head_shuffled')) and
            all(results[t]['contrasts']['frozen_scalar']['mean_net_queries'] >= 0 for t in ('gsv-adaptation-holdout', 'msls-val')))
        write(out/'summary.json', json_ready(dict(results=results, seeds=c['seeds'], primary_controls=PRIMARY,
            descriptive_independent_gate_signal=signal, verdict='FINITE_STUDY_COMPLETE',
            independent_test_used=False, novelty_proven=False, encoder_retrained=False,
            note='Count signal, uncertainty, historical development exposure and mechanism controls are distinct evidence. No winner selection.')))
        finish(out)
        print('FINITE_STUDY_COMPLETE', 'descriptive_independent_gate_signal', signal, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('preflight', 'smoke', 'train', 'eval', 'aggregate'))
    p.add_argument('--allow-training', action='store_true')
    p.add_argument('--allow-shared-gpu', action='store_true')
    p.add_argument('--resume', action='store_true')
    args, c = p.parse_args(), configuration()
    if args.command in ('smoke', 'train') and not args.allow_training:
        p.error('Explicit training consent required')
    if args.command in ('preflight', 'aggregate'):
        {'preflight': preflight, 'aggregate': aggregate}[args.command](c, args)
        return
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        p.error('Physical GPU1 only')
    import fcntl
    import subprocess
    with (ROOT/'logs/query_relation_screen_v1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        text = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=20)
        source.check_gpu_capacity(*map(int, text.strip().split(',')), args.allow_shared_gpu, .2)
        source.old.key.gpu(source.configuration())
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        train_all(c, args) if args.command in ('smoke', 'train') else evaluate(c, args)


if __name__ == '__main__':
    main()
