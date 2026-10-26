"""Six-cell scale/gate ablation with paired official and innovation1 families."""
import argparse
from contextlib import ExitStack
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from scripts import scale_agreement_repeat as prior
from scripts import scale_agreement_fusion as source
from scripts.summarize_scale_agreement_repeat import json_ready
from src.models.official_boq import load_official, WEIGHT_SHA
from src.models.scale_agreement_fusion import ARMS, mix, image_seeds
from src.models.complementary_covariance import permute_positions
from src.scale_factorial_statistics import factorial
from src.scale_repeat_statistics import contrast, positive_components
from src.semantic_region_protocol import read, write, sha, owned_run, state_hash, seed_all

SCHEMA = 'scale_factorial_v1'
FILES = ('scripts/scale_factorial.py', 'scripts/launch_scale_factorial.py', 'src/scale_factorial_statistics.py',
    'config/scale_factorial.json', 'tests/test_scale_factorial.py', 'doc/SCALE_FACTORIAL_V1.md')
VARIANTS = ('base_322', 'base_518', 'descriptor_average', *ARMS, 'train_mean_gate', 'same_head_shuffled')


def configuration():
    c = read(ROOT/'config/scale_factorial.json')
    expected = dict(schema=SCHEMA, runs='logs/scale_factorial_v1', cache='.cache/scale_factorial_v1', seeds=[43, 44, 45],
        pilot_steps=1024, smoke_steps=8, learning_rate=.001, microbatch=2, shard_size=64,
        bootstrap_repetitions=2000, bootstrap_seed=20261026)
    if c != expected:
        raise ValueError('Fixed six-cell protocol changed')
    return c


def root(c):
    return ROOT/c['runs']


def cache_root(c):
    return ROOT/c['cache']


def reference(tag):
    return ROOT/'logs/single_dino_interaction_v1/eval'/(tag+'.json')


def identity(c, stage, seed=None):
    base = source.configuration()
    result = dict(source.identity(base), schema=SCHEMA, stage=stage, factorial_policy=c,
        factorial_sources=source.old.key.source_hashes((*FILES, 'scripts/scale_agreement_repeat.py',
            'src/scale_repeat_statistics.py', 'scripts/summarize_scale_agreement_repeat.py')),
        paired_innovation1_seals={str(s): {t: sha(prior.root(prior.configuration())/f'seed{s}'/t/'completed.json')
            for t in ('smoke', 'train', 'eval')} for s in c['seeds']},
        official_322_reference_reports={t: sha(reference(t)) for t in source.TAGS},
        scope='Fixed prior128parameter gate, two frozen model families. Original exposed development sets. No reliability teacher or automatic expansion.')
    if seed is not None:
        import hashlib
        import json
        result.update(fusion_seed=seed, schedule_sha256=hashlib.sha256(json.dumps(prior.training_schedule(seed), separators=(',', ':')).encode()).hexdigest())
        if stage in ('smoke', 'train'):
            result['official_train_cache_seal_sha256'] = sha(cache_root(c)/'train_official/completed.json')
    return result


def finish(out):
    files = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*'))
        if p.is_file() and p.name not in ('.lock', 'completed.json') and not p.name.endswith('.tmp')}
    write(out/'completed.json', dict(complete=True, schema=SCHEMA, files=files))


def verified(out):
    seal = read(out/'completed.json')
    if not seal.get('complete') or seal.get('schema') != SCHEMA:
        raise ValueError('Incomplete six-cell artifact')
    for name, digest in seal['files'].items():
        p = (out/name).resolve()
        if not p.is_relative_to(out.resolve()) or sha(p) != digest:
            raise ValueError('Changed six-cell artifact: '+name)
    return read(out/'contract.json')


def preflight(c, args):
    checks = {}
    for s in c['seeds']:
        for t in ('smoke', 'train', 'eval'):
            p = prior.root(prior.configuration())/f'seed{s}'/t
            if prior.verified(p) != (dict(prior.stage_contract(prior.configuration(), 'eval', s),
                    checkpoint_sha256=sha(p.parent/'train/last.pt')) if t == 'eval' else prior.stage_contract(prior.configuration(), t, s)):
                raise ValueError('Historical paired source identity changed')
            checks[f'paired_{s}_{t}'] = all(read(p/'summary.json')['checks'].values())
    for name in ('train_high', *('eval_'+t for t in source.TAGS)):
        source.verified(ROOT/'.cache/scale_agreement_fusion_v1'/name)
        checks['cache_'+name] = True
    source.rm.verified(source.LOW_CACHE)
    base = source.configuration()
    plan = source.old.key.open_plan(base)
    if len(source.rows_for(plan, base)) != 32768:
        raise ValueError('Training population changed')
    for d in source.populations(base):
        r = read(reference(d['tag']))
        if r['dataset'] != source.population_contract(d):
            raise ValueError('Historical official population mismatch')
    if shutil.disk_usage(ROOT).free < 12*1024**3:
        raise RuntimeError('Need at least12GiB disk headroom')
    with owned_run(root(c)/'preflight', identity(c, 'preflight'), args.resume) as out:
        if (out/'completed.json').exists():
            verified(out)
            return
        write(out/'summary.json', dict(checks=checks, new_train_images=32768, new_eval_images=41315,
            new_official_forward_images=148166, historical_innovation1_training_reused=True,
            original_weight_sha256=WEIGHT_SHA, encoder_loaded=False, training_started=False))
        if not all(checks.values()):
            raise ValueError('Prior checks failed')
        finish(out)


def official(c):
    b = source.configuration()
    return load_official(source.old.key.path(b, 'official_cache'), source.old.key.path(b, 'dino_source'), 'cuda:0')


def encode(model, pixels, c):
    return torch.cat([model(x.to('cuda:0')) for x in pixels.split(c['microbatch'])])


def probe(c, args):
    from scripts.cross_resolution_detail import ScaleImages
    b = source.configuration()
    plan = source.old.key.open_plan(b)
    rows = source.rows_for(plan, b)[:16]
    with owned_run(root(c)/'probe', identity(c, 'probe'), args.resume) as out:
        if (out/'completed.json').exists():
            verified(out)
            return
        seed_all(42)
        model = official(c)
        initial = state_hash(model)
        ds = ScaleImages(source.old.key.path(b, 'dataset_root'), [plan['images'][r]['path'] for r in rows],
            [plan['images'][r]['image_sha256'] for r in rows])
        batch = next(iter(DataLoader(ds, batch_size=16, num_workers=0)))
        with torch.inference_mode():
            encode(model, batch[0][:2], c)
            encode(model, batch[1][:2], c)
            torch.cuda.synchronize()
            started = time.monotonic()
            low, high = encode(model, batch[0], c), encode(model, batch[1], c)
            torch.cuda.synchronize()
            seconds = time.monotonic()-started
            source.rm.validate_vectors(low.cpu().numpy(), 16)
            source.rm.validate_vectors(high.cpu().numpy(), 16)
            check = torch.cat([model(batch[0][i:i+2].to('cuda:0')) for i in range(0, 16, 2)])
            error = float((check-low).abs().max())
        checks = dict(frozen_unchanged=initial == state_hash(model), no_trainable_encoder_parameters=not any(p.requires_grad for p in model.parameters()),
            exact_microbatch_repeat=error == 0., no_innovation1_neck=not hasattr(model, 'neck'))
        write(out/'summary.json', dict(checks=checks, official_state_sha256=initial, microbatch_repeat_error=error,
            two_scale_images=16, encoder_seconds=seconds, paired_images_per_second=16/seconds,
            pure_forward_minutes_for74083=74083*seconds/16/60,
            note='Warm encoder-only estimate; excludes I/O, hashing, cache compression, head training and retrieval.'))
        if not all(checks.values()):
            raise ValueError('Encoder probe failed')
        finish(out)
        print('OFFICIAL_PROBE', read(out/'summary.json'), flush=True)


def existing_hashes(d):
    p = ROOT/'.cache/scale_agreement_fusion_v1'/('eval_'+d['tag'])
    source.verified(p)
    hashes = []
    for i in range((len(d['names'])+63)//64):
        with np.load(p/f'{i:04d}.npz') as shard:
            hashes.extend(shard['hashes'].tolist())
    if len(hashes) != len(d['names']):
        raise ValueError('Prior image population mismatch')
    return hashes


def cache(c, args):
    from scripts.cross_resolution_detail import ScaleImages
    from scripts import eval_clip_key_study as metrics
    if verified(root(c)/'preflight') != identity(c, 'preflight') or verified(root(c)/'probe') != identity(c, 'probe'):
        raise ValueError('Same-source preflight and real probe required')
    base = source.configuration()
    plan = source.old.key.open_plan(base)
    if args.command == 'cache_train':
        rows = source.rows_for(plan, base)
        jobs = [dict(tag='train_official', root=source.old.key.path(base, 'dataset_root'),
            names=[plan['images'][r]['path'] for r in rows], hashes=[plan['images'][r]['image_sha256'] for r in rows], rows=rows)]
    else:
        jobs = [dict(d, tag='eval_'+d['tag'], hashes=existing_hashes(d)) for d in source.populations(base)]
    model, initial = None, None
    for d in jobs:
        contract = dict(identity(c, args.command), names_sha256=metrics.order_digest(d['names']),
            image_hashes_sha256=metrics.order_digest(d['hashes']), images=len(d['names']), cache_tag=d['tag'],
            official_probe_seal_sha256=sha(root(c)/'probe/completed.json'))
        if args.command == 'cache_eval':
            contract['dataset'] = source.population_contract(d)
        out = cache_root(c)/d['tag']
        with owned_run(out, contract, args.resume):
            if (out/'completed.json').exists():
                verified(out)
                continue
            if model is None:
                model = official(c)
                initial = state_hash(model)
                if initial != read(root(c)/'probe/summary.json')['official_state_sha256']:
                    raise ValueError('Official model differs from real probe')
            n = len(d['names'])
            todo = [i for i in range((n+63)//64) if not source.shard_ready(out, i)]
            ds = ScaleImages(d['root'], d['names'], d['hashes'])
            loader = DataLoader(Subset(ds, [j for i in todo for j in range(i*64, min(n, (i+1)*64))]),
                batch_size=64, num_workers=base['workers'], shuffle=False)
            started = time.monotonic()
            added = 0
            with torch.inference_mode():
                for i, (lo, hi, ids, hashes, _) in zip(todo, loader):
                    if ids.tolist() != list(range(i*64, min(n, (i+1)*64))):
                        raise ValueError('Official cache row mismatch')
                    low, high = encode(model, lo, c).cpu().numpy(), encode(model, hi, c).cpu().numpy()
                    source.rm.validate_vectors(low, len(lo))
                    source.rm.validate_vectors(high, len(hi))
                    source.shard_write(out, i, low=low, high=high, indices=ids.numpy(), hashes=np.asarray(hashes),
                        row_ids=np.asarray(d['rows'][i*64:(i+1)*64] if args.command == 'cache_train' else ids.numpy()))
                    added += len(lo)
                    elapsed = time.monotonic()-started
                    write(out/'progress.json', dict(images=int(ids[-1])+1, total=n, new_images_this_attempt=added,
                        seconds=elapsed, paired_images_per_second=added/elapsed,
                        estimated_remaining_minutes=(n-int(ids[-1])-1)*elapsed/added/60))
                    print('CACHE', d['tag'], int(ids[-1])+1, '/', n, 'paired_images_per_sec', round(added/elapsed, 2), flush=True)
            if state_hash(model) != initial:
                raise ValueError('Frozen official weights changed')
            write(out/'summary.json', dict(images=n, official_state_sha256=initial, frozen_unchanged=True,
                no_innovation1_neck=True, precision='fp32', shape=[n, 12288], scales=[322, 518]))
            finish(out)


def official_bank(c):
    out = cache_root(c)/'train_official'
    verified(out)
    b = source.configuration()
    rows = source.rows_for(source.old.key.open_plan(b), b)
    low, high = np.empty((32768, 12288), np.float32), np.empty((32768, 12288), np.float32)
    for i in range(512):
        with np.load(out/f'{i:04d}.npz') as d:
            if d['row_ids'].tolist() != rows[i*64:(i+1)*64] or d['indices'].tolist() != list(range(i*64, (i+1)*64)):
                raise ValueError('Training cache alignment differs')
            low[i*64:(i+1)*64], high[i*64:(i+1)*64] = d['low'], d['high']
    return low, high


def private_trainer():
    # Inject output/provenance/sealing helpers into a PRIVATE module instance.
    # The source module and all original experiments remain untouched.
    spec = importlib.util.spec_from_file_location('_scale_factorial_private_trainer', ROOT/'scripts/scale_agreement_repeat.py')
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.root, engine.stage_contract, engine.verified, engine.finish = root, identity, verified, finish
    return engine


def train(c, args):
    if verified(root(c)/'preflight') != identity(c, 'preflight'):
        raise ValueError('Preflight required')
    low, high = official_bank(c)
    base = source.configuration()
    plan = source.old.key.open_plan(base)
    engine = private_trainer()
    for seed in c['seeds']:
        write(root(c)/(args.command+'_progress.json'), dict(seed=seed, seeds=c['seeds']))
        engine.train_seed(c, args, seed, low, high, plan, base)
        actual = read(root(c)/f'seed{seed}'/args.command/'initial.json')['heads']
        if actual != read(prior.root(prior.configuration())/f'seed{seed}'/args.command/'initial.json')['heads']:
            raise ValueError('Official/F1 paired head initializations differ')


def remap_historical(reports):
    return {('base_322' if a == 'innovation1' else 'base_518' if a == 'innovation1_518' else a): r for a, r in reports.items()}


def evaluate(c, args):
    from scripts import eval_clip_key_study as metrics
    with ExitStack() as stack:
        states = {}
        for family in ('official', 'innovation1'):
            for seed in c['seeds']:
                train = (root(c) if family == 'official' else prior.root(prior.configuration()))/f'seed{seed}/train'
                (verified if family == 'official' else prior.verified)(train)
                if not all(read(train/'summary.json')['checks'].values()):
                    raise ValueError('Invalid final training artifact')
                out = root(c)/family/f'seed{seed}/eval'
                contract = dict(identity(c, 'eval', seed), family=family, checkpoint_sha256=sha(train/'last.pt'),
                    eval_cache_seals={t: sha((cache_root(c) if family == 'official' else ROOT/'.cache/scale_agreement_fusion_v1')/('eval_'+t)/'completed.json') for t in source.TAGS})
                stack.enter_context(owned_run(out, contract, args.resume))
                if (out/'completed.json').exists():
                    verified(out)
                    continue
                heads = source.models(dict(source.configuration(), seed=seed))
                payload = torch.load(train/'last.pt', map_location='cpu', weights_only=True)
                if payload['state']['step'] != c['pilot_steps'] or payload['state']['contract_sha256'] != sha(train/'contract.json'):
                    raise ValueError('Not fixed-final checkpoint')
                for a, m in heads.items():
                    m.load_state_dict(payload['heads'][a], strict=True)
                    if state_hash(m) != read(train/'summary.json')['final_head_sha256'][a]:
                        raise ValueError('Head state changed')
                    m.requires_grad_(False).eval()
                with np.load(train/'train_mean_gate.npz') as z:
                    mean = torch.tensor(z['mean'], device='cuda:0')[None]
                states[family, seed] = dict(out=out, train=train, heads=heads, mean=mean, contract=contract, results={})
        for family in ('official', 'innovation1'):
            selected = {s: states[family, s] for s in c['seeds'] if (family, s) in states}
            for d in source.populations(source.configuration()) if selected else []:
                cache = (cache_root(c) if family == 'official' else ROOT/'.cache/scale_agreement_fusion_v1')/('eval_'+d['tag'])
                (verified if family == 'official' else source.verified)(cache)
                n = len(d['names'])
                low, high = np.empty((n, 12288), np.float32), np.empty((n, 12288), np.float32)
                hashes = []
                for i in range((n+63)//64):
                    with np.load(cache/f'{i:04d}.npz') as shard:
                        if shard['indices'].tolist() != list(range(i*64, min(n, (i+1)*64))):
                            raise ValueError('Evaluation cache alignment differs')
                        low[i*64:(i+1)*64], high[i*64:(i+1)*64] = shard['low'], shard['high']
                        hashes.extend(shard['hashes'].tolist())
                if hashes != existing_hashes(d):
                    raise ValueError('Official/F1 image contents differ')
                with torch.inference_mode():
                    fixed = {}
                    for a in ('base_322', 'base_518', 'descriptor_average'):
                        values = low if a == 'base_322' else high if a == 'base_518' else np.empty_like(low)
                        if a == 'descriptor_average':
                            for i in range(0, n, 64):
                                values[i:i+64] = F.normalize(torch.tensor(low[i:i+64], device='cuda:0')+torch.tensor(high[i:i+64], device='cuda:0'), dim=1).cpu().numpy()
                        fixed[a] = prior.ranking_report(values, d)
                        del values
                    if family == 'official' and fixed['base_322'] != read(reference(d['tag']))['variants']['official']:
                        raise ValueError('Full historical official322 Top20 not reproduced')
                    for seed, state in selected.items():
                        target, ready = state['out']/(d['tag']+'.json'), state['out']/(d['tag']+'.ready.json')
                        if ready.exists():
                            if sha(target) != read(ready)['sha256']:
                                raise ValueError('Report changed')
                            state['results'][d['tag']] = read(target)['comparison']
                            continue
                        reports = dict(fixed)
                        for a in (*ARMS, 'train_mean_gate', 'same_head_shuffled'):
                            values = np.empty_like(low)
                            for i in range(0, n, 64):
                                l, h = torch.tensor(low[i:i+64], device='cuda:0'), torch.tensor(high[i:i+64], device='cuda:0')
                                seeds = image_seeds(hashes[i:i+64])
                                if a == 'train_mean_gate':
                                    g = state['mean'].expand(len(l), -1)
                                elif a == 'same_head_shuffled':
                                    g = permute_positions(state['heads']['group_gate'].gates(l, h)[:, :, None], seeds).squeeze(-1)
                                else:
                                    g = state['heads'][a].gates(l, h, seeds)
                                z = mix(l, h, g).cpu().numpy()
                                source.rm.validate_vectors(z, len(l))
                                values[i:i+64] = z
                            reports[a] = prior.ranking_report(values, d)
                            del values
                        if family == 'innovation1':
                            old = remap_historical(read(prior.root(prior.configuration())/f'seed{seed}/eval'/(d['tag']+'.json'))['variants'])
                            if any(reports[a] != r for a, r in old.items()):
                                raise ValueError('Full historical F1 results not reproduced')
                        pairs = {a: metrics.paired_report(v, reports['group_gate']) for a, v in reports.items() if a != 'group_gate'}
                        comparison = dict(correct={a: v['correct'] for a, v in reports.items()}, candidate_vs_others=pairs)
                        write(target, dict(dataset=source.population_contract(d), family=family, seed=seed, variants=reports, comparison=comparison))
                        write(ready, dict(sha256=sha(target)))
                        state['results'][d['tag']] = comparison
                        print('EVAL', family, seed, d['tag'], comparison['correct'], flush=True)
                del low, high
        for (family, seed), state in states.items():
            checks = dict(checkpoint_unchanged=sha(state['train']/'last.pt') == state['contract']['checkpoint_sha256'],
                protected_sources_unchanged=source.old.key.source_hashes(source.old.key.PROTECTED) == state['contract']['protected_sources'])
            write(state['out']/'summary.json', dict(checks=checks, results=state['results'], family=family, seed=seed, verdict='FACTORIAL_EVALUATION_COMPLETE'))
            if not all(checks.values()):
                raise ValueError('Evaluation integrity failed')
            finish(state['out'])


def aggregate(c, args):
    from scripts import eval_clip_key_study as metrics
    with owned_run(root(c)/'aggregate', identity(c, 'aggregate'), args.resume) as out:
        if (out/'completed.json').exists():
            verified(out)
            return
        for family in ('official', 'innovation1'):
            for s in c['seeds']:
                verified(root(c)/family/f'seed{s}/eval')
        results = {}
        audited = 0
        for d in source.populations(source.configuration()):
            families = {}
            for family in ('official', 'innovation1'):
                reports = []
                for s in c['seeds']:
                    r = read(root(c)/family/f'seed{s}/eval'/(d['tag']+'.json'))
                    if r['dataset'] != source.population_contract(d):
                        raise ValueError('Population differs')
                    for v in r['variants'].values():
                        if metrics.retrieval_report(v['predictions'], d['ground_truth'], d['references']) != v:
                            raise ValueError('Saved predictions do not reproduce GT')
                        audited += 1
                    pairs = {a: metrics.paired_report(v, r['variants']['group_gate']) for a, v in r['variants'].items() if a != 'group_gate'}
                    if pairs != r['comparison']['candidate_vs_others']:
                        raise ValueError('Paired fixes/harms differ')
                    reports.append(r['variants'])
                families[family] = reports
            clusters = positive_components(d['ground_truth'])
            result = factorial(families['official'], families['innovation1'], clusters, c['bootstrap_repetitions'], c['bootstrap_seed'])
            result['all_variant_counts'] = {f: {a: [r[a]['correct'] for r in rs] for a in VARIANTS} for f, rs in families.items()}
            result['gate_controls'] = {f: {a: contrast([r['group_gate']['outcomes'] for r in rs], [r[a]['outcomes'] for r in rs],
                clusters, c['bootstrap_repetitions'], c['bootstrap_seed']) for a in ('static_gate', 'global_gate', 'shuffled_gate', 'train_mean_gate', 'same_head_shuffled')} for f, rs in families.items()}
            results[d['tag']] = result
        write(out/'summary.json', json_ready(dict(results=results, seeds=c['seeds'], verdict='SIX_CELL_STUDY_COMPLETE',
            checks=dict(all_saved_predictions_recomputed_from_GT=True, all_paired_fix_harm_reports_verified=True),
            audited_version_reports=audited, innovation1_training_reused=True, official_training_completed=True,
            independent_test_used=False, novelty_proven=False,
            note='Package effects, scale-only effects, gate-only effects and paired interactions are reported separately. No seed/checkpoint selection.')))
        finish(out)
        print('SIX_CELL_STUDY_COMPLETE', {t: r['cell_correct'] for t, r in results.items()}, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('preflight', 'probe', 'cache_train', 'smoke', 'train', 'cache_eval', 'eval', 'aggregate'))
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
        memory = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=20)
        source.check_gpu_capacity(*map(int, memory.strip().split(',')), args.allow_shared_gpu, .2)
        source.old.key.gpu(source.configuration())
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        {'probe': probe, 'cache_train': cache, 'cache_eval': cache, 'smoke': train, 'train': train, 'eval': evaluate}[args.command](c, args)


if __name__ == '__main__':
    main()
