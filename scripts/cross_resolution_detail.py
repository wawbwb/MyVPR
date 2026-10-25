"""Finite cross-resolution residual study on the frozen innovation-one model."""
import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scripts import single_dino_interaction as old
from scripts.spatial_resample import mined_loss, replay
from scripts.semantic_region_study import transform
from scripts.single_dino_supervision import check_gpu_capacity
from src.models.cross_resolution_detail import ARMS, CrossResolutionVPR, image_seeds
from src.semantic_region_protocol import (read, write, sha, state_hash, owned_run,
    seed_all, save_torch, rng_state, restore_rng)

SCHEMA = 'cross_resolution_detail_v1'
NEW_FILES = ('src/models/cross_resolution_detail.py', 'scripts/cross_resolution_detail.py',
    'scripts/launch_cross_resolution_detail.py', 'config/cross_resolution_detail.json',
    'tests/test_cross_resolution_detail.py', 'doc/CROSS_RESOLUTION_DETAIL_V1.md')
SOURCES = (*NEW_FILES, 'scripts/spatial_resample.py', 'scripts/semantic_region_study.py',
    'src/models/structural_channel_adapter.py', 'src/models/complementary_covariance.py',
    'scripts/eval_spatial_adapter_repeat.py', 'scripts/eval_clip_key_study.py',
    'scripts/single_dino_supervision.py', 'src/models/official_boq.py')
VARIANTS = ('innovation1', 'innovation1_518', 'descriptor_average', *ARMS)


def configuration():
    extra = read(ROOT/'config/cross_resolution_detail.json')
    expected = dict(schema=SCHEMA, runs='logs/cross_resolution_detail_v1',
        learning_rate=.0001, pilot_steps=1024, smoke_steps=8, keep_weight=0.)
    if extra != expected:
        raise ValueError('Registered detail study changed')
    return dict(old.configuration(), **extra)


def identity(c):
    value = old.identity(c)
    value.update(schema=SCHEMA, new_sources=old.key.source_hashes(SOURCES),
        encoder_inventory='ONE frozen weight set; 322/native518/degraded518 forwards; candidate inference TWO forwards',
        mechanism='Area-aligned cross-resolution difference, bounded residual before original BoQ',
        scope='Fixed1024step single-seed development screen; no blind test or novelty claim',
        baseline_reports={tag: sha(ROOT/'logs/structural_channel_v1/eval'/(tag+'.json'))
            for tag in ('gsv-adaptation-holdout', 'msls-val', 'pitts30k-val')})
    return value


def finish(output):
    files = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*'))
        if p.is_file() and p.name not in ('.lock', 'completed.json') and not p.name.endswith('.tmp')}
    write(output/'completed.json', dict(complete=True, schema=SCHEMA, files=files))


def verified(output):
    seal = read(output/'completed.json')
    if seal.get('complete') is not True or seal.get('schema') != SCHEMA:
        raise ValueError('Incomplete/wrong experiment')
    for name, digest in seal['files'].items():
        p = (output/name).resolve()
        if not p.is_relative_to(output.resolve()) or sha(p) != digest:
            raise ValueError('Artifact changed: '+name)
    return read(output/'contract.json')


class ScaleImages(Dataset):
    def __init__(self, root, names, hashes):
        self.root = Path(root).resolve()
        self.names, self.hashes = names, hashes
        self.low, self.high = transform(322), transform(518)
        if hashes is not None and len(hashes) != len(names):
            raise ValueError('Hash population differs')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        p = (self.root/self.names[i]).resolve()
        if not p.is_relative_to(self.root) or not p.is_file():
            raise ValueError('Missing/unsafe image')
        digest = sha(p)
        if self.hashes is not None and digest != self.hashes[i]:
            raise ValueError('Sealed image changed')
        with Image.open(p) as im:
            rgb = im.convert('RGB')
            return self.low(rgb), self.high(rgb), i, digest, torch.tensor(rgb.size)


class ScaleSteps(Dataset):
    def __init__(self, c, plan, steps):
        self.plan, self.steps = plan, steps
        self.images = ScaleImages(old.key.path(c, 'dataset_root'),
            [r['path'] for r in plan['images']], [r['image_sha256'] for r in plan['images']])

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, i):
        ids = [r for p in self.steps[i] for r in self.plan['places'][p]['rows']]
        values = [self.images[r] for r in ids]
        return (torch.stack([v[0] for v in values]), torch.stack([v[1] for v in values]),
            torch.tensor(self.steps[i], dtype=torch.long).repeat_interleave(4),
            torch.tensor([self.plan['images'][r]['coordinates'] for r in ids], dtype=torch.float64),
            torch.tensor(ids, dtype=torch.long))


def build(c):
    anchor = old.make_anchor(c)
    models = {}
    for arm in ARMS:
        seed_all(c['seed'])
        models[arm] = CrossResolutionVPR(anchor, arm).to(c['device'])
    if len({state_hash(m.adapter) for m in models.values()}) != 1:
        raise ValueError('Adapter initial states differ')
    ids = [set(map(id, m.adapter.parameters())) for m in models.values()]
    if any(ids[i] & ids[j] for i in range(len(ids)) for j in range(i)):
        raise ValueError('Trainable parameters shared')
    return models


def train(c, args):
    from src.losses.vpr_losses import VPRLossFunction
    plan = old.key.open_plan(c)
    runs = old.key.path(c, 'runs')
    contract = dict(stage=args.command, **identity(c))
    if len(plan['schedule']) < c['pilot_steps']:
        raise ValueError('Insufficient sealed schedule')
    if args.command == 'train':
        smoke = verified(runs/'smoke')
        if ({k: v for k, v in smoke.items() if k != 'stage'} !=
                {k: v for k, v in contract.items() if k != 'stage'} or
                not all(read(runs/'smoke/summary.json')['checks'].values())):
            raise ValueError('Same-source smoke required')
    output = runs/args.command
    with owned_run(output, contract, args.resume):
        if (output/'completed.json').exists():
            verified(output)
            print('Already complete')
            return
        old.key.gpu(c)
        models = build(c)
        first = models[ARMS[0]]
        initial = dict(frozen=state_hash(first.anchor),
            adapters={a: state_hash(m.adapter) for a, m in models.items()},
            parameters={a: sum(p.numel() for p in m.adapter.parameters()) for a, m in models.items()})
        if (output/'initial.json').exists() and read(output/'initial.json') != initial:
            raise ValueError('Initial state changed')
        write(output/'initial.json', initial)
        sample = ScaleSteps(c, plan, plan['schedule'][:1])[0]
        seeds = image_seeds([plan['images'][int(i)]['image_sha256'] for i in sample[4][:2]])
        with torch.no_grad():
            f = first.encode(sample[0][:2].to(c['device']), sample[1][:2].to(c['device']))
            errors = {a: float((m.describe(f, seeds)[0]-f[1]).abs().max()) for a, m in models.items()}
            gaps = dict(real=float(f[2].square().mean().sqrt()), degraded=float(f[3].square().mean().sqrt()),
                distinct=float((f[2]-f[3]).abs().max()) > 1e-5)
        del sample, f
        optimizers = {a: torch.optim.AdamW(m.adapter.parameters(), lr=c['learning_rate'], weight_decay=0.)
            for a, m in models.items()}
        state = dict(step=0, history=[], contract_sha256=sha(output/'contract.json'),
            gradients={a: {n: False for n, p in m.adapter.named_parameters()} for a, m in models.items()})

        def save():
            save_torch(output/'last.pt', dict(adapters={a: m.adapter.state_dict() for a, m in models.items()},
                optimizers={a: o.state_dict() for a, o in optimizers.items()}, state=state, rng=rng_state()))

        if (output/'last.pt').exists():
            saved = torch.load(output/'last.pt', map_location='cpu', weights_only=True)
            if saved['state']['contract_sha256'] != state['contract_sha256']:
                raise ValueError('Resume contract changed')
            for a, m in models.items():
                m.adapter.load_state_dict(saved['adapters'][a], strict=True)
                optimizers[a].load_state_dict(saved['optimizers'][a])
            state = saved['state']
            restore_rng(saved['rng'])
        else:
            seed_all(c['seed'])
            save()
        total = c['smoke_steps'] if args.command == 'smoke' else c['pilot_steps']
        if not 0 <= state['step'] <= total:
            raise ValueError('Invalid cursor')
        loader = DataLoader(ScaleSteps(c, plan, plan['schedule'][state['step']:total]), batch_size=None,
            num_workers=c['workers'], generator=torch.Generator().manual_seed(c['seed']))
        lossfn = VPRLossFunction()
        for m in models.values():
            m.train()
        for low, high, labels, coords, ids in tqdm(loader, total=total, initial=state['step'], desc='detail '+args.command):
            started = time.monotonic()
            labels = labels.to(c['device'])
            seeds = image_seeds([plan['images'][int(i)]['image_sha256'] for i in ids])
            with torch.no_grad():
                chunks = [first.encode(low[i:i+2].to(c['device']), high[i:i+2].to(c['device']))
                    for i in range(0, len(low), 2)]
            encoded_seconds = time.monotonic()-started
            reference = torch.cat([f[1] for f in chunks])
            row = {}
            for arm, m in models.items():
                with torch.no_grad():
                    temporary, offset = [], 0
                    for f in chunks:
                        n = len(f[0])
                        temporary.append(m.describe(f, seeds[offset:offset+n]))
                        offset += n
                    leaf = torch.cat([v[0] for v in temporary]).detach().requires_grad_()
                    relative = torch.cat([v[1]['relative_delta'] for v in temporary])
                    difference = sum(float(v[1]['difference_rms']) for v in temporary)/len(temporary)
                    saturation = sum(float(v[1]['saturation']) for v in temporary)/len(temporary)
                del temporary
                loss, mining = mined_loss(lossfn, leaf, labels, coords)
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite loss')
                gradient = torch.autograd.grad(loss, leaf)[0]
                optimizers[arm].zero_grad(set_to_none=True)
                replay(m, chunks, seeds, gradient)
                for name, p in m.adapter.named_parameters():
                    state['gradients'][arm][name] |= p.grad is not None and bool((p.grad != 0).any())
                norm = torch.nn.utils.clip_grad_norm_(m.adapter.parameters(), c['gradient_clip'], error_if_nonfinite=True)
                optimizers[arm].step()
                row[arm] = dict(main_loss=float(loss.detach()), main_loss_active=float(loss.detach()) > 0,
                    gradient_norm=float(norm), relative_delta_mean=float(relative.mean()),
                    relative_delta_max=float(relative.max()), difference_rms=difference, saturation=saturation,
                    descriptor_cosine_drift=float((1-(leaf.detach()*reference).sum(1)).clamp_min(0).mean()), mining=mining)
                del leaf, loss, gradient
            torch.cuda.synchronize()
            state['step'] += 1
            row.update(step=state['step'], seconds=time.monotonic()-started, encoded_seconds=encoded_seconds,
                peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
            state['history'].append(row)
            write(output/'progress.json', dict(phase='training', total=total, **row))
            if state['step'] % c['checkpoint_every'] == 0:
                save()
            del chunks, reference
        save()
        checks = dict(complete_steps=state['step'] == total, zero_start=max(errors.values()) <= 1e-6,
            real_and_degraded_features_distinct=gaps['distinct'],
            frozen_anchor_unchanged=state_hash(first.anchor) == initial['frozen'],
            capacity_matched=len(set(initial['parameters'].values())) == 1,
            all_adapters_updated=all(state_hash(m.adapter) != initial['adapters'][a] for a, m in models.items()),
            all_parameters_gradient_seen=all(all(v.values()) for v in state['gradients'].values()),
            innovation1_sources_unchanged=old.key.source_hashes(old.key.PROTECTED) == contract['protected_sources'])
        write(output/'summary.json', dict(checks=checks, steps=state['step'], zero_errors=errors, initial_gaps=gaps,
            effective_loss_steps={a: sum(r[a]['main_loss_active'] for r in state['history']) for a in ARMS},
            final_adapter_sha256={a: state_hash(m.adapter) for a, m in models.items()}, parameters=initial['parameters'],
            encoder_weight_sets=1, candidate_encoder_forwards=2, study_encoder_forwards=3, descriptor_dimensions=12288,
            verdict='SMOKE_DISCARD' if args.command == 'smoke' else 'TRAINED_NOT_EVALUATED'))
        if not all(checks.values()):
            raise ValueError('Mechanical checks failed: '+str(checks))
        finish(output)


def evaluate(c, args):
    from scripts import eval_clip_key_study as metrics
    run = old.key.path(c, 'runs')/'train'
    if verified(run) != dict(stage='train', **identity(c)) or not all(read(run/'summary.json')['checks'].values()):
        raise ValueError('Training incomplete/changed')
    datasets = metrics.populations(c, old.key.open_plan(c))
    contract = dict(stage='eval', **identity(c), checkpoint_sha256=sha(run/'last.pt'),
        datasets={d['tag']: dict(references=d['references'], queries=d['queries'],
            order=metrics.order_digest(d['names']), gt=metrics.gt_digest(d['ground_truth'])) for d in datasets})
    output = old.key.path(c, 'runs')/'eval'
    with owned_run(output, contract, args.resume):
        if (output/'completed.json').exists():
            verified(output)
            print('Already evaluated')
            return
        old.key.gpu(c)
        models = build(c)
        first = models[ARMS[0]]
        frozen = state_hash(first.anchor)
        saved = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
        if saved['state']['step'] != c['pilot_steps'] or saved['state']['contract_sha256'] != sha(run/'contract.json'):
            raise ValueError('Not registered final checkpoint')
        for a, m in models.items():
            m.adapter.load_state_dict(saved['adapters'][a], strict=True)
            if state_hash(m.adapter) != read(run/'summary.json')['final_adapter_sha256'][a]:
                raise ValueError('Adapter changed')
            m.requires_grad_(False).eval()
        results = {}
        for d in datasets:
            tag = d['tag']
            target, ready = output/(tag+'.json'), output/(tag+'.ready.json')
            if ready.exists():
                if sha(target) != read(ready)['sha256']:
                    raise ValueError('Report changed')
                results[tag] = read(target)['comparison']
                continue
            started = time.monotonic()
            arrays = {a: np.empty((len(d['names']), 12288), np.float32) for a in VARIANTS}
            stats = {a: dict(images=0, relative_sum=0., max_relative_delta=0., difference_sum=0., saturation_sum=0.) for a in ARMS}
            native = dict(images=0, both_axes_above_322=0, both_axes_at_least_518=0)
            contents = []
            loader = DataLoader(ScaleImages(d['root'], d['names'], d['hashes']), batch_size=16,
                num_workers=c['workers'], shuffle=False)
            with torch.inference_mode():
                for batch, (low, high, ids, hashes, sizes) in enumerate(tqdm(loader, desc='detail '+tag)):
                    if ids.tolist() != list(range(batch*16, batch*16+len(low))):
                        raise ValueError('Extraction order changed')
                    contents.extend(hashes)
                    native['images'] += len(low)
                    native['both_axes_above_322'] += int((sizes.min(1).values > 322).sum())
                    native['both_axes_at_least_518'] += int((sizes.min(1).values >= 518).sum())
                    for i in range(0, len(low), 2):
                        f = first.encode(low[i:i+2].to(c['device']), high[i:i+2].to(c['device']))
                        seeds = image_seeds(hashes[i:i+2])
                        values = dict(innovation1=f[1], innovation1_518=f[4], descriptor_average=F.normalize(f[1]+f[4], dim=1))
                        for a, m in models.items():
                            values[a], s = m.describe(f, seeds)
                            n, v = len(values[a]), stats[a]
                            v['images'] += n
                            v['relative_sum'] += float(s['relative_delta'].sum())
                            v['max_relative_delta'] = max(v['max_relative_delta'], float(s['relative_delta'].max()))
                            v['difference_sum'] += n*float(s['difference_rms'])
                            v['saturation_sum'] += n*float(s['saturation'])
                        for a, z in values.items():
                            if not torch.isfinite(z).all() or float((z.norm(dim=1)-1).abs().max()) > 1e-4:
                                raise ValueError('Invalid descriptor')
                            arrays[a][ids[i:i+2].numpy()] = z.cpu().numpy()
                        del f, values
                    write(output/'progress.json', dict(phase='evaluation', dataset=tag, images=int(ids[-1])+1,
                        total=len(d['names']), elapsed_seconds=time.monotonic()-started))
                reports = {}
                for a in VARIANTS:
                    db = torch.tensor(arrays[a][:d['references']], device=c['device'])
                    queries = arrays[a][d['references']:]
                    ranks = np.concatenate([(torch.tensor(q, device=c['device'])@db.T).topk(20, dim=1).indices.cpu().numpy()
                        for q in np.array_split(queries, max(1, (len(queries)+15)//16))])
                    reports[a] = metrics.retrieval_report(ranks, d['ground_truth'], d['references'])
                    del db
            baseline = ROOT/'logs/structural_channel_v1/eval'/(tag+'.json')
            if sha(baseline) != contract['baseline_reports'][tag] or reports['innovation1'] != read(baseline)['variants']['innovation1']:
                raise ValueError('Baseline full Top20 differs')
            comparison = dict(correct={a: r['correct'] for a, r in reports.items()},
                vs_innovation1={a: metrics.paired_report(reports['innovation1'], r) for a, r in reports.items() if a != 'innovation1'},
                candidate_vs_controls={a: metrics.paired_report(reports[a], reports['aligned_detail'])
                    for a in ('degraded_detail', 'shuffled_detail', 'innovation1_518', 'descriptor_average')})
            write(target, dict(dataset=contract['datasets'][tag], variants=reports, comparison=comparison,
                detail_diagnostics=stats, native_resolution=native,
                image_content_digest=metrics.order_digest([n+':'+h for n, h in zip(d['names'], contents)]),
                elapsed_seconds=time.monotonic()-started))
            write(ready, dict(sha256=sha(target)))
            results[tag] = comparison
            print(tag, comparison['correct'], flush=True)
            del arrays, loader
        checks = dict(frozen_anchor_unchanged=state_hash(first.anchor) == frozen,
            innovation1_sources_unchanged=old.key.source_hashes(old.key.PROTECTED) == contract['protected_sources'],
            checkpoint_unchanged=sha(run/'last.pt') == contract['checkpoint_sha256'])
        primary, guard = results['pitts30k-val'], results['msls-val']
        signal = (primary['vs_innovation1']['aligned_detail']['net'] > 0 and
            all(v['net'] > 0 for v in primary['candidate_vs_controls'].values()) and
            guard['vs_innovation1']['aligned_detail']['net'] >= 0 and
            all(v['net'] >= 0 for v in guard['candidate_vs_controls'].values()) and
            results['gsv-adaptation-holdout']['vs_innovation1']['aligned_detail']['net'] >= 0)
        write(output/'summary.json', dict(checks=checks, results=results, stable_gain_proven=False, blind_test_used=False,
            primary_dataset='pitts30k-val', msls_guard='Observed counts only, not statistical noninferiority',
            verdict='EXPLORATORY_CROSS_RESOLUTION_DETAIL_SIGNAL' if signal else 'NO_MATCHED_CROSS_RESOLUTION_DETAIL_SIGNAL'))
        if not all(checks.values()):
            raise ValueError('Integrity failed')
        finish(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('plan', 'smoke', 'train', 'eval'))
    p.add_argument('--allow-training', action='store_true')
    p.add_argument('--allow-shared-gpu', action='store_true')
    p.add_argument('--resume', action='store_true')
    args, c = p.parse_args(), configuration()
    if args.command in ('smoke', 'train') and not args.allow_training:
        p.error('Training consent required')
    if args.command == 'plan':
        old.key.open_plan(c)
        print(dict(identity=identity(c), training_started=False))
        return
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        p.error('Physical GPU1 only')
    import fcntl
    import subprocess
    with (ROOT/'logs/query_relation_screen_v1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        memory = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=memory.total,memory.used,memory.free',
            '--format=csv,noheader,nounits'], text=True, timeout=20)
        check_gpu_capacity(*map(int, memory.strip().split(',')), args.allow_shared_gpu, c['cuda_memory_fraction'])
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if args.command == 'eval':
            evaluate(c, args)
        else:
            train(c, args)


if __name__ == '__main__':
    main()
