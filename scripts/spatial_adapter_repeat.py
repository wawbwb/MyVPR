"""Three paired random seeds: frozen official -> 3x3 vs 1x1 adapters.

Only new files/outputs. Same split/views, paired shuffle/init, fixed final epoch.
One frozen backbone pass shared within a seed; independent student gradients.
"""
import argparse
from importlib import metadata
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts import clip_key_study as key
from scripts.semantic_region_study import PlaceSteps
from src.models.spatial_adapter_repeat import ARMS, paired_models, adapter_inventory
from src.models.clip_preboq import replay_backward
from src.models.official_boq import load_official, native_feature_layout, verify_download
from src.semantic_region_protocol import (owned_run, read, write, sha, state_hash, seed_all,
    save_torch, rng_state, restore_rng, gps_retrieval, schedule)

SCHEMA = 'spatial_adapter_repeat_v1'
SOURCES = ('src/models/spatial_adapter_repeat.py', 'scripts/spatial_adapter_repeat.py',
    'scripts/eval_spatial_adapter_repeat.py', 'scripts/launch_spatial_adapter_repeat.py',
    'config/spatial_adapter_repeat.yaml', 'tests/test_spatial_adapter_repeat.py')
path = key.path


def configuration(p):
    import yaml
    c = yaml.safe_load(p.read_text(encoding='utf-8'))
    if (c['schema'] != SCHEMA or c['seeds'] != [42, 43, 44] or c['split_seed'] != 42
            or c['views'] != 4 or c['student_size'] != 322 or c['precision'] != 'fp32'
            or c['device'] != 'cuda:0' or c['primary_dataset'] != 'pitts30k-val'):
        raise ValueError('Expected locked three-seed FP32/322/four-view protocol')
    for k in ('train_places', 'places_per_batch', 'epochs', 'hidden', 'microbatch', 'workers',
              'checkpoint_every', 'smoke_steps'):
        if type(c[k]) is not int or c[k] < 1: raise ValueError('Positive integer required: '+k)
    for k in ('adapter_gain', 'learning_rate', 'keep_weight', 'gradient_clip', 'cuda_memory_fraction'):
        if not np.isfinite(c[k]) or c[k] <= 0: raise ValueError('Positive finite setting required: '+k)
    if c['cuda_memory_fraction'] > .2 or c['adapter_gain'] > 1 or c['hidden'] != 64:
        raise ValueError('Adapter/memory protocol differs')
    return c


def open_plan(c, seed):
    if seed not in c['seeds']: raise ValueError('Unregistered seed')
    # The original sealed split was generated with seed42. Vary only NEW training
    # shuffles/init, never regenerate the sampled places, views or holdout.
    base = key.open_plan(dict(c, seed=c['split_seed']))
    return dict(base, schedule=schedule(c['train_places'], c['epochs'], c['places_per_batch'], seed))


def identity(c):
    dino = path(c, 'dino_source')
    return dict(schema=SCHEMA, config=c, code=key.source_hashes(SOURCES),
        frozen_helpers=key.source_hashes((*key.SOURCES, 'src/models/clip_preboq.py',
            'src/models/clip_nuisance.py', 'scripts/eval_clip_key_study.py')),
        protected_sources=key.source_hashes(key.PROTECTED),
        official=verify_download(path(c, 'official_cache')),
        dino_code={p.relative_to(dino).as_posix(): sha(p) for p in sorted(dino.rglob('*.py'))},
        plan_complete_sha256=sha(path(c, 'plan')/'completed.json'),
        plan_sha256=sha(path(c, 'plan')/'plan.json'),
        versions={n: metadata.version(n) for n in ('torch', 'torchvision', 'numpy', 'pillow', 'pytorch-metric-learning')})


def finish(output):
    files = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*'))
             if p.is_file() and p.name not in ('.lock', 'completed.json') and not p.name.endswith('.tmp')}
    write(output/'completed.json', dict(complete=True, schema=SCHEMA, files=files))


def verified(output):
    value = read(output/'completed.json')
    if value.get('schema') != SCHEMA or value.get('complete') is not True: raise ValueError('Incomplete repeat artifact')
    for n, digest in value['files'].items():
        p = (output/n).resolve()
        if not p.is_relative_to(output.resolve()) or not p.is_file() or sha(p) != digest:
            raise ValueError('Repeat artifact checksum mismatch: '+str(p))
    return read(output/'contract.json')


def contract_for(c, seed, stage):
    import hashlib, json
    plan = open_plan(c, seed)
    return dict(stage=stage, seed=seed, **identity(c),
        schedule_sha256=hashlib.sha256(json.dumps(plan['schedule'], separators=(',', ':')).encode()).hexdigest(),
        inference='RGB -> frozen DINO -> spatial OR channel adapter -> frozen BoQ; no CLIP/adversary',
        selection='Fixed final epoch5/2560 steps; no best-epoch/seed selection; paired shared frozen encoding')


def build(c, seed):
    seed_all(seed)
    official = load_official(path(c, 'official_cache'), path(c, 'dino_source'), c['device'])
    return {a: m.to(c['device']) for a, m in paired_models(official, c['hidden'], c['adapter_gain']).items()}


def common_initial(models):
    s, c = models['spatial'].adapter, models['channel'].adapter
    return (state_hash(s.down) == state_hash(c.down) and state_hash(s.up) == state_hash(c.up)
        and torch.equal(s.spatial.weight[:, :, 1:2, 1:2], c.spatial.weight)
        and torch.equal(s.spatial.bias, c.spatial.bias))


def train(c, a):
    from src.losses.vpr_losses import VPRLossFunction
    seed, stage = a.seed, a.command
    plan = open_plan(c, seed)
    if stage == 'train':
        smoke = path(c, 'runs')/'smoke_seed42'
        if verified(smoke) != contract_for(c, 42, 'smoke') or not all(read(smoke/'summary.json')['checks'].values()):
            raise ValueError('Paired smoke must pass first')
    output = path(c, 'runs')/('smoke_seed42' if stage == 'smoke' else f'train_seed{seed}')
    if stage == 'smoke' and seed != 42: raise ValueError('Only registered seed42 smoke')
    contract = contract_for(c, seed, stage)
    with owned_run(output, contract, a.resume):
        if (output/'completed.json').exists(): verified(output); print('Already complete', flush=True); return
        key.gpu(c); models = build(c, seed); visual = models['spatial'].visual
        if not common_initial(models): raise ValueError('Paired shared initialization differs')
        active = {arm: {n: p for n, p in m.named_parameters() if p.requires_grad} for arm, m in models.items()}
        if any(not p or any(not n.startswith('adapter.') for n in p) for p in active.values()):
            raise ValueError('Only new adapter parameters may train')
        if set(map(id, active['spatial'].values())) & set(map(id, active['channel'].values())):
            raise ValueError('Student parameters must be independent')
        initial = dict(official=state_hash(visual), adapters={arm: state_hash(m.adapter) for arm, m in models.items()},
            inventory=adapter_inventory(models), common_initial=True)
        if (output/'initial.json').exists() and read(output/'initial.json') != initial: raise ValueError('Initial states differ')
        write(output/'initial.json', initial)
        pixels = PlaceSteps(c, plan, plan['schedule'][:1])[0][0][:2].to(c['device'])
        with torch.no_grad():
            reference = visual(pixels)
            errors = {arm: float((m(pixels)-reference).abs().max()) for arm, m in models.items()}
        if max(errors.values()) > 1e-6: raise ValueError('Zero start differs from official')
        del pixels, reference
        optimizers = {arm: torch.optim.AdamW(p.values(), lr=c['learning_rate'], weight_decay=0.) for arm, p in active.items()}
        state = dict(step=0, history=[], contract_sha256=sha(output/'contract.json'),
                     gradients={arm: dict(up=False, inner=False) for arm in ARMS})
        if (output/'last.pt').exists():
            saved = torch.load(output/'last.pt', map_location='cpu', weights_only=True)
            if saved['state']['contract_sha256'] != state['contract_sha256']: raise ValueError('Resume contract differs')
            for arm in ARMS:
                models[arm].adapter.load_state_dict(saved['adapters'][arm], strict=True)
                optimizers[arm].load_state_dict(saved['optimizers'][arm])
            state = saved['state']; restore_rng(saved['rng'])
        else: seed_all(seed)
        steps = plan['schedule'][:c['smoke_steps']] if stage == 'smoke' else plan['schedule']
        if not 0 <= state['step'] <= len(steps): raise ValueError('Invalid resumed step')

        def save():
            save_torch(output/'last.pt', dict(adapters={arm: m.adapter.state_dict() for arm, m in models.items()},
                optimizers={arm: opt.state_dict() for arm, opt in optimizers.items()}, state=state, rng=rng_state()))

        if not (output/'last.pt').exists(): save()
        loader = DataLoader(PlaceSteps(c, plan, steps[state['step']:]), batch_size=None, num_workers=c['workers'],
                            generator=torch.Generator().manual_seed(seed))
        lossfn = VPRLossFunction()
        for m in models.values(): m.train()
        for images, labels, coords, _ in tqdm(loader, total=len(steps), initial=state['step'], desc=f'paired seed{seed} {stage}'):
            started = time.monotonic(); labels = labels.to(c['device']); chunks, bases = [], []
            with torch.no_grad():
                for start in range(0, len(images), c['microbatch']):
                    f = visual.backbone(images[start:start+c['microbatch']].to(c['device']))
                    chunks.append(f); bases.append(visual.aggregator(native_feature_layout(f))[0])
            features = torch.cat(chunks); base = torch.cat(bases); del bases
            row = {}
            for arm, m in models.items():
                opt = optimizers[arm]; opt.zero_grad(set_to_none=True)
                with torch.no_grad(): zs = [m.describe(f)[0] for f in chunks]
                z = torch.cat(zs).detach().requires_grad_(); del zs
                retrieval = gps_retrieval(lossfn, z, labels, coords)
                keep = (1-(z*base).sum(1)).clamp_min(0).mean()
                loss = retrieval + c['keep_weight']*keep
                if not torch.isfinite(loss): raise ValueError('Nonfinite objective')
                loss.backward()
                replay_backward(m, features, z.grad, torch.zeros(len(features), 768, device=c['device']), c['microbatch'])
                for n, p in active[arm].items():
                    if p.grad is not None and bool((p.grad != 0).any()):
                        state['gradients'][arm]['up' if n.startswith('adapter.up.') else 'inner'] = True
                norm = torch.nn.utils.clip_grad_norm_(active[arm].values(), c['gradient_clip'], error_if_nonfinite=True)
                opt.step()
                row[arm] = dict(retrieval=float(retrieval.detach()), keep=float(keep.detach()), gradient_norm=float(norm))
                del z, retrieval, keep, loss
            torch.cuda.synchronize(); state['step'] += 1
            row.update(step=state['step'], seconds=time.monotonic()-started, peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
            state['history'].append(row)
            write(output/'progress.json', dict(phase='training', seed=seed, total=len(steps), **row))
            if state['step'] % c['checkpoint_every'] == 0: save()
            del chunks, features, base
        save()
        checks = dict(zero_start=max(errors.values()) <= 1e-6, complete_steps=state['step'] == len(steps),
            official_unchanged=state_hash(visual) == initial['official'],
            both_adapters_updated=all(state_hash(m.adapter) != initial['adapters'][arm] for arm, m in models.items()),
            gradients_connected=all(all(v.values()) for v in state['gradients'].values()),
            innovation1_unchanged=key.source_hashes(key.PROTECTED) == contract['protected_sources'])
        write(output/'summary.json', dict(seed=seed, steps=state['step'], checks=checks, zero_start_errors=errors,
            adapter_sha256={arm: state_hash(m.adapter) for arm, m in models.items()},
            verdict=('SMOKE_PASS_DISCARD' if stage == 'smoke' else 'TRAINED_NOT_EVALUATED') if all(checks.values()) else 'MECHANICAL_FAIL'))
        if not all(checks.values()): raise ValueError('Mechanical failure; artifacts retained')
        finish(output); print(stage, 'complete', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('smoke', 'train', 'eval', 'aggregate'))
    p.add_argument('--config', type=Path, default=ROOT/'config/spatial_adapter_repeat.yaml')
    p.add_argument('--seed', type=int, choices=(42, 43, 44), default=42)
    p.add_argument('--resume', action='store_true'); p.add_argument('--allow-training', action='store_true')
    p.add_argument('--evaluate', action='store_true')
    a = p.parse_args()
    if a.command in ('smoke', 'train') and not a.allow_training: p.error('Need --allow-training')
    if a.command in ('eval', 'aggregate') and not a.evaluate: p.error('Need --evaluate')
    c = configuration(a.config)
    if a.command in ('smoke', 'train'): train(c, a)
    else:
        from scripts.eval_spatial_adapter_repeat import evaluate, aggregate
        (evaluate if a.command == 'eval' else aggregate)(c, a)


if __name__ == '__main__': main()
