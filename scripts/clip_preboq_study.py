"""Three matched pre-BoQ adapters; proxy/adversary are training-only.

Independent experiment: never bypasses or modifies the failed global probe.
Reuses sealed train/heldout places and variation targets, not old students.
"""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts import clip_nuisance_study as old
from scripts import clip_nuisance_representation as diagnosis
from src.models.clip_preboq import ARMS, PreBoQ, MeanAdversary, replay_backward
from src.models.clip_nuisance import center_four
from src.models.official_boq import load_official, native_feature_layout
from src.semantic_region_protocol import (owned_run, read, write, sha, state_hash, seed_all,
    save_torch, rng_state, restore_rng, gps_retrieval)

SCHEMA = 'clip_preboq_variation_v1'
SOURCES = ('src/models/clip_preboq.py', 'scripts/clip_preboq_study.py',
    'scripts/eval_clip_preboq.py', 'scripts/launch_clip_preboq.py',
    'config/clip_preboq.yaml', 'tests/test_clip_preboq.py')


def configuration(p):
    import yaml
    c = yaml.safe_load(p.read_text(encoding='utf-8'))
    if (c['schema'] != SCHEMA or c['precision'] != 'fp32' or c['views'] != 4
            or c['student_size'] != 322 or c['device'] != 'cuda:0'):
        raise ValueError('Fixed FP32/322/four-view/GPU1 protocol required')
    for k in ('seed', 'train_places', 'places_per_batch', 'epochs', 'hidden', 'predictor_hidden',
              'target_dim', 'predictor_updates', 'adversarial_ramp_steps', 'microbatch', 'workers',
              'checkpoint_every', 'smoke_steps', 'probe_epochs', 'probe_batch', 'bootstrap_samples'):
        if type(c[k]) is not int or c[k] < 1:
            raise ValueError('Positive integer required: ' + k)
    for k in ('adapter_gain', 'learning_rate', 'predictor_lr', 'keep_weight', 'adversarial_weight',
              'gradient_clip', 'probe_lr', 'cuda_memory_fraction'):
        if not np.isfinite(c[k]) or c[k] <= 0:
            raise ValueError('Positive finite setting required: ' + k)
    if c['cuda_memory_fraction'] > .2 or c['adapter_gain'] > 1:
        raise ValueError('Memory/adapter gain exceeds fixed budget')
    return c


path = old.path


def identity(c):
    s = old.configuration(path(c, 'source_config'))
    d = diagnosis.configuration(path(c, 'diagnostic_config'))
    return dict(schema=SCHEMA, config=c, code=old.source_hashes(SOURCES), source=old.identity(s),
        prepared_sha256=sha(path(s, 'prepared')/'completed.json'),
        diagnostic_sha256=sha(path(d, 'output')/'completed.json'),
        feature_cache_sha256=sha(path(d, 'cache')/'completed.json'))


def prerequisites(c):
    s = old.configuration(path(c, 'source_config'))
    _, targets, _, donors = old.prepared(s)
    d = diagnosis.configuration(path(c, 'diagnostic_config'))
    if diagnosis.verified(path(d, 'cache')) != dict(stage='frozen_feature_extraction', **diagnosis.identity(d)):
        raise ValueError('Frozen feature cache identity differs')
    expected = dict(stage='representation_probes', **diagnosis.identity(d),
                    feature_complete_sha256=sha(path(d, 'cache')/'completed.json'))
    if diagnosis.verified(path(d, 'output')) != expected:
        raise ValueError('Representation diagnostic identity differs')
    report = read(path(d, 'output')/'summary.json')['results']['patch_mean']
    if not report['diagnostic_signal']:
        raise ValueError('Pre-BoQ association gate did not pass')
    means = np.load(path(d, 'cache')/'patch_mean.npy', mmap_mode='r')
    plan = old.open_plan(c)
    scale = center_four(means[:plan['train_rows']]).std(0).clip(1e-4)
    if targets.shape != (len(plan['images']), c['target_dim']):
        raise ValueError('Target population differs')
    return plan, targets, donors, scale


def finish(output):
    files = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*'))
             if p.is_file() and p.name not in ('.lock', 'completed.json') and not p.name.endswith('.tmp')}
    write(output/'completed.json', dict(complete=True, schema=SCHEMA, files=files))


def verified(output):
    value = read(output/'completed.json')
    if value.get('schema') != SCHEMA or value.get('complete') is not True:
        raise ValueError('Incomplete pre-BoQ artifact')
    for n, digest in value['files'].items():
        p = (output/n).resolve()
        if not p.is_relative_to(output.resolve()) or not p.is_file() or sha(p) != digest:
            raise ValueError('Corrupt pre-BoQ artifact: ' + str(p))
    return read(output/'contract.json')


def contract_for(c, arm, stage):
    return dict(stage=stage, arm=arm, **identity(c),
        inference='RGB -> frozen DINO -> trainable residual adapter -> frozen BoQ. No CLIP or adversary.',
        selection='Fixed final epoch5, 2560 updates; exploratory single seed, no best-epoch tuning',
        corruption='Fixed within-place other-view derangement, NOT independent noise; residuals can be anticorrelated')


def build(c):
    seed_all(c['seed'])
    official = load_official(path(c, 'official_cache'), path(c, 'dino_source'), c['device'])
    return PreBoQ(official, hidden=c['hidden'], gain=c['adapter_gain']).to(c['device'])


def train(c, a):
    from src.losses.vpr_losses import VPRLossFunction
    plan, targets, donors, scale = prerequisites(c)
    arm, stage = a.arm, a.command
    if stage == 'train':
        initial_smoke = None
        for name in ARMS:
            p = path(c, 'runs')/('smoke_' + name)
            if verified(p) != contract_for(c, name, 'smoke') or not all(read(p/'summary.json')['checks'].values()):
                raise ValueError('All three mechanical smokes must pass')
            current = read(p/'initial.json')
            if initial_smoke is not None and current != initial_smoke:
                raise ValueError('Three-arm initialization differs')
            initial_smoke = current
    output = path(c, 'runs')/(('smoke_' if stage == 'smoke' else '') + arm)
    contract = contract_for(c, arm, stage)
    with owned_run(output, contract, a.resume):
        if (output/'completed.json').exists():
            verified(output); print('Already complete', flush=True); return
        old.gpu(c)
        model = build(c)
        predictor = MeanAdversary(scale, c['predictor_hidden'], c['target_dim']).to(c['device'])
        active = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if not active or any(not n.startswith('adapter.') for n in active):
            raise ValueError('Only new adapter may train')
        initial = dict(official=state_hash(model.visual), adapter=state_hash(model.adapter),
            predictor=state_hash(predictor), student_parameters=sum(p.numel() for p in active.values()),
            predictor_parameters=sum(p.numel() for p in predictor.parameters()))
        if (output/'initial.json').exists() and read(output/'initial.json') != initial:
            raise ValueError('Initial states changed')
        write(output/'initial.json', initial)
        pixels = old.PlaceSteps(c, plan, plan['schedule'][:1])[0][0][:2].to(c['device'])
        with torch.no_grad():
            error = float((model(pixels) - model.visual(pixels)).abs().max())
        if error > 1e-6:
            raise ValueError('Zero start does not reproduce original')
        del pixels
        optimizer = torch.optim.AdamW(active.values(), lr=c['learning_rate'], weight_decay=0.)
        adversary = torch.optim.AdamW(predictor.parameters(), lr=c['predictor_lr'], weight_decay=0.)
        state = dict(step=0, history=[], contract_sha256=sha(output/'contract.json'),
                     up_gradient=False, inner_gradient=False, adversarial_gradient=False, predictor_gradient=False)
        if (output/'last.pt').exists():
            saved = torch.load(output/'last.pt', map_location='cpu', weights_only=True)
            if saved['state']['contract_sha256'] != state['contract_sha256'] or set(saved['student']) != set(active):
                raise ValueError('Resume checkpoint identity differs')
            with torch.no_grad():
                for n, p in active.items(): p.copy_(saved['student'][n].to(p.device))
            predictor.load_state_dict(saved['predictor'], strict=True)
            optimizer.load_state_dict(saved['optimizer']); adversary.load_state_dict(saved['adversary'])
            state = saved['state']; restore_rng(saved['rng'])
        else:
            seed_all(c['seed'])
        steps = plan['schedule'][:c['smoke_steps']] if stage == 'smoke' else plan['schedule']
        if not 0 <= state['step'] <= len(steps): raise ValueError('Bad resumed step')

        def save():
            save_torch(output/'last.pt', dict(student={n: p.detach().cpu() for n, p in active.items()},
                predictor=predictor.state_dict(), optimizer=optimizer.state_dict(), adversary=adversary.state_dict(),
                state=state, rng=rng_state()))

        if not (output/'last.pt').exists(): save()
        loader = DataLoader(old.PlaceSteps(c, plan, steps[state['step']:]), batch_size=None,
            num_workers=c['workers'], generator=torch.Generator().manual_seed(c['seed']))
        lossfn = VPRLossFunction(); model.train()
        for images, labels, coords, ids in tqdm(loader, total=len(steps), initial=state['step'], desc=arm+' '+stage):
            started = time.monotonic(); optimizer.zero_grad(set_to_none=True)
            y = torch.from_numpy(np.asarray(targets[donors[ids.numpy()]] if arm == 'shuffled' else targets[ids.numpy()])).to(c['device'])
            labels = labels.to(c['device'])
            pieces, zs, ms, bs = [], [], [], []
            with torch.no_grad():
                for start in range(0, len(images), c['microbatch']):
                    f = model.encode(images[start:start+c['microbatch']].to(c['device']))
                    z, m = model.describe(f)
                    pieces.append(f); zs.append(z); ms.append(m)
                    bs.append(model.visual.aggregator(native_feature_layout(f))[0])
            # Copy concatenated raw features only; dense features are never persisted.
            features = torch.cat(pieces); del pieces
            z = torch.cat(zs).detach().requires_grad_(); means = torch.cat(ms).detach().requires_grad_()
            base = torch.cat(bs); del zs, ms, bs
            predictor.requires_grad_(True)
            for _ in range(c['predictor_updates']):
                adversary.zero_grad(set_to_none=True)
                prediction_loss = F.mse_loss(predictor(means.detach(), labels), y)
                prediction_loss.backward()
                state['predictor_gradient'] |= any(p.grad is not None and bool((p.grad != 0).any()) for p in predictor.parameters())
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1., error_if_nonfinite=True)
                adversary.step()
            adversary.zero_grad(set_to_none=True); predictor.requires_grad_(False)
            retrieval = gps_retrieval(lossfn, z, labels, coords)
            keep = (1 - (z*base).sum(1)).clamp_min(0).mean()
            nuisance = F.mse_loss(predictor(means, labels, adversarial=True), y)
            coefficient = 0. if arm == 'visual' else c['adversarial_weight'] * min(1., (state['step']+1)/c['adversarial_ramp_steps'])
            if state['step'] == 0:
                g = torch.autograd.grad(nuisance, means, retain_graph=True)[0]
                state['adversarial_gradient'] = bool((g != 0).any())
            loss = retrieval + c['keep_weight']*keep + coefficient*nuisance
            if not torch.isfinite(loss): raise ValueError('Nonfinite objective')
            loss.backward()
            replay_backward(model, features, z.grad, means.grad, c['microbatch'])
            for n, p in active.items():
                if p.grad is not None and bool((p.grad != 0).any()):
                    state['up_gradient' if n.startswith('adapter.up.') else 'inner_gradient'] = True
            norm = torch.nn.utils.clip_grad_norm_(active.values(), c['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); torch.cuda.synchronize(); state['step'] += 1
            row = dict(step=state['step'], retrieval=float(retrieval.detach()), keep=float(keep.detach()),
                predictor_mse=float(prediction_loss.detach()), adversarial_mse=float(nuisance.detach()),
                adversarial_weight=coefficient, gradient_norm=float(norm), seconds=time.monotonic()-started,
                mean_feature_shift_rms=float((means.detach()-features.mean((2, 3))).square().mean().sqrt()),
                peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
            state['history'].append(row)
            write(output/'progress.json', dict(phase='training', arm=arm, total=len(steps), **row))
            if state['step'] % c['checkpoint_every'] == 0: save()
            del features, z, means, base, loss, retrieval, keep, nuisance
        save()
        checks = dict(zero_start=error <= 1e-6, complete_steps=state['step'] == len(steps),
            official_unchanged=state_hash(model.visual) == initial['official'],
            adapter_updated=state_hash(model.adapter) != initial['adapter'],
            up_gradient=state['up_gradient'], inner_gradient=state['inner_gradient'],
            adversarial_path_connected=state['adversarial_gradient'], predictor_gradient=state['predictor_gradient'],
            innovation1_unchanged=old.source_hashes(old.PROTECTED) == contract['source']['base']['protected_sources'])
        write(output/'summary.json', dict(arm=arm, steps=state['step'], checks=checks, zero_start_error=error,
            adapter_sha256=state_hash(model.adapter), clip_required_at_inference=False,
            verdict=('SMOKE_PASS_DISCARD' if stage == 'smoke' else 'TRAINED_NOT_EVALUATED') if all(checks.values()) else 'MECHANICAL_FAIL'))
        if not all(checks.values()): raise ValueError('Mechanical failure; report retained')
        finish(output); print(stage, 'complete', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('smoke', 'train', 'eval'))
    p.add_argument('--config', type=Path, default=ROOT/'config/clip_preboq.yaml')
    p.add_argument('--arm', choices=ARMS, default='aligned')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--allow-training', action='store_true')
    p.add_argument('--evaluate', action='store_true')
    a = p.parse_args()
    if a.command in ('smoke', 'train') and not a.allow_training: p.error('Need --allow-training')
    if a.command == 'eval' and not a.evaluate: p.error('Need --evaluate')
    c = configuration(a.config)
    if a.command == 'eval':
        from scripts.eval_clip_preboq import evaluate
        evaluate(c, a)
    else: train(c, a)


if __name__ == '__main__': main()
