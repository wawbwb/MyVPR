"""Prepared four-arm study. Explicit authorization AND passed teacher required.

No launcher is provided. Default command is plan (CPU/file-only). The current
v1 teacher audit is NOT PASS, so this entry refuses smoke/train before CUDA.
"""
import argparse
import hashlib
import json
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

from scripts import clip_key_study as key
from scripts import cv_slot_teacher_audit as audit
from scripts.semantic_region_study import PlaceSteps
from src.cv_slot_protocol import correspondence
from src.models.cv_slot_boq import (ARMS, CVSlotBoQ, correspondence_loss,
    slot_gram_loss, replay_backward)
from src.models.official_boq import load_official, native_feature_layout
from src.semantic_region_protocol import (owned_run, read, write, sha, save_torch,
    seed_all, state_hash, gps_retrieval, rng_state, restore_rng)

SCHEMA = 'cv_slot_boq_v1'
SOURCES = ('src/models/cv_slot_boq.py', 'scripts/cv_slot_study.py', 'config/cv_slot_boq.yaml',
    'tests/test_cv_slot_boq.py')


def configuration(path):
    import yaml
    c = yaml.safe_load(path.read_text(encoding='utf-8'))
    locked = dict(schema=SCHEMA, seed=42, train_places=8192, places_per_batch=16,
        views=4, epochs=5, student_size=322, hidden=64, adapter_gain=.1, bias_cap=2.,
        grid=14, topk=8, dino_temperature=.1, clip_temperature=.2, row_peak_floor=.25,
        learning_rate=.0003, keep_weight=.1, gram_weight=.05, correspondence_weight=.1,
        gradient_clip=1., microbatch=2, workers=2, checkpoint_every=16, smoke_steps=8,
        device='cuda:0', cuda_memory_fraction=.2, precision='fp32')
    if any(c.get(k) != v for k, v in locked.items()):
        raise ValueError('Prepared matched-arm protocol differs')
    return c


def require_teacher(c):
    output = key.path(c, 'teacher_audit')
    contract = audit.verify(output)
    if contract['code'] != key.source_hashes(audit.SOURCES):
        raise ValueError('Teacher audit implementation differs from sealed version')
    summary = read(output/'summary.json')
    if (summary.get('verdict') != 'PASS'
            or not all(summary['partitions']['holdout']['checks'].values())):
        raise ValueError('Teacher gate NOT PASS: '+str(summary.get('verdict'))+
            '. Student training is blocked; no force/ignore-gate option.')
    if (contract['plan_sha256'] != sha(key.path(c, 'plan')/'plan.json')
            or contract['config']['topk'] != c['topk']
            or contract['config']['dino_temperature'] != c['dino_temperature']
            or contract['config']['clip_temperature'] != c['clip_temperature']):
        raise ValueError('Teacher/data/temperatures differ')
    return dict(completed_sha256=sha(output/'completed.json'), contract_sha256=sha(output/'contract.json'))


def prepared(c):
    plan = key.open_plan(c)
    teacher = key.path(c, 'teacher_audit')
    exists = (teacher/'completed.json').is_file()
    if exists:
        audit.verify(teacher)
    status = read(teacher/'summary.json')['verdict'] if exists else 'NOT_RUN'
    return dict(schema=SCHEMA, status='CODE_PREPARED_NOT_TRAINED', teacher_verdict=status,
        training_blocked=status != 'PASS', arms=list(ARMS),
        steps_per_arm=len(plan['schedule']), final_epoch=c['epochs'],
        inference='RGB -> frozen DINO -> channel adapter + visual BoQ routing -> one12288D descriptor',
        no_clip_at_inference=True, no_auto_launch=True,
        note='No CUDA, extraction, smoke updates or training are executed by plan.')


def identity(c):
    result = key.identity(c)
    result.update(schema=SCHEMA, code=key.source_hashes(SOURCES),
        frozen_helpers=key.source_hashes((*key.SOURCES, *audit.SOURCES,
            'scripts/cv_slot_teacher_audit.py')))
    return result


def build(c):
    official = load_official(key.path(c, 'official_cache'), key.path(c, 'dino_source'), c['device'])
    models = {}
    for arm in ARMS:
        seed_all(c['seed'])
        models[arm] = CVSlotBoQ(official, c['hidden'], c['adapter_gain'], c['bias_cap'], c['grid']).to(c['device'])
    return models


def student_hash(model):
    return dict(adapter=state_hash(model.adapter), routers=state_hash(model.routers))


def make_graphs(features, keys, ids, c):
    if len(features) % 4 or len(features) != len(ids) or len(keys) != len(features):
        raise ValueError('Need contiguous four views per place')
    dino = F.normalize(F.adaptive_avg_pool2d(features.detach(), (c['grid'], c['grid']))
                       .flatten(2).transpose(1, 2), dim=-1)
    clip = keys.detach().flatten(2).transpose(1, 2)
    def graph(a, b):
        # Global image row IDs fix the permutation independently of batch order.
        seed = int.from_bytes(hashlib.sha256(f"{c['seed']}:{int(ids[a])}:{int(ids[b])}".encode()).digest()[:8], 'little')
        return correspondence(dino[a], dino[b], clip[a], clip[b], topk=c['topk'],
            dino_temperature=c['dino_temperature'], clip_temperature=c['clip_temperature'], seed=seed)
    return [(a, b, graph(a, b), graph(b, a)) for start in range(0, len(features), 4)
            for a, b in ((start, start+1), (start+2, start+3))]


def train(c, a):
    # Both checks precede CUDA initialization, output creation, and model loads.
    if not a.allow_training:
        raise ValueError('Explicit --allow-training is required even for smoke')
    teacher = require_teacher(c)
    plan = key.open_plan(c)
    cache = key.path(c, 'key_cache')
    key_contract = key.verified(cache)
    if (key_contract['plan_sha256'] != sha(key.path(c, 'plan')/'plan.json')
            or key_contract['clip_sha256'] != key.identity(c)['clip_sha256']):
        raise ValueError('Existing continuous CLIP cache belongs to different data/weights')
    contract = dict(stage=a.command, **identity(c), teacher_gate=teacher,
        key_cache_complete_sha256=sha(cache/'completed.json'),
        selection='Fixed final epoch5; same initialization/schedule for all4 arms; no best epoch/arm selection')
    output = key.path(c, 'runs')/a.command
    if a.command == 'train':
        smoke = key.path(c, 'runs')/'smoke'
        smoke_contract = read(smoke/'contract.json')
        _verify(smoke)
        if smoke_contract != dict(contract, stage='smoke') or not all(read(smoke/'summary.json')['checks'].values()):
            raise ValueError('Matched smoke must pass first')
    with owned_run(output, contract, a.resume):
        if (output/'completed.json').exists():
            _verify(output)
            print('Already complete', flush=True)
            return
        from src.losses.vpr_losses import VPRLossFunction
        key.gpu(c)
        models = build(c)
        official = models['dino'].visual
        active = {arm: {n: p for n, p in model.named_parameters() if p.requires_grad}
                  for arm, model in models.items()}
        if any(not params or any(not n.startswith(('adapter.', 'routers.')) for n in params)
               for params in active.values()):
            raise ValueError('Only independent new student parameters may train')
        for arm in ARMS:
            for other in ARMS:
                if arm != other and set(map(id, active[arm].values())) & set(map(id, active[other].values())):
                    raise ValueError('Trainable weights shared between arms')
        hashes = {arm: student_hash(m) for arm, m in models.items()}
        if any(v != hashes[ARMS[0]] for v in hashes.values()):
            raise ValueError('Matched initialization differs')
        initial = dict(official=state_hash(official), student=hashes,
            trainable={arm: sum(p.numel() for p in params.values()) for arm, params in active.items()})
        if (output/'initial.json').exists() and read(output/'initial.json') != initial:
            raise ValueError('Initial states changed')
        write(output/'initial.json', initial)
        steps = plan['schedule'][:c['smoke_steps']] if a.command == 'smoke' else plan['schedule']
        with torch.no_grad():
            images = PlaceSteps(c, plan, steps[:1])[0][0][:2].to(c['device'])
            features = official.backbone(images)
            reference = official.aggregator(features)[0]
            errors = {arm: float((model.describe(features)[0]-reference).abs().max()) for arm, model in models.items()}
        if max(errors.values()) > 1e-6:
            raise ValueError('Zero start does not reproduce official descriptor')
        del images, features, reference
        optimizers = {arm: torch.optim.AdamW(params.values(), lr=c['learning_rate'], weight_decay=0.)
                      for arm, params in active.items()}
        state = dict(step=0, history=[], contract_sha256=sha(output/'contract.json'))
        if (output/'last.pt').exists():
            saved = torch.load(output/'last.pt', map_location='cpu', weights_only=True)
            if saved['state']['contract_sha256'] != state['contract_sha256']:
                raise ValueError('Checkpoint contract mismatch')
            for arm, m in models.items():
                m.load_student(saved['students'][arm])
                optimizers[arm].load_state_dict(saved['optimizers'][arm])
            state = saved['state']
            restore_rng(saved['rng'])
        else:
            seed_all(c['seed'])
        if not 0 <= state['step'] <= len(steps):
            raise ValueError('Invalid checkpoint step')
        def save():
            save_torch(output/'last.pt', dict(students={arm: m.student_state() for arm, m in models.items()},
                optimizers={arm: opt.state_dict() for arm, opt in optimizers.items()}, state=state, rng=rng_state()))
        if not (output/'last.pt').exists():
            save()
        keys = np.load(cache/'keys.npy', mmap_mode='r', allow_pickle=False)
        if keys.shape != (plan['train_rows'], 512, 14, 14) or keys.dtype != np.float16:
            raise ValueError('Unexpected dense key array')
        loader = DataLoader(PlaceSteps(c, plan, steps[state['step']:]), batch_size=None, num_workers=c['workers'],
                            generator=torch.Generator().manual_seed(c['seed']))
        lossfn = VPRLossFunction()
        for m in models.values():
            m.train()
        for images, labels, coords, ids in tqdm(loader, total=len(steps), initial=state['step'], desc='matched4 '+a.command):
            started = time.monotonic()
            chunks, bases, baseline_slots = [], [], []
            with torch.no_grad():
                for start in range(0, len(images), c['microbatch']):
                    f = official.backbone(images[start:start+c['microbatch']].to(c['device']))
                    z0, s0 = models['dino'].describe(f, baseline=True)
                    chunks.append(f)
                    bases.append(z0)
                    baseline_slots.append(s0)
                features, base, slots0 = torch.cat(chunks), torch.cat(bases), torch.cat(baseline_slots)
                semantics = torch.from_numpy(np.array(keys[np.asarray(ids)], copy=True)).float().to(c['device'])
                graphs = make_graphs(features, semantics, ids, c)
            row = {}
            for arm, model in models.items():
                optimizers[arm].zero_grad(set_to_none=True)
                with torch.no_grad():
                    values = [model.describe(f) for f in chunks]
                z = torch.cat([v[0] for v in values]).detach().requires_grad_()
                slots = torch.cat([v[1] for v in values]).detach().requires_grad_()
                retrieval = gps_retrieval(lossfn, z, labels.to(c['device']), coords)
                keep = (1-(z*base).sum(-1)).clamp_min(0).mean()
                gram = slot_gram_loss(slots, slots0)
                relation = correspondence_loss(slots, graphs, arm, c['row_peak_floor'])
                loss = retrieval+c['keep_weight']*keep+c['gram_weight']*gram+c['correspondence_weight']*relation
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite matched objective')
                loss.backward()
                replay_backward(model, features, z.grad, slots.grad, c['microbatch'])
                norm = torch.nn.utils.clip_grad_norm_(active[arm].values(), c['gradient_clip'], error_if_nonfinite=True)
                optimizers[arm].step()
                row[arm] = dict(retrieval=float(retrieval.detach()), keep=float(keep.detach()),
                    slot_gram=float(gram.detach()), correspondence=float(relation.detach()), gradient_norm=float(norm))
                del values, z, slots, loss, retrieval, keep, gram, relation
            torch.cuda.synchronize()
            state['step'] += 1
            row.update(step=state['step'], seconds=time.monotonic()-started,
                       peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
            state['history'].append(row)
            write(output/'progress.json', dict(phase='training', total=len(steps), **row))
            if state['step'] % c['checkpoint_every'] == 0:
                save()
            del features, base, slots0, semantics, graphs, chunks, bases, baseline_slots
        save()
        checks = dict(complete_steps=state['step'] == len(steps), zero_start=max(errors.values()) <= 1e-6,
            official_unchanged=state_hash(official) == initial['official'],
            innovation1_unchanged=key.source_hashes(key.PROTECTED) == contract['protected_sources'],
            all_students_updated=all(student_hash(m) != initial['student'][arm] for arm, m in models.items()))
        write(output/'summary.json', dict(checks=checks, steps=state['step'], zero_start_errors=errors,
            verdict='TRAINED_NOT_EVALUATED' if a.command == 'train' else 'SMOKE_DISCARD',
            student_hash={arm: student_hash(m) for arm, m in models.items()}))
        if not all(checks.values()):
            raise ValueError('Mechanical checks failed; keep artifacts')
        _finish(output)


def _finish(output):
    files = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*'))
        if p.is_file() and p.name not in ('.lock', 'completed.json') and not p.name.endswith('.tmp')}
    write(output/'completed.json', dict(schema=SCHEMA, complete=True, files=files))


def _verify(output):
    value = read(output/'completed.json')
    if value.get('schema') != SCHEMA or value.get('complete') is not True:
        raise ValueError('Incomplete student run')
    for name, digest in value['files'].items():
        p = (output/name).resolve()
        if not p.is_relative_to(output.resolve()) or not p.is_file() or sha(p) != digest:
            raise ValueError('Student artifact checksum mismatch')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('plan', 'smoke', 'train'), nargs='?', default='plan')
    p.add_argument('--config', type=Path, default=ROOT/'config/cv_slot_boq.yaml')
    p.add_argument('--allow-training', action='store_true')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.command in ('smoke', 'train') and not a.allow_training:
        p.error('Need --allow-training; nothing started')
    c = configuration(a.config)
    if a.command == 'plan':
        print(json.dumps(prepared(c), ensure_ascii=False, indent=2))
    else:
        train(c, a)


if __name__ == '__main__':
    main()
