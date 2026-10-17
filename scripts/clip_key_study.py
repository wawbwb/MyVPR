"""Isolated, matched CLIP-key BoQ pilot. Training requires explicit consent.

No automatic benchmark evaluation; no blind test-set access. Original DINO
and BoQ are frozen. Existing innovation-1 and failed region artifacts are read
only. Reuses sealed GSV split/views and identical five-epoch batch schedule.
"""
import argparse
from importlib import metadata
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from scripts.semantic_region_study import ImageRows, PlaceSteps
from src.models.clip_dense_keys import DenseClipKeys
from src.models.clip_key_boq import ARMS, ClipKeyBoQ, permute_patches
from src.models.official_boq import load_official, verify_download, WEIGHT_SHA
from src.models.sclip_region_teacher import CLIP_SHA256
from src.semantic_region_protocol import (gps_retrieval, owned_run, read, restore_rng,
    rng_state, save_torch, schedule, seed_all, sha, state_hash, verify as verify_old_plan, write)

SCHEMA = 'clip_key_boq_v1'
SOURCES = ('src/models/clip_key_boq.py', 'src/models/clip_dense_keys.py',
    'scripts/clip_key_study.py', 'scripts/launch_clip_key_study.py',
    'config/clip_key_boq.yaml', 'tests/test_clip_key_boq.py',
    'src/models/official_boq.py', 'src/models/sclip_region_teacher.py',
    'src/semantic_region_protocol.py', 'scripts/semantic_region_study.py',
    'src/losses/vpr_losses.py')
PROTECTED = ('src/models/conditional_complement.py', 'src/conditional_complement_protocol.py',
    'scripts/conditional_complement.py', 'src/models/hcen_ablation.py',
    'src/models/hcen_vpr.py', 'src/models/conv_residual_ablation.py',
    'src/models/official_boq.py')


def configuration(path):
    import yaml
    c = yaml.safe_load(path.read_text(encoding='utf-8'))
    if c['schema'] != SCHEMA or c['precision'] != 'fp32' or c['student_size'] != 322 or c['views'] != 4:
        raise ValueError('Expected sealed FP32/322/four-view pilot')
    for key in ('seed', 'train_places', 'places_per_batch', 'epochs', 'hidden', 'microbatch',
                'cache_batch', 'cache_flush_rows', 'smoke_steps', 'checkpoint_every', 'workers'):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError('Positive integer required: ' + key)
    for key in ('bias_cap', 'key_temperature', 'learning_rate', 'gradient_clip', 'cuda_memory_fraction'):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError('Positive finite scale required: ' + key)
    if c['cuda_memory_fraction'] > 1 or c['train_places'] % c['places_per_batch'] or c['cache_flush_rows'] % c['cache_batch']:
        raise ValueError('Invalid memory fraction or batch divisibility')
    return c


def path(c, key):
    p = Path(c[key]).expanduser()
    return p if p.is_absolute() else ROOT / p


def source_hashes(names):
    return {n: hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for n in names}


def finish(output):
    files = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*'))
             if p.is_file() and p.name not in ('.lock', 'completed.json') and not p.name.endswith('.tmp')}
    write(output/'completed.json', dict(complete=True, schema=SCHEMA, files=files))


def verified(output):
    result = read(output/'completed.json')
    if result.get('schema') != SCHEMA or result.get('complete') is not True:
        raise ValueError('Incomplete/wrong-schema artifact: ' + str(output))
    for name, digest in result['files'].items():
        target = (output/name).resolve()
        if not target.is_relative_to(output.resolve()) or not target.is_file() or sha(target) != digest:
            raise ValueError('Artifact checksum mismatch: ' + str(target))
    return read(output/'contract.json')


def open_plan(c):
    verify_old_plan(path(c, 'plan'))
    plan = read(path(c, 'plan')/'plan.json')
    n = c['train_places']
    expected = schedule(n, c['epochs'], c['places_per_batch'], c['seed'])
    if (plan['train_rows'] != n * 4 or plan['schedule'] != expected
            or any(p['split'] != 'train' or len(p['rows']) != 4 for p in plan['places'][:n])):
        raise ValueError('Sealed GSV training plan differs')
    train = {p['place'] for p in plan['places'][:n]}
    holdout = {p['place'] for p in plan['places'][n:]}
    if train & holdout or len(train) != n:
        raise ValueError('Place split overlap/duplicates')
    if any(r['split'] != 'train' for r in plan['images'][:plan['train_rows']]):
        raise ValueError('Nontraining image in key cache population')
    return plan


def identity(c):
    import open_clip
    dino = path(c, 'dino_source')
    clip = Path(open_clip.__file__).parent
    if sha(path(c, 'clip_checkpoint')) != CLIP_SHA256:
        raise ValueError('CLIP checkpoint differs')
    return dict(schema=SCHEMA, config=c, code=source_hashes(SOURCES),
        protected_sources=source_hashes(PROTECTED),
        official=verify_download(path(c, 'official_cache')),
        dino_code={p.relative_to(dino).as_posix(): sha(p) for p in sorted(dino.rglob('*.py'))},
        clip_code={p.relative_to(clip).as_posix(): sha(p) for p in sorted(clip.rglob('*.py'))},
        versions={n: metadata.version(n) for n in ('torch','torchvision','open_clip_torch','numpy',
                                                   'pillow','pytorch-metric-learning')},
        clip_sha256=CLIP_SHA256, plan_complete_sha256=sha(path(c,'plan')/'completed.json'),
        plan_sha256=sha(path(c,'plan')/'plan.json'))


def gpu(c):
    if not torch.cuda.is_available() or c['device'] != 'cuda:0':
        raise ValueError('Expose physical GPU1 before starting process; use logical cuda:0')
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('This pilot is restricted to physical GPU1')
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(c['cuda_memory_fraction'], 0)


def validate_keys(value, count):
    if value.shape != (count,512,14,14) or value.dtype != np.float16 or not np.isfinite(value).all():
        raise ValueError('Invalid continuous CLIP key cache shape/dtype/finiteness')
    if np.max(np.abs(np.linalg.norm(value.astype(np.float32), axis=1)-1)) > .003:
        raise ValueError('Continuous patch vectors must have unit channel norm')


def cache_keys(c, args):
    plan = open_plan(c)
    contract = dict(stage='continuous_key_cache', **identity(c),
        representation='SCLIP-equation continuous projected512 channels BEFORE class/text softmax',
        inference='Frozen CLIP remains required; this training-only cache is an efficiency aid')
    output = path(c, 'key_cache')
    with owned_run(output, contract, args.resume):
        if (output/'completed.json').exists():
            verified(output)
            print('Continuous key cache already complete', flush=True)
            return
        gpu(c)
        count = plan['train_rows']
        target = output/'keys.npy'
        temporary = output/'keys.partial.npy'
        cursor = read(output/'cursor.json') if (output/'cursor.json').exists() else dict(next=0, chunks=[])
        if not 0 <= cursor['next'] <= count:
            raise ValueError('Invalid key-cache resume cursor')
        if target.exists():
            if cursor['next'] != count:
                raise ValueError('Final array with incomplete cursor')
            temporary = target
        if temporary.exists():
            array = np.load(temporary, mmap_mode='r+', allow_pickle=False)
            if array.shape != (count,512,14,14) or array.dtype != np.float16:
                raise ValueError('Existing key-cache array layout differs')
        else:
            if cursor['next']:
                raise ValueError('Missing partially completed key array')
            array = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float16, shape=(count,512,14,14))
        end = 0
        for chunk in cursor['chunks']:
            start, stop = chunk['start'], chunk['stop']
            if start != end or not start < stop <= cursor['next']:
                raise ValueError('Invalid cache chunk inventory')
            value = np.asarray(array[start:stop])
            validate_keys(value, stop-start)
            if hashlib.sha256(value.tobytes()).hexdigest() != chunk['sha256']:
                raise ValueError('Damaged existing key rows; refusing silent overwrite')
            end = stop
        if end != cursor['next']:
            raise ValueError('Incomplete cache chunk inventory')
        rows = plan['images'][:count]
        dataset = ImageRows(path(c,'dataset_root'), rows, 224, teacher=True)
        teacher = DenseClipKeys(path(c,'clip_checkpoint'), c['device'])
        for start in tqdm(range(cursor['next'],count,c['cache_flush_rows']), desc='Continuous CLIP keys (training only)'):
            stop = min(start+c['cache_flush_rows'],count)
            loader = DataLoader(Subset(dataset,range(start,stop)), batch_size=c['cache_batch'],
                                num_workers=c['workers'], shuffle=False)
            parts = []
            for images, _ in loader:
                parts.append(teacher(images.to(c['device'])).cpu().numpy().astype(np.float16))
            value = np.concatenate(parts)
            validate_keys(value, stop-start)
            array[start:stop] = value
            array.flush()
            cursor['chunks'].append(dict(start=start,stop=stop,sha256=hashlib.sha256(value.tobytes()).hexdigest()))
            cursor['next'] = stop
            write(output/'cursor.json', cursor)
            write(output/'progress.json', dict(phase='cache',images=stop,total=count,training_started=False,
                peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20))
        del array
        if temporary != target:
            temporary.replace(target)
        write(output/'summary.json',dict(images=count,bytes=count*512*14*14*2,
            grid=[14,14],channels=512,source='Continuous CLIP, no text prompts/class labels',
            teacher_accuracy='Not a validated segmentation/region ground truth',
            peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20))
        finish(output)
        print('Continuous key cache sealed; no student updates yet', flush=True)


def contract_for(c, arm, stage):
    return dict(stage=stage,arm=arm,**identity(c),
        key_cache_complete_sha256=sha(path(c,'key_cache')/'completed.json'),
        inference='Frozen DINO + frozen continuous CLIP -> CLIP-key biased frozen BoQ -> single global descriptor',
        training='Only added query/key projections and bounded gates; no auxiliary semantic loss',
        selection='Fixed final epoch; no benchmark/best-epoch selection; one-seed pilot')


def make_model(c):
    seed_all(c['seed'])
    official = load_official(path(c,'official_cache'),path(c,'dino_source'),c['device'])
    model = ClipKeyBoQ(official, c['hidden'], c['bias_cap'], c['key_temperature'], c['seed']).to(c['device'])
    seed_all(c['seed'])
    return model


def key_rows(array, ids, plan, arm, seed, device):
    # Advanced indexing copies only the current microbatch, not the full 6-GiB file.
    value = torch.from_numpy(np.array(array[ids], copy=True)).float().to(device)
    if arm == 'shuffled':
        value = permute_patches(value, [plan['images'][i]['path'] for i in ids], seed)
    return value


def train(c, args):
    from src.losses.vpr_losses import VPRLossFunction
    plan = open_plan(c)
    key_contract = verified(path(c,'key_cache'))
    if any(key_contract[k] != v for k,v in identity(c).items()):
        raise ValueError('Key cache code/config/data/teacher differs')
    stage, arm = args.command, args.arm
    output = path(c,'runs')/(('smoke_' if stage == 'smoke' else '')+arm)
    contract = contract_for(c,arm,stage)
    if stage == 'train':
        for name in ARMS:
            smoke = path(c,'runs')/('smoke_'+name)
            if verified(smoke) != contract_for(c,name,'smoke') or not all(read(smoke/'summary.json')['checks'].values()):
                raise ValueError('All three current-code smoke tests must pass first')
    with owned_run(output,contract,args.resume):
        if (output/'completed.json').exists():
            verified(output)
            print('Already complete',flush=True)
            return
        gpu(c)
        model = make_model(c)
        array = np.load(path(c,'key_cache')/'keys.npy',mmap_mode='r',allow_pickle=False)
        if array.shape != (plan['train_rows'],512,14,14) or array.dtype != np.float16:
            raise ValueError('Wrong key-cache population/layout')
        params = {n:p for n,p in model.named_parameters() if p.requires_grad}
        if not params or any(not n.startswith('routers.') for n in params):
            raise ValueError('Only new routers may train')
        before = state_hash(model.visual)
        initial = dict(official_initial_sha256=before,router_initial_sha256=state_hash(model.routers),
            trainable_parameters=sum(p.numel() for p in params.values()),official_weight_sha256=WEIGHT_SHA)
        if (output/'initial.json').exists() and read(output/'initial.json') != initial:
            raise ValueError('Initial model changed')
        write(output/'initial.json',initial)
        # Real official kernel/layout equivalence in inference and gradient modes.
        probe,_,_,probe_ids = PlaceSteps(c,plan,plan['schedule'][:1])[0]
        probe, probe_ids = probe[:2].to(c['device']), probe_ids[:2].tolist()
        clip = key_rows(array,probe_ids,plan,arm,c['seed'],c['device'])
        model.eval()
        with torch.inference_mode():
            reference = model.visual(probe)
            error = float((reference-model(probe,clip,arm)).abs().max())
        model.train()
        error = max(error,float((reference-model(probe,clip,arm)).detach().abs().max()))
        if error > 1e-6:
            raise ValueError(f'Official zero-start not reproduced: {error}')
        del probe, clip, reference
        optimizer = torch.optim.AdamW(params.values(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
        state = dict(step=0,contract_sha256=sha(output/'contract.json'),history=[],
                     gate_gradient=False,projection_gradient=False,nonzero_steps=0)
        if (output/'last.pt').exists():
            saved = torch.load(output/'last.pt',map_location='cpu',weights_only=True)
            if saved['state']['contract_sha256'] != state['contract_sha256'] or set(saved['parameters']) != set(params):
                raise ValueError('Resume contract/parameter inventory differs')
            with torch.no_grad():
                for n,p in params.items():
                    p.copy_(saved['parameters'][n].to(p.device))
            optimizer.load_state_dict(saved['optimizer'])
            state = saved['state']
            restore_rng(saved['rng'])
        else:
            seed_all(c['seed'])
        steps = plan['schedule'][:c['smoke_steps']] if stage == 'smoke' else plan['schedule']
        if not 0 <= state['step'] <= len(steps):
            raise ValueError('Invalid resumed training cursor')

        def save():
            save_torch(output/'last.pt',dict(parameters={n:p.detach().cpu() for n,p in params.items()},
                optimizer=optimizer.state_dict(),state=state,rng=rng_state()))

        if not (output/'last.pt').exists():
            save()
        loader = DataLoader(PlaceSteps(c,plan,steps[state['step']:]),batch_size=None,num_workers=c['workers'],
                            generator=torch.Generator().manual_seed(c['seed']))
        lossfn = VPRLossFunction()
        bar = tqdm(loader,total=len(steps),initial=state['step'],desc=f'{arm} {stage}')
        for images,labels,coordinates,ids in bar:
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            descriptors, diagnostics = [], []
            for start in range(0,len(images),c['microbatch']):
                stop = min(start+c['microbatch'],len(images))
                keys = key_rows(array,ids[start:stop].tolist(),plan,arm,c['seed'],c['device'])
                features = model.encode(images[start:stop].to(c['device']))
                value,stats = model.aggregate(features,keys,arm,auxiliary=True)
                descriptors.append(value)
                diagnostics.append(stats)
            descriptor = torch.cat(descriptors)
            if not torch.isfinite(descriptor).all() or float((descriptor.detach().norm(dim=1)-1).abs().max()) > 1e-4:
                raise ValueError('Invalid descriptors')
            loss = gps_retrieval(lossfn,descriptor,labels.to(c['device']),coordinates)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite retrieval loss')
            loss.backward()
            for n,p in params.items():
                if p.grad is not None and bool((p.grad != 0).any()):
                    if n.endswith('.beta'):
                        state['gate_gradient'] = True
                    else:
                        state['projection_gradient'] = True
            norm = torch.nn.utils.clip_grad_norm_(params.values(),c['gradient_clip'],error_if_nonfinite=True)
            state['nonzero_steps'] += int(float(norm) > 0)
            optimizer.step()
            torch.cuda.synchronize()
            state['step'] += 1
            bias_mean,bias_max,beta = torch.stack(diagnostics).mean(0).cpu().tolist()
            row = dict(step=state['step'],loss=float(loss.detach()),gradient_norm=float(norm),
                bias_abs_mean=bias_mean,bias_max=bias_max,beta_tanh_max=beta,
                seconds=time.monotonic()-started,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
            state['history'].append(row)
            write(output/'progress.json',dict(phase='training',arm=arm,total=len(steps),**row))
            bar.set_postfix(loss=f'{row["loss"]:.5f}',bias=f'{bias_mean:.4g}',active=state['nonzero_steps'])
            if state['step'] % c['checkpoint_every'] == 0:
                save()
        save()
        # Sensitivity is not a fair trained ablation; it only detects disconnection.
        probe,_,_,probe_ids = PlaceSteps(c,plan,steps[:1])[0]
        probe = probe[:2].to(c['device'])
        keys = key_rows(array,probe_ids[:2].tolist(),plan,'aligned',c['seed'],c['device'])
        with torch.inference_mode():
            features = model.encode(probe)
            trained_keys = permute_patches(keys,[plan['images'][i]['path'] for i in probe_ids[:2].tolist()],c['seed']) if arm == 'shuffled' else keys
            trained = model.aggregate(features,trained_keys,arm)
            drift = float((trained-model.visual(probe)).abs().max())
            wrong = permute_patches(keys,[plan['images'][i]['path'] for i in probe_ids[:2].tolist()],c['seed'])
            sensitivity = float((model.aggregate(features,keys,'aligned')-
                                 model.aggregate(features,wrong,'shuffled')).abs().max())
        checks = dict(zero_start=error <= 1e-6,frozen_entire_official_unchanged=state_hash(model.visual) == before,
            gate_gradient=state['gate_gradient'],projection_gradient=state['projection_gradient'],
            descriptor_drift=drift > 1e-8,semantic_key_sensitivity=sensitivity > 1e-8,
            complete_steps=state['step'] == len(steps),
            protected_sources_unchanged=source_hashes(PROTECTED) == contract['protected_sources'])
        write(output/'summary.json',dict(arm=arm,steps=state['step'],checks=checks,zero_start_max_abs_error=error,
            descriptor_max_abs_drift=drift,aligned_vs_shuffled_sensitivity=sensitivity,
            nonzero_gradient_steps=state['nonzero_steps'],zero_gradient_fraction=1-state['nonzero_steps']/state['step'],
            final_router_sha256=state_hash(model.routers),peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
            verdict=('SMOKE_PASS_UPDATES_DISCARDED' if stage == 'smoke' else 'TRAINED_NOT_EVALUATED')
                if all(checks.values()) else 'MECHANICAL_FAILURE',
            scope='Exploratory one-seed pilot, no test-set access or demonstrated retrieval benefit'))
        if not all(checks.values()):
            raise ValueError('Mechanical checks failed; checkpoint/report retained; do not proceed')
        finish(output)
        print(stage,'complete; benchmark evaluation NOT started',flush=True)


def preflight(c,args):
    plan = open_plan(c)
    value = identity(c)
    print(json.dumps(dict(status='FILES_READY_NOT_MODEL_TESTED',schema=SCHEMA,
        train_images=plan['train_rows'],steps=len(plan['schedule']),
        new_key_cache_gib=plan['train_rows']*512*14*14*2/2**30,
        protected_sources=value['protected_sources'],physical_gpu=1,
        inference_requires_clip=True,training_started=False),indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('preflight','cache','smoke','train'))
    parser.add_argument('--config',type=Path,default=ROOT/'config/clip_key_boq.yaml')
    parser.add_argument('--arm',choices=ARMS,default='aligned')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--allow-training',action='store_true')
    args = parser.parse_args()
    if args.command in ('smoke','train') and not args.allow_training:
        parser.error('Real updates require --allow-training, including smoke')
    c = configuration(args.config)
    {'preflight':preflight,'cache':cache_keys,'smoke':train,'train':train}[args.command](c,args)


if __name__ == '__main__':
    main()
