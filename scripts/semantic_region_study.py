"""CLIP-guided spatial region pooling: explicit prepare/cache/smoke/train/eval.

Default is a CPU/file-only preflight. No mode launches training implicitly.
Both smoke (real, discarded updates) and train require --allow-training.
"""
import argparse
from collections import Counter
from importlib import metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

from src.models.official_boq import load_official, verify_download, file_sha, WEIGHT_SHA
from src.models.semantic_region_pool import ARMS, SemanticRegionBoQ, guidance_loss, permute_target
from src.models.sclip_region_teacher import CLASSES, TEMPLATES, CLIP_SHA256, SCLIPRegionTeacher
from src.semantic_region_protocol import (
    SCHEMA, atomic_bytes, code_inventory, gps_retrieval, owned_run, paired, read,
    restore_rng, rng_state, save_npz, save_torch, schedule, seal, seed_all, sha,
    state_hash, verify, write)


def configuration(path):
    import yaml
    c = yaml.safe_load(path.read_text(encoding='utf-8'))
    if c['schema'] != SCHEMA or c['views'] != 4 or c['student_size'] != 322 or c['teacher_size'] != 224:
        raise ValueError('Expected fixed 322-student/224-teacher/four-view protocol')
    if c['precision'] != 'fp32' or c['holdout_places'] < 2:
        raise ValueError('FP32 and a nonempty place holdout are required')
    for key in ('train_places', 'holdout_places', 'places_per_batch', 'epochs',
                'teacher_batch_size', 'teacher_shard_rows', 'hidden', 'microbatch',
                'smoke_steps', 'checkpoint_every'):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError('Positive integer required: '+key)
    if c['train_places'] % c['places_per_batch'] or c['teacher_shard_rows'] % c['teacher_batch_size']:
        raise ValueError('Place/shard batch sizes must divide their populations')
    for key in ('rho', 'guidance_weight', 'pool_lr', 'boq_lr', 'gradient_clip'):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError('Positive finite setting required: '+key)
    return c


def paths(c, key):
    p = Path(c[key]).expanduser()
    return p if p.is_absolute() else ROOT/p


def versions():
    return {name: metadata.version(name) for name in
            ('torch', 'torchvision', 'open_clip_torch', 'numpy', 'pillow', 'pandas', 'pytorch-metric-learning')}


def data_metadata(c):
    files = sorted((paths(c, 'dataset_root')/'Dataframes').glob('*.csv'))
    if not files:
        raise FileNotFoundError('Missing GSV city CSVs')
    return {p.name: sha(p) for p in files}


def identity(c):
    dino = paths(c, 'dino_source')
    source = {p.relative_to(dino).as_posix(): sha(p) for p in sorted(dino.rglob('*.py'))}
    if not source or not (dino/'hubconf.py').is_file():
        raise ValueError('Existing local DINOv2 source required; no network fallback')
    return dict(schema=SCHEMA, config=c, code=code_inventory(ROOT),
        official=verify_download(paths(c, 'official_cache')), dino_source_sha256=source,
        gsv_csv_sha256=data_metadata(c), versions=versions())


def transform(size, teacher=False):
    from torchvision import transforms as T
    mean, std = ((.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711)) if teacher else (
        (.485, .456, .406), (.229, .224, .225))
    # Same full-image square geometry; no crop, flip or stochastic augmentation.
    return T.Compose([T.ToTensor(), T.Resize((size, size),
        interpolation=T.InterpolationMode.BICUBIC, antialias=True), T.Normalize(mean, std)])


def preflight(c, args):
    info = identity(c)
    clip = paths(c, 'clip_checkpoint')
    if not clip.is_file() or file_sha(clip) != CLIP_SHA256:
        raise ValueError('Pinned local CLIP checkpoint is missing/different')
    print(json.dumps(dict(status='PREPARATION_FILES_READY_NOT_MODEL_TESTED',
        versions=info['versions'], official_sha256=WEIGHT_SHA,
        clip_sha256=CLIP_SHA256, city_csvs=len(info['gsv_csv_sha256']),
        region_slots=len(CLASSES)*4,
        teacher_probability_bytes=c['train_places']*c['views']*len(CLASSES)*14*14*2,
        train_steps=c['epochs']*c['train_places']//c['places_per_batch'],
        training_started=False, model_constructed=False,
        note='File checks only. Teacher quality, zero-start and gradients remain untested.'), indent=2))


def image_name(row):
    # Match the existing GSVCitiesDataset original-row filename convention.
    values = [str(row['city_id']), str(int(row['place_id']) % 10**5).zfill(7),
        str(row['year']).zfill(4), str(row['month']).zfill(2), str(row['northdeg']).zfill(3),
        str(row['lat']), str(row['lon']), str(row['panoid'])]
    return '_'.join(values)+'.jpg'


def prepare(c, args):
    import pandas as pd
    from tqdm import tqdm
    contract = dict(stage='plan', **identity(c))
    with owned_run(paths(c, 'plan'), contract, args.resume) as output:
        if (output/'completed.json').exists():
            verify(output)
            print('Plan already complete; no training', flush=True)
            return
        candidates = []
        root = paths(c, 'dataset_root')
        for csv in sorted((root/'Dataframes').glob('*.csv')):
            frame = pd.read_csv(csv)
            for place_id, group in frame.groupby('place_id', sort=True):
                if len(group) < c['views']:
                    continue
                key = f'{csv.stem}:{int(place_id)}'
                selected = []
                for _, row in group.iterrows():
                    relative = (Path('Images')/str(row['city_id'])/image_name(row)).as_posix()
                    order = __import__('hashlib').sha256(f'srp-view:{c["seed"]}:{relative}'.encode()).hexdigest()
                    selected.append((order, relative, [float(row['lat']), float(row['lon'])]))
                selected.sort()
                split_key = __import__('hashlib').sha256(f'srp-split:{c["seed"]}:{key}'.encode()).hexdigest()
                candidates.append((split_key, key, csv.stem, selected[:c['views']]))
        candidates.sort()
        total = c['holdout_places']+c['train_places']
        if len(candidates) < total:
            raise ValueError('Insufficient eligible GSV places')
        populations = {'holdout': candidates[:c['holdout_places']],
                       'train': candidates[c['holdout_places']:total]}
        rows, places = [], []
        # Train rows first, so the teacher only ever sees the first train_rows.
        for split in ('train', 'holdout'):
            for _, key, city, selected in tqdm(populations[split], desc=f'Plan {split} image SHA256'):
                indices = []
                for _, relative, coords in selected:
                    image = (root/relative).resolve()
                    if not image.is_relative_to(root.resolve()) or not image.is_file():
                        raise FileNotFoundError(image)
                    if not np.isfinite(coords).all() or abs(coords[0])>90 or abs(coords[1])>180:
                        raise ValueError('Invalid GSV coordinates')
                    indices.append(len(rows))
                    rows.append(dict(path=relative, place=key, split=split,
                                     coordinates=coords, image_sha256=sha(image)))
                places.append(dict(place=key, split=split, city=city, rows=indices))
        train = {p['place'] for p in places if p['split']=='train'}
        holdout = {p['place'] for p in places if p['split']=='holdout'}
        if train & holdout or len({r['path'] for r in rows}) != len(rows):
            raise ValueError('Image/place overlap in plan')
        plan = dict(places=places, images=rows, train_rows=c['train_places']*c['views'],
                    schedule=schedule(c['train_places'], c['epochs'], c['places_per_batch'], c['seed']))
        write(output/'plan.json', plan)
        write(output/'summary.json', dict(train_places=len(train), holdout_places=len(holdout),
            total_images=len(rows), train_holdout_place_disjoint=True,
            fixed_four_views=True, full_gsv_training=False,
            train_city_counts=dict(Counter(p['city'] for p in places if p['split']=='train')),
            note='GSV place holdout is disjoint from this adaptation, not necessarily official pretraining.'))
        seal(output)
        print('Plan sealed; no training started', flush=True)


def open_plan(c):
    contract = verify(paths(c, 'plan'))
    current = identity(c)
    if any(contract[k] != v for k, v in current.items()):
        raise ValueError('Plan config/code/data/runtime changed; use a new study/config')
    return read(paths(c, 'plan')/'plan.json')


class ImageRows(Dataset):
    def __init__(self, root, rows, size, teacher=False, verify_images=True):
        self.root, self.rows = root, rows
        self.transform = transform(size, teacher)
        self.verify_images = verify_images

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.root/row['path']
        if self.verify_images and sha(path) != row['image_sha256']:
            raise ValueError('Image bytes changed: '+str(path))
        with Image.open(path) as image:
            pixels = self.transform(image.convert('RGB'))
        return pixels, index


def teacher_identity(c):
    import open_clip
    base = Path(open_clip.__file__).parent
    return dict(weight_sha256=CLIP_SHA256, classes=list(CLASSES), templates=list(TEMPLATES),
        temperature_scale=40., native_grid=[14,14], dtype='float16 probabilities',
        implementation='Own OpenCLIP SCLIP-equation adapter, no sliding window/PAMR; not exact official reproduction',
        package_sources={p.relative_to(base).as_posix():sha(p) for p in sorted(base.rglob('*.py'))})


def validate_probability(value, rows):
    if (value.shape != (rows,len(CLASSES),14,14) or value.dtype != np.float16
            or not np.isfinite(value).all() or value.min()<0
            or np.max(np.abs(value.astype(np.float32).sum(1)-1))>.002):
        raise ValueError('Invalid teacher probabilities')


def cache_teacher(c, args):
    from tqdm import tqdm
    plan = open_plan(c)
    if sha(paths(c, 'clip_checkpoint')) != CLIP_SHA256:
        raise ValueError('CLIP checkpoint changed')
    contract = dict(stage='teacher_cache', **identity(c), teacher=teacher_identity(c),
        plan_completed_sha256=sha(paths(c, 'plan')/'completed.json'))
    with owned_run(paths(c, 'teacher_cache'), contract, args.resume) as output:
        if (output/'completed.json').exists():
            verify(output)
            print('Teacher cache already complete; no training', flush=True)
            return
        rows = plan['images'][:plan['train_rows']]
        dataset = ImageRows(paths(c,'dataset_root'), rows,224,teacher=True)
        teacher = None
        histogram = np.zeros(len(CLASSES), np.float64)
        entropy_sum = 0.
        shard_rows = c['teacher_shard_rows']
        for start in tqdm(range(0,len(rows),shard_rows),desc='Frozen SCLIP targets (no training)'):
            stop = min(start+shard_rows,len(rows))
            target, record = output/f'shard_{start//shard_rows:05d}.npz', output/f'shard_{start//shard_rows:05d}.json'
            if record.exists():
                report = read(record)
                if not target.is_file() or sha(target)!=report['sha256'] or report['start']!=start or report['stop']!=stop:
                    raise ValueError('Damaged sealed shard; retained for inspection: '+str(target))
                with np.load(target, allow_pickle=False) as z:
                    probability = z['probabilities']
                    if not np.array_equal(z['indices'],np.arange(start,stop)):
                        raise ValueError('Teacher shard image order differs')
            else:
                if teacher is None:
                    teacher = SCLIPRegionTeacher(paths(c,'clip_checkpoint'),c['device'])
                subset = torch.utils.data.Subset(dataset,range(start,stop))
                loader = DataLoader(subset,batch_size=c['teacher_batch_size'],num_workers=c['workers'],shuffle=False)
                parts = []
                for images, indices in loader:
                    with torch.inference_mode():
                        parts.append(teacher(images.to(c['device'])).cpu().numpy().astype(np.float16))
                probability = np.concatenate(parts)
                validate_probability(probability,stop-start)
                save_npz(target,probabilities=probability,indices=np.arange(start,stop))
                write(record,dict(start=start,stop=stop,sha256=sha(target)))
            validate_probability(probability,stop-start)
            p = probability.astype(np.float32)
            histogram += p.sum((0,2,3))
            entropy_sum += float(-(p*np.log(np.maximum(p,1e-8))).sum())
            write(output/'progress.json',dict(phase='teacher_cache',images=stop,total=len(rows),training_started=False))
        write(output/'summary.json',dict(images=len(rows),native_grid=[14,14],
            class_mass=(histogram/histogram.sum()).tolist(),
            mean_entropy=entropy_sum/(len(rows)*14*14),training_started=False,
            quality_claim='Numerical cache only; segmentation accuracy is NOT validated.'))
        seal(output)
        print('Teacher cache sealed; no student training started',flush=True)


def open_teacher(c):
    contract = verify(paths(c,'teacher_cache'))
    expected = dict(stage='teacher_cache', **identity(c),teacher=teacher_identity(c),
        plan_completed_sha256=sha(paths(c,'plan')/'completed.json'))
    if contract != expected:
        raise ValueError('Teacher cache provenance changed')
    # At ~220 MiB in this pilot, keep targets in RAM instead of per-step ZIP I/O.
    parts = []
    plan = read(paths(c,'plan')/'plan.json')
    for start in range(0,plan['train_rows'],c['teacher_shard_rows']):
        with np.load(paths(c,'teacher_cache')/f'shard_{start//c["teacher_shard_rows"]:05d}.npz',allow_pickle=False) as z:
            value = z['probabilities']
            validate_probability(value,min(c['teacher_shard_rows'],plan['train_rows']-start))
            parts.append(value)
    return torch.from_numpy(np.concatenate(parts))


class PlaceSteps(Dataset):
    def __init__(self,c,plan,steps):
        self.plan,self.steps=plan,steps
        self.images=ImageRows(paths(c,'dataset_root'),plan['images'],c['student_size'])

    def __len__(self):
        return len(self.steps)

    def __getitem__(self,index):
        ids = [i for place in self.steps[index] for i in self.plan['places'][place]['rows']]
        pixels = torch.stack([self.images[i][0] for i in ids])
        labels = torch.tensor(self.steps[index],dtype=torch.long).repeat_interleave(4)
        coordinates = torch.tensor([self.plan['images'][i]['coordinates'] for i in ids],dtype=torch.float64)
        return pixels,labels,coordinates,torch.tensor(ids,dtype=torch.long)


class AdaptationHoldout(ImageRows):
    """One reference + three queries per GSV adaptation-heldout place.

    Exact place-ID positives, not a claim of unseen foundation-model data.
    Different place IDs can overlap geographically; this is a matched proxy.
    """
    def __init__(self,c,plan):
        places=[p for p in plan['places'] if p['split']=='holdout']
        db=[plan['images'][p['rows'][-1]] for p in places]
        queries=[plan['images'][i] for p in places for i in p['rows'][:-1]]
        super().__init__(paths(c,'dataset_root'),db+queries,c['student_size'])
        self.dbImages=np.asarray([r['path'] for r in db])
        self.qImages=np.asarray([r['path'] for r in queries])
        self.dataset_name='gsv-adaptation-holdout'
        self.num_references=len(db)
        self.num_queries=len(queries)
        self.ground_truth=[np.asarray([i//3]) for i in range(len(queries))]


def official(c):
    return load_official(paths(c,'official_cache'),paths(c,'dino_source'),c['device'])


def student(c):
    seed_all(c['seed'])
    model=SemanticRegionBoQ(official(c),classes=len(CLASSES),hidden=c['hidden'],rho=c['rho']).to(c['device'])
    seed_all(c['seed'])  # construction must not move matched batch RNG streams
    return model


def training_contract(c,arm,stage):
    return dict(stage=stage,arm=arm,**identity(c),
        plan_completed_sha256=sha(paths(c,'plan')/'completed.json'),
        teacher_completed_sha256=sha(paths(c,'teacher_cache')/'completed.json'),
        inference='RGB -> frozen official DINO -> learned semantic/spatial pooling -> trainable BoQ -> one 12288D vector; no CLIP',
        training_scope='Single seed, fixed final five epochs on fixed8192 GSV adaptation places; not full GSV or stable benefit')


def train(c,args):
    from tqdm import tqdm
    from src.losses.vpr_losses import VPRLossFunction
    plan = open_plan(c)
    targets = open_teacher(c)  # same verified cache, including visual control
    stage = args.command
    output = paths(c,'runs')/(('smoke_' if stage=='smoke' else '')+args.arm)
    contract = training_contract(c,args.arm,stage)
    if stage=='train':
        for arm in ARMS:
            smoke = paths(c,'runs')/('smoke_'+arm)
            previous = verify(smoke)
            expected = training_contract(c,arm,'smoke')
            if previous!=expected or not all(read(smoke/'summary.json')['checks'].values()):
                raise ValueError('Three current-code smoke runs must pass; updates are never reused')
    with owned_run(output,contract,args.resume):
        if (output/'completed.json').exists():
            verify(output)
            print('Already complete; no additional training',flush=True)
            return
        model = student(c)
        trainable = {n:p for n,p in model.named_parameters() if p.requires_grad}
        if any(not n.startswith(('pool.','visual.aggregator.')) for n in trainable):
            raise ValueError('Unexpected trainable backbone/teacher')
        frozen_before = state_hash(model.visual.backbone)
        pool_initial = state_hash(model.pool)
        official_aggregator_initial = state_hash(model.visual.aggregator)
        probe = PlaceSteps(c,plan,plan['schedule'][:1])[0][0][:2].to(c['device'])
        model.eval()
        with torch.inference_mode():
            direct = model.visual(probe)
            adapted = model(probe)
            error = float((direct-adapted).abs().max())
        if error>1e-6:
            raise ValueError('Zero-start does not reproduce official descriptor')
        initial = dict(zero_start_error=error,pool_initial_sha256=pool_initial,
            official_aggregator_initial_sha256=official_aggregator_initial,
            frozen_backbone_sha256=frozen_before,
            trainable_parameters=sum(p.numel() for p in trainable.values()))
        if (output/'initial.json').exists() and read(output/'initial.json')!=initial:
            raise ValueError('Reconstructed initial parameters changed')
        write(output/'initial.json',initial)
        optimizer = torch.optim.AdamW([
            dict(params=model.pool.parameters(),lr=c['pool_lr']),
            dict(params=model.visual.aggregator.parameters(),lr=c['boq_lr'])],weight_decay=c['weight_decay'])
        state = dict(step=0,contract_sha256=sha(output/'contract.json'),history=[],
                     gradient_seen=dict(router=False,pool_up=False,boq=False))
        if (output/'last.pt').exists():
            payload = torch.load(output/'last.pt',map_location='cpu',weights_only=True)
            if payload['state']['contract_sha256']!=state['contract_sha256']:
                raise ValueError('Resume checkpoint contract differs')
            active = payload['parameters']
            if set(active)!=set(trainable):
                raise ValueError('Resume trainable parameters differ')
            with torch.no_grad():
                for n,p in trainable.items():
                    p.copy_(active[n].to(p.device))
            optimizer.load_state_dict(payload['optimizer'])
            state=payload['state']
            restore_rng(payload['rng'])
        else:
            seed_all(c['seed'])
        all_steps = plan['schedule'][:c['smoke_steps']] if stage=='smoke' else plan['schedule']
        if not 0<=state['step']<=len(all_steps):
            raise ValueError('Invalid resumed step')

        def save_checkpoint():
            save_torch(output/'last.pt',dict(parameters={n:p.detach().cpu() for n,p in trainable.items()},
                optimizer=optimizer.state_dict(),state=state,rng=rng_state()))

        if not (output/'last.pt').exists():
            save_checkpoint()
        # Private loader RNG avoids consuming the restored model RNG at resume.
        loader = DataLoader(PlaceSteps(c,plan,all_steps[state['step']:]),batch_size=None,
            num_workers=c['workers'],generator=torch.Generator().manual_seed(c['seed']))
        lossfn=VPRLossFunction()
        model.train()
        for images,labels,coords,ids in tqdm(loader,total=len(all_steps),initial=state['step'],desc=f'{args.arm} {stage}'):
            optimizer.zero_grad(set_to_none=True)
            descriptors,semantic,diagnostics=[],[],[]
            target = targets[ids].float()
            if args.arm=='shuffled':
                target=permute_target(target,[plan['images'][i]['path'] for i in ids.tolist()],c['seed'])
            for start in range(0,len(images),c['microbatch']):
                stop=min(start+c['microbatch'],len(images))
                # Checkpoint trainable BoQ+pool to retain full P*K metric batch
                # without storing every decoder graph. Backbone is frozen.
                features=model.encode(images[start:stop].to(c['device']))
                def forward(x):
                    descriptor,logits,stats=model.aggregate(x,auxiliary=True)
                    return descriptor,logits,torch.stack(list(stats.values()))
                d,logits,stats=checkpoint(forward,features,use_reentrant=False)
                descriptors.append(d)
                diagnostics.append(stats.detach())
                semantic.append(guidance_loss(logits,target[start:stop].to(c['device']))[0])
            descriptor=torch.cat(descriptors)
            if not torch.isfinite(descriptor).all() or float((descriptor.detach().norm(dim=1)-1).abs().max())>1e-4:
                raise ValueError('Nonfinite/nonunit descriptor')
            retrieval=gps_retrieval(lossfn,descriptor,labels.to(c['device']),coords)
            # Weight by actual microbatch size (including a short final chunk).
            sizes=[min(c['microbatch'],len(images)-s) for s in range(0,len(images),c['microbatch'])]
            semantic_loss=sum(x*n for x,n in zip(semantic,sizes))/len(images)
            loss=retrieval+(0. if args.arm=='visual' else c['guidance_weight'])*semantic_loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss; last durable checkpoint retained')
            loss.backward()
            for group,prefix in [('router','pool.router.'),('pool_up','pool.up.'),('boq','visual.aggregator.')]:
                state['gradient_seen'][group] |= any(p.grad is not None and bool((p.grad!=0).any())
                    for n,p in trainable.items() if n.startswith(prefix))
            if any(p.grad is not None for p in model.visual.backbone.parameters()):
                raise ValueError('Frozen backbone received gradient')
            norm=torch.nn.utils.clip_grad_norm_(list(trainable.values()),c['gradient_clip'],error_if_nonfinite=True)
            optimizer.step()
            state['step']+=1
            row=dict(step=state['step'],loss=float(loss.detach()),retrieval=float(retrieval.detach()),
                semantic=float(semantic_loss.detach()),gradient_norm=float(norm),
                diagnostics=torch.stack(diagnostics).mean(0).cpu().tolist())
            state['history'].append(row)
            write(output/'progress.json',dict(phase='training',arm=args.arm,step=state['step'],
                total=len(all_steps),epoch=(state['step']-1)//(c['train_places']//c['places_per_batch'])+1,**{
                    k:v for k,v in row.items() if k!='step'}))
            if state['step']%c['checkpoint_every']==0:
                save_checkpoint()
        save_checkpoint()
        checks=dict(zero_start=error<=1e-6,complete_steps=state['step']==len(all_steps),
            frozen_backbone_unchanged=state_hash(model.visual.backbone)==frozen_before,
            router_gradient=state['gradient_seen']['router'],pool_gradient=state['gradient_seen']['pool_up'],
            boq_gradient=state['gradient_seen']['boq'],
            boq_updated=state_hash(model.visual.aggregator)!=official_aggregator_initial,
            teacher_not_in_student=not any('clip' in n.lower() or 'teacher' in n.lower() for n in model.state_dict()))
        write(output/'summary.json',dict(arm=args.arm,steps=state['step'],checks=checks,
            verdict=('SMOKE_PASS_DISCARD_UPDATES' if stage=='smoke' else 'TRAINING_COMPLETE_NOT_EVALUATED')
                if all(checks.values()) else 'MECHANICAL_CHECK_FAILED',
            final_pool_sha256=state_hash(model.pool), final_boq_sha256=state_hash(model.visual.aggregator),
            selection='Fixed final step; no best checkpoint selection',
            scope=contract['training_scope']))
        if not all(checks.values()):
            raise ValueError('Mechanical checks failed; report and checkpoint retained')
        seal(output)
        print(stage,'complete; no benchmark evaluation started automatically',flush=True)


def load_trained(c,arm):
    output=paths(c,'runs')/arm
    contract=verify(output)
    if contract!=training_contract(c,arm,'train'):
        raise ValueError('Trained student contract changed')
    model=student(c)
    payload=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
    if payload['state']['contract_sha256']!=sha(output/'contract.json'):
        raise ValueError('Checkpoint training contract mismatch')
    active={n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(active)!=set(payload['parameters']):
        raise ValueError('Trained parameter inventory differs')
    with torch.no_grad():
        for n,p in active.items():
            p.copy_(payload['parameters'][n].to(p.device))
    return model


def evaluate(c,args):
    from tqdm import tqdm
    from src.dataloaders.valid.mapillary_sls import MapillarySLSDataset
    from src.dataloaders.valid.pittsburgh import PittsburghDataset
    plan=open_plan(c)
    # Deliberately no argument accepting pitts30k-test; preserve the blind test.
    datasets=(AdaptationHoldout(c,plan),
              MapillarySLSDataset(paths(c,'msls_root'),transform(322)),
              PittsburghDataset(paths(c,'pitts_val_root'),transform(322)))
    if any(d.dataset_name not in ('gsv-adaptation-holdout','msls-val','pitts30k-val') for d in datasets):
        raise ValueError('Only exposed development sets allowed in this command')
    contract=dict(stage='development_eval',arm=args.arm,**identity(c),
        dataset_metadata={str(p.resolve()):sha(p) for key in ('msls_root','pitts_val_root')
                          for p in sorted(paths(c,key).glob('*.npy'))},
        training_completed_sha256=None if args.arm=='official' else sha(paths(c,'runs')/args.arm/'completed.json'))
    output=paths(c,'runs')/('eval_'+args.arm)
    with owned_run(output,contract,args.resume):
        if (output/'completed.json').exists():
            verify(output)
            print('Development evaluation already complete',flush=True)
            return
        model=(official(c) if args.arm=='official' else load_trained(c,args.arm)).eval()
        for dataset in datasets:
            tag=dataset.dataset_name
            target=output/(tag+'.json')
            ready=output/(tag+'.ready.json')
            all_paths=[str(p) for p in np.concatenate((dataset.dbImages,dataset.qImages))]
            order_sha=__import__('hashlib').sha256('\n'.join(all_paths).encode()).hexdigest()
            if ready.exists():
                if not target.exists() or sha(target)!=read(ready)['sha256']:
                    raise ValueError('Finished evaluation checksum differs')
                previous=read(target)
                if (previous['image_order_sha256']!=order_sha or previous['queries']!=dataset.num_queries
                        or len(previous['outcomes'])!=dataset.num_queries):
                    raise ValueError('Existing finished evaluation population differs')
                print('Reuse finished development evaluation:',tag,flush=True)
                continue
            features=np.empty((len(dataset),12288),np.float32)
            loader=DataLoader(dataset,batch_size=16,num_workers=c['workers'],shuffle=False)
            with torch.inference_mode():
                for batch,(images,indices) in enumerate(tqdm(loader,desc=f'{args.arm} {tag}')):
                    if indices.tolist()!=list(range(batch*16,batch*16+len(images))):
                        raise ValueError('Extraction order differs')
                    value=torch.cat([model(chunk.to(c['device'])) for chunk in images.split(c['microbatch'])])
                    if not torch.isfinite(value).all() or float((value.norm(dim=1)-1).abs().max())>1e-4:
                        raise ValueError('Invalid evaluation descriptor')
                    features[indices.numpy()]=value.cpu().numpy()
                database=torch.from_numpy(features[:dataset.num_references]).to(c['device'])
                predictions=np.concatenate([(torch.from_numpy(chunk).to(c['device'])@database.T).topk(20,dim=1).indices.cpu().numpy()
                    for chunk in np.array_split(features[dataset.num_references:],max(1,(dataset.num_queries+15)//16))])
            hits=np.asarray([[np.isin(row[:k],gt).any() for k in (1,5,10,20)]
                              for row,gt in zip(predictions,dataset.ground_truth)],bool)
            write(target,dict(queries=len(hits),correct=int(hits[:,0].sum()),outcomes=hits[:,0].tolist(),
                recall_k=[1,5,10,20],recall=hits.mean(0).tolist(),predictions=predictions.tolist(),
                image_order_sha256=order_sha,input_size=322,
                scope=('Adaptation-place holdout, exact GSV ID positives; official pretraining may include these places. Different IDs may overlap geographically.'
                    if tag=='gsv-adaptation-holdout' else 'Exposed development set, fixed final checkpoint; not a blind test.')))
            write(ready,dict(sha256=sha(target)))
            del features,database
        seal(output)


def summarize(c,args):
    variants={arm:paths(c,'runs')/('eval_'+arm) for arm in ('official',*ARMS)}
    common=None
    for path in variants.values():
        report_contract=verify(path)
        if any(report_contract[k]!=v for k,v in identity(c).items()):
            raise ValueError('Development run code/data/config/runtime differs')
        if common is None:
            common=report_contract['dataset_metadata']
        elif report_contract['dataset_metadata']!=common:
            raise ValueError('Development run metadata differs')
    # Train/smoke construction equality is checked across arms, not merely claimed.
    initials=[read(paths(c,'runs')/arm/'initial.json') for arm in ARMS]
    if any(value!=initials[0] for value in initials[1:]):
        raise ValueError('B/C/D did not start with identical parameters/capacity')
    result={}
    for tag in ('gsv-adaptation-holdout','msls-val','pitts30k-val'):
        reports={arm:read(path/(tag+'.json')) for arm,path in variants.items()}
        if len({r['image_order_sha256'] for r in reports.values()})!=1:
            raise ValueError('Query/database identity differs')
        result[tag]=dict(correct={a:r['correct'] for a,r in reports.items()},
            aligned_vs={a:paired(reports[a]['outcomes'],reports['aligned']['outcomes'])
                        for a in ('official','visual','shuffled')})
    comparison=dict(results=result,
        scope='Single-seed exploratory evidence; compare C-B and C-D, not only C-A.',
        next='Only a positive signal merits matched seed replication and then a sealed Pitts30k-test evaluation.')
    write(paths(c,'runs')/'comparison.json',comparison)
    print(json.dumps(comparison,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',nargs='?',default='preflight',
        choices=('preflight','plan','cache','smoke','train','eval','summary'))
    parser.add_argument('--config',type=Path,default=ROOT/'config/semantic_region_study.yaml')
    parser.add_argument('--arm',choices=('official',*ARMS),default='aligned')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--allow-training',action='store_true')
    args=parser.parse_args()
    if args.command in ('smoke','train'):
        if not args.allow_training:
            parser.error('No updates authorized: explicitly pass --allow-training (smoke also updates weights)')
        if args.arm=='official':
            parser.error('Official baseline is evaluation-only')
    c=configuration(args.config)
    # Keep relative paths anchored to the repository, not the shell's cwd.
    if args.command in ('cache','smoke','train','eval'):
        device=torch.device(c['device'])
        if device.type!='cuda' or not torch.cuda.is_available():
            parser.error('CUDA required for this explicit command; preflight/plan/summary use no GPU')
        # Never edit CUDA_VISIBLE_DEVICES in a running process.
        torch.cuda.set_device(device)
    handlers=dict(preflight=preflight,plan=prepare,cache=cache_teacher,
                  smoke=train,train=train,eval=evaluate,summary=summarize)
    handlers[args.command](c,args)


if __name__=='__main__':
    main()
