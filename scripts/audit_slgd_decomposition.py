"""Reuse fixed checkpoints to separate initial fusion, trained fusion and drift."""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_slgd import RU_SHA,read,write,sha,code_hash,verify,seal,seed,parameter_hash
from scripts.slgd_mid_screen import dataset,sources,load_visual,current_code,open_contract
from scripts.audit_slgd_hard import PlannedPlaces,collate,shard_rows,save_npz
from src.models.slgd import LocalProjector,SLGDStudent
from src.slgd_hard_protocol import distance_metres
from src.slgd_decomposition import (HEADS,VARIANTS,GLOBAL_DIM,LOCAL_DIM,
    descriptor_variant,validate_shard,paired_summary)

SOURCES=('scripts/audit_slgd_decomposition.py','src/slgd_decomposition.py')
POLICY=dict(revision=1,beta=.2,microbatch=4,negative_exclusion_metres=25.,
    variants=list(VARIANTS),training=False,weight_selection=False,
    scope='Exploratory same-data fixed-checkpoint inference decomposition, not a fair retrained frozen-global ablation or semantic benefit')


def build_models(args,data,plan,contracts):
    seed(42)
    visual=load_visual(args)
    projector=LocalProjector().cuda().eval().requires_grad_(False)
    projector.load_state_dict(torch.load(args.teacher/'last.pt',map_location='cpu',weights_only=True)['parameters'],strict=True)
    seed(42)
    initial=SLGDStudent(visual,projector,slots=16,beta=.2).cuda().eval()
    init_hash=parameter_hash(initial)
    for mode in HEADS[1:]:
        saved=read(args.runs/(mode+'_'+args.stage)/'summary.json')
        if init_hash!=saved['initial_parameter_sha256']:
            raise ValueError('Initial parameters could not be reconstructed: '+mode)
    images=PlannedPlaces(data,plan)[0][0]
    with torch.inference_mode():
        probe=initial(images.cuda()).cpu().numpy()
    actual=hashlib.sha256(probe.tobytes()).hexdigest()
    for mode in HEADS[1:]:
        original=read(args.runs/(mode+'_'+args.stage)/'initial.json')
        if actual!=original['descriptor_sha256']:
            raise ValueError('Initial descriptor reconstruction differs: '+mode)
    models={'initial':initial}
    for mode in HEADS[1:]:
        model=copy.deepcopy(initial)
        path=args.runs/(mode+'_'+args.stage)
        payload=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
        active={name:p for name,p in model.named_parameters() if p.requires_grad}
        if set(active)!=set(payload['parameters']) or payload['state']['contract_sha256']!=sha(path/'contract.json'):
            raise ValueError('Checkpoint identity differs: '+mode)
        with torch.no_grad():
            for name,p in active.items():p.copy_(payload['parameters'][name].to(p.device))
        if parameter_hash(model)!=read(path/'summary.json')['final_parameter_sha256']:
            raise ValueError('Trained parameters could not be restored: '+mode)
        models[mode]=model
        del payload
    for model in models.values():model.requires_grad_(False).eval()
    return models


def validate_existing(path,indices):
    if sha(path)!=read(path.with_suffix('.json'))['sha256']:
        raise ValueError('Diagnostic cache hash differs; retained, not overwritten: '+str(path))
    with np.load(path,allow_pickle=False) as z:arrays={name:z[name] for name in z.files}
    validate_shard(arrays,indices)
    return arrays


def evaluate_variant(descriptors,coordinates,plan,name):
    from tqdm import tqdm
    qrows=np.asarray(plan['query_rows']);grows=np.asarray(plan['gallery_rows'])
    positive=np.asarray(plan['positive_gallery_indices'])
    gallery=torch.from_numpy(descriptor_variant({h:d[grows] for h,d in descriptors.items()},name)).cuda()
    predictions=[]
    for start in tqdm(range(0,len(qrows),16),desc=name+' fixed retrieval'):
        rows=qrows[start:start+16]
        query=descriptor_variant({h:d[rows] for h,d in descriptors.items()},name)
        scores=torch.from_numpy(query).cuda() @ gallery.T
        allowed=distance_metres(coordinates[rows],coordinates[grows])>=25
        allowed[np.arange(len(rows)),positive[start:start+16]]=True
        scores.masked_fill_(~torch.from_numpy(allowed).cuda(),-torch.inf)
        predictions.extend(scores.argmax(1).cpu().tolist())
    return predictions,(np.asarray(predictions)==positive).tolist()


def run(args):
    from tqdm import tqdm
    teacher,heldout,metadata=sources(args)
    contracts={mode:verify(args.runs/(mode+'_'+args.stage)) for mode in HEADS[1:]}
    reference=dict(contracts['retrieval']);reference.pop('mode')
    for mode,c in contracts.items():
        actual=dict(c);actual.pop('mode')
        if actual!=reference or c['mode']!=mode or c['stage']!=args.stage or c['code']!=current_code():
            raise ValueError('Original trained controls differ')
        if c['source_completed_sha256']!=sha(args.source/'completed.json') or c['teacher_completed_sha256']!=sha(args.teacher/'completed.json'):
            raise ValueError('Original source identity differs')
    contract=dict(policy=POLICY,stage=args.stage,source_completed_sha256=sha(args.source/'completed.json'),
        runs_completed_sha256={mode:sha(args.runs/(mode+'_'+args.stage)/'completed.json') for mode in HEADS[1:]},
        dataset_metadata=metadata,original_code=current_code(),code={p:code_hash(ROOT/p) for p in SOURCES},
        versions=dict(torch=torch.__version__,numpy=np.__version__))
    open_contract(args.output,contract,args.resume)
    open_contract(args.cache,contract,args.resume)
    if (args.output/'completed.json').exists():
        verify(args.output);verify(args.cache)
        verify_shards(args.cache,heldout['plan'])
        print('Already complete',flush=True);return
    plan=heldout['plan'];data=dataset(args)
    if [str(data.places_ids[i]) for i in plan['dataset_indices']]!=plan['query_places']+plan['distractor_places']:
        raise ValueError('Original evaluation identities changed')
    descriptors={h:np.empty((plan['rows'],GLOBAL_DIM+LOCAL_DIM),np.float32) for h in HEADS}
    coordinates=np.empty((plan['rows'],2),np.float64)
    shards=args.cache/'shards';shards.mkdir(exist_ok=True)
    loader=DataLoader(PlannedPlaces(data,plan),batch_size=16,num_workers=args.workers,collate_fn=collate)
    models=None;hashes={}
    print('No training; encode DINO once per image for four fixed heads. Descriptor cache GiB:',
          sum(d.nbytes for d in descriptors.values())/1024**3,flush=True)
    for batch,(images,indices,coords) in enumerate(tqdm(loader,desc='Fixed initial + three trained heads')):
        path=shards/f'{batch:05d}.npz';rows=indices.numpy()
        write(args.output/'progress.json',dict(phase='encode',batch=batch,total=len(loader)))
        if path.exists():
            arrays=validate_existing(path,rows)
        else:
            if models is None:models=build_models(args,data,plan,contracts)
            parts={h:[] for h in HEADS}
            with torch.inference_mode():
                for chunk in images.cuda().split(4):
                    features=models['initial'].visual.backbone(chunk)
                    if isinstance(features,tuple):features=features[0]
                    for h,model in models.items():
                        parts[h].append(model.aggregate(features,return_training_tokens=False)[0].cpu().numpy())
            arrays={h:np.concatenate(parts[h]).astype(np.float32) for h in HEADS}
            arrays.update(indices=rows,coordinates=coords.numpy().astype(np.float64))
            validate_shard(arrays,rows)
            save_npz(path,arrays)
        for h in HEADS:descriptors[h][rows]=arrays[h]
        coordinates[rows]=arrays['coordinates'];hashes[path.name]=sha(path)
    if models is not None:
        for model in models.values():
            if any(p.grad is not None for p in model.parameters()):raise ValueError('Unexpected diagnostic gradient')
    del models,loader
    torch.cuda.empty_cache()
    write(args.cache/'manifest.json',dict(shards=hashes))
    seal(args.cache)
    predictions={};outcomes={}
    for name in VARIANTS:
        write(args.output/'progress.json',dict(phase='retrieval',variant=name,total=len(VARIANTS)))
        predictions[name],outcomes[name]=evaluate_variant(descriptors,coordinates,plan,name)
        write(args.output/'partial_predictions.json',dict(predictions=predictions,outcomes=outcomes))
    original=read(args.source/'outcomes.json')
    if outcomes['ru_global']!=original['ru']:
        raise ValueError('Frozen RU outcomes do not reproduce original source')
    for mode in HEADS[1:]:
        old=read(args.runs/(mode+'_'+args.stage)/'final.json')
        if predictions[mode+'_hybrid']!=old['predictions'] or outcomes[mode+'_hybrid']!=old['outcomes']:
            raise ValueError('Trained hybrid predictions do not reproduce original final: '+mode)
    partitions=paired_summary(outcomes,plan['query_place_groups'],original['hard'],plan['calibration_places'])
    drift={}
    for mode in HEADS[1:]:
        drift[mode]={}
        for name,sl,scale in [('global',slice(0,GLOBAL_DIM),.8),('local',slice(GLOBAL_DIM,None),.2)]:
            cosine=np.sum(descriptors['initial'][:,sl]*descriptors[mode][:,sl],axis=1)/scale
            distance=1-cosine
            drift[mode][name]=dict(cosine_distance_mean=float(distance.mean()),
                cosine_distance_p95=float(np.quantile(distance,.95)),
                note='Float32 roundoff can give tiny negative cosine distances')
    write(args.output/'predictions.json',dict(predictions=predictions,outcomes=outcomes))
    write(args.output/'summary.json',dict(verdict='DIAGNOSTIC_COMPLETE',partitions=partitions,drift=drift,
        checks=dict(initial_parameter_and_descriptor_reconstructed=True,frozen_ru_outcomes_reproduced=True,
                    trained_hybrid_predictions_reproduced=True,no_training=True),
        scope=POLICY['scope'],note='Global-only and frozen-RU-global/trained-local are inference counterfactuals; no weight search or further training'))
    write(args.output/'progress.json',dict(phase='complete'))
    seal(args.output)
    print('DIAGNOSTIC_COMPLETE',flush=True)
    print({name:partitions['all']['variants'][name]['correct'] for name in VARIANTS},flush=True)


def verify_shards(cache,plan):
    expected=(len(plan['dataset_indices'])+15)//16
    hashes=read(cache/'manifest.json')['shards']
    if set(hashes)!={f'{i:05d}.npz' for i in range(expected)}:
        raise ValueError('Incomplete diagnostic shard set')
    for batch in range(expected):
        path=cache/'shards'/f'{batch:05d}.npz'
        if sha(path)!=hashes[path.name]:raise ValueError('Diagnostic manifest hash differs')
        validate_existing(path,shard_rows(plan,batch))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['smoke','pilot'],required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--teacher',type=Path,default=Path('logs/slgd_v1/teacher_pilot'))
    p.add_argument('--runs',type=Path,default=Path('logs/slgd_mid_v1'))
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,default=Path('datasets/gsv_cities'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1' or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        p.error('Expose physical GPU1 only')
    if not args.checkpoint.is_file() or sha(args.checkpoint)!=RU_SHA:p.error('Original RU identity differs')
    import fcntl
    args.output.parent.mkdir(parents=True,exist_ok=True)
    lock=(args.output.parent/'slgd_decomposition_worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(.65,0)
    run(args)


if __name__=='__main__':main()
