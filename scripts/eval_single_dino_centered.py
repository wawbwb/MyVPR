"""Full-DB V2 fixed-checkpoint centering intervention. No training command.

Exactly two exposed development sets (MSLS-val / Pitts30k-val), same 322/FP32.
No subset oracle, weight sweep, best-checkpoint choice or additional optimizer.
"""
import argparse
from pathlib import Path
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scripts import single_dino_supervision as study
from scripts import eval_clip_key_study as metrics
from scripts.eval_spatial_adapter_repeat import RGBImages
from src.single_dino_centering import VARIANTS,decompose
from src.single_dino_diagnosis import stats
from src.semantic_region_protocol import read,write,sha,state_hash,owned_run

SCHEMA='single_dino_centered_v2_eval'
OUTPUT=ROOT/'logs/single_dino_centered_v2_eval'
SOURCES=('src/single_dino_centering.py','scripts/eval_single_dino_centered.py',
    'tests/test_single_dino_centering.py','src/single_dino_diagnosis.py')


def datasets():
    from src.dataloaders.valid.mapillary_sls import MapillarySLSDataset
    from src.dataloaders.valid.pittsburgh import PittsburghDataset
    result=[]
    for tag,cls in [('msls-val',MapillarySLSDataset),('pitts30k-val',PittsburghDataset)]:
        root=ROOT/'datasets'/tag;d=cls(root)
        names=np.concatenate((d.dbImages,d.qImages)).tolist()
        if d.dataset_name!=tag or len(names)!=d.num_references+d.num_queries:
            raise ValueError('Unexpected development dataset population')
        if len(d.ground_truth)!=d.num_queries:raise ValueError('GT/query population differs')
        for gt in d.ground_truth:
            gt=np.asarray(gt)
            if not len(gt) or (gt<0).any() or (gt>=d.num_references).any():raise ValueError('Invalid GT')
        result.append(dict(tag=tag,root=root,names=names,references=d.num_references,
            queries=d.num_queries,gt=d.ground_truth))
    return result


def dataset_identity(d):
    return dict(references=d['references'],queries=d['queries'],order=metrics.order_digest(d['names']),gt=metrics.gt_digest(d['gt']))


def mechanics_summary(arrays,references):
    return {role:{name:stats(values[selection]) for name,values in arrays.items()}
        for role,selection in [('all',slice(None)),('reference',slice(0,references)),('query',slice(references,None))]}


def compare(reports):
    return dict(correct={v:r['correct'] for v,r in reports.items()},
        recall={v:r['recall'] for v,r in reports.items()},
        vs_innovation1={v:metrics.paired_report(reports['innovation1'],reports[v]) for v in VARIANTS[1:]},
        vs_aligned={v:metrics.paired_report(reports['aligned'],reports[v]) for v in ('centered','mean_only')})


def validate_dataset(report,d,original):
    if report['dataset']!=dataset_identity(d) or set(report['variants'])!=set(VARIANTS):raise ValueError('Wrong population/variants')
    for v,r in report['variants'].items():
        if metrics.retrieval_report(r['predictions'],d['gt'],d['references'])!=r:raise ValueError('Saved GT outcomes inconsistent')
    # Exact full Top20 reproduction, not merely equal rounded recall.
    for v in ('innovation1','aligned'):
        if report['variants'][v]!=original['variants'][v]:raise ValueError('Original full-DB rankings not reproduced: '+v)
    if compare(report['variants'])!=report['comparison']:raise ValueError('Paired counts inconsistent')
    if not report['checks'] or not all(report['checks'].values()):raise ValueError('Dataset integrity failed')


def verified():
    s=read(OUTPUT/'completed.json')
    if s.get('complete') is not True or s.get('schema')!=SCHEMA:raise ValueError('Incomplete output')
    for name,h in s['files'].items():
        p=(OUTPUT/name).resolve()
        if not p.is_relative_to(OUTPUT.resolve()) or sha(p)!=h:raise ValueError('Changed diagnostic artifact')


def run(c,a):
    trained=study.old.key.path(c,'runs')/'train';evaluation=trained.parent/'eval'
    if study.verified(trained)!=study.contract_for(c,'train'):raise ValueError('V2 training identity differs')
    evaluation_contract=study.verified(evaluation)
    for stage in (trained,evaluation):
        if not all(read(stage/'summary.json')['checks'].values()):raise ValueError('Original V2 mechanical check failed')
    if evaluation_contract['checkpoint_sha256']!=sha(trained/'last.pt'):raise ValueError('V2 eval checkpoint differs')
    sets=datasets();identity=study.identity(c)
    for d in sets:
        if evaluation_contract['datasets'][d['tag']]!=dataset_identity(d):raise ValueError('Original eval population changed')
    contract=dict(schema=SCHEMA,source_identity=identity,code=study.old.key.source_hashes(SOURCES),
        train_seal=sha(trained/'completed.json'),eval_seal=sha(evaluation/'completed.json'),checkpoint=sha(trained/'last.pt'),
        datasets={d['tag']:dataset_identity(d) for d in sets},variants=list(VARIANTS),
        policy='Fixed final V2 aligned checkpoint, full databases, no training, no rho scaling, no best-variant selection',
        allow_shared_gpu=a.allow_shared_gpu,blind_test_used=False)
    with owned_run(OUTPUT,contract,a.resume):
        if (OUTPUT/'completed.json').exists():verified();print('Already complete');return
        study.old.key.gpu(c);models=study.build(c);model=models['aligned']
        payload=torch.load(trained/'last.pt',map_location='cpu',weights_only=True)
        if payload['state']['step']!=c['pilot_steps'] or payload['state']['contract_sha256']!=sha(trained/'contract.json'):
            raise ValueError('Not the fixed-final V2 checkpoint')
        model.fusion.load_state_dict(payload['fusion']['aligned'],strict=True)
        frozen=state_hash(model.anchor);head=state_hash(model.fusion)
        if head!=read(trained/'summary.json')['final_fusion_sha256']['aligned']:raise ValueError('V2 aligned head differs')
        model.requires_grad_(False).eval();results={};mechanisms={}
        for d in sets:
            tag=d['tag'];target=OUTPUT/(tag+'.json');ready=OUTPUT/(tag+'.ready.json')
            original=read(evaluation/(tag+'.json'))
            if ready.exists():
                if read(ready)['sha256']!=sha(target):raise ValueError('Changed finished dataset')
                finished=read(target);validate_dataset(finished,d,original)
                results[tag]=finished['comparison'];mechanisms[tag]=finished['mechanisms'];continue
            total=len(d['names']);started=time.monotonic()
            # In-memory descriptor buffers only; no multi-GB cache is written.
            available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
            required=len(VARIANTS)*total*12288*4
            if available<required+2*1024**3:raise RuntimeError('Insufficient host RAM; no huge cache created')
            arrays={v:np.empty((total,12288),np.float32) for v in VARIANTS}
            measurements={n:np.empty(total,np.float64) for n in
                ('residual_relative','centered_relative','dc_relative','dc_energy_fraction','residual_nonzero')}
            contents=[]
            loader=DataLoader(RGBImages(d['root'],d['names'],None),batch_size=16,num_workers=c['workers'],shuffle=False)
            with torch.inference_mode():
                for batch,(images,ids,hashes) in enumerate(tqdm(loader,desc='Full-DB centering '+tag)):
                    if ids.tolist()!=list(range(batch*16,batch*16+len(images))):raise ValueError('Image order differs')
                    contents.extend(hashes)
                    for start in range(0,len(images),c['microbatch']):
                        pixels=images[start:start+c['microbatch']].to(c['device'])
                        anchor=model.encode(pixels)[2];adapted=model.fusion(anchor)[0]
                        maps,values=decompose(anchor,adapted)
                        indices=ids[start:start+c['microbatch']].numpy()
                        for v,m in maps.items():
                            z=model.anchor.readout(m)
                            if not torch.isfinite(z).all() or float((z.norm(dim=1)-1).abs().max())>1e-4:raise ValueError('Invalid unit descriptor')
                            arrays[v][indices]=z.cpu().numpy()
                        for n,x in values.items():measurements[n][indices]=x.cpu().numpy()
                    write(OUTPUT/'progress.json',dict(phase='evaluation',dataset=tag,images=int(ids[-1])+1,total=total,
                        elapsed_seconds=time.monotonic()-started,training_started=False))
                reports={}
                for v,x in arrays.items():
                    db=torch.tensor(x[:d['references']],device=c['device'])
                    ranks=np.concatenate([(torch.tensor(q,device=c['device'])@db.T).topk(20,dim=1).indices.cpu().numpy()
                        for q in np.array_split(x[d['references']:],max(1,(d['queries']+15)//16))])
                    reports[v]=metrics.retrieval_report(ranks,d['gt'],d['references']);del db
            checks=dict(innovation1_full_top20_reproduced=reports['innovation1']==original['variants']['innovation1'],
                aligned_full_top20_reproduced=reports['aligned']==original['variants']['aligned'],
                original_image_content_reproduced=metrics.order_digest([n+':'+h for n,h in zip(d['names'],contents)])==original['image_content_digest'],
                centered_frobenius_nonexpansive=bool((measurements['centered_relative']<=measurements['residual_relative']+1e-6).all()),
                finite_mechanics=all(np.isfinite(x).all() for x in measurements.values()),
                dc_energy_bounded=bool(((measurements['dc_energy_fraction']>=0)&(measurements['dc_energy_fraction']<=1+1e-5)).all()))
            value=dict(dataset=dataset_identity(d),variants=reports,comparison=compare(reports),checks=checks,
                mechanisms=mechanics_summary(measurements,d['references']),elapsed_seconds=time.monotonic()-started,
                scope='Full-DB fixed-checkpoint sensitivity only; not retrained ablation or independent test')
            validate_dataset(value,d,original)
            write(target,value);write(ready,dict(sha256=sha(target)))
            results[tag]=value['comparison'];mechanisms[tag]=value['mechanisms']
            print(tag,value['comparison']['correct'],flush=True)
            del arrays,measurements,loader
        checks=dict(frozen_anchor_unchanged=state_hash(model.anchor)==frozen,
            fusion_tensors_unchanged=state_hash(model.fusion)==head,checkpoint_unchanged=sha(trained/'last.pt')==contract['checkpoint'],
            train_seal_unchanged=sha(trained/'completed.json')==contract['train_seal'],
            eval_seal_unchanged=sha(evaluation/'completed.json')==contract['eval_seal'],
            protected_sources_unchanged=study.old.key.source_hashes(study.old.key.PROTECTED)==identity['protected_sources'],
            new_source_unchanged=study.old.key.source_hashes(SOURCES)==contract['code'])
        if not all(checks.values()):raise ValueError('Final integrity check failed')
        signal=all(x['vs_innovation1']['centered']['net']>0 and x['vs_aligned']['centered']['net']>0 for x in results.values())
        write(OUTPUT/'summary.json',dict(checks=checks,results=results,mechanisms=mechanisms,
            verdict='EXPLORATORY_FULL_DB_CENTERING_SIGNAL' if signal else 'NO_CONSISTENT_FULL_DB_CENTERING_GAIN',
            training_started=False,stable_gain_proven=False,blind_test_used=False,
            scope='Two exposed validation sets, fixed-checkpoint intervention; no claim that centered training works'))
        write(OUTPUT/'completed.json',dict(complete=True,schema=SCHEMA,
            files={p.relative_to(OUTPUT).as_posix():sha(p) for p in sorted(OUTPUT.rglob('*'))
                if p.is_file() and p.name not in ('.lock','completed.json') and not p.name.endswith('.tmp')}))
        print('CENTERING EVALUATION COMPLETE',str(OUTPUT),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('plan','run'))
    p.add_argument('--allow-shared-gpu',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();c=study.configuration()
    if a.command=='plan':
        sets=datasets()
        print(dict(training=False,datasets={d['tag']:dict(references=d['references'],queries=d['queries']) for d in sets},
            variants=VARIANTS,output=str(OUTPUT)));return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':p.error('Physical GPU1 only')
    import fcntl,subprocess
    with (ROOT/'logs/query_relation_screen_v1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        memory=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.total,memory.used,memory.free',
            '--format=csv,noheader,nounits'],text=True,timeout=20)
        study.check_gpu_capacity(*map(int,memory.strip().split(',')),a.allow_shared_gpu,c['cuda_memory_fraction'])
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        run(c,a)


if __name__=='__main__':main()
