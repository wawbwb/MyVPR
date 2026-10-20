"""Single-DINO experiment; EVERY GPU stage requires the dual experiment complete.

Reuse the sealed trainer in a PRIVATE module namespace, injecting only model,
identity and freeze checks. Never edit/patch the running upstream source/module.
"""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts import clip_key_study as key
from src.models.single_dino_interaction import ARMS, SingleDinoVPR
from src.semantic_region_protocol import read, write, sha, state_hash, seed_all, owned_run
from src.single_dino_dependency import require_ready, dependency_status

SCHEMA='single_dino_interaction_v1'
NEW_SOURCES=('src/models/single_dino_interaction.py','src/single_dino_dependency.py',
    'scripts/single_dino_interaction.py','scripts/queue_single_dino_interaction.py',
    'config/single_dino_interaction.json','tests/test_single_dino_interaction.py')
_ENGINE=None


def configuration():
    c=read(ROOT/'config/single_dino_interaction.json')
    expected=dict(schema=SCHEMA,seed=42,student_size=322,pilot_steps=1024,train_places=8192,
        places_per_batch=16,views=4,epochs=5,dim=64,rank=16,rho=.025,
        microbatch=2,smoke_steps=8,checkpoint_every=16,device='cuda:0',cuda_memory_fraction=.2,
        precision='fp32',learning_rate=.0001,keep_weight=.1,aux_weight=.1,
        wait_for_run='logs/hetero_interleave_v1',wait_for_launcher='.cache/hetero_interleave_launcher_v1',
        wait_for_train_contract_sha256='0ce69b4f89199b583485e16d6eb6dd29c1ae4f6fc602f85d57808c3e6d35395b')
    if any(c.get(k)!=v for k,v in expected.items()) or any('resnet' in k for k in c):
        raise ValueError('Single-DINO/dependency protocol differs')
    return c


def make_anchor(c):
    # This function constructs only the old DINO/innovation-1, never calls the
    # dual experiment's build() or a ResNet loader.
    from scripts.hetero_interleave_study import make_anchor as frozen_anchor
    return frozen_anchor(c)


def build(c):
    anchor=make_anchor(c)
    models={}
    for arm in ARMS:
        seed_all(c['seed'])
        models[arm]=SingleDinoVPR(anchor,arm,dim=c['dim'],rank=c['rank'],rho=c['rho']).to(c['device'])
    if len({state_hash(m.fusion) for m in models.values()})!=1:
        raise ValueError('New fusion initialization differs across arms')
    ids=[set(map(id,m.fusion.parameters())) for m in models.values()]
    if any(ids[i]&ids[j] for i in range(3) for j in range(i)):
        raise ValueError('Trainable fusion parameters shared')
    if any(hasattr(m,'resnet') for m in models.values()):
        raise ValueError('No second encoder is allowed')
    return models


def identity(c):
    dependency=require_ready(ROOT,c)
    from scripts.train_slgd import verify as old_verify
    from src.models.official_boq import verify_download
    run=key.path(c,'anchor_run');old_verify(run)
    report=read(run/'summary.json')
    if report['final_parameter_sha256']!=c['anchor_parameter_sha256'] or not all(report['checks'].values()):
        raise ValueError('Archived innovation-1 differs')
    from scripts.hetero_interleave_study import SOURCES as frozen_helpers
    sources=tuple(dict.fromkeys((*NEW_SOURCES,*frozen_helpers)))
    return dict(schema=SCHEMA,config=c,code=key.source_hashes(sources),
        protected_sources=key.source_hashes(key.PROTECTED),dependency=dependency,
        anchor_checkpoint_sha256=sha(run/'last.pt'),anchor_completed_sha256=sha(run/'completed.json'),
        official=verify_download(key.path(c,'official_cache')),
        plan_sha256=sha(key.path(c,'plan')/'plan.json'),plan_completed_sha256=sha(key.path(c,'plan')/'completed.json'),
        dino_sources={p.relative_to(key.path(c,'dino_source')).as_posix():sha(p)
            for p in sorted(key.path(c,'dino_source').rglob('*.py'))},
        versions={n:__import__('importlib.metadata',fromlist=['version']).version(n)
            for n in ('torch','torchvision','numpy','pillow','pytorch-metric-learning')},
        encoder_inventory='ONE frozen DINOv2 + archived innovation-1; no ResNet instantiated/loaded',
        private_reference_sha256='98034e48211800766f4aa0230560315660182636ae50858187c2933ceebac63c')


def contract_for(c,stage):
    return dict(stage=stage,**identity(c),
        schedule='Same first1024 sealed GSV seed42 rows as dual pilot; same four views, no holdout training',
        scope='Single-DINO local/global operator interaction; fixed single-seed pilot, development data, not independent/stable/novelty proof')


def finish(output):
    # The reused training loop has a legacy diagnostic key mentioning CNN.
    # Rename that key before sealing, without changing its Boolean result.
    report=output/'summary.json'
    if report.exists():
        value=read(report);checks=value.get('checks',{})
        if 'frozen_anchor_and_cnn_unchanged' in checks:
            checks['frozen_anchor_unchanged']=checks.pop('frozen_anchor_and_cnn_unchanged')
            value['encoder_count']=1
            write(report,value)
    files={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*'))
        if p.is_file() and p.name not in ('.lock','completed.json') and not p.name.endswith('.tmp')}
    write(output/'completed.json',dict(complete=True,schema=SCHEMA,files=files))


def evaluate(c,a):
    from scripts import eval_clip_key_study as metrics
    from scripts.eval_spatial_adapter_repeat import RGBImages
    engine=bound_engine();run=key.path(c,'runs')/'train'
    if engine.verified(run)!=contract_for(c,'train') or not all(read(run/'summary.json')['checks'].values()):
        raise ValueError('Single-DINO training not complete')
    plan=key.open_plan(c);datasets=metrics.populations(c,plan)
    contract=dict(stage='eval',**identity(c),checkpoint_sha256=sha(run/'last.pt'),
        datasets={d['tag']:dict(references=d['references'],queries=d['queries'],
            order=metrics.order_digest(d['names']),gt=metrics.gt_digest(d['ground_truth'])) for d in datasets})
    output=key.path(c,'runs')/'eval'
    with owned_run(output,contract,a.resume):
        if (output/'completed.json').exists():engine.verified(output);print('Already evaluated');return
        key.gpu(c);models=build(c);first=models[ARMS[0]];frozen=state_hash(first.anchor)
        payload=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        if payload['state']['step']!=c['pilot_steps'] or payload['state']['contract_sha256']!=sha(run/'contract.json'):
            raise ValueError('Not the fixed-final single-DINO checkpoint')
        for arm,m in models.items():
            m.fusion.load_state_dict(payload['fusion'][arm],strict=True)
            if state_hash(m.fusion)!=read(run/'summary.json')['final_fusion_sha256'][arm]:raise ValueError('Fusion state changed')
            m.requires_grad_(False).eval()
        variants=('official','innovation1',*ARMS);results={}
        for d in datasets:
            tag=d['tag'];target=output/(tag+'.json');ready=output/(tag+'.ready.json')
            if ready.exists():
                if sha(target)!=read(ready)['sha256']:raise ValueError('Changed evaluation report')
                results[tag]=read(target)['comparison'];continue
            started=time.monotonic();contents=[]
            loader=DataLoader(RGBImages(d['root'],d['names'],d['hashes']),batch_size=16,num_workers=c['workers'],shuffle=False)
            arrays={arm:np.empty((len(d['names']),12288),np.float32) for arm in variants}
            with torch.inference_mode():
                for batch,(images,ids,hashes) in enumerate(tqdm(loader,desc='single-DINO '+tag)):
                    if ids.tolist()!=list(range(batch*16,batch*16+len(images))):raise ValueError('Image order differs')
                    contents.extend(hashes)
                    for start in range(0,len(images),c['microbatch']):
                        pixels=images[start:start+c['microbatch']].to(c['device']);features=first.encode(pixels)
                        values=dict(official=first.anchor.visual(pixels),innovation1=first.anchor.readout(features[2]),
                            **{arm:m.describe(features)[0] for arm,m in models.items()})
                        for arm,z in values.items():
                            if not torch.isfinite(z).all() or float((z.norm(dim=1)-1).abs().max())>1e-4:raise ValueError('Invalid descriptor')
                            arrays[arm][ids[start:start+c['microbatch']].numpy()]=z.cpu().numpy()
                    write(output/'progress.json',dict(phase='evaluation',dataset=tag,images=int(ids[-1])+1,total=len(d['names']),elapsed_seconds=time.monotonic()-started))
                reports={}
                for arm in variants:
                    db=torch.tensor(arrays[arm][:d['references']],device=c['device']);queries=arrays[arm][d['references']:]
                    ranks=np.concatenate([(torch.tensor(q,device=c['device'])@db.T).topk(20,dim=1).indices.cpu().numpy()
                        for q in np.array_split(queries,max(1,(len(queries)+15)//16))])
                    reports[arm]=metrics.retrieval_report(ranks,d['ground_truth'],d['references']);del db
            if tag in ('msls-val','pitts30k-val') and reports['innovation1']['outcomes']!=read(key.path(c,'anchor_run')/(tag+'_322.json'))['outcomes']:
                raise ValueError('Archived innovation-1 baseline not reproduced')
            comparison=dict(correct={arm:v['correct'] for arm,v in reports.items()},
                vs_innovation1={arm:metrics.paired_report(reports['innovation1'],reports[arm]) for arm in ARMS},
                aligned_vs_separate=metrics.paired_report(reports['separate'],reports['aligned']),
                aligned_vs_permuted=metrics.paired_report(reports['permuted'],reports['aligned']))
            write(target,dict(dataset=contract['datasets'][tag],variants=reports,comparison=comparison,
                image_content_digest=metrics.order_digest([n+':'+h for n,h in zip(d['names'],contents)]),elapsed_seconds=time.monotonic()-started))
            write(ready,dict(sha256=sha(target)));results[tag]=comparison;print(tag,comparison['correct'],flush=True)
            del arrays,loader
        checks=dict(frozen_anchor_unchanged=state_hash(first.anchor)==frozen,
            innovation1_sources_unchanged=key.source_hashes(key.PROTECTED)==contract['protected_sources'],
            train_checkpoint_unchanged=sha(run/'last.pt')==contract['checkpoint_sha256'],single_encoder_only=all(not hasattr(m,'resnet') for m in models.values()))
        if not all(checks.values()):raise ValueError('Evaluation integrity check failed')
        signal=all(results[tag]['vs_innovation1']['aligned']['net']>0 and results[tag]['aligned_vs_separate']['net']>0
            and results[tag]['aligned_vs_permuted']['net']>0 for tag in ('msls-val','pitts30k-val'))
        write(output/'summary.json',dict(checks=checks,results=results,
            verdict='EXPLORATORY_SINGLE_DINO_INTERACTION_SIGNAL' if signal else 'NO_MATCHED_SINGLE_DINO_INTERACTION_GAIN',
            stable_gain_proven=False,blind_test_used=False))
        engine.finish(output)


def bound_engine():
    global _ENGINE
    if _ENGINE is None:
        spec=importlib.util.spec_from_file_location('_single_dino_private_training_engine',ROOT/'scripts/hetero_interleave_study.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.SCHEMA=SCHEMA
        module.ARMS=ARMS
        module.build=build
        module.identity=identity
        module.contract_for=contract_for
        module.finish=finish
        module.frozen_hashes=lambda model:dict(anchor=state_hash(model.anchor))
        _ENGINE=module
    return _ENGINE


def main():
    import os
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('plan','smoke','train','eval'))
    p.add_argument('--allow-training',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();c=configuration()
    if a.command=='plan':
        print(dict(training_started=False,dependency=dependency_status(ROOT,c),single_encoder=True));return
    # Explicit independent guard, BEFORE importing the reusable engine,
    # constructing models, touching GPU or creating experiment output.
    require_ready(ROOT,c)
    if a.command in ('smoke','train') and not a.allow_training:
        p.error('Explicit --allow-training required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':p.error('Physical GPU1 only')
    import fcntl
    with (ROOT/'logs/query_relation_screen_v1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require_ready(ROOT,c)
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        engine=bound_engine()
        {'smoke':engine.train,'train':engine.train,'eval':evaluate}[a.command](c,a)


if __name__=='__main__':main()
