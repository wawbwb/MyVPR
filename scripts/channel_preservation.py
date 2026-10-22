"""Same fixed warm adapter, three loss-only repair groups; no benchmark labels."""
import argparse
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts import structural_channel as prior
from scripts.single_dino_supervision import check_gpu_capacity
from src.models.structural_channel_adapter import StructuralChannelVPR,image_seeds
from src.channel_rank_preservation import ARMS,teacher_contract,ranking_penalty
from src.semantic_region_protocol import (read,write,sha,state_hash,owned_run,seed_all,
    save_torch,rng_state,restore_rng,gps_retrieval)

SCHEMA='channel_preservation_v1'
SOURCES=('src/channel_rank_preservation.py','scripts/channel_preservation.py',
    'scripts/launch_channel_preservation.py','tests/test_channel_preservation.py','config/channel_preservation.json',
    *prior.SOURCES)


def configuration():
    extra=read(ROOT/'config/channel_preservation.json')
    expected=dict(schema=SCHEMA,runs='logs/channel_preservation_v1',warm_run='logs/structural_channel_v1/train',
        warm_arm='structure_gate',pilot_steps=512,smoke_steps=8,learning_rate=.0003,rank_weight=.1,rank_cap=.05)
    if extra!=expected:raise ValueError('Registered repair protocol differs')
    return dict(prior.configuration(),**extra)


def warm_identity(c):
    run=prior.old.key.path(c,'warm_run')
    if prior.verified(run)!=prior.contract_for(prior.configuration(),'train'):raise ValueError('Warm predecessor changed')
    summary=read(run/'summary.json')
    if summary['steps']!=2560 or not all(summary['checks'].values()):raise ValueError('Warm training incomplete')
    return dict(checkpoint_sha256=sha(run/'last.pt'),completed_sha256=sha(run/'completed.json'),
        adapter_sha256=summary['final_adapter_sha256'][c['warm_arm']])


def identity(c):
    value=prior.old.identity(c)
    value.update(schema=SCHEMA,new_sources=prior.old.key.source_hashes(tuple(dict.fromkeys(SOURCES))),
        warm=warm_identity(c),inference='Identical single-DINO structure-gate architecture12288D; only training losses differ',
        scope='One fixed512-step repair from common warm head. Previously inspected development sets, no blind confirmation')
    return value


def contract_for(c,stage):return dict(stage=stage,**identity(c))


def finish(output):
    files={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*'))
        if p.is_file() and p.name not in ('.lock','completed.json') and not p.name.endswith('.tmp')}
    write(output/'completed.json',dict(complete=True,schema=SCHEMA,files=files))


def verified(output):
    d=read(output/'completed.json')
    if d.get('complete') is not True or d.get('schema')!=SCHEMA:raise ValueError('Incomplete/wrong repair artifact')
    for name,h in d['files'].items():
        p=(output/name).resolve()
        if not p.is_relative_to(output.resolve()) or sha(p)!=h:raise ValueError('Repair artifact changed')
    return read(output/'contract.json')


def build(c):
    anchor=prior.old.make_anchor(c);models={}
    info=warm_identity(c)
    payload=torch.load(prior.old.key.path(c,'warm_run')/'last.pt',map_location='cpu',weights_only=True)
    for arm in ARMS:
        seed_all(c['seed'])
        m=StructuralChannelVPR(anchor,'structure_gate',hidden=c['hidden'],rank=c['gate_rank'],gain=c['adapter_gain']).to(c['device'])
        m.adapter.load_state_dict(payload['adapters'][c['warm_arm']],strict=True)
        if state_hash(m.adapter)!=info['adapter_sha256']:raise ValueError('Warm tensors differ')
        models[arm]=m
    if len({state_hash(m.adapter) for m in models.values()})!=1:raise ValueError('Warm initialization differs')
    ids=[set(map(id,m.adapter.parameters())) for m in models.values()]
    if any(ids[i]&ids[j] for i in range(3) for j in range(i)):raise ValueError('Trainable parameters shared')
    return models


def train(c,args):
    from scripts.semantic_region_study import PlaceSteps
    from src.losses.vpr_losses import VPRLossFunction
    plan=prior.old.key.open_plan(c);contract=contract_for(c,args.command);runs=prior.old.key.path(c,'runs')
    if args.command=='train':
        smoke=verified(runs/'smoke')
        if ({k:v for k,v in smoke.items() if k!='stage'}!={k:v for k,v in contract.items() if k!='stage'}
                or not all(read(runs/'smoke/summary.json')['checks'].values())):raise ValueError('Same-version smoke required')
    output=runs/args.command
    with owned_run(output,contract,args.resume):
        if (output/'completed.json').exists():verified(output);print('Already complete');return
        prior.old.key.gpu(c);models=build(c);first=models[ARMS[0]]
        initial=dict(frozen=state_hash(first.anchor),adapters={a:state_hash(m.adapter) for a,m in models.items()},
            parameters={a:sum(p.numel() for p in m.adapter.parameters()) for a,m in models.items()})
        if (output/'initial.json').exists() and read(output/'initial.json')!=initial:raise ValueError('Warm state changed')
        write(output/'initial.json',initial)
        sample=PlaceSteps(c,plan,plan['schedule'][:1])[0]
        seeds=image_seeds([plan['images'][int(i)]['image_sha256'] for i in sample[3][:2]])
        with torch.no_grad():
            f=first.encode(sample[0][:2].to(c['device']));expected=first.describe(f,seeds)[0]
            errors={a:float((m.describe(f,seeds)[0]-expected).abs().max()) for a,m in models.items()}
        del sample,f,expected
        optimizers={a:torch.optim.AdamW(m.adapter.parameters(),lr=c['learning_rate'],weight_decay=0.) for a,m in models.items()}
        state=dict(step=0,history=[],contract_sha256=sha(output/'contract.json'),
            gradients={a:{n:False for n,p in m.adapter.named_parameters()} for a,m in models.items()})
        def save():save_torch(output/'last.pt',dict(adapters={a:m.adapter.state_dict() for a,m in models.items()},
            optimizers={a:o.state_dict() for a,o in optimizers.items()},state=state,rng=rng_state()))
        if (output/'last.pt').exists():
            payload=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
            if payload['state']['contract_sha256']!=state['contract_sha256']:raise ValueError('Resume contract differs')
            for a,m in models.items():
                m.adapter.load_state_dict(payload['adapters'][a],strict=True);optimizers[a].load_state_dict(payload['optimizers'][a])
            state=payload['state'];restore_rng(payload['rng'])
        else:seed_all(c['seed']);save()
        total=c['smoke_steps'] if args.command=='smoke' else c['pilot_steps']
        if not 0<=state['step']<=total:raise ValueError('Invalid resume cursor')
        loader=DataLoader(PlaceSteps(c,plan,plan['schedule'][state['step']:total]),batch_size=None,
            num_workers=c['workers'],generator=torch.Generator().manual_seed(c['seed']))
        lossfn=VPRLossFunction()
        for m in models.values():m.train()
        for images,labels,coords,row_ids in tqdm(loader,total=total,initial=state['step'],desc='channel preservation '+args.command):
            start=time.monotonic();labels=labels.to(c['device'])
            seeds=image_seeds([plan['images'][int(i)]['image_sha256'] for i in row_ids])
            with torch.no_grad():chunks=[first.encode(x.to(c['device'])) for x in images.split(c['microbatch'])]
            reference=torch.cat([f[1] for f in chunks]);teacher=teacher_contract(reference,labels,coords,c['rank_cap']);row={}
            for arm,m in models.items():
                with torch.no_grad():
                    temporary=[];offset=0
                    for f in chunks:
                        n=len(f[0]);temporary.append(m.describe(f,seeds[offset:offset+n]));offset+=n
                    leaf=torch.cat([v[0] for v in temporary]).detach().requires_grad_()
                    relative=torch.cat([v[1]['relative_delta'] for v in temporary])
                del temporary
                rank,stats=ranking_penalty(leaf,teacher,arm)
                losses=dict(main=gps_retrieval(lossfn,leaf,labels,coords),
                    keep=c['keep_weight']*(1-(leaf*reference).sum(1)).clamp_min(0).mean(),rank=c['rank_weight']*rank)
                if not all(bool(torch.isfinite(v)) for v in losses.values()):raise ValueError('Nonfinite loss')
                gradients=torch.autograd.grad(sum(losses.values()),leaf)[0]
                optimizers[arm].zero_grad(set_to_none=True);prior.replay(m,chunks,seeds,gradients)
                for name,p in m.adapter.named_parameters():state['gradients'][arm][name]|=p.grad is not None and bool((p.grad!=0).any())
                norm=torch.nn.utils.clip_grad_norm_(m.adapter.parameters(),c['gradient_clip'],error_if_nonfinite=True)
                optimizers[arm].step()
                row[arm]=dict(losses={n:float(v.detach()) for n,v in losses.items()},gradient_norm=float(norm),
                    relative_delta_mean=float(relative.mean()),ranking=stats)
                del leaf,losses,gradients
            torch.cuda.synchronize();state['step']+=1
            row.update(step=state['step'],seconds=time.monotonic()-start,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
            state['history'].append(row);write(output/'progress.json',dict(phase='training',total=total,**row))
            if state['step']%c['checkpoint_every']==0:save()
            del chunks,reference,teacher
        save()
        checks=dict(complete_steps=state['step']==total,warm_start_matched=max(errors.values())==0,
            frozen_anchor_unchanged=state_hash(first.anchor)==initial['frozen'],matched_capacity=len(set(initial['parameters'].values()))==1,
            all_adapters_updated=all(state_hash(m.adapter)!=initial['adapters'][a] for a,m in models.items()),
            all_parameters_gradient_seen=all(all(v.values()) for v in state['gradients'].values()),
            innovation1_sources_unchanged=prior.old.key.source_hashes(prior.old.key.PROTECTED)==contract['protected_sources'])
        activity={a:dict(protected_rows=sum(v[a]['ranking']['protected'] for v in state['history']),
            violating_rows=sum(v[a]['ranking']['violating'] for v in state['history']),
            nonzero_penalty_steps=sum(v[a]['losses']['rank']>0 for v in state['history'])) for a in ARMS}
        write(output/'summary.json',dict(checks=checks,steps=state['step'],warm_start_errors=errors,penalty_activity=activity,
            final_adapter_sha256={a:state_hash(m.adapter) for a,m in models.items()},parameters=initial['parameters'],
            encoder_count=1,descriptor_dimensions=12288,verdict='SMOKE_DISCARD' if args.command=='smoke' else 'TRAINED_NOT_EVALUATED'))
        if not all(checks.values()):raise ValueError('Mechanical check failed: '+str(checks))
        finish(output)


def evaluate(c,args):
    from scripts import eval_clip_key_study as metrics
    from scripts.eval_spatial_adapter_repeat import RGBImages
    run=prior.old.key.path(c,'runs')/'train'
    if verified(run)!=contract_for(c,'train') or not all(read(run/'summary.json')['checks'].values()):raise ValueError('Train incomplete/changed')
    datasets=metrics.populations(c,prior.old.key.open_plan(c))
    contract=dict(stage='eval',**identity(c),checkpoint_sha256=sha(run/'last.pt'),
        datasets={d['tag']:dict(references=d['references'],queries=d['queries'],order=metrics.order_digest(d['names']),gt=metrics.gt_digest(d['ground_truth'])) for d in datasets})
    output=prior.old.key.path(c,'runs')/'eval'
    with owned_run(output,contract,args.resume):
        if (output/'completed.json').exists():verified(output);print('Already evaluated');return
        prior.old.key.gpu(c);models=build(c);first=models[ARMS[0]];frozen=state_hash(first.anchor)
        warm=StructuralChannelVPR(first.anchor,'structure_gate',hidden=c['hidden'],rank=c['gate_rank'],gain=c['adapter_gain']).to(c['device'])
        warm.adapter.load_state_dict(first.adapter.state_dict(),strict=True);warm.requires_grad_(False).eval()
        payload=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        if payload['state']['step']!=c['pilot_steps'] or payload['state']['contract_sha256']!=sha(run/'contract.json'):raise ValueError('Not fixed-final')
        for a,m in models.items():
            m.adapter.load_state_dict(payload['adapters'][a],strict=True)
            if state_hash(m.adapter)!=read(run/'summary.json')['final_adapter_sha256'][a]:raise ValueError('Adapter changed')
            m.requires_grad_(False).eval()
        results={};variants=('innovation1','warm',*ARMS)
        for d in datasets:
            tag=d['tag'];target=output/(tag+'.json');ready=output/(tag+'.ready.json')
            if ready.exists():
                if sha(target)!=read(ready)['sha256']:raise ValueError('Report changed')
                results[tag]=read(target)['comparison'];continue
            start=time.monotonic();contents=[]
            arrays={a:np.empty((len(d['names']),12288),np.float32) for a in variants}
            loader=DataLoader(RGBImages(d['root'],d['names'],d['hashes']),batch_size=16,num_workers=c['workers'],shuffle=False)
            with torch.inference_mode():
                for batch,(images,ids,hashes) in enumerate(tqdm(loader,desc='channel preservation '+tag)):
                    if ids.tolist()!=list(range(batch*16,batch*16+len(images))):raise ValueError('Order differs')
                    contents.extend(hashes)
                    for i in range(0,len(images),c['microbatch']):
                        f=first.encode(images[i:i+c['microbatch']].to(c['device']));seeds=image_seeds(hashes[i:i+c['microbatch']])
                        values=dict(innovation1=f[1],warm=warm.describe(f,seeds)[0],**{a:m.describe(f,seeds)[0] for a,m in models.items()})
                        for a,z in values.items():
                            if not torch.isfinite(z).all() or float((z.norm(dim=1)-1).abs().max())>1e-4:raise ValueError('Invalid descriptor')
                            arrays[a][ids[i:i+c['microbatch']].numpy()]=z.cpu().numpy()
                    write(output/'progress.json',dict(phase='evaluation',dataset=tag,images=int(ids[-1])+1,total=len(d['names']),elapsed_seconds=time.monotonic()-start))
                reports={}
                for a in variants:
                    db=torch.tensor(arrays[a][:d['references']],device=c['device']);queries=arrays[a][d['references']:]
                    ranks=np.concatenate([(torch.tensor(q,device=c['device'])@db.T).topk(20,dim=1).indices.cpu().numpy()
                        for q in np.array_split(queries,max(1,(len(queries)+15)//16))])
                    reports[a]=metrics.retrieval_report(ranks,d['ground_truth'],d['references']);del db
            previous=read(prior.old.key.path(prior.configuration(),'runs')/'eval'/(tag+'.json'))
            content_digest=metrics.order_digest([n+':'+h for n,h in zip(d['names'],contents)])
            if content_digest!=previous['image_content_digest']:raise ValueError('Previous image population changed')
            if reports['innovation1']['predictions']!=previous['variants']['innovation1']['predictions']:raise ValueError('Frozen baseline ranks changed')
            if reports['warm']['predictions']!=previous['variants']['structure_gate']['predictions']:raise ValueError('Warm ranks not reproduced')
            comparison=dict(correct={a:r['correct'] for a,r in reports.items()},
                vs_innovation1={a:metrics.paired_report(reports['innovation1'],reports[a]) for a in ARMS},
                vs_warm={a:metrics.paired_report(reports['warm'],reports[a]) for a in ARMS},
                selective_vs_controls={a:metrics.paired_report(reports[a],reports['correct_margin']) for a in ('cosine_only','all_margin')})
            write(target,dict(dataset=contract['datasets'][tag],variants=reports,comparison=comparison,
                image_content_digest=content_digest,elapsed_seconds=time.monotonic()-start))
            write(ready,dict(sha256=sha(target)));results[tag]=comparison;print(tag,comparison['correct'],flush=True)
            del arrays,loader
        checks=dict(frozen_anchor_unchanged=state_hash(first.anchor)==frozen,
            innovation1_sources_unchanged=prior.old.key.source_hashes(prior.old.key.PROTECTED)==contract['protected_sources'],
            checkpoint_unchanged=sha(run/'last.pt')==contract['checkpoint_sha256'])
        primary=results['pitts30k-val'];guard=results['msls-val']
        activity=read(run/'summary.json')['penalty_activity']['correct_margin']['nonzero_penalty_steps']
        signal=(activity>0 and primary['vs_innovation1']['correct_margin']['net']>0 and
            all(v['net']>0 for v in primary['selective_vs_controls'].values()) and
            guard['vs_innovation1']['correct_margin']['net']>=0 and all(v['net']>=0 for v in guard['selective_vs_controls'].values()))
        write(output/'summary.json',dict(checks=checks,results=results,stable_gain_proven=False,blind_test_used=False,
            primary_dataset='pitts30k-val',selective_constraint_active_steps=activity,
            verdict='EXPLORATORY_SELECTIVE_PRESERVATION_SIGNAL' if signal else 'NO_MATCHED_SELECTIVE_PRESERVATION_SIGNAL'))
        if not all(checks.values()):raise ValueError('Integrity check failed')
        finish(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('plan','smoke','train','eval'))
    p.add_argument('--allow-training',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--allow-shared-gpu',action='store_true')
    args=p.parse_args();c=configuration()
    if args.command in ('smoke','train') and not args.allow_training:p.error('Explicit consent required')
    if args.command=='plan':prior.old.key.open_plan(c);print(dict(identity=identity(c),training_started=False));return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':p.error('Physical GPU1 only')
    import fcntl,subprocess
    with (ROOT/'logs/query_relation_screen_v1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        memory=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=20)
        check_gpu_capacity(*map(int,memory.strip().split(',')),args.allow_shared_gpu,c['cuda_memory_fraction'])
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        if args.command=='eval':evaluate(c,args)
        else:train(c,args)


if __name__=='__main__':main()
