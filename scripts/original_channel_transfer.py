"""Archived original channel initialization and checkpoint transfer onto innovation one."""
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

from scripts import single_dino_interaction as old
from scripts.single_dino_supervision import check_gpu_capacity
from src.models.original_channel_transfer import ARMS, OriginalChannelVPR, image_seeds
from src.semantic_region_protocol import (read,write,sha,state_hash,owned_run,seed_all,
    save_torch,rng_state,restore_rng,gps_retrieval)

SCHEMA='original_channel_transfer_v1'
SOURCES=('src/models/original_channel_transfer.py','scripts/original_channel_transfer.py',
    'scripts/launch_original_channel_transfer.py','tests/test_original_channel_transfer.py','config/original_channel_transfer.json',
    'src/models/complementary_covariance.py','scripts/single_dino_supervision.py',
    'scripts/eval_clip_key_study.py','scripts/eval_spatial_adapter_repeat.py',
    'src/models/spatial_adapter_repeat.py','src/models/clip_preboq.py',
    'src/models/structural_channel_adapter.py','scripts/spatial_adapter_repeat.py',
    'config/spatial_adapter_repeat.yaml','doc/ORIGINAL_CHANNEL_TRANSFER_V1.md')


def configuration():
    extra=read(ROOT/'config/original_channel_transfer.json')
    expected=dict(schema=SCHEMA,runs='logs/original_channel_transfer_v1',hidden=64,
        adapter_gain=.1,learning_rate=.0003,pilot_steps=2560,smoke_steps=8,keep_weight=.1)
    if extra!=expected:raise ValueError('Registered channel experiment changed')
    return dict(old.configuration(),**extra)


def identity(c):
    import hashlib,json
    from scripts import spatial_adapter_repeat as original
    source=ROOT/'logs/spatial_adapter_repeat_v1/train_seed42'
    source_contract=original.verified(source)
    digest=hashlib.sha256(json.dumps(old.key.open_plan(c)['schedule'],separators=(',',':')).encode()).hexdigest()
    if digest!=source_contract['schedule_sha256']:raise ValueError('Original seed42 training order differs')
    value=old.identity(c)
    value.update(schema=SCHEMA,new_sources=old.key.source_hashes(SOURCES),
        mechanism='Exact original channel initializer and archived seed42 checkpoint on frozen innovation1',
        scope='One seed, fixed5epoch development experiment; same old data, not blind confirmation',
        prior_channel_aggregate_sha256=sha(ROOT/'logs/spatial_adapter_repeat_v1/aggregate/summary.json'),
        source_checkpoint_sha256=sha(source/'last.pt'),source_completed_sha256=sha(source/'completed.json'),
        original_schedule_sha256=digest,
        baseline_reports={tag:sha(ROOT/'logs/structural_channel_v1/eval'/(tag+'.json'))
            for tag in ('gsv-adaptation-holdout','msls-val','pitts30k-val')})
    return value


def contract_for(c,stage):return dict(stage=stage,**identity(c))


def finish(output):
    files={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*'))
        if p.is_file() and p.name not in ('.lock','completed.json') and not p.name.endswith('.tmp')}
    write(output/'completed.json',dict(complete=True,schema=SCHEMA,files=files))


def verified(output):
    d=read(output/'completed.json')
    if d.get('complete') is not True or d.get('schema')!=SCHEMA:raise ValueError('Incomplete/wrong experiment')
    for name,digest in d['files'].items():
        p=(output/name).resolve()
        if not p.is_relative_to(output.resolve()) or sha(p)!=digest:raise ValueError('Artifact changed: '+name)
    return read(output/'contract.json')


def build(c, include_frozen=False):
    from copy import deepcopy
    from scripts import spatial_adapter_repeat as original
    from src.models.structural_channel_adapter import StructuralChannelAdapter
    source=ROOT/'logs/spatial_adapter_repeat_v1/train_seed42'
    original.verified(source)
    oc=original.configuration(ROOT/'config/spatial_adapter_repeat.yaml')
    paired=original.build(oc,42)
    adapter=deepcopy(paired['channel'].adapter).cpu()
    if state_hash(adapter)!=read(source/'initial.json')['adapters']['channel']:
        raise ValueError('Original initial tensors not exactly reproduced')
    del paired
    torch.cuda.empty_cache()
    anchor=old.make_anchor(c)
    seed_all(c['seed'])
    direct=StructuralChannelAdapter('channel',hidden=c['hidden'],gain=c['adapter_gain'])
    direct_adapter=deepcopy(adapter)
    direct_adapter.down.load_state_dict(direct.down.state_dict())
    direct_adapter.spatial.load_state_dict(direct.middle.state_dict())
    direct_adapter.up.load_state_dict(direct.up.state_dict())
    models=dict(direct=OriginalChannelVPR(anchor,direct_adapter),
        original_init=OriginalChannelVPR(anchor,adapter),transferred=OriginalChannelVPR(anchor,adapter))
    payload=torch.load(source/'last.pt',map_location='cpu',weights_only=True)
    if payload['state']['step']!=2560:raise ValueError('Source not final epoch')
    models['transferred'].adapter.load_state_dict(payload['adapters']['channel'],strict=True)
    if state_hash(models['transferred'].adapter)!=read(source/'summary.json')['adapter_sha256']['channel']:
        raise ValueError('Transferred weights differ')
    if include_frozen:models['transferred_frozen']=OriginalChannelVPR(anchor,models['transferred'].adapter)
    for m in models.values():m.to(c['device'])
    ids=[set(map(id,m.adapter.parameters())) for m in models.values()]
    if any(ids[i]&ids[j] for i in range(len(ids)) for j in range(i)):raise ValueError('Shared adapter tensors')
    return models

def replay(model,chunks,seeds,gradients):
    offset=0
    for f in chunks:
        n=len(f[0]);z,_=model.describe(f,seeds[offset:offset+n])
        z.backward(gradients[offset:offset+n]);offset+=n
    if offset!=len(seeds):raise ValueError('Replay coverage differs')


def train(c,args):
    from scripts.semantic_region_study import PlaceSteps
    from src.losses.vpr_losses import VPRLossFunction
    plan=old.key.open_plan(c);contract=contract_for(c,args.command);runs=old.key.path(c,'runs')
    if len(plan['schedule'])!=c['pilot_steps']:raise ValueError('Expected full5epoch sealed schedule')
    if args.command=='train':
        smoke=verified(runs/'smoke')
        if ({k:v for k,v in smoke.items() if k!='stage'}!={k:v for k,v in contract.items() if k!='stage'}
                or not all(read(runs/'smoke/summary.json')['checks'].values())):raise ValueError('Same-version smoke required')
    output=runs/args.command
    with owned_run(output,contract,args.resume):
        if (output/'completed.json').exists():verified(output);print('Already complete');return
        old.key.gpu(c);models=build(c);first=models[ARMS[0]]
        initial=dict(frozen=state_hash(first.anchor),adapters={a:state_hash(m.adapter) for a,m in models.items()},
            parameters={a:sum(p.numel() for p in m.adapter.parameters()) for a,m in models.items()})
        if (output/'initial.json').exists() and read(output/'initial.json')!=initial:raise ValueError('Initial states changed')
        write(output/'initial.json',initial)
        sample=PlaceSteps(c,plan,plan['schedule'][:1])[0]
        seeds=image_seeds([plan['images'][int(i)]['image_sha256'] for i in sample[3][:2]])
        with torch.no_grad():
            f=first.encode(sample[0][:2].to(c['device']))
            errors={a:float((m.describe(f,seeds)[0]-f[1]).abs().max()) for a,m in models.items()}
        del sample,f
        optimizers={a:torch.optim.AdamW(m.adapter.parameters(),lr=c['learning_rate'],weight_decay=0.) for a,m in models.items()}
        state=dict(step=0,history=[],contract_sha256=sha(output/'contract.json'),
            gradients={a:{n:False for n,p in m.adapter.named_parameters()} for a,m in models.items()})
        def save():
            save_torch(output/'last.pt',dict(adapters={a:m.adapter.state_dict() for a,m in models.items()},
                optimizers={a:o.state_dict() for a,o in optimizers.items()},state=state,rng=rng_state()))
        if (output/'last.pt').exists():
            payload=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
            if payload['state']['contract_sha256']!=state['contract_sha256']:raise ValueError('Resume changed')
            for a,m in models.items():
                m.adapter.load_state_dict(payload['adapters'][a],strict=True);optimizers[a].load_state_dict(payload['optimizers'][a])
            state=payload['state'];restore_rng(payload['rng'])
        else:seed_all(c['seed']);save()
        total=c['smoke_steps'] if args.command=='smoke' else c['pilot_steps']
        if not 0<=state['step']<=total:raise ValueError('Invalid cursor')
        loader=DataLoader(PlaceSteps(c,plan,plan['schedule'][state['step']:total]),batch_size=None,
            num_workers=c['workers'],generator=torch.Generator().manual_seed(c['seed']))
        lossfn=VPRLossFunction()
        for m in models.values():m.train()
        for images,labels,coords,row_ids in tqdm(loader,total=total,initial=state['step'],desc='original channel transfer '+args.command):
            start=time.monotonic();labels=labels.to(c['device'])
            seeds=image_seeds([plan['images'][int(i)]['image_sha256'] for i in row_ids])
            with torch.no_grad():chunks=[first.encode(x.to(c['device'])) for x in images.split(c['microbatch'])]
            reference=torch.cat([f[1] for f in chunks]);row={}
            for arm,m in models.items():
                with torch.no_grad():
                    temporary=[];offset=0
                    for f in chunks:
                        n=len(f[0]);temporary.append(m.describe(f,seeds[offset:offset+n]));offset+=n
                    leaf=torch.cat([v[0] for v in temporary]).detach().requires_grad_()
                    relative=torch.cat([v[1]['relative_delta'] for v in temporary])
                del temporary
                losses=dict(main=gps_retrieval(lossfn,leaf,labels,coords),
                    keep=c['keep_weight']*(1-(leaf*reference).sum(1)).clamp_min(0).mean())
                if not all(bool(torch.isfinite(v)) for v in losses.values()):raise ValueError('Nonfinite loss')
                gradients=torch.autograd.grad(sum(losses.values()),leaf)[0]
                optimizers[arm].zero_grad(set_to_none=True);replay(m,chunks,seeds,gradients)
                for name,p in m.adapter.named_parameters():state['gradients'][arm][name]|=p.grad is not None and bool((p.grad!=0).any())
                norm=torch.nn.utils.clip_grad_norm_(m.adapter.parameters(),c['gradient_clip'],error_if_nonfinite=True)
                optimizers[arm].step()
                row[arm]=dict(losses={n:float(v.detach()) for n,v in losses.items()},gradient_norm=float(norm),
                    relative_delta_mean=float(relative.mean()))
                del leaf,losses,gradients
            torch.cuda.synchronize();state['step']+=1
            row.update(step=state['step'],seconds=time.monotonic()-start,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
            state['history'].append(row);write(output/'progress.json',dict(phase='training',total=total,**row))
            if state['step']%c['checkpoint_every']==0:save()
            del chunks,reference
        save()
        checks=dict(complete_steps=state['step']==total,zero_start=max(errors[a] for a in ('direct','original_init'))<=1e-6,
            frozen_anchor_unchanged=state_hash(first.anchor)==initial['frozen'],
            capacity_matched=len({initial['parameters'][a] for a in ARMS})==1,
            all_adapters_updated=all(state_hash(m.adapter)!=initial['adapters'][a] for a,m in models.items()),
            all_parameters_gradient_seen=all(all(v.values()) for v in state['gradients'].values()),
            innovation1_sources_unchanged=old.key.source_hashes(old.key.PROTECTED)==contract['protected_sources'])
        write(output/'summary.json',dict(checks=checks,steps=state['step'],zero_errors=errors,
            final_adapter_sha256={a:state_hash(m.adapter) for a,m in models.items()},parameters=initial['parameters'],
            encoder_count=1,descriptor_dimensions=12288,verdict='SMOKE_DISCARD' if args.command=='smoke' else 'TRAINED_NOT_EVALUATED'))
        if not all(checks.values()):raise ValueError('Mechanical checks failed: '+str(checks))
        finish(output)


def evaluate(c,args):
    from scripts import eval_clip_key_study as metrics
    from scripts.eval_spatial_adapter_repeat import RGBImages
    run=old.key.path(c,'runs')/'train'
    if verified(run)!=contract_for(c,'train') or not all(read(run/'summary.json')['checks'].values()):raise ValueError('Train incomplete/changed')
    datasets=metrics.populations(c,old.key.open_plan(c))
    contract=dict(stage='eval',**identity(c),checkpoint_sha256=sha(run/'last.pt'),
        datasets={d['tag']:dict(references=d['references'],queries=d['queries'],order=metrics.order_digest(d['names']),gt=metrics.gt_digest(d['ground_truth'])) for d in datasets})
    output=old.key.path(c,'runs')/'eval'
    with owned_run(output,contract,args.resume):
        if (output/'completed.json').exists():verified(output);print('Already evaluated');return
        old.key.gpu(c);models=build(c,include_frozen=True);first=models[ARMS[0]];frozen=state_hash(first.anchor)
        payload=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        if payload['state']['step']!=c['pilot_steps'] or payload['state']['contract_sha256']!=sha(run/'contract.json'):raise ValueError('Not fixed-final')
        for a,m in models.items():
            if a!='transferred_frozen':
                m.adapter.load_state_dict(payload['adapters'][a],strict=True)
                if state_hash(m.adapter)!=read(run/'summary.json')['final_adapter_sha256'][a]:raise ValueError('Adapter changed')
            m.requires_grad_(False).eval()
        results={};variants=('innovation1',*models)
        for d in datasets:
            tag=d['tag'];target=output/(tag+'.json');ready=output/(tag+'.ready.json')
            if ready.exists():
                if sha(target)!=read(ready)['sha256']:raise ValueError('Report changed')
                results[tag]=read(target)['comparison'];continue
            start=time.monotonic();contents=[]
            arrays={a:np.empty((len(d['names']),12288),np.float32) for a in variants}
            loader=DataLoader(RGBImages(d['root'],d['names'],d['hashes']),batch_size=16,num_workers=c['workers'],shuffle=False)
            with torch.inference_mode():
                for batch,(images,ids,hashes) in enumerate(tqdm(loader,desc='original channel transfer '+tag)):
                    if ids.tolist()!=list(range(batch*16,batch*16+len(images))):raise ValueError('Order changed')
                    contents.extend(hashes)
                    for i in range(0,len(images),c['microbatch']):
                        f=first.encode(images[i:i+c['microbatch']].to(c['device']));seeds=image_seeds(hashes[i:i+c['microbatch']])
                        values=dict(innovation1=f[1],**{a:m.describe(f,seeds)[0] for a,m in models.items()})
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
            if tag in ('msls-val','pitts30k-val') and reports['innovation1']['outcomes']!=read(old.key.path(c,'anchor_run')/(tag+'_322.json'))['outcomes']:raise ValueError('Baseline differs')
            baseline=ROOT/'logs/structural_channel_v1/eval'/(tag+'.json')
            if sha(baseline)!=contract['baseline_reports'][tag]:raise ValueError('Baseline archive changed')
            if reports['innovation1']!=read(baseline)['variants']['innovation1']:raise ValueError('Baseline Top20 differs')
            comparison=dict(correct={a:r['correct'] for a,r in reports.items()},
                vs_innovation1={a:metrics.paired_report(reports['innovation1'],reports[a]) for a in models},
                vs_direct={a:metrics.paired_report(reports['direct'],reports[a]) for a in ('original_init','transferred')},
                training_vs_frozen=metrics.paired_report(reports['transferred_frozen'],reports['transferred']))
            write(target,dict(dataset=contract['datasets'][tag],variants=reports,comparison=comparison,
                image_content_digest=metrics.order_digest([n+':'+h for n,h in zip(d['names'],contents)]),elapsed_seconds=time.monotonic()-start))
            write(ready,dict(sha256=sha(target)));results[tag]=comparison;print(tag,comparison['correct'],flush=True)
            del arrays,loader
        checks=dict(frozen_anchor_unchanged=state_hash(first.anchor)==frozen,
            innovation1_sources_unchanged=old.key.source_hashes(old.key.PROTECTED)==contract['protected_sources'],
            checkpoint_unchanged=sha(run/'last.pt')==contract['checkpoint_sha256'],
            transferred_frozen_unchanged=state_hash(models['transferred_frozen'].adapter)==read(ROOT/'logs/spatial_adapter_repeat_v1/train_seed42/summary.json')['adapter_sha256']['channel'])
        primary=results['pitts30k-val'];guard=results['msls-val']
        signals={a:(primary['vs_innovation1'][a]['net']>0 and guard['vs_innovation1'][a]['net']>=0
            and results['gsv-adaptation-holdout']['vs_innovation1'][a]['net']>=0)
            for a in ('original_init','transferred','transferred_frozen')}
        write(output/'summary.json',dict(checks=checks,results=results,anchor_gain=signals,
            initialization_advantage=signals['original_init'] and primary['vs_direct']['original_init']['net']>0,
            warm_start_advantage=signals['transferred'] and primary['vs_direct']['transferred']['net']>0,
            stable_gain_proven=False,blind_test_used=False,
            verdict='EXPLORATORY_TRANSFER_GAIN' if any(signals.values()) else 'NO_ORIGINAL_CHANNEL_TRANSFER_GAIN'))
        if not all(checks.values()):raise ValueError('Integrity failed')
        finish(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('plan','smoke','train','eval'))
    p.add_argument('--allow-training',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--allow-shared-gpu',action='store_true')
    args=p.parse_args();c=configuration()
    if args.command in ('smoke','train') and not args.allow_training:p.error('Explicit training consent required')
    if args.command=='plan':old.key.open_plan(c);print(dict(identity=identity(c),training_started=False));return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':p.error('Physical GPU1 only')
    import fcntl,subprocess
    with (ROOT/'logs/query_relation_screen_v1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        memory=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=20)
        check_gpu_capacity(*map(int,memory.strip().split(',')),args.allow_shared_gpu,c['cuda_memory_fraction'])
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        if args.command=='eval':evaluate(c,args)
        else:train(c,args)


if __name__=='__main__':main()
