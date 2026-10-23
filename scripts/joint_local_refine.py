"""Joint adaptation of innovation-one neck and bounded local-global refinement."""
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
from src.models.joint_local_refine import ARMS, JointLocalVPR, image_seeds
from src.semantic_region_protocol import (read,write,sha,state_hash,owned_run,seed_all,
    save_torch,rng_state,restore_rng,gps_retrieval)

SCHEMA='joint_local_refine_v1'
SOURCES=('src/models/joint_local_refine.py','scripts/joint_local_refine.py',
    'scripts/launch_joint_local_refine.py','tests/test_joint_local_refine.py','config/joint_local_refine.json',
    'src/models/complementary_covariance.py','scripts/single_dino_supervision.py',
    'scripts/eval_clip_key_study.py','scripts/eval_spatial_adapter_repeat.py',
    'src/models/structural_channel_adapter.py','src/models/official_boq.py',
    'doc/JOINT_LOCAL_REFINE_V1.md')


def configuration():
    extra=read(ROOT/'config/joint_local_refine.json')
    expected=dict(schema=SCHEMA,runs='logs/joint_local_refine_v1',
        learning_rate=.0001,neck_learning_rate=.00001,pilot_steps=1024,smoke_steps=8,keep_weight=0.)
    if extra!=expected:raise ValueError('Registered channel experiment changed')
    return dict(old.configuration(),**extra)


def identity(c):
    value=old.identity(c)
    value.update(schema=SCHEMA,new_sources=old.key.source_hashes(SOURCES),
        mechanism='Joint copied innovation-one neck with bounded one-percent local-global post-fusion residual',
        scope='One seed, fixed1024-step development screen; copied neck updates, original checkpoint immutable',
        prior_channel_aggregate_sha256=sha(ROOT/'logs/spatial_adapter_repeat_v1/aggregate/summary.json'),
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


def build(c):
    anchor=old.make_anchor(c);models={}
    for arm in ARMS:
        seed_all(c['seed']);models[arm]=JointLocalVPR(anchor,arm).to(c['device'])
    if state_hash(models['joint_refine'].adapter)!=state_hash(models['joint_permuted'].adapter):raise ValueError('Geometry controls initialization differs')
    ids=[set(map(id,m.adapter.parameters())) for m in models.values()]
    if any(ids[i]&ids[j] for i in range(4) for j in range(i)):raise ValueError('Shared trainable parameters')
    return models

def optimizer_groups(model,c):
    groups=[dict(params=[p for p in model.adapter['neck'].parameters() if p.requires_grad],lr=c['neck_learning_rate'])]
    if 'refine' in model.adapter:groups.append(dict(params=list(model.adapter['refine'].parameters()),lr=c['learning_rate']))
    return [g for g in groups if g['params']]

def replay(model,chunks,seeds,gradients):
    offset=0
    for f in chunks:
        n=len(f[0]);z,_=model.describe(f,seeds[offset:offset+n])
        z.backward(gradients[offset:offset+n]);offset+=n
    if offset!=len(seeds):raise ValueError('Replay coverage differs')


def mined_loss(lossfn,z,labels,coordinates):
    ap,positive,an,negative=lossfn.miner(z,labels)
    xy=torch.deg2rad(coordinates.double().to(z.device));left,right=xy[an],xy[negative]
    a=torch.sin((left[:,0]-right[:,0])/2).square()
    a+=torch.cos(left[:,0])*torch.cos(right[:,0])*torch.sin((left[:,1]-right[:,1])/2).square()
    keep=6371000.*2*torch.asin(a.clamp(0,1).sqrt())>=25.
    return lossfn.loss_fn(z,labels,(ap,positive,an[keep],negative[keep])),dict(
        mined_positive_pairs=len(positive),safe_mined_negative_pairs=int(keep.sum()),
        mined_negative_anchors=int(an[keep].unique().numel()))


def train(c,args):
    from scripts.semantic_region_study import PlaceSteps
    from src.losses.vpr_losses import VPRLossFunction
    plan=old.key.open_plan(c);contract=contract_for(c,args.command);runs=old.key.path(c,'runs')
    if len(plan['schedule'])<c['pilot_steps']:raise ValueError('Insufficient sealed schedule')
    if args.command=='train':
        smoke=verified(runs/'smoke')
        if ({k:v for k,v in smoke.items() if k!='stage'}!={k:v for k,v in contract.items() if k!='stage'}
                or not all(read(runs/'smoke/summary.json')['checks'].values())):raise ValueError('Same-version smoke required')
    output=runs/args.command
    with owned_run(output,contract,args.resume):
        if (output/'completed.json').exists():verified(output);print('Already complete');return
        old.key.gpu(c);models=build(c);first=models[ARMS[0]]
        initial=dict(frozen=state_hash(first.anchor),adapters={a:state_hash(m.adapter) for a,m in models.items()},
            frozen_neck=state_hash(models['frozen_refine'].adapter['neck']),
            auxiliary={a:state_hash(m.adapter['neck'].aux) for a,m in models.items()},
            parameters={a:sum(p.numel() for p in m.adapter.parameters() if p.requires_grad) for a,m in models.items()})
        if (output/'initial.json').exists() and read(output/'initial.json')!=initial:raise ValueError('Initial states changed')
        write(output/'initial.json',initial)
        sample=PlaceSteps(c,plan,plan['schedule'][:1])[0]
        seeds=image_seeds([plan['images'][int(i)]['image_sha256'] for i in sample[3][:2]])
        with torch.no_grad():
            f=first.encode(sample[0][:2].to(c['device']))
            errors={a:float((m.describe(f,seeds)[0]-f[1]).abs().max()) for a,m in models.items()}
        del sample,f
        optimizers={a:torch.optim.AdamW(optimizer_groups(m,c),weight_decay=0.) for a,m in models.items()}
        state=dict(step=0,history=[],contract_sha256=sha(output/'contract.json'),
            gradients={a:{n:False for n,p in m.adapter.named_parameters() if p.requires_grad} for a,m in models.items()})
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
        for images,labels,coords,row_ids in tqdm(loader,total=total,initial=state['step'],desc='joint local refine '+args.command):
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
                    neck_drift=sum(float(v[1]['neck_feature_drift']) for v in temporary)/len(temporary)
                del temporary
                main,mining=mined_loss(lossfn,leaf,labels,coords)
                drift=float((1-(leaf.detach()*reference).sum(1)).clamp_min(0).mean())
                losses=dict(main=main,
                    keep=c['keep_weight']*(1-(leaf*reference).sum(1)).clamp_min(0).mean())
                if not all(bool(torch.isfinite(v)) for v in losses.values()):raise ValueError('Nonfinite loss')
                gradients=torch.autograd.grad(sum(losses.values()),leaf)[0]
                optimizers[arm].zero_grad(set_to_none=True);replay(m,chunks,seeds,gradients)
                for name,p in m.adapter.named_parameters():
                    if p.requires_grad:state['gradients'][arm][name]|=p.grad is not None and bool((p.grad!=0).any())
                norm=torch.nn.utils.clip_grad_norm_(m.adapter.parameters(),c['gradient_clip'],error_if_nonfinite=True)
                optimizers[arm].step()
                row[arm]=dict(losses={n:float(v.detach()) for n,v in losses.items()},gradient_norm=float(norm),
                    relative_delta_mean=float(relative.mean()),neck_feature_drift=neck_drift,
                    descriptor_cosine_drift=drift,mining=mining,
                    main_loss_active=float(losses["main"].detach())>0)
                del leaf,losses,gradients
            torch.cuda.synchronize();state['step']+=1
            row.update(step=state['step'],seconds=time.monotonic()-start,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
            state['history'].append(row);write(output/'progress.json',dict(phase='training',total=total,**row))
            if state['step']%c['checkpoint_every']==0:save()
            del chunks,reference
        save()
        checks=dict(complete_steps=state['step']==total,zero_start=max(errors.values())<=1e-6,
            copied_frozen_neck_unchanged=state_hash(models['frozen_refine'].adapter['neck'])==initial['frozen_neck'],
            auxiliary_unchanged=all(state_hash(m.adapter['neck'].aux)==initial['auxiliary'][a] for a,m in models.items()),
            frozen_anchor_unchanged=state_hash(first.anchor)==initial['frozen'],
            geometry_capacity_matched=initial['parameters']['joint_refine']==initial['parameters']['joint_permuted'],
            all_adapters_updated=all(state_hash(m.adapter)!=initial['adapters'][a] for a,m in models.items()),
            all_parameters_gradient_seen=all(all(v.values()) for v in state['gradients'].values()),
            innovation1_sources_unchanged=old.key.source_hashes(old.key.PROTECTED)==contract['protected_sources'])
        write(output/'summary.json',dict(checks=checks,steps=state['step'],zero_errors=errors,
            effective_loss_steps={a:sum(r[a]['main_loss_active'] for r in state['history']) for a in ARMS},
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
        old.key.gpu(c);models=build(c);first=models[ARMS[0]];frozen=state_hash(first.anchor)
        payload=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        if payload['state']['step']!=c['pilot_steps'] or payload['state']['contract_sha256']!=sha(run/'contract.json'):raise ValueError('Not fixed-final')
        for a,m in models.items():
            m.adapter.load_state_dict(payload['adapters'][a],strict=True)
            if state_hash(m.adapter)!=read(run/'summary.json')['final_adapter_sha256'][a]:raise ValueError('Adapter changed')
            m.requires_grad_(False).eval()
        results={};variants=('innovation1',*ARMS)
        for d in datasets:
            tag=d['tag'];target=output/(tag+'.json');ready=output/(tag+'.ready.json')
            if ready.exists():
                if sha(target)!=read(ready)['sha256']:raise ValueError('Report changed')
                results[tag]=read(target)['comparison'];continue
            start=time.monotonic();contents=[]
            arrays={a:np.empty((len(d['names']),12288),np.float32) for a in variants}
            loader=DataLoader(RGBImages(d['root'],d['names'],d['hashes']),batch_size=16,num_workers=c['workers'],shuffle=False)
            with torch.inference_mode():
                for batch,(images,ids,hashes) in enumerate(tqdm(loader,desc='joint local refine '+tag)):
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
            if reports['innovation1']!=read(baseline)['variants']['innovation1']:raise ValueError('Baseline full Top20 differs')
            comparison=dict(correct={a:r['correct'] for a,r in reports.items()},
                vs_innovation1={a:metrics.paired_report(reports['innovation1'],reports[a]) for a in ARMS},
                joint_vs_controls={a:metrics.paired_report(reports[a],reports['joint_refine']) for a in ('neck_continue','frozen_refine','joint_permuted')})
            write(target,dict(dataset=contract['datasets'][tag],variants=reports,comparison=comparison,
                image_content_digest=metrics.order_digest([n+':'+h for n,h in zip(d['names'],contents)]),elapsed_seconds=time.monotonic()-start))
            write(ready,dict(sha256=sha(target)));results[tag]=comparison;print(tag,comparison['correct'],flush=True)
            del arrays,loader
        checks=dict(frozen_anchor_unchanged=state_hash(first.anchor)==frozen,
            innovation1_sources_unchanged=old.key.source_hashes(old.key.PROTECTED)==contract['protected_sources'],
            checkpoint_unchanged=sha(run/'last.pt')==contract['checkpoint_sha256'])
        primary=results['pitts30k-val'];guard=results['msls-val']
        signal=(primary['vs_innovation1']['joint_refine']['net']>0 and
            all(v['net']>0 for v in primary['joint_vs_controls'].values()) and
            guard['vs_innovation1']['joint_refine']['net']>=0 and
            all(v['net']>=0 for v in guard['joint_vs_controls'].values()) and results['gsv-adaptation-holdout']['vs_innovation1']['joint_refine']['net']>=0)
        write(output/'summary.json',dict(checks=checks,results=results,stable_gain_proven=False,blind_test_used=False,
            primary_dataset='pitts30k-val',msls_guard='Noninferior observed count only, NOT statistical noninferiority proof',
            verdict='EXPLORATORY_JOINT_LOCAL_REFINE_SIGNAL' if signal else 'NO_MATCHED_JOINT_LOCAL_REFINE_SIGNAL'))
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
