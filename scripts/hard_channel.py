"""Train-only hard-place batching x preservation, fixed512 steps and exposure."""
import argparse
import io
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
from tqdm import tqdm

from scripts import channel_preservation as repair
from scripts import clip_key_study as key
from scripts.single_dino_supervision import check_gpu_capacity
from src.models.structural_channel_adapter import StructuralChannelVPR,image_seeds
from src.channel_rank_preservation import teacher_contract,ranking_penalty
from src.hard_channel_batching import ARMS,validate_partition,pack_places,difficulty_gate
from src.semantic_region_protocol import (read,write,sha,state_hash,owned_run,seed_all,
    save_torch,save_npz,rng_state,restore_rng,gps_retrieval)

SCHEMA='hard_channel_v1'
SOURCES=('src/hard_channel_batching.py','scripts/hard_channel.py','scripts/launch_hard_channel.py',
    'config/hard_channel.json','tests/test_hard_channel.py',*repair.SOURCES)


def configuration():
    extra=read(ROOT/'config/hard_channel.json')
    expected=dict(schema=SCHEMA,runs='logs/hard_channel_v1',mining_cache='.cache/hard_channel_v1',
        pilot_steps=512,smoke_steps=8,neighbor_count=64,cache_places_per_shard=16,graph_query_chunk=128)
    if extra!=expected:raise ValueError('Registered hard-batching experiment changed')
    return dict(repair.configuration(),**extra)


def identity(c):
    value=repair.identity(c)
    value.update(schema=SCHEMA,hard_sources=key.source_hashes(tuple(dict.fromkeys(SOURCES))),
        exposure='Exactly same8192 train places x4 views, once per group. Only within-batch co-occurrence changes.',
        scope='2x2 train sampling x preservation, one fixed warm head/seed,512 steps; not a new architecture or blind validation')
    return value


def finish(output):
    files={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*'))
        if p.is_file() and p.name not in ('.lock','completed.json') and not p.name.endswith('.tmp')}
    write(output/'completed.json',dict(complete=True,schema=SCHEMA,files=files))


def verified(output):
    d=read(output/'completed.json')
    if d.get('complete') is not True or d.get('schema')!=SCHEMA:raise ValueError('Incomplete/wrong hard-batch artifact')
    for name,h in d['files'].items():
        p=(output/name).resolve()
        if not p.is_relative_to(output.resolve()) or sha(p)!=h:raise ValueError('Hard-batch artifact changed: '+name)
    return read(output/'contract.json')


def warm_model(c,anchor=None):
    if anchor is None:anchor=repair.prior.old.make_anchor(c)
    seed_all(c['seed'])
    m=StructuralChannelVPR(anchor,'structure_gate',hidden=c['hidden'],rank=c['gate_rank'],gain=c['adapter_gain']).to(c['device'])
    payload=torch.load(key.path(c,'warm_run')/'last.pt',map_location='cpu',weights_only=True)
    m.adapter.load_state_dict(payload['adapters'][c['warm_arm']],strict=True)
    if state_hash(m.adapter)!=repair.warm_identity(c)['adapter_sha256']:raise ValueError('Warm tensors changed')
    return m


def build(c):
    first=warm_model(c);models={ARMS[0]:first}
    for arm in ARMS[1:]:models[arm]=warm_model(c,first.anchor)
    if len({state_hash(m.adapter) for m in models.values()})!=1:raise ValueError('Warm heads differ')
    ids=[set(map(id,m.adapter.parameters())) for m in models.values()]
    if any(ids[i]&ids[j] for i in range(4) for j in range(i)):raise ValueError('Heads shared')
    return models


def cache(c,args):
    from scripts.semantic_region_study import PlaceSteps
    plan=key.open_plan(c);output=key.path(c,'mining_cache')/'descriptors'
    contract=dict(stage='train_only_descriptor_cache',**identity(c),format='place-major4 unit descriptors, FP16 storage for mining ONLY')
    with owned_run(output,contract,args.resume):
        if (output/'completed.json').exists():verified(output);print('Train descriptor cache complete');return
        count=c['train_places'];shard_places=c['cache_places_per_shard'];done=[];todo=[]
        def expected_ids(index):return [i for p in range(index*shard_places,(index+1)*shard_places) for i in plan['places'][p]['rows']]
        def load(index):
            target=output/f'{index:04d}.npz';record=output/f'{index:04d}.ready.json'
            if sha(target)!=read(record)['sha256']:raise ValueError('Cache shard damaged: '+str(index))
            with np.load(target,allow_pickle=False) as z:
                x=z['descriptors'];ids=z['row_ids']
                if x.shape!=(shard_places*4,12288) or x.dtype!=np.float16 or ids.tolist()!=expected_ids(index) or not np.isfinite(x).all():raise ValueError('Invalid train shard')
                if np.max(np.abs(np.linalg.norm(x.astype(np.float32),axis=1)-1))>.002:raise ValueError('Nonunit train descriptor')
        for index in range(count//shard_places):
            if (output/f'{index:04d}.ready.json').exists():load(index);done.append(index)
            else:todo.append(index)
        print(f'Train descriptors: reuse{len(done)} shards; extract{len(todo)}; total storage about768MiB; no patches',flush=True)
        if todo:
            key.gpu(c);model=warm_model(c).requires_grad_(False).eval();frozen=state_hash(model.anchor)
            dataset=PlaceSteps(c,plan,[[p] for p in range(count)])
            positions=[p for i in todo for p in range(i*shard_places,(i+1)*shard_places)]
            loader=DataLoader(Subset(dataset,positions),batch_size=shard_places,num_workers=c['workers'],shuffle=False)
            with torch.inference_mode():
                for index,(images,labels,coords,row_ids) in zip(todo,tqdm(loader,total=len(todo),desc='GSV train-only descriptors')):
                    flat=images.flatten(0,1);ids=row_ids.flatten().tolist()
                    if ids!=expected_ids(index):raise ValueError('Train cache order changed')
                    z=torch.cat([model.encode(x.to(c['device']))[1] for x in flat.split(c['microbatch'])])
                    save_npz(output/f'{index:04d}.npz',descriptors=z.cpu().numpy().astype(np.float16),row_ids=np.asarray(ids,np.int64))
                    write(output/f'{index:04d}.ready.json',dict(sha256=sha(output/f'{index:04d}.npz')));load(index)
                    write(output/'progress.json',dict(phase='cache',shards_complete=len(done)+todo.index(index)+1,total_shards=count//shard_places))
            if state_hash(model.anchor)!=frozen:raise ValueError('Frozen extraction weights changed')
        write(output/'summary.json',dict(train_places=count,images=count*4,dimensions=12288,storage_dtype='float16',
            training_started=False,no_holdout=True,no_patch_cache=True))
        finish(output)


def bank(c):
    root=key.path(c,'mining_cache')/'descriptors';verified(root)
    values=np.empty((c['train_places']*4,12288),np.float16);width=c['cache_places_per_shard']*4
    for index in range(c['train_places']//c['cache_places_per_shard']):
        with np.load(root/f'{index:04d}.npz',allow_pickle=False) as z:values[index*width:(index+1)*width]=z['descriptors']
    return values


def audit_partition(values,batches,plan,c):
    summary=dict(queries=0,valid=0,errors=0,low_margin=0)
    for batch in tqdm(batches,desc='Audit batch difficulty'):
        ids=[4*p+v for p in batch for v in range(4)]
        rows=[i for p in batch for i in plan['places'][p]['rows']]
        descriptors=F.normalize(torch.from_numpy(values[ids]).to(c['device']).float(),dim=1)
        labels=torch.tensor(batch,device=c['device']).repeat_interleave(4)
        coordinates=torch.tensor([plan['images'][i]['coordinates'] for i in rows],dtype=torch.float64)
        teacher=teacher_contract(descriptors,labels,coordinates,c['rank_cap'])
        scores=descriptors@descriptors.T
        gap=scores.masked_fill(~teacher['positive'],-torch.inf).amax(1)-scores.masked_fill(~teacher['negative'],-torch.inf).amax(1)
        valid=teacher['valid'];summary['queries']+=len(ids);summary['valid']+=int(valid.sum())
        summary['errors']+=int((valid & (gap<=0)).sum());summary['low_margin']+=int((valid & (gap<=c['rank_cap'])).sum())
    summary['teacher_correct_fraction']=1-summary['errors']/max(1,summary['valid'])
    return summary


def mine(c,args):
    plan=key.open_plan(c);base=key.path(c,'mining_cache');verified(base/'descriptors')
    contract=dict(stage='equal_exposure_mining',**identity(c),cache_completed_sha256=sha(base/'descriptors/completed.json'))
    output=base/'mining'
    with owned_run(output,contract,args.resume):
        if (output/'completed.json').exists():verified(output);print('Mining complete');return
        key.gpu(c);values=bank(c)
        prototypes=values.reshape(c['train_places'],4,12288).mean(1,dtype=np.float32)
        prototypes/=np.maximum(np.linalg.norm(prototypes,axis=1,keepdims=True),1e-8)
        proto=torch.from_numpy(prototypes).to(c['device'])
        nearest=[]
        with torch.inference_mode():
            for start in tqdm(range(0,len(proto),c['graph_query_chunk']),desc='Frozen train place neighbors'):
                stop=min(start+c['graph_query_chunk'],len(proto));scores=proto[start:stop]@proto.T
                scores[torch.arange(stop-start,device=c['device']),torch.arange(start,stop,device=c['device'])]=-torch.inf
                nearest.append(scores.topk(c['neighbor_count'],dim=1).indices.cpu().numpy())
            neighbors=np.concatenate(nearest)
            def fallback(anchor,unused):
                scores=(proto[anchor:anchor+1]@proto[unused].T).flatten().cpu().numpy()
                return np.asarray(unused)[np.lexsort((np.asarray(unused),-scores))].tolist()
            hard,stats=pack_places(neighbors,c['seed'],fallback,c['places_per_batch'])
            random=plan['schedule'][:c['pilot_steps']]
            validate_partition(random,c['train_places'],c['places_per_batch'])
            random_audit=audit_partition(values,random,plan,c);hard_audit=audit_partition(values,hard,plan,c)
        gate=difficulty_gate(random_audit,hard_audit,c['train_places']*4)
        write(output/'schedule.json',dict(random=random,hard=hard))
        save_npz(output/'neighbors.npz',neighbors=neighbors.astype(np.int32))
        write(output/'summary.json',dict(random=random_audit,hard=hard_audit,gate=gate,packing=stats,
            note='FP16 cached descriptors normalized toFP32 for mining/audit; training recomputes FP32. GPS safety applied in audit/loss, not in prototype graph.',
            verdict='READY_FOR_MATCHED_TRAINING' if gate['pass_all'] else 'INSUFFICIENT_BATCH_DIFFICULTY'))
        finish(output);print(read(output/'summary.json'),flush=True)


def mining_ready(c):
    base=key.path(c,'mining_cache');verified(base/'descriptors');verified(base/'mining')
    report=read(base/'mining/summary.json')
    if not report['gate']['pass_all']:raise ValueError('Hard batches failed registered difficulty gate; NO training launched')
    schedules=read(base/'mining/schedule.json')
    for batches in schedules.values():validate_partition(batches,c['train_places'],c['places_per_batch'])
    return schedules


def training_contract(c,stage):
    mining_ready(c)
    return dict(stage=stage,**identity(c),mining_completed_sha256=sha(key.path(c,'mining_cache')/'mining/completed.json'))


def train(c,args):
    from scripts.semantic_region_study import PlaceSteps
    from src.losses.vpr_losses import VPRLossFunction
    plan=key.open_plan(c);schedules=mining_ready(c);contract=training_contract(c,args.command);runs=key.path(c,'runs')
    if args.command=='train':
        smoke=verified(runs/'smoke')
        if ({k:v for k,v in smoke.items() if k!='stage'}!={k:v for k,v in contract.items() if k!='stage'}
                or not all(read(runs/'smoke/summary.json')['checks'].values())):raise ValueError('Same-version smoke required')
    output=runs/args.command
    with owned_run(output,contract,args.resume):
        if (output/'completed.json').exists():verified(output);print('Already complete');return
        key.gpu(c);models=build(c);first=models[ARMS[0]]
        initial=dict(frozen=state_hash(first.anchor),adapters={a:state_hash(m.adapter) for a,m in models.items()},
            parameters={a:sum(p.numel() for p in m.adapter.parameters()) for a,m in models.items()})
        if (output/'initial.json').exists() and read(output/'initial.json')!=initial:raise ValueError('Warm state changed')
        write(output/'initial.json',initial)
        optimizers={a:torch.optim.AdamW(m.adapter.parameters(),lr=c['learning_rate'],weight_decay=0.) for a,m in models.items()}
        state=dict(step=0,history=[],contract_sha256=sha(output/'contract.json'),
            gradients={a:{n:False for n,p in m.adapter.named_parameters()} for a,m in models.items()})
        def save():save_torch(output/'last.pt',dict(adapters={a:m.adapter.state_dict() for a,m in models.items()},
            optimizers={a:o.state_dict() for a,o in optimizers.items()},state=state,rng=rng_state()))
        if (output/'last.pt').exists():
            payload=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
            if payload['state']['contract_sha256']!=state['contract_sha256']:raise ValueError('Resume contract changed')
            for a,m in models.items():
                m.adapter.load_state_dict(payload['adapters'][a],strict=True);optimizers[a].load_state_dict(payload['optimizers'][a])
            state=payload['state'];restore_rng(payload['rng'])
        else:seed_all(c['seed']);save()
        total=c['smoke_steps'] if args.command=='smoke' else c['pilot_steps']
        if not 0<=state['step']<=total:raise ValueError('Invalid cursor')
        loaders={name:DataLoader(PlaceSteps(c,plan,schedule[state['step']:total]),batch_size=None,
            num_workers=c['workers'],generator=torch.Generator().manual_seed(c['seed'])) for name,schedule in schedules.items()}
        lossfn=VPRLossFunction()
        for m in models.values():m.train()
        for random_batch,hard_batch in tqdm(zip(loaders['random'],loaders['hard']),total=total,initial=state['step'],desc='hard batch '+args.command):
            start=time.monotonic();row={}
            for sampling,batch in (('random',random_batch),('hard',hard_batch)):
                images,labels,coords,row_ids=batch;labels=labels.to(c['device'])
                seeds=image_seeds([plan['images'][int(i)]['image_sha256'] for i in row_ids])
                with torch.no_grad():chunks=[first.encode(x.to(c['device'])) for x in images.split(c['microbatch'])]
                reference=torch.cat([f[1] for f in chunks]);teacher=teacher_contract(reference,labels,coords,c['rank_cap'])
                for suffix in ('ms','protect'):
                    arm=sampling+'_'+suffix;m=models[arm]
                    with torch.no_grad():
                        temporary=[];offset=0
                        for f in chunks:
                            n=len(f[0]);temporary.append(m.describe(f,seeds[offset:offset+n]));offset+=n
                        leaf=torch.cat([v[0] for v in temporary]).detach().requires_grad_()
                        relative=torch.cat([v[1]['relative_delta'] for v in temporary])
                    del temporary
                    rank,stats=ranking_penalty(leaf,teacher,'correct_margin' if suffix=='protect' else 'cosine_only')
                    losses=dict(main=gps_retrieval(lossfn,leaf,labels,coords),
                        keep=c['keep_weight']*(1-(leaf*reference).sum(1)).clamp_min(0).mean(),rank=c['rank_weight']*rank)
                    if not all(bool(torch.isfinite(v)) for v in losses.values()):raise ValueError('Nonfinite loss')
                    rank_gradient=torch.autograd.grad(losses['rank'],leaf,retain_graph=True)[0]
                    rank_gradient_norm=float(rank_gradient.norm().detach())
                    gradients=torch.autograd.grad(sum(losses.values()),leaf)[0]
                    optimizers[arm].zero_grad(set_to_none=True);repair.prior.replay(m,chunks,seeds,gradients)
                    for name,p in m.adapter.named_parameters():state['gradients'][arm][name]|=p.grad is not None and bool((p.grad!=0).any())
                    norm=torch.nn.utils.clip_grad_norm_(m.adapter.parameters(),c['gradient_clip'],error_if_nonfinite=True)
                    optimizers[arm].step()
                    row[arm]=dict(losses={n:float(v.detach()) for n,v in losses.items()},gradient_norm=float(norm),ranking=stats,
                        relative_delta_mean=float(relative.mean()),rank_descriptor_gradient_norm=rank_gradient_norm)
                    del leaf,losses,gradients,rank_gradient
                del chunks,reference,teacher
            torch.cuda.synchronize();state['step']+=1
            row.update(step=state['step'],seconds=time.monotonic()-start,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
            state['history'].append(row);write(output/'progress.json',dict(phase='training',total=total,**row))
            if state['step']%c['checkpoint_every']==0:save()
        save()
        checks=dict(complete_steps=state['step']==total,frozen_anchor_unchanged=state_hash(first.anchor)==initial['frozen'],
            matched_capacity=len(set(initial['parameters'].values()))==1,
            all_adapters_updated=all(state_hash(m.adapter)!=initial['adapters'][a] for a,m in models.items()),
            all_parameters_gradient_seen=all(all(v.values()) for v in state['gradients'].values()),
            innovation1_sources_unchanged=key.source_hashes(key.PROTECTED)==contract['protected_sources'])
        activity={a:dict(teacher_errors=sum(v[a]['ranking']['valid']-v[a]['ranking']['teacher_correct'] for v in state['history']),
            violating_rows=sum(v[a]['ranking']['violating'] for v in state['history']),
            nonzero_penalty_steps=sum(v[a]['losses']['rank']>0 for v in state['history']),
            nonzero_rank_gradient_steps=sum(v[a]['rank_descriptor_gradient_norm']>0 for v in state['history'])) for a in ARMS}
        write(output/'summary.json',dict(checks=checks,steps=state['step'],penalty_activity=activity,
            final_adapter_sha256={a:state_hash(m.adapter) for a,m in models.items()},parameters=initial['parameters'],
            encoder_count=1,descriptor_dimensions=12288,verdict='SMOKE_DISCARD' if args.command=='smoke' else 'TRAINED_NOT_EVALUATED'))
        if not all(checks.values()):raise ValueError('Mechanical check failed: '+str(checks))
        finish(output)


def evaluate(c,args):
    from scripts import eval_clip_key_study as metrics
    from scripts.eval_spatial_adapter_repeat import RGBImages
    run=key.path(c,'runs')/'train'
    if verified(run)!=training_contract(c,'train') or not all(read(run/'summary.json')['checks'].values()):raise ValueError('Train incomplete/changed')
    datasets=metrics.populations(c,key.open_plan(c))
    contract=dict(stage='eval',**identity(c),checkpoint_sha256=sha(run/'last.pt'),
        datasets={d['tag']:dict(references=d['references'],queries=d['queries'],order=metrics.order_digest(d['names']),gt=metrics.gt_digest(d['ground_truth'])) for d in datasets})
    output=key.path(c,'runs')/'eval'
    with owned_run(output,contract,args.resume):
        if (output/'completed.json').exists():verified(output);print('Already evaluated');return
        key.gpu(c);models=build(c);first=models[ARMS[0]];frozen=state_hash(first.anchor)
        warm=warm_model(c,first.anchor).requires_grad_(False).eval()
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
            start=time.monotonic();contents=[];arrays={a:np.empty((len(d['names']),12288),np.float32) for a in variants}
            loader=DataLoader(RGBImages(d['root'],d['names'],d['hashes']),batch_size=16,num_workers=c['workers'],shuffle=False)
            with torch.inference_mode():
                for batch,(images,ids,hashes) in enumerate(tqdm(loader,desc='hard batch '+tag)):
                    if ids.tolist()!=list(range(batch*16,batch*16+len(images))):raise ValueError('Order changed')
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
            previous=read(key.path(repair.prior.configuration(),'runs')/'eval'/(tag+'.json'))
            digest=metrics.order_digest([n+':'+h for n,h in zip(d['names'],contents)])
            if digest!=previous['image_content_digest']:raise ValueError('Previous image bytes/order changed')
            if reports['innovation1']['predictions']!=previous['variants']['innovation1']['predictions']:raise ValueError('Baseline not reproduced')
            if reports['warm']['predictions']!=previous['variants']['structure_gate']['predictions']:raise ValueError('Warm not reproduced')
            comparison=dict(correct={a:r['correct'] for a,r in reports.items()},
                vs_innovation1={a:metrics.paired_report(reports['innovation1'],reports[a]) for a in ARMS},
                hard_vs_random_ms=metrics.paired_report(reports['random_ms'],reports['hard_ms']),
                hard_vs_random_protect=metrics.paired_report(reports['random_protect'],reports['hard_protect']),
                protect_within_hard=metrics.paired_report(reports['hard_ms'],reports['hard_protect']))
            write(target,dict(dataset=contract['datasets'][tag],variants=reports,comparison=comparison,image_content_digest=digest,elapsed_seconds=time.monotonic()-start))
            write(ready,dict(sha256=sha(target)));results[tag]=comparison;print(tag,comparison['correct'],flush=True);del arrays,loader
        checks=dict(frozen_anchor_unchanged=state_hash(first.anchor)==frozen,
            innovation1_sources_unchanged=key.source_hashes(key.PROTECTED)==contract['protected_sources'],
            checkpoint_unchanged=sha(run/'last.pt')==contract['checkpoint_sha256'])
        primary=results['pitts30k-val'];guard=results['msls-val'];activity=read(run/'summary.json')['penalty_activity']
        sampling=(primary['vs_innovation1']['hard_ms']['net']>0 and primary['hard_vs_random_ms']['net']>0
            and guard['vs_innovation1']['hard_ms']['net']>=0 and guard['hard_vs_random_ms']['net']>=0)
        protection=(primary['vs_innovation1']['hard_protect']['net']>0 and primary['protect_within_hard']['net']>0
            and primary['hard_vs_random_protect']['net']>0 and guard['vs_innovation1']['hard_protect']['net']>=0
            and guard['protect_within_hard']['net']>=0 and guard['hard_vs_random_protect']['net']>=0
            and activity['hard_protect']['nonzero_rank_gradient_steps']>0)
        write(output/'summary.json',dict(checks=checks,results=results,stable_gain_proven=False,blind_test_used=False,
            sampling_signal=sampling,protection_signal=protection,penalty_activity=activity,
            verdict='EXPLORATORY_HARD_BATCH_SIGNAL' if sampling or protection else 'NO_MATCHED_HARD_BATCH_SIGNAL'))
        if not all(checks.values()):raise ValueError('Integrity failure')
        finish(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('plan','cache','mine','smoke','train','eval'))
    p.add_argument('--allow-training',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--allow-shared-gpu',action='store_true')
    args=p.parse_args();c=configuration()
    if args.command in ('smoke','train') and not args.allow_training:p.error('Explicit training consent required')
    if args.command=='plan':
        key.open_plan(c);print(dict(identity=identity(c),training_started=False,estimated_cache_mib=768));return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':p.error('Physical GPU1 only')
    import fcntl,subprocess
    with (ROOT/'logs/query_relation_screen_v1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        memory=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=20)
        check_gpu_capacity(*map(int,memory.strip().split(',')),args.allow_shared_gpu,c['cuda_memory_fraction'])
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        {'cache':cache,'mine':mine,'smoke':train,'train':train,'eval':evaluate}[args.command](c,args)


if __name__=='__main__':main()
