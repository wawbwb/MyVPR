"""Global image negatives, freshly re-encoded on both sides; matched2x2 study."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm

from scripts import hard_channel as old
from scripts import clip_key_study as key
from scripts.single_dino_supervision import check_gpu_capacity
from src.image_negative_pairs import ARMS,full_train_negatives,random_train_negatives,paired_contract,paired_losses
from src.hard_channel_batching import validate_partition,difficulty_gate
from src.models.structural_channel_adapter import image_seeds
from src.semantic_region_protocol import (read,write,sha,state_hash,owned_run,seed_all,
    save_torch,save_npz,rng_state,restore_rng,gps_retrieval)

SCHEMA='image_negative_v1'
SOURCES=('src/image_negative_pairs.py','scripts/image_negative.py','scripts/launch_image_negative.py',
    'config/image_negative.json','tests/test_image_negative.py',*old.SOURCES)


def configuration():
    extra=read(ROOT/'config/image_negative.json')
    expected=dict(schema=SCHEMA,runs='logs/image_negative_v1',pair_cache='.cache/image_negative_v1',
        pilot_steps=512,smoke_steps=8,pair_weight=.1,pair_margin=.05,pair_temperature=.05)
    if extra!=expected:raise ValueError('Registered image-negative protocol changed')
    return dict(old.configuration(),**extra)


def cache_identity(c):
    root=key.path(c,'mining_cache')/'descriptors';actual=old.verified(root)
    expected=dict(stage='train_only_descriptor_cache',**old.identity(old.configuration()),
        format='place-major4 unit descriptors, FP16 storage for mining ONLY')
    if actual!=expected:raise ValueError('Original frozen descriptor cache contract changed')
    return sha(root/'completed.json')


def identity(c):
    value=old.identity(c)
    value.update(schema=SCHEMA,image_negative_sources=key.source_hashes(tuple(dict.fromkeys(SOURCES))),
        descriptor_cache_sha256=cache_identity(c),
        exposure='Same64 base queries +64 explicit negative slots per step/arm; negatives can repeat, fresh student encoding on BOTH sides',
        scope='Global image mining x correct-margin protection,512step single-seed development study; NOT a new encoder/module')
    return value


def finish(output):
    summary=output/'summary.json'
    if summary.exists():
        d=read(summary)
        if d.get('verdict')=='EXPLORATORY_HARD_BATCH_SIGNAL':d['verdict']='EXPLORATORY_IMAGE_NEGATIVE_SIGNAL'
        elif d.get('verdict')=='NO_MATCHED_HARD_BATCH_SIGNAL':d['verdict']='NO_MATCHED_IMAGE_NEGATIVE_SIGNAL'
        write(summary,d)
    files={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*'))
        if p.is_file() and p.name not in ('.lock','completed.json') and not p.name.endswith('.tmp')}
    write(output/'completed.json',dict(complete=True,schema=SCHEMA,files=files))


def verified(output):
    d=read(output/'completed.json')
    if d.get('complete') is not True or d.get('schema')!=SCHEMA:raise ValueError('Incomplete/wrong pair artifact')
    for name,h in d['files'].items():
        p=(output/name).resolve()
        if not p.is_relative_to(output.resolve()) or sha(p)!=h:raise ValueError('Pair artifact changed: '+name)
    return read(output/'contract.json')


def population(plan,c):
    rows=[i for p in plan['places'][:c['train_places']] for i in p['rows']]
    if len(rows)!=c['train_places']*4 or len(set(rows))!=len(rows) or any(plan['images'][i]['split']!='train' for i in rows):
        raise ValueError('Only unique sealed training images may enter mining')
    return rows


def mine(c,args):
    plan=key.open_plan(c);rows=population(plan,c);output=key.path(c,'pair_cache')
    contract=dict(stage='global_image_negative_mining',**identity(c))
    with owned_run(output,contract,args.resume):
        if (output/'completed.json').exists():verified(output);print('Image pairs complete');return
        key.gpu(c);values=old.bank(c)
        vectors=torch.from_numpy(values).to(device=c['device'],dtype=torch.float32)
        vectors.div_(vectors.norm(dim=1,keepdim=True).clamp_min(1e-8))
        labels=torch.arange(c['train_places'],device=c['device']).repeat_interleave(4)
        coords=torch.tensor([plan['images'][i]['coordinates'] for i in rows],device=c['device'],dtype=torch.float64)
        hard,gaps=full_train_negatives(vectors,labels,coords,c['graph_query_chunk'])
        random=random_train_negatives(labels.cpu().numpy(),coords.cpu().numpy(),c['seed'])
        schedule=plan['schedule'][:c['pilot_steps']];validate_partition(schedule,c['train_places'],c['places_per_batch'])
        reports={}
        with torch.inference_mode():
            for name,negatives in (('random',random),('hard',hard)):
                report=dict(queries=0,valid=0,errors=0,low_margin=0)
                for batch in tqdm(schedule,desc='Audit direct image pairs '+name):
                    base=np.asarray([4*p+v for p in batch for v in range(4)])
                    ids=np.concatenate((base,negatives[base]));idx=torch.from_numpy(ids).to(c['device'])
                    teacher=paired_contract(vectors[idx],labels[idx],coords[idx],c['rank_cap'])
                    report['queries']+=len(base);report['valid']+=len(base)
                    report['errors']+=int((~teacher['correct']).sum());report['low_margin']+=int((teacher['margin']<=c['rank_cap']).sum())
                report['unique_negative_images']=int(len(np.unique(negatives)));reports[name]=report
        gate=difficulty_gate(reports['random'],reports['hard'],len(rows))
        save_npz(output/'pairs.npz',random=random.astype(np.int32),hard=hard.astype(np.int32),row_ids=np.asarray(rows,np.int64),full_train_margins=gaps)
        write(output/'summary.json',dict(random=reports['random'],hard=reports['hard'],gate=gate,
            full_train=dict(queries=len(rows),errors=int((gaps<=0).sum()),low_margin=int((gaps<=c['rank_cap']).sum())),
            verdict='READY_FOR_MATCHED_TRAINING' if gate['pass_all'] else 'INSUFFICIENT_IMAGE_PAIR_DIFFICULTY',
            note='FP16-cache selection/audit only. BOTH query and chosen negative images freshly encoded and differentiated in training.'))
        finish(output);print(read(output/'summary.json'),flush=True)


def pairs_ready(c):
    root=key.path(c,'pair_cache');verified(root)
    report=read(root/'summary.json')
    if not report['gate']['pass_all']:raise ValueError('Image pair difficulty gate failed; no training')
    with np.load(root/'pairs.npz',allow_pickle=False) as z:
        return {k:z[k].copy() for k in ('random','hard','row_ids')}


def training_contract(c,stage):
    pairs_ready(c)
    return dict(stage=stage,**identity(c),pairs_completed_sha256=sha(key.path(c,'pair_cache')/'completed.json'))


class PairSteps(Dataset):
    def __init__(self,c,plan,steps,negative_ids):
        from scripts.semantic_region_study import ImageRows
        self.steps,self.plan,self.rows,self.negatives=steps,plan,population(plan,c),np.asarray(negative_ids)
        if self.negatives.shape!=(len(self.rows),) or not np.issubdtype(self.negatives.dtype,np.integer) or ((self.negatives<0)|(self.negatives>=len(self.rows))).any():
            raise ValueError('Invalid pair mapping')
        self.images=ImageRows(key.path(c,'dataset_root'),plan['images'],c['student_size'])

    def __len__(self):return len(self.steps)

    def __getitem__(self,index):
        base=np.asarray([4*p+v for p in self.steps[index] for v in range(4)])
        ids=np.concatenate((base,self.negatives[base]));rows=[self.rows[int(i)] for i in ids]
        pixels=torch.stack([self.images[i][0] for i in rows])
        labels=torch.from_numpy(ids//4).long()
        coordinates=torch.tensor([self.plan['images'][i]['coordinates'] for i in rows],dtype=torch.float64)
        return pixels,labels,coordinates,torch.tensor(rows,dtype=torch.long)


def train(c,args):
    from src.losses.vpr_losses import VPRLossFunction
    plan=key.open_plan(c);pairs=pairs_ready(c);contract=training_contract(c,args.command);runs=key.path(c,'runs')
    if pairs['row_ids'].tolist()!=population(plan,c):raise ValueError('Pair source rows changed')
    if args.command=='train':
        smoke=verified(runs/'smoke')
        if ({k:v for k,v in smoke.items() if k!='stage'}!={k:v for k,v in contract.items() if k!='stage'}
                or not all(read(runs/'smoke/summary.json')['checks'].values())):raise ValueError('Same-version smoke required')
    output=runs/args.command
    with owned_run(output,contract,args.resume):
        if (output/'completed.json').exists():verified(output);print('Already complete');return
        key.gpu(c);models=old.build(c);first=models[ARMS[0]]
        initial=dict(frozen=state_hash(first.anchor),adapters={a:state_hash(m.adapter) for a,m in models.items()},
            parameters={a:sum(p.numel() for p in m.adapter.parameters()) for a,m in models.items()})
        if (output/'initial.json').exists() and read(output/'initial.json')!=initial:raise ValueError('Warm initialization changed')
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
        steps=plan['schedule'][state['step']:total]
        loaders={name:DataLoader(PairSteps(c,plan,steps,pairs[name]),batch_size=None,num_workers=c['workers'],
            generator=torch.Generator().manual_seed(c['seed'])) for name in ('random','hard')}
        lossfn=VPRLossFunction()
        for m in models.values():m.train()
        for random_batch,hard_batch in tqdm(zip(loaders['random'],loaders['hard']),total=total,initial=state['step'],desc='image negatives '+args.command):
            start=time.monotonic();row={}
            for sampling,batch in (('random',random_batch),('hard',hard_batch)):
                images,labels,coords,row_ids=batch;labels=labels.to(c['device']);n=len(images)//2
                seeds=image_seeds([plan['images'][int(i)]['image_sha256'] for i in row_ids])
                with torch.no_grad():chunks=[first.encode(x.to(c['device'])) for x in images.split(c['microbatch'])]
                reference=torch.cat([f[1] for f in chunks]);teacher=paired_contract(reference,labels,coords,c['rank_cap'])
                for suffix in ('ms','protect'):
                    arm=sampling+'_'+suffix;m=models[arm]
                    with torch.no_grad():
                        temporary=[];offset=0
                        for f in chunks:
                            k=len(f[0]);temporary.append(m.describe(f,seeds[offset:offset+k]));offset+=k
                        leaf=torch.cat([v[0] for v in temporary]).detach().requires_grad_()
                    del temporary
                    pair,rank,stats=paired_losses(leaf,teacher,suffix=='protect',c['pair_margin'],c['pair_temperature'])
                    losses=dict(main=gps_retrieval(lossfn,leaf[:n],labels[:n],coords[:n]),
                        keep=c['keep_weight']*(1-(leaf*reference).sum(1)).clamp_min(0).mean(),
                        pair=c['pair_weight']*pair,rank=c['rank_weight']*rank)
                    if not all(bool(torch.isfinite(v)) for v in losses.values()):raise ValueError('Nonfinite loss')
                    rank_gradient=torch.autograd.grad(losses['rank'],leaf,retain_graph=True)[0]
                    gradients=torch.autograd.grad(sum(losses.values()),leaf)[0]
                    optimizers[arm].zero_grad(set_to_none=True);old.repair.prior.replay(m,chunks,seeds,gradients)
                    for name,p in m.adapter.named_parameters():state['gradients'][arm][name]|=p.grad is not None and bool((p.grad!=0).any())
                    norm=torch.nn.utils.clip_grad_norm_(m.adapter.parameters(),c['gradient_clip'],error_if_nonfinite=True)
                    optimizers[arm].step()
                    row[arm]=dict(losses={name:float(v.detach()) for name,v in losses.items()},gradient_norm=float(norm),
                        ranking=stats,rank_descriptor_gradient_norm=float(rank_gradient.norm()),input_slots=len(images),
                        negative_descriptor_gradient_norm=float(gradients[n:].norm()))
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
            negative_end_gradient_seen=all(any(v[a]['negative_descriptor_gradient_norm']>0 for v in state['history']) for a in ARMS),
            innovation1_sources_unchanged=key.source_hashes(key.PROTECTED)==contract['protected_sources'])
        activity={a:dict(teacher_errors=sum(v[a]['ranking']['teacher_errors'] for v in state['history']),
            violating_rows=sum(v[a]['ranking']['violating'] for v in state['history']),
            nonzero_rank_gradient_steps=sum(v[a]['rank_descriptor_gradient_norm']>0 for v in state['history'])) for a in ARMS}
        write(output/'summary.json',dict(checks=checks,steps=state['step'],penalty_activity=activity,
            final_adapter_sha256={a:state_hash(m.adapter) for a,m in models.items()},parameters=initial['parameters'],
            verdict='SMOKE_DISCARD' if args.command=='smoke' else 'TRAINED_NOT_EVALUATED',encoder_count=1,descriptor_dimensions=12288))
        if not all(checks.values()):raise ValueError('Mechanical checks failed: '+str(checks))
        finish(output)


def evaluate(c,args):
    # Reuse ONLY the existing full-database evaluator in a private namespace.
    # Four model names and checkpoint format match; old code/files stay intact.
    spec=importlib.util.spec_from_file_location('_image_negative_private_eval',ROOT/'scripts/hard_channel.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.identity=identity;module.training_contract=training_contract;module.verified=verified;module.finish=finish
    module.evaluate(c,args)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('plan','mine','smoke','train','eval'))
    p.add_argument('--allow-training',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--allow-shared-gpu',action='store_true')
    args=p.parse_args();c=configuration()
    if args.command in ('smoke','train') and not args.allow_training:p.error('Explicit training consent required')
    if args.command=='plan':key.open_plan(c);print(dict(identity=identity(c),training_started=False));return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':p.error('Physical GPU1 only')
    import fcntl,subprocess
    with (ROOT/'logs/query_relation_screen_v1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        memory=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=20)
        check_gpu_capacity(*map(int,memory.strip().split(',')),args.allow_shared_gpu,c['cuda_memory_fraction'])
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        {'mine':mine,'smoke':train,'train':train,'eval':evaluate}[args.command](c,args)


if __name__=='__main__':main()
