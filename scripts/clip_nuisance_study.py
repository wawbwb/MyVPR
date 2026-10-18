"""Within-place CLIP residual disentanglement: prepare/probe/smoke/train/eval.

Three matched students; all CLIP information is training-only. These are
candidate variation proxies, NOT verified pure nuisance/appearance labels.
"""
import argparse
import hashlib
import json
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

from scripts.clip_key_study import (configuration as key_configuration,identity as key_identity,
    open_plan,path,source_hashes,gpu,verified as verified_keys,PROTECTED)
from scripts.semantic_region_study import PlaceSteps
from scripts.eval_clip_key_study import (DualImages,populations,retrieval_report,
    compare_reports,order_digest,gt_digest,validate_completed_dataset)
from src.models.clip_nuisance import (ARMS,InvariantBoQ,ResidualPredictor,Regressor,
    center_four,projection,whitening,donor_indices)
from src.models.clip_dense_keys import DenseClipKeys
from src.models.official_boq import load_official
from src.semantic_region_protocol import (owned_run,read,write,sha,state_hash,seed_all,
    save_torch,save_npz,rng_state,restore_rng,gps_retrieval)

SCHEMA='within_place_clip_residual_v1'
SOURCES=('src/models/clip_nuisance.py','scripts/clip_nuisance_study.py',
    'scripts/eval_clip_nuisance.py','scripts/launch_clip_nuisance.py','config/clip_nuisance.yaml','tests/test_clip_nuisance.py',
    'scripts/eval_clip_key_study.py','src/dataloaders/valid/mapillary_sls.py',
    'src/dataloaders/valid/pittsburgh.py','src/utils/config_manager.py')
VARIANTS=('official',*ARMS)


def configuration(p):
    import yaml
    c=yaml.safe_load(p.read_text(encoding='utf-8'))
    if c['schema']!=SCHEMA or c['precision']!='fp32' or c['views']!=4 or c['student_size']!=322:
        raise ValueError('Need fixed FP32/322/four-view protocol')
    for k in ('seed','train_places','places_per_batch','epochs','projection_dim','target_dim',
              'predictor_hidden','predictor_updates','adversarial_ramp_steps','microbatch',
              'prepare_batch','flush_rows','checkpoint_every','smoke_steps','workers',
              'probe_epochs','probe_batch','hidden'):
        if type(c[k])is not int or c[k]<1:raise ValueError('Positive integer required: '+k)
    for k in ('learning_rate','predictor_lr','keep_weight','adversarial_weight','gradient_clip',
              'bias_cap','key_temperature','probe_lr','probe_min_r2','probe_min_gap','cuda_memory_fraction'):
        if not np.isfinite(c[k]) or c[k]<=0:raise ValueError('Positive finite setting required: '+k)
    if c['train_places']%c['places_per_batch'] or c['flush_rows']%c['prepare_batch'] or c['cuda_memory_fraction']>1:
        raise ValueError('Invalid batch layout/memory budget')
    return c


def identity(c):
    return dict(schema=SCHEMA,config=c,code=source_hashes(SOURCES),
        base=key_identity(key_configuration(ROOT/'config/clip_key_boq.yaml')),
        source_keys_completed_sha256=sha(path(c,'source_keys')/'completed.json'))


def finish(output):
    files={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*'))
           if p.is_file() and p.name not in ('.lock','completed.json') and not p.name.endswith('.tmp')}
    write(output/'completed.json',dict(complete=True,schema=SCHEMA,files=files))


def verified(output):
    report=read(output/'completed.json')
    if report.get('complete')is not True or report.get('schema')!=SCHEMA:raise ValueError('Incomplete artifact')
    for n,digest in report['files'].items():
        p=(output/n).resolve()
        if not p.is_relative_to(output.resolve()) or not p.is_file() or sha(p)!=digest:
            raise ValueError('Artifact checksum differs: '+str(p))
    return read(output/'contract.json')


def prepared(c):
    output=path(c,'prepared')
    if verified(output)!=dict(stage='prepared',**identity(c)):
        raise ValueError('Prepared code/config/data identity changed')
    x=np.load(output/'baseline_projected.npy',mmap_mode='r')
    y=np.load(output/'targets.npy',allow_pickle=False)
    scale=np.load(output/'scale.npy',allow_pickle=False)
    permutation=np.load(output/'donor_indices.npy',allow_pickle=False)
    return x,y,scale,permutation


def prepare(c,args):
    plan=open_plan(c)
    old=verified_keys(path(c,'source_keys'))
    base=key_identity(key_configuration(ROOT/'config/clip_key_boq.yaml'))
    if any(old[k]!=v for k,v in base.items()):raise ValueError('Existing CLIP cache provenance differs')
    for i,p in enumerate(plan['places']):
        if p['rows']!=list(range(i*4,i*4+4)):raise ValueError('Need original four-view row grouping')
    output=path(c,'prepared')
    with owned_run(output,dict(stage='prepared',**identity(c)),args.resume):
        if (output/'completed.json').exists():verified(output);print('Preparation complete',flush=True);return
        gpu(c)
        rows=plan['images'];count=len(rows)
        cursor=read(output/'cursor.json') if (output/'cursor.json').exists() else dict(next=0,chunks=[])
        arrays={}
        for name,channels in (('baseline_projected',c['projection_dim']),('scene',512)):
            p=output/(name+'.npy')
            if p.exists():
                a=np.load(p,mmap_mode='r+')
                if a.shape!=(count,channels) or a.dtype!=np.float32:raise ValueError('Partial array layout differs')
            else:
                if cursor['next']:raise ValueError('Missing partial array')
                a=np.lib.format.open_memmap(p,mode='w+',dtype=np.float32,shape=(count,channels))
            arrays[name]=a
        end=0
        for chunk in cursor['chunks']:
            start,stop=chunk['start'],chunk['stop']
            if start!=end or not start<stop<=cursor['next']:raise ValueError('Bad cursor inventory')
            for name,a in arrays.items():
                if hashlib.sha256(a[start:stop].tobytes()).hexdigest()!=chunk[name]:raise ValueError('Damaged prepared rows')
            end=stop
        if end!=cursor['next'] or not 0<=end<=count:raise ValueError('Bad resume cursor')
        official=load_official(path(c,'official_cache'),path(c,'dino_source'),c['device'])
        teacher=None
        fixed_projection=projection(12288,c['projection_dim'],c['seed']).to(c['device'])
        cached=np.load(path(c,'source_keys')/'keys.npy',mmap_mode='r')
        dataset=DualImages(path(c,'dataset_root'),[r['path'] for r in rows],[r['image_sha256'] for r in rows])
        for start in tqdm(range(cursor['next'],count,c['flush_rows']),desc='Frozen baseline/CLIP proxy preparation'):
            stop=min(start+c['flush_rows'],count)
            loader=DataLoader(Subset(dataset,range(start,stop)),batch_size=c['prepare_batch'],num_workers=c['workers'],shuffle=False)
            xs,ss=[],[]
            with torch.inference_mode():
                for images,clip_images,ids,_ in loader:
                    descriptors=official(images.to(c['device']))
                    xs.append((descriptors@fixed_projection).cpu().numpy())
                    indices=ids.numpy()
                    if bool((indices<plan['train_rows']).all()):
                        keys=torch.from_numpy(np.array(cached[indices],dtype=np.float32,copy=True))
                    elif bool((indices>=plan['train_rows']).all()):
                        if teacher is None:teacher=DenseClipKeys(path(c,'clip_checkpoint'),c['device'])
                        keys=teacher(clip_images.to(c['device'])).half().float().cpu()
                    else:raise ValueError('Training and holdout crossed inside one preparation batch')
                    ss.append(F.normalize(keys.mean((2,3)),dim=1).numpy())
            values=dict(baseline_projected=np.concatenate(xs),scene=np.concatenate(ss))
            entry=dict(start=start,stop=stop)
            for name,value in values.items():
                if not np.isfinite(value).all():raise ValueError('Nonfinite proxy/descriptor')
                arrays[name][start:stop]=value;arrays[name].flush()
                entry[name]=hashlib.sha256(value.tobytes()).hexdigest()
            cursor['chunks'].append(entry);cursor['next']=stop
            write(output/'cursor.json',cursor)
            write(output/'progress.json',dict(phase='preparing',images=stop,total=count,student_updates=0))
        matrix,targets,stats=whitening(arrays['scene'],plan['train_rows'],c['target_dim'])
        scale=center_four(arrays['baseline_projected'][:plan['train_rows']]).std(0).clip(1e-4)
        for name,value in (('targets',targets),('scale',scale),('donor_indices',donor_indices([p['place'] for p in plan['places']],c['seed']))):
            p=output/(name+'.npy');temp=output/(name+'.tmp')
            with temp.open('wb') as stream:np.save(stream,value,allow_pickle=False)
            temp.replace(p)
        save_npz(output/'whitening.npz',matrix=matrix)
        write(output/'summary.json',dict(images=count,train_rows=plan['train_rows'],target_dim=c['target_dim'],
            statistics=stats,student_updates=0,scope='Within-place pooled SCLIP variation proxy, NOT pure nuisance labels',
            new_array_mib=sum(p.stat().st_size for p in output.glob('*.npy'))/2**20))
        del arrays
        finish(output)
        print('Preparation sealed; no student updates',flush=True)


def fit_probe(c,features,targets,train_rows,seed):
    """Fresh CPU probe, own TRAIN-only normalization; fixed epochs, no selection."""
    x=center_four(features)
    scale=x[:train_rows].std(0).clip(1e-4)
    x=torch.from_numpy(np.asarray(x/scale,np.float32))
    y=torch.from_numpy(np.asarray(targets,np.float32))
    seed_all(seed)
    head=Regressor(c['projection_dim'],c['predictor_hidden'],c['target_dim'])
    optimizer=torch.optim.AdamW(head.parameters(),lr=c['probe_lr'],weight_decay=0.)
    for epoch in range(c['probe_epochs']):
        order=torch.randperm(train_rows,generator=torch.Generator().manual_seed(seed+epoch))
        for ids in order.split(c['probe_batch']):
            optimizer.zero_grad(set_to_none=True)
            loss=F.mse_loss(head(x[ids]),y[ids])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
    with torch.inference_mode():
        prediction=head(x[train_rows:]);truth=y[train_rows:]
        mse=float(F.mse_loss(prediction,truth))
        null=float((truth-y[:train_rows].mean(0)).square().mean())
    return dict(holdout_mse=mse,null_mse=null,holdout_r2=1-mse/max(null,1e-8),
                train_rows=train_rows,holdout_rows=len(y)-train_rows,fixed_epochs=c['probe_epochs'])


def probe(c,args):
    x,y,_,donors=prepared(c)
    output=path(c,'runs')/'preflight_probe'
    contract=dict(stage='preflight_probe',**identity(c),prepared_sha256=sha(path(c,'prepared')/'completed.json'))
    with owned_run(output,contract,args.resume):
        if (output/'completed.json').exists():verified(output);print(read(output/'summary.json'));return
        plan=open_plan(c);n=plan['train_rows']
        aligned=fit_probe(c,x,y,n,c['seed']+8101)
        shuffled=fit_probe(c,x,y[donors],n,c['seed']+8101)
        gap=aligned['holdout_r2']-shuffled['holdout_r2']
        checks=dict(nontrivial_predictability=aligned['holdout_r2']>=c['probe_min_r2'],
                    beats_view_permuted_proxy=gap>=c['probe_min_gap'])
        report=dict(aligned=aligned,shuffled=shuffled,r2_gap=gap,checks=checks,
            verdict='PROXY_SIGNAL_PASS' if all(checks.values()) else 'NO_PREDICTABLE_PROXY_STOP',
            scope='Proxy association only, not nuisance ground truth or retrieval improvement; probe updates discarded')
        write(output/'summary.json',report);finish(output)
        print(json.dumps(report,indent=2),flush=True)


def training_contract(c,arm,stage):
    return dict(stage=stage,arm=arm,**identity(c),prepared_sha256=sha(path(c,'prepared')/'completed.json'),
        probe_sha256=sha(path(c,'runs')/'preflight_probe/completed.json'),
        inference='RGB -> frozen DINO -> visual-only router -> frozen BoQ; no CLIP/predictor at inference',
        selection='Fixed final epoch5/2560 steps; single-seed exploratory screen')


def make_student(c):
    seed_all(c['seed'])
    official=load_official(path(c,'official_cache'),path(c,'dino_source'),c['device'])
    model=InvariantBoQ(official,c['hidden'],c['bias_cap'],c['key_temperature'],c['seed']).to(c['device'])
    return model


def train(c,args):
    from src.losses.vpr_losses import VPRLossFunction
    _,targets,scale,donors=prepared(c)
    p=path(c,'runs')/'preflight_probe'
    verified(p)
    if read(p/'summary.json')['verdict']!='PROXY_SIGNAL_PASS':raise ValueError('No useful proxy signal; stop before student training')
    plan=open_plan(c);stage=args.command;arm=args.arm
    if stage=='train':
        for a in ARMS:
            smoke=path(c,'runs')/('smoke_'+a)
            if verified(smoke)!=training_contract(c,a,'smoke') or not all(read(smoke/'summary.json')['checks'].values()):
                raise ValueError('All matched smoke runs must pass')
    output=path(c,'runs')/(('smoke_' if stage=='smoke' else '')+arm)
    contract=training_contract(c,arm,stage)
    with owned_run(output,contract,args.resume):
        if (output/'completed.json').exists():verified(output);print('Already complete',flush=True);return
        gpu(c);model=make_student(c)
        predictor=ResidualPredictor(scale,hidden=c['predictor_hidden'],targets=c['target_dim'],seed=c['seed']).to(c['device'])
        active={n:p for n,p in model.named_parameters() if p.requires_grad}
        if any(not n.startswith('routers.') for n in active):raise ValueError('Original network must be frozen')
        frozen=state_hash(model.visual)
        initial=dict(official=frozen,routers=state_hash(model.routers),predictor=state_hash(predictor),
                     student_parameters=sum(p.numel() for p in active.values()),predictor_parameters=sum(p.numel() for p in predictor.parameters()))
        if (output/'initial.json').exists() and read(output/'initial.json')!=initial:raise ValueError('Initial states differ')
        write(output/'initial.json',initial)
        probe_images=PlaceSteps(c,plan,plan['schedule'][:1])[0][0][:2].to(c['device'])
        with torch.inference_mode():error=float((model(probe_images)-model.visual(probe_images)).abs().max())
        if error>1e-6:raise ValueError('Zero start differs')
        del probe_images
        optimizer=torch.optim.AdamW(active.values(),lr=c['learning_rate'],weight_decay=0.)
        adversary=torch.optim.AdamW(predictor.parameters(),lr=c['predictor_lr'],weight_decay=0.)
        state=dict(step=0,history=[],contract_sha256=sha(output/'contract.json'),gate_gradient=False,
                   projection_gradient=False,predictor_gradient=False,adversarial_gradient=False)
        if (output/'last.pt').exists():
            saved=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
            if saved['state']['contract_sha256']!=state['contract_sha256'] or set(saved['student'])!=set(active):raise ValueError('Resume identity differs')
            with torch.no_grad():
                for n,p in active.items():p.copy_(saved['student'][n].to(p.device))
            predictor.load_state_dict(saved['predictor'],strict=True)
            optimizer.load_state_dict(saved['optimizer']);adversary.load_state_dict(saved['adversary'])
            state=saved['state'];restore_rng(saved['rng'])
        else:seed_all(c['seed'])
        steps=plan['schedule'][:c['smoke_steps']] if stage=='smoke' else plan['schedule']
        if not 0<=state['step']<=len(steps):raise ValueError('Invalid resumed step')

        def save():
            save_torch(output/'last.pt',dict(student={n:p.detach().cpu() for n,p in active.items()},predictor=predictor.state_dict(),
                optimizer=optimizer.state_dict(),adversary=adversary.state_dict(),state=state,rng=rng_state()))

        if not (output/'last.pt').exists():save()
        loader=DataLoader(PlaceSteps(c,plan,steps[state['step']:]),batch_size=None,num_workers=c['workers'],
                          generator=torch.Generator().manual_seed(c['seed']))
        lossfn=VPRLossFunction();model.train()
        for images,labels,coords,ids in tqdm(loader,total=len(steps),initial=state['step'],desc=f'{arm} {stage}'):
            started=time.monotonic();optimizer.zero_grad(set_to_none=True)
            ys=targets[donors[ids.numpy()]] if arm=='shuffled' else targets[ids.numpy()]
            y=torch.from_numpy(np.asarray(ys)).to(c['device']);labels=labels.to(c['device'])
            values,bases=[],[]
            for start in range(0,len(images),c['microbatch']):
                pixels=images[start:start+c['microbatch']].to(c['device'])
                features=model.encode(pixels)
                with torch.no_grad():bases.append(model.visual.aggregator(features)[0])
                values.append(model.describe(features))
            z,base=torch.cat(values),torch.cat(bases)
            if not torch.isfinite(z).all():raise ValueError('Nonfinite descriptor')
            predictor.requires_grad_(True)
            for _ in range(c['predictor_updates']):
                adversary.zero_grad(set_to_none=True)
                prediction_loss=F.mse_loss(predictor(z.detach(),labels),y)
                prediction_loss.backward()
                state['predictor_gradient']|=any(p.grad is not None and bool((p.grad!=0).any()) for p in predictor.parameters())
                torch.nn.utils.clip_grad_norm_(predictor.parameters(),1.,error_if_nonfinite=True);adversary.step()
            predictor.requires_grad_(False)
            retrieval=gps_retrieval(lossfn,z,labels,coords)
            keep=(1-(z*base).sum(1)).clamp_min(0).mean()
            nuisance=F.mse_loss(predictor(z,labels,adversarial=True),y)
            coefficient=0. if arm=='visual' else c['adversarial_weight']*min(1.,(state['step']+1)/c['adversarial_ramp_steps'])
            if state['step']==0:
                gradients=torch.autograd.grad(nuisance,list(active.values()),retain_graph=True,allow_unused=True)
                state['adversarial_gradient']=any(g is not None and bool((g!=0).any()) for g in gradients)
            loss=retrieval+c['keep_weight']*keep+coefficient*nuisance
            if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
            loss.backward()
            for n,p in active.items():
                if p.grad is not None and bool((p.grad!=0).any()):state['gate_gradient' if n.endswith('.beta') else 'projection_gradient']=True
            norm=torch.nn.utils.clip_grad_norm_(active.values(),c['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize();state['step']+=1
            row=dict(step=state['step'],retrieval=float(retrieval.detach()),keep=float(keep.detach()),
                predictor_mse=float(prediction_loss.detach()),adversarial_mse=float(nuisance.detach()),
                adversarial_weight=coefficient,gradient_norm=float(norm),seconds=time.monotonic()-started,
                peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
            state['history'].append(row)
            write(output/'progress.json',dict(phase='training',arm=arm,total=len(steps),**row))
            if state['step']%c['checkpoint_every']==0:save()
        save()
        checks=dict(zero_start=error<=1e-6,complete_steps=state['step']==len(steps),
            official_unchanged=state_hash(model.visual)==frozen,gate_gradient=state['gate_gradient'],
            projection_gradient=state['projection_gradient'],predictor_gradient=state['predictor_gradient'],
            nuisance_gradient_connected=state['adversarial_gradient'],
            router_updated=state_hash(model.routers)!=initial['routers'],
            innovation1_unchanged=source_hashes(PROTECTED)==contract['base']['protected_sources'])
        write(output/'summary.json',dict(arm=arm,steps=state['step'],checks=checks,zero_start_error=error,
            final_router_sha256=state_hash(model.routers),student_requires_clip_at_inference=False,
            verdict=('SMOKE_PASS_DISCARD' if stage=='smoke' else 'TRAINED_NOT_EVALUATED') if all(checks.values()) else 'MECHANICAL_FAIL'))
        if not all(checks.values()):raise ValueError('Mechanical check failed; retained report/checkpoint')
        finish(output);print(stage,'complete',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','probe','smoke','train','eval'))
    p.add_argument('--config',type=Path,default=ROOT/'config/clip_nuisance.yaml')
    p.add_argument('--arm',choices=ARMS,default='aligned')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--allow-training',action='store_true')
    p.add_argument('--evaluate',action='store_true')
    args=p.parse_args()
    if args.command in ('probe','smoke','train') and not args.allow_training:p.error('Updates require --allow-training')
    if args.command=='eval' and not args.evaluate:p.error('Evaluation requires --evaluate')
    c=configuration(args.config)
    if args.command=='eval':
        from scripts.eval_clip_nuisance import evaluate
        evaluate(c,args)
    else:{'prepare':prepare,'probe':probe,'smoke':train,'train':train}[args.command](c,args)


if __name__=='__main__':main()
