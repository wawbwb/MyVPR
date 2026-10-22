"""Finite FP32 frozen-descriptor metric experiment; no old checkpoint updates."""
import argparse
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
from scripts import image_negative as previous
from scripts import clip_key_study as key
from scripts import single_dino_interaction as frozen
from scripts.single_dino_supervision import check_gpu_capacity
from src.models.relation_metric import ARMS,RelationMetric,relation_basis,whitening,calibrated,canonical_columns,orthogonal_check
from src.image_negative_pairs import paired_contract,paired_losses
from src.semantic_region_protocol import read,write,sha,state_hash,owned_run,seed_all,save_torch,save_npz,rng_state,restore_rng,gps_retrieval

SCHEMA='relation_metric_v1'
FILES=('src/models/relation_metric.py','scripts/relation_metric.py','scripts/launch_relation_metric.py',
    'config/relation_metric.json','tests/test_relation_metric.py')


def configuration():
    extra=read(ROOT/'config/relation_metric.json')
    expected=dict(schema=SCHEMA,runs='logs/relation_metric_v1',metric_cache='.cache/relation_metric_v1',
        pilot_steps=512,smoke_steps=8,pca_dim=256,pca_oversampling=16,pca_iterations=2,
        metric_rank=64,metric_hidden=32,metric_cap=.05,learning_rate=.0003)
    if extra!=expected:raise ValueError('Fixed protocol changed')
    return dict(previous.configuration(),**extra)


def finish(out):
    write(out/'completed.json',dict(complete=True,schema=SCHEMA,files={p.relative_to(out).as_posix():sha(p)
        for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('.lock','completed.json') and p.suffix!='.tmp'}))


def verified(out):
    d=read(out/'completed.json')
    if d.get('schema')!=SCHEMA or not d.get('complete'):raise ValueError('Incomplete metric artifact')
    for name,h in d['files'].items():
        p=(out/name).resolve()
        if not p.is_relative_to(out.resolve()) or sha(p)!=h:raise ValueError('Changed artifact: '+name)
    return read(out/'contract.json')


def identity(c,stage):
    base=previous.configuration();pairs=previous.pairs_ready(base);plan=key.open_plan(c)
    if pairs['row_ids'].tolist()!=previous.population(plan,c):raise ValueError('Training rows differ')
    return dict(schema=SCHEMA,stage=stage,config=c,code=key.source_hashes(FILES),
        dependency=previous.identity(base),pair_sha256=sha(key.path(base,'pair_cache')/'completed.json'),
        scope='Frozen innovation1 FP32 descriptors. Fixed final512 steps, single seed, development only; no warm adapter.')


def encode(anchor,images):
    from src.hcen_plain_protocol import canonical_features
    layers=canonical_features(anchor.features(images))
    enhanced,_,_=anchor.neck(layers,auxiliary=False)
    anchor.neck.last_raw=None;anchor.neck.last_conditional_stats=None
    return anchor.readout(enhanced)


def validate_vectors(x,n):
    if x.shape!=(n,12288) or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('Need finite FP32 descriptors')
    if np.max(np.abs(np.linalg.norm(x,axis=1)-1))>1e-4:raise ValueError('Nonunit descriptor')


def cache(c,args):
    from scripts.semantic_region_study import ImageRows
    plan=key.open_plan(c);rows=previous.population(plan,c)
    out=key.path(c,'metric_cache')/'train';contract=identity(c,'train_fp32_cache')
    if (out/'completed.json').exists():
        original=verified(out)
        # A sealed frozen-anchor cache remains valid after downstream metric fixes.
        # Keep its original code provenance; never rewrite its contract or seal.
        if {k:v for k,v in original.items() if k!='code'}!={k:v for k,v in contract.items() if k!='code'}:
            raise ValueError('Frozen cache configuration/dependencies changed')
        print('Reusing verified FP32 cache with original code provenance',sha(out/'completed.json'),flush=True)
        return
    with owned_run(out,contract,args.resume):
        if (out/'completed.json').exists():verified(out);print('FP32 cache verified');return
        width=64;todo=[]
        def check(i):
            target=out/f'{i:04d}.npz'
            if sha(target)!=read(out/f'{i:04d}.ready.json')['sha256']:raise ValueError('Corrupt FP32 shard')
            with np.load(target,allow_pickle=False) as d:
                validate_vectors(d['descriptors'],width)
                if d['row_ids'].tolist()!=rows[i*width:(i+1)*width]:raise ValueError('FP32 row order changed')
        for i in range(len(rows)//width):
            if (out/f'{i:04d}.ready.json').exists():check(i)
            else:todo.append(i)
        if todo:
            anchor=frozen.make_anchor(c).requires_grad_(False).eval();before=state_hash(anchor)
            dataset=ImageRows(key.path(c,'dataset_root'),plan['images'],c['student_size'])
            indices=[j for i in todo for j in rows[i*width:(i+1)*width]]
            loader=DataLoader(Subset(dataset,indices),batch_size=width,num_workers=c['workers'],shuffle=False)
            sentinel=None
            with torch.inference_mode():
                for i,(images,ids) in zip(todo,loader):
                    if ids.tolist()!=rows[i*width:(i+1)*width]:raise ValueError('Extraction order changed')
                    x=torch.cat([encode(anchor,t.to(c['device'])) for t in images.split(c['microbatch'])])
                    if sentinel is None:
                        single=encode(anchor,images[:1].to(c['device']))
                        sentinel=float((single-x[:1]).abs().max())
                        if sentinel>1e-5:raise ValueError('Descriptor depends on batch shape')
                    save_npz(out/f'{i:04d}.npz',descriptors=x.cpu().numpy(),row_ids=ids.numpy())
                    write(out/f'{i:04d}.ready.json',dict(sha256=sha(out/f'{i:04d}.npz')));check(i)
                    write(out/'progress.json',dict(shards=i+1,total=len(rows)//width))
                    print('FP32 train',i+1,'/',len(rows)//width,flush=True)
            if state_hash(anchor)!=before:raise ValueError('Frozen anchor changed')
            write(out/'extraction_checks.json',dict(anchor_sha256=before,batch_sentinel_max_error=sentinel))
        if key.source_hashes(key.PROTECTED)!=contract['dependency']['protected_sources']:raise ValueError('Innovation1 source changed')
        finish(out)


def bank(c):
    path=key.path(c,'metric_cache')/'train';verified(path)
    x=np.empty((c['train_places']*4,12288),np.float32)
    for i in range(len(x)//64):
        with np.load(path/f'{i:04d}.npz',allow_pickle=False) as d:x[i*64:(i+1)*64]=d['descriptors']
    validate_vectors(x,len(x));return x


def basis(c,args):
    src=key.path(c,'metric_cache')/'train';verified(src)
    contract=dict(identity(c,'fit_train_only_bases'),cache_sha256=sha(src/'completed.json'))
    out=key.path(c,'metric_cache')/'basis'
    with owned_run(out,contract,args.resume):
        if (out/'completed.json').exists():verified(out);return
        seed_all(c['seed']);z=torch.from_numpy(bank(c)).to(c['device']);mu=z.mean(0)
        # Center explicitly once; no full DxD covariance or all-pairs NxN allocation.
        z.sub_(mu)
        print('Fit randomized PCA on training FP32 bank only',flush=True)
        with torch.no_grad():
            _,sv,v=torch.pca_lowrank(z,q=c['pca_dim']+c['pca_oversampling'],center=False,niter=c['pca_iterations'])
            p=canonical_columns(v[:,:c['pca_dim']]);y=z@p;del z
            negatives=previous.pairs_ready(previous.configuration())['hard']
            u,w,b,eigen=relation_basis(y,p,torch.as_tensor(negatives,device=c['device']),c['metric_rank'])
            wu,wa=whitening(p,w,c['metric_rank'],c['metric_cap'])
            generator=torch.Generator(device=c['device']).manual_seed(c['seed'])
            random=canonical_columns(torch.linalg.qr(torch.randn((12288,c['metric_rank']),generator=generator,device=c['device']),mode='reduced')[0])
            for a in (p,random,u,wu):orthogonal_check(a)
        payload=dict(random_basis=random.cpu(),pca_basis=p[:,:c['metric_rank']].cpu(),relation_basis=u.cpu(),
            whitening_basis=wu.cpu(),whitening_coefficients=wa.cpu(),training_mean=mu.cpu(),
            within=w.cpu(),between=b.cpu(),relation_eigenvalues=eigen.cpu(),pca_singular_values=sv.cpu())
        save_torch(out/'basis.pt',payload)
        write(out/'summary.json',dict(training_images=len(y),pca_dim=c['pca_dim'],rank=c['metric_rank'],
            within_trace=float(w.trace()),between_trace=float(b.trace()),eigenvalues=eigen.cpu().tolist(),
            train_only=True,training_started=False));finish(out)


def load_bases(c):
    path=key.path(c,'metric_cache')/'basis';verified(path)
    return torch.load(path/'basis.pt',map_location=c['device'],weights_only=True)


def build(c,bases):
    models={}
    for arm in ARMS:
        seed_all(c['seed']);u=bases['relation_basis'] if arm.startswith('relation_') else bases[arm]
        models[arm]=RelationMetric(u,static=arm=='relation_static',hidden=c['metric_hidden'],cap=c['metric_cap']).to(c['device'])
    initials=[state_hash(m.gate) for m in models.values() if m.gate is not None]
    if len(set(initials))!=1:raise ValueError('MLP initialization differs')
    return models


def train(c,args):
    from src.losses.vpr_losses import VPRLossFunction
    bases=load_bases(c);root=key.path(c,'metric_cache');out=key.path(c,'runs')/args.command
    contract=dict(identity(c,args.command),basis_sha256=sha(root/'basis/completed.json'),cache_sha256=sha(root/'train/completed.json'))
    if args.command=='train':
        smoke=key.path(c,'runs')/'smoke';sc=verified(smoke)
        if {k:v for k,v in sc.items() if k!='stage'}!={k:v for k,v in contract.items() if k!='stage'} or not all(read(smoke/'summary.json')['checks'].values()):
            raise ValueError('Matching successful smoke required')
    with owned_run(out,contract,args.resume):
        if (out/'completed.json').exists():verified(out);return
        values=bank(c);plan=key.open_plan(c);pairs=previous.pairs_ready(previous.configuration())
        coordinates=np.array([plan['images'][int(i)]['coordinates'] for i in pairs['row_ids']],np.float64)
        models=build(c,bases);opts={a:torch.optim.AdamW(m.parameters(),lr=c['learning_rate'],weight_decay=0.) for a,m in models.items()}
        sample=torch.from_numpy(values[:128]).to(c['device'])
        for m in models.values():
            if not torch.equal(m(sample),sample):raise ValueError('Zero calibration is not exact')
        initial={a:state_hash(m) for a,m in models.items()}
        state=dict(step=0,history=[],contract_sha256=sha(out/'contract.json'),
            gradients={a:{k:False for k,p in m.named_parameters()} for a,m in models.items()})
        def save():save_torch(out/'last.pt',dict(models={a:m.state_dict() for a,m in models.items()},
            optimizers={a:o.state_dict() for a,o in opts.items()},state=state,rng=rng_state()))
        if (out/'last.pt').exists():
            payload=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
            if payload['state']['contract_sha256']!=state['contract_sha256']:raise ValueError('Resume contract changed')
            for a,m in models.items():m.load_state_dict(payload['models'][a]);opts[a].load_state_dict(payload['optimizers'][a])
            state=payload['state'];restore_rng(payload['rng'])
        else:save()
        total=c['smoke_steps'] if args.command=='smoke' else c['pilot_steps'];lossfn=VPRLossFunction()
        for step in range(state['step'],total):
            start=time.monotonic();base=np.array([4*p+v for p in plan['schedule'][step] for v in range(4)])
            ids=np.concatenate((base,pairs['hard'][base]));z=torch.from_numpy(values[ids]).to(c['device'])
            labels=torch.as_tensor(ids//4,device=c['device']);coords=torch.from_numpy(coordinates[ids]);n=len(base)
            teacher=paired_contract(z,labels,coords,c['rank_cap']);row=dict(step=step+1)
            for arm,m in models.items():
                m.train();output,delta,a=m(z,True)
                main=gps_retrieval(lossfn,output[:n],labels[:n],coords[:n]);pair,_,stats=paired_losses(output,teacher,False,c['pair_margin'],c['pair_temperature'])
                keep=(1-(output*z).sum(1)).clamp_min(0).mean();loss=main+c['pair_weight']*pair+c['keep_weight']*keep
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                endpoint=torch.autograd.grad(loss,output,retain_graph=True)[0]
                opts[arm].zero_grad(set_to_none=True);loss.backward()
                for k,p in m.named_parameters():state['gradients'][arm][k]|=p.grad is not None and bool((p.grad!=0).any())
                norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opts[arm].step()
                if float(delta.norm(dim=1).max())>c['metric_cap']+1e-5 or float((output.norm(dim=1)-1).abs().max())>1e-5:raise ValueError('Calibration bound/norm failed')
                row[arm]=dict(main=float(main),pair=float(pair),keep=float(keep),gradient_norm=float(norm),
                    negative_gradient_norm=float(endpoint[n:].norm()),teacher_errors=stats['teacher_errors'],
                    delta_mean=float(delta.norm(dim=1).mean()),delta_max=float(delta.norm(dim=1).max()),
                    gate_mean=float(a.mean()),gate_std=float(a.std()),saturation_fraction=float((a.abs()>.99*c['metric_cap']).float().mean()))
            state['step']=step+1;row['seconds']=time.monotonic()-start;state['history'].append(row)
            write(out/'progress.json',dict(total=total,**row))
            if state['step']%16==0:save();print(args.command,state['step'],'/',total,flush=True)
        save()
        checks=dict(complete_steps=state['step']==total,zero_initialization_exact=True,
            all_adapters_updated=all(state_hash(m)!=initial[a] for a,m in models.items()),
            all_parameters_gradient_seen=all(all(v.values()) for v in state['gradients'].values()),
            negative_end_gradient_seen=all(any(r[a]['negative_gradient_norm']>0 for r in state['history']) for a in ARMS),
            protected_sources_unchanged=key.source_hashes(key.PROTECTED)==contract['dependency']['protected_sources'])
        write(out/'summary.json',dict(checks=checks,steps=state['step'],parameters={a:sum(p.numel() for p in m.parameters()) for a,m in models.items()},
            final_sha256={a:state_hash(m) for a,m in models.items()},verdict='SMOKE_DISCARD' if args.command=='smoke' else 'TRAINED_NOT_EVALUATED'))
        if not all(checks.values()):raise ValueError('Training mechanics failed: '+str(checks))
        finish(out)


def verdict(results):
    candidate='relation_conditional';controls=('random_basis','pca_basis','relation_static','bounded_whitening')
    counts={tag:r['correct'] for tag,r in results.items()};p=counts['pitts30k-val'];m=counts['msls-val'];g=counts['gsv-adaptation-holdout']
    gain=p[candidate]>p['innovation1'] and m[candidate]>=m['innovation1'] and g[candidate]>=g['innovation1']
    mechanism=gain and all(p[candidate]>p[a] and m[candidate]>=m[a] for a in controls)
    return dict(anchor_gain=bool(gain),matched_mechanism_signal=bool(mechanism),stable_gain_proven=False,blind_test_used=False,
        verdict='EXPLORATORY_RELATION_METRIC_SIGNAL' if mechanism else 'NO_MATCHED_RELATION_METRIC_SIGNAL')


def evaluate(c,args):
    from scripts import eval_clip_key_study as metrics
    from scripts.eval_spatial_adapter_repeat import RGBImages
    trainroot=key.path(c,'runs')/'train';verified(trainroot)
    if not all(read(trainroot/'summary.json')['checks'].values()):raise ValueError('Training checks failed')
    sets=metrics.populations(c,key.open_plan(c));bases=load_bases(c);models=build(c,bases)
    payload=torch.load(trainroot/'last.pt',map_location=c['device'],weights_only=True)
    if payload['state']['step']!=512 or payload['state']['contract_sha256']!=sha(trainroot/'contract.json'):raise ValueError('Wrong final checkpoint')
    for a,m in models.items():
        m.load_state_dict(payload['models'][a]);m.requires_grad_(False).eval()
        if state_hash(m)!=read(trainroot/'summary.json')['final_sha256'][a]:raise ValueError('Changed final tensors')
    out=key.path(c,'runs')/'eval';prior=key.path(previous.configuration(),'runs')/'eval';previous.verified(prior)
    contract=dict(identity(c,'eval'),train_sha256=sha(trainroot/'completed.json'),prior_sha256=sha(prior/'completed.json'),
        datasets={d['tag']:dict(references=d['references'],queries=d['queries'],order=metrics.order_digest(d['names']),gt=metrics.gt_digest(d['ground_truth'])) for d in sets})
    with owned_run(out,contract,args.resume):
        if (out/'completed.json').exists():verified(out);return
        anchor=frozen.make_anchor(c).requires_grad_(False).eval();frozen_hash=state_hash(anchor);results={}
        for d in sets:
            tag=d['tag'];target=out/(tag+'.json');ready=out/(tag+'.ready.json')
            if ready.exists():
                if sha(target)!=read(ready)['sha256']:raise ValueError('Changed report')
                results[tag]=read(target)['comparison'];continue
            start=time.monotonic();values=np.empty((len(d['names']),12288),np.float32);contents=[]
            loader=DataLoader(RGBImages(d['root'],d['names'],d['hashes']),batch_size=16,num_workers=c['workers'],shuffle=False)
            with torch.inference_mode():
                for batch,(images,ids,hashes) in enumerate(loader):
                    if ids.tolist()!=list(range(batch*16,batch*16+len(images))):raise ValueError('Eval order changed')
                    contents.extend(hashes)
                    z=torch.cat([encode(anchor,t.to(c['device'])) for t in images.split(c['microbatch'])]);values[ids.numpy()]=z.cpu().numpy()
                    write(out/'progress.json',dict(phase='extraction',dataset=tag,images=int(ids[-1])+1,total=len(values)))
                    if batch%64==0:print('eval',tag,int(ids[-1])+1,'/',len(values),flush=True)
                validate_vectors(values,len(values));digest=metrics.order_digest([n+':'+h for n,h in zip(d['names'],contents)])
                before=read(prior/(tag+'.json'))
                if digest!=before['image_content_digest']:raise ValueError('Previous image bytes/order differ')
                reports={};diagnostics={}
                for arm in ('innovation1',*ARMS,'bounded_whitening'):
                    def transform(z):
                        if arm=='innovation1':return z
                        if arm=='bounded_whitening':return calibrated(z,bases['whitening_basis'],bases['whitening_coefficients'])[0]
                        return models[arm](z)
                    transformed=np.empty_like(values)
                    for i in range(0,len(values),128):
                        z=torch.from_numpy(values[i:i+128]).to(c['device']);z=transform(z)
                        transformed[i:i+128]=z.cpu().numpy()
                    validate_vectors(transformed,len(values));db=torch.from_numpy(transformed[:d['references']]).to(c['device'])
                    pred=np.concatenate([(torch.from_numpy(q).to(c['device'])@db.T).topk(20,dim=1).indices.cpu().numpy()
                        for q in np.array_split(transformed[d['references']:],max(1,(d['queries']+15)//16))]);del db
                    reports[arm]=metrics.retrieval_report(pred,d['ground_truth'],d['references'])
                    drift=np.linalg.norm(transformed-values,axis=1);diagnostics[arm]=dict(mean_descriptor_drift=float(drift.mean()),max_descriptor_drift=float(drift.max()))
                    if arm=='innovation1' and reports[arm]!=before['variants']['innovation1']:raise ValueError('Innovation1 full Top20/metrics not reproduced')
                    # Per-image inference should not depend on other batch members.
                    probe=torch.from_numpy(values[:8]).to(c['device'])
                    err=float((transform(probe)-torch.cat([transform(t[None]) for t in probe])).abs().max())
                    if err>1e-5:raise ValueError('Calibration depends on batch composition')
                    diagnostics[arm]['batch_independence_max_error']=err
                    del transformed
                comparison=dict(correct={a:r['correct'] for a,r in reports.items()},
                    vs_innovation1={a:metrics.paired_report(reports['innovation1'],reports[a]) for a in reports if a!='innovation1'},
                    conditional_vs_controls={a:metrics.paired_report(reports[a],reports['relation_conditional']) for a in reports if a!='relation_conditional'})
                write(target,dict(dataset=contract['datasets'][tag],variants=reports,comparison=comparison,diagnostics=diagnostics,
                    image_content_digest=digest,elapsed_seconds=time.monotonic()-start));write(ready,dict(sha256=sha(target)))
                results[tag]=comparison;print(tag,comparison['correct'],flush=True)
        checks=dict(anchor_unchanged=state_hash(anchor)==frozen_hash,
            protected_sources_unchanged=key.source_hashes(key.PROTECTED)==contract['dependency']['protected_sources'])
        write(out/'summary.json',dict(checks=checks,results=results,**verdict(results)))
        if not all(checks.values()):raise ValueError('Integrity failure')
        finish(out)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('plan','cache','basis','smoke','train','eval'))
    p.add_argument('--resume',action='store_true');p.add_argument('--allow-training',action='store_true');p.add_argument('--allow-shared-gpu',action='store_true')
    args=p.parse_args();c=configuration()
    if args.command in ('smoke','train') and not args.allow_training:p.error('Training requires authorization')
    if args.command=='plan':print(identity(c,'plan'));return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':p.error('Physical GPU1 only')
    import fcntl,subprocess
    with (ROOT/'logs/query_relation_screen_v1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        memory=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=20)
        check_gpu_capacity(*map(int,memory.strip().split(',')),args.allow_shared_gpu,c['cuda_memory_fraction']);key.gpu(c)
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        {'cache':cache,'basis':basis,'smoke':train,'train':train,'eval':evaluate}[args.command](c,args)


if __name__=='__main__':main()
