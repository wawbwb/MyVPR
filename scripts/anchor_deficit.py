"""Finite frozen-anchor complementary embedding experiment with training admission."""
import argparse
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts import relation_metric as old
from scripts.audit_relation_metric import compare_margin
from src.models.anchor_deficit import ARMS,Complement,fuse,required_margin,balanced_loss
from src.semantic_region_protocol import read,write,sha,state_hash,owned_run,seed_all,save_torch,save_npz,rng_state,restore_rng

SCHEMA='anchor_deficit_v1'
FILES=('src/models/anchor_deficit.py','scripts/anchor_deficit.py','scripts/launch_anchor_deficit.py',
       'config/anchor_deficit.json','tests/test_anchor_deficit.py','scripts/audit_relation_metric.py')


def config():
    c=read(ROOT/'config/anchor_deficit.json')
    expected=dict(schema=SCHEMA,runs='logs/anchor_deficit_v1',cache='.cache/anchor_deficit_v1',seed=42,steps=512,
        smoke_steps=8,pca_dim=256,side_dim=128,hidden=256,score_weight=.1,goal=.01,target_cap=1.8,
        hard_threshold=.05,hard_extra=16,learning_rate=.0003)
    if c!=expected:raise ValueError('Registered protocol changed')
    return c


def finish(out):
    write(out/'completed.json',dict(schema=SCHEMA,complete=True,files={p.name:sha(p) for p in out.iterdir()
        if p.is_file() and p.name not in ('.lock','completed.json') and p.suffix!='.tmp'}))


def verify(out):
    seal=read(out/'completed.json')
    if seal.get('schema')!=SCHEMA or not seal.get('complete'):raise ValueError('Incomplete artifact')
    for n,h in seal['files'].items():
        path=(out/n).resolve()
        if not path.is_relative_to(out.resolve()) or sha(path)!=h:raise ValueError('Corrupt artifact: '+n)
    return read(out/'contract.json')


def identity(c,stage):
    oc=old.configuration();src=old.key.path(oc,'metric_cache')/'train';old.verified(src)
    return dict(schema=SCHEMA,stage=stage,config=c,code=old.key.source_hashes(FILES),
        upstream=old.identity(oc,'anchor_deficit_source'),cache_sha256=sha(src/'completed.json'))


def prepare(c,args):
    out=ROOT/c['cache'];contract=identity(c,'prepare')
    with owned_run(out,contract,args.resume):
        if (out/'completed.json').exists():verify(out);return
        oc=old.configuration();plan=old.key.open_plan(oc);pairs=old.previous.pairs_ready(old.previous.configuration())
        seed_all(c['seed']);z=torch.from_numpy(old.bank(oc)).to('cuda:0');n=len(z)
        views=z.reshape(-1,4,12288);s=views@views.transpose(1,2);s.diagonal(dim1=1,dim2=2).fill_(-torch.inf)
        best,pi=s.max(2);positive=(4*torch.arange(len(views),device=z.device)[:,None]+pi).flatten().cpu().numpy()
        negative=pairs['hard'].astype(np.int64);negscore=[]
        coords=np.asarray([plan['images'][int(i)]['coordinates'] for i in pairs['row_ids']],np.float64)
        q=np.deg2rad(coords);r=q[negative]
        a=np.sin((q[:,0]-r[:,0])/2)**2+np.cos(q[:,0])*np.cos(r[:,0])*np.sin((q[:,1]-r[:,1])/2)**2
        if ((np.arange(n)//4==negative//4)|(6371000*2*np.arcsin(np.sqrt(np.clip(a,0,1)))<25)).any():raise ValueError('Illegal negative')
        for start in range(0,n,256):negscore.append((z[start:start+256]*z[negative[start:start+256]]).sum(1))
        gap=(best.flatten()-torch.cat(negscore));target,clipped=required_margin(gap,c['score_weight'],c['goal'],c['target_cap'])
        hard=gap<=c['hard_threshold'];hardids=hard.nonzero().flatten().cpu().numpy()
        if not len(hardids):raise ValueError('No hard training pairs')
        rng=np.random.default_rng(c['seed']);schedule=np.asarray([[4*p+v for p in b for v in range(4)] for b in plan['schedule'][:c['steps']]])
        if sorted(schedule.flatten().tolist())!=list(range(n)):raise ValueError('Training schedule coverage changed')
        queries=np.concatenate((schedule,rng.choice(hardids,(c['steps'],c['hard_extra']),replace=True)),axis=1)
        shuffled=target.clone()
        # Preserve target distribution separately in hard/easy pools.
        for mask in (hard,~hard):
            ids=mask.nonzero().flatten();perm=torch.tensor(rng.permutation(len(ids)),device=z.device)
            shuffled[ids]=target[ids[perm]]
        mu=z.mean(0);del views,s,pi,best
        z.sub_(mu);print('Fit training-only PCA256',flush=True)
        _,_,v=torch.pca_lowrank(z,q=c['pca_dim']+16,center=False,niter=2)
        p=old.canonical_columns(v[:,:c['pca_dim']]);x=z@p
        save_torch(out/'projection.pt',dict(mean=mu.cpu(),pca=p.cpu()))
        save_npz(out/'train.npz',x=x.cpu().numpy(),positive=positive,negative=negative,gap=gap.cpu().numpy(),
            target=target.cpu().numpy(),shuffled=shuffled.cpu().numpy(),hard=hard.cpu().numpy(),queries=queries,row_ids=pairs['row_ids'])
        write(out/'summary.json',dict(images=n,hard=len(hardids),errors=int((gap<=0).sum()),
            clipped_targets=int(clipped.sum()),target_mean=float(target.mean()),
            targets_changed_by_shuffle=int((target!=shuffled).sum()),positive_policy='Frozen-anchor best of3; fixed throughout.'))
        finish(out)


def data(c):
    out=ROOT/c['cache'];verify(out)
    with np.load(out/'train.npz',allow_pickle=False) as f:return {k:torch.from_numpy(f[k].copy()).to('cuda:0') for k in f.files}


def build(c):
    models={}
    for arm in ARMS:
        seed_all(c['seed']);models[arm]=Complement(c['pca_dim'],c['hidden'],c['side_dim']).to('cuda:0')
    if len({state_hash(m) for m in models.values()})!=1:raise ValueError('Initialization differs')
    return models


def train(c,args):
    out=ROOT/c['runs']/args.command;contract=dict(identity(c,args.command),prepared_sha256=sha(ROOT/c['cache']/'completed.json'))
    if args.command=='train':
        sc=verify(ROOT/c['runs']/'smoke')
        if {k:v for k,v in sc.items() if k!='stage'}!={k:v for k,v in contract.items() if k!='stage'}:raise ValueError('Smoke mismatch')
        if not all(read(ROOT/c['runs']/'smoke/summary.json')['checks'].values()):raise ValueError('Smoke failed')
    with owned_run(out,contract,args.resume):
        if (out/'completed.json').exists():verify(out);return
        d=data(c);models=build(c);initial={a:state_hash(m) for a,m in models.items()}
        opts={a:torch.optim.AdamW(m.parameters(),lr=c['learning_rate'],weight_decay=0) for a,m in models.items()}
        state=dict(step=0,history=[],contract_sha256=sha(out/'contract.json'),gradients={a:{n:False for n,p in m.named_parameters()} for a,m in models.items()})
        def save():save_torch(out/'last.pt',dict(models={a:m.state_dict() for a,m in models.items()},
            optimizers={a:o.state_dict() for a,o in opts.items()},state=state,rng=rng_state()))
        if (out/'last.pt').exists():
            p=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
            if p['state']['contract_sha256']!=state['contract_sha256']:raise ValueError('Resume contract mismatch')
            for a,m in models.items():m.load_state_dict(p['models'][a]);opts[a].load_state_dict(p['optimizers'][a])
            state=p['state'];restore_rng(p['rng'])
        total=c['smoke_steps'] if args.command=='smoke' else c['steps']
        for step in range(state['step'],total):
            q=d['queries'][step];ids=torch.cat((q,d['positive'][q],d['negative'][q]));row=dict(step=step+1)
            for arm,m in models.items():
                h=m(d['x'][ids]);hq,hp,hn=h.chunk(3);gap=(hq*hp).sum(1)-(hq*hn).sum(1)
                target=torch.full_like(gap,.1) if arm=='constant_target' else d['shuffled' if arm=='shuffled_target' else 'target'][q]
                loss=balanced_loss(gap,target,d['hard'][q])
                if not torch.isfinite(loss):raise ValueError('Nonfinite training')
                opts[arm].zero_grad(set_to_none=True);loss.backward()
                for n,p in m.named_parameters():state['gradients'][arm][n]|=p.grad is not None and bool((p.grad!=0).any())
                norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opts[arm].step()
                row[arm]=dict(loss=float(loss.detach()),gradient_norm=float(norm),side_gap_mean=float(gap.detach().mean()))
            state['step']=step+1;state['history'].append(row)
            write(out/'progress.json',dict(total=total,**row))
            if state['step']%16==0:save();print(args.command,state['step'],'/',total,flush=True)
        save();checks=dict(complete=state['step']==total,all_parameters_gradient_seen=all(all(g.values()) for g in state['gradients'].values()),
            all_heads_updated=all(state_hash(m)!=initial[a] for a,m in models.items()),
            protected_sources_unchanged=old.key.source_hashes(old.key.PROTECTED)==contract['upstream']['dependency']['protected_sources'])
        write(out/'summary.json',dict(checks=checks,final_sha256={a:state_hash(m) for a,m in models.items()},
            parameters={a:sum(p.numel() for p in m.parameters()) for a,m in models.items()},steps=total))
        if not all(checks.values()):raise ValueError('Training mechanics failed')
        finish(out)


def trained(c):
    out=ROOT/c['runs']/'train';verify(out)
    if not all(read(out/'summary.json')['checks'].values()):raise ValueError('Training checks failed')
    models=build(c);p=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
    if p['state']['step']!=c['steps'] or p['state']['contract_sha256']!=sha(out/'contract.json'):raise ValueError('Wrong checkpoint')
    for a,m in models.items():
        m.load_state_dict(p['models'][a]);m.eval().requires_grad_(False)
        if state_hash(m)!=read(out/'summary.json')['final_sha256'][a]:raise ValueError('Changed tensors')
    return models


def audit(c,args):
    out=ROOT/c['runs']/'audit';contract=dict(identity(c,'audit'),train_sha256=sha(ROOT/c['runs']/'train/completed.json'))
    with owned_run(out,contract,args.resume),torch.inference_mode():
        if (out/'completed.json').exists():verify(out);return
        d=data(c);models=trained(c);initial=build(c)['constant_target'];before=d['gap'].cpu().numpy();reports={};margins={}
        for arm,m in [('initial_random',initial),*models.items()]:
            h=torch.cat([m(x) for x in d['x'].split(256)])
            side=(h*h[d['positive']]).sum(1)-(h*h[d['negative']]).sum(1)
            margin=((1-c['score_weight'])*d['gap']+c['score_weight']*side).cpu().numpy()
            margins[arm]=margin;reports[arm]=compare_margin(before,margin)
            reports[arm]['net']=reports[arm]['corrections']-reports[arm]['regressions']
            probe=d['x'][:8];error=float((m(probe)-torch.cat([m(x[None]) for x in probe])).abs().max())
            if error>1e-5:raise ValueError('Batch dependence')
            reports[arm]['batch_independence_max_error']=error
        r=reports['deficit_target'];hard=before<=c['hard_threshold']
        conditions=dict(at_least10_corrections=r['corrections']>=10,regressions_at_most_half=r['regressions']<=r['corrections']/2,
            greater_net_than_controls=all(r['net']>reports[a]['net'] for a in ('constant_target','shuffled_target')),
            hard_mean_margin_gain=float((margins['deficit_target'][hard]-before[hard]).mean())>=.001)
        summary=dict(reports=reports,conditions=conditions,admission=all(conditions.values()),
            scope='Training fixed-pair admission only; NOT full-bank or validation accuracy.',
            verdict='ADMIT_DEVELOPMENT_EVAL' if all(conditions.values()) else 'STOP_NO_TRAINING_MECHANISM_SIGNAL')
        save_npz(out/'margins.npz',innovation1=before,**margins);write(out/'summary.json',summary);finish(out)
        print(summary,flush=True)


def evaluate(c,args):
    from scripts import eval_clip_key_study as metrics
    from scripts.eval_spatial_adapter_repeat import RGBImages
    from torch.utils.data import DataLoader
    admission=ROOT/c['runs']/'audit';verify(admission)
    if not read(admission/'summary.json')['admission']:print('STOP: training gate failed; no validation extraction');return
    oc=old.configuration();models=trained(c);models['initial_random']=build(c)['constant_target'].requires_grad_(False).eval()
    projection=torch.load(ROOT/c['cache']/'projection.pt',map_location='cuda:0',weights_only=True)
    sets=metrics.populations(oc,old.key.open_plan(oc));prior=ROOT/'logs/relation_metric_v1/eval';old.verified(prior)
    out=ROOT/c['runs']/'eval';contract=dict(identity(c,'eval'),train_sha256=sha(ROOT/c['runs']/'train/completed.json'),
        admission_sha256=sha(admission/'completed.json'),prior_sha256=sha(prior/'completed.json'),
        datasets={d['tag']:dict(references=d['references'],queries=d['queries'],order=metrics.order_digest(d['names']),gt=metrics.gt_digest(d['ground_truth'])) for d in sets})
    with owned_run(out,contract,args.resume),torch.inference_mode():
        if (out/'completed.json').exists():verify(out);return
        anchor=old.frozen.make_anchor(oc).requires_grad_(False).eval();frozen_hash=state_hash(anchor);results={}
        for d in sets:
            tag=d['tag'];target=out/(tag+'.json');ready=out/(tag+'.ready.json')
            if ready.exists():
                if sha(target)!=read(ready)['sha256']:raise ValueError('Changed report')
                results[tag]=read(target)['comparison'];continue
            values=np.empty((len(d['names']),12288),np.float32);contents=[]
            loader=DataLoader(RGBImages(d['root'],d['names'],d['hashes']),batch_size=16,num_workers=oc['workers'],shuffle=False)
            for batch,(images,ids,hashes) in enumerate(loader):
                if ids.tolist()!=list(range(batch*16,batch*16+len(images))):raise ValueError('Image order changed')
                contents.extend(hashes);values[ids.numpy()]=torch.cat([old.encode(anchor,x.to('cuda:0')) for x in images.split(oc['microbatch'])]).cpu().numpy()
                write(out/'progress.json',dict(dataset=tag,images=int(ids[-1])+1,total=len(values)))
                if batch%64==0:print(tag,int(ids[-1])+1,'/',len(values),flush=True)
            old.validate_vectors(values,len(values));digest=metrics.order_digest([n+':'+h for n,h in zip(d['names'],contents)])
            before=read(prior/(tag+'.json'))
            if digest!=before['image_content_digest']:raise ValueError('Image bytes changed')
            reports={}
            for arm in ('innovation1',*models):
                transformed=values if arm=='innovation1' else np.empty((len(values),12288+c['side_dim']),np.float32)
                if arm!='innovation1':
                    for i in range(0,len(values),128):
                        z=torch.from_numpy(values[i:i+128]).to('cuda:0');h=models[arm]((z-projection['mean'])@projection['pca'])
                        transformed[i:i+len(z)]=fuse(z,h,c['score_weight']).cpu().numpy()
                if not np.isfinite(transformed).all() or np.max(np.abs(np.linalg.norm(transformed,axis=1)-1))>1e-4:raise ValueError('Nonunit descriptor')
                db=torch.from_numpy(transformed[:d['references']]).to('cuda:0')
                pred=np.concatenate([(torch.from_numpy(q).to('cuda:0')@db.T).topk(20,dim=1).indices.cpu().numpy()
                    for q in np.array_split(transformed[d['references']:],(d['queries']+15)//16)]);del db
                reports[arm]=metrics.retrieval_report(pred,d['ground_truth'],d['references'])
                if arm=='innovation1' and reports[arm]!=before['variants']['innovation1']:raise ValueError('Baseline Top20 not reproduced')
                del transformed
            comparison=dict(correct={a:r['correct'] for a,r in reports.items()},
                vs_innovation1={a:metrics.paired_report(reports['innovation1'],r) for a,r in reports.items() if a!='innovation1'},
                deficit_vs_controls={a:metrics.paired_report(reports[a],reports['deficit_target']) for a in ('constant_target','shuffled_target')})
            write(target,dict(dataset=contract['datasets'][tag],variants=reports,comparison=comparison,image_content_digest=digest))
            write(ready,dict(sha256=sha(target)));results[tag]=comparison;print(tag,comparison['correct'],flush=True)
        checks=dict(anchor_unchanged=state_hash(anchor)==frozen_hash,
            protected_sources_unchanged=old.key.source_hashes(old.key.PROTECTED)==contract['upstream']['dependency']['protected_sources'])
        gain=all(results[t]['correct']['deficit_target']>results[t]['correct']['innovation1'] for t in ('msls-val','pitts30k-val'))
        mechanism=gain and all(results[t]['correct']['deficit_target']>results[t]['correct'][a] for t in ('msls-val','pitts30k-val') for a in ('constant_target','shuffled_target'))
        write(out/'summary.json',dict(checks=checks,results=results,anchor_gain=gain,matched_mechanism_signal=mechanism,stable_gain_proven=False))
        if not all(checks.values()):raise ValueError('Integrity failure')
        finish(out)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('plan','prepare','smoke','train','audit','eval'))
    p.add_argument('--resume',action='store_true');p.add_argument('--allow-training',action='store_true');p.add_argument('--allow-shared-gpu',action='store_true')
    args=p.parse_args();c=config()
    if args.command in ('train','smoke') and not args.allow_training:p.error('Training permission required')
    if args.command=='plan':print(identity(c,'plan'));return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':p.error('Physical GPU1 required')
    import fcntl,subprocess
    with (ROOT/'logs/query_relation_screen_v1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        memory=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=20)
        old.check_gpu_capacity(*map(int,memory.strip().split(',')),args.allow_shared_gpu,.2);old.key.gpu(old.configuration())
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        {'prepare':prepare,'smoke':train,'train':train,'audit':audit,'eval':evaluate}[args.command](c,args)


if __name__=='__main__':main()
