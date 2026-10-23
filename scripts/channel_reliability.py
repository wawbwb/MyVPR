"""Train-population branch reliability admission; frozen features, finite budget."""
import argparse
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts import relation_metric as rm
from scripts import spatial_adapter_repeat as original
from src.semantic_region_protocol import read,write,sha,owned_run,state_hash,save_npz

CACHE=ROOT/'.cache/channel_reliability_v1'
REPORT=ROOT/'doc/channel_reliability_v1'
FILES=('scripts/channel_reliability.py','scripts/launch_channel_reliability.py',
    'tests/test_channel_reliability.py','doc/CHANNEL_RELIABILITY_V1.md')


def finish(out):
    write(out/'completed.json',dict(complete=True,files={p.name:sha(p) for p in out.iterdir()
        if p.is_file() and p.name not in ('.lock','completed.json')}))


def verify(out):
    v=read(out/'completed.json')
    if not v['complete']:raise ValueError('Incomplete')
    for n,h in v['files'].items():
        if Path(n).name!=n or sha(out/n)!=h:raise ValueError('Corrupt '+n)


def identity(stage):
    source=ROOT/'logs/spatial_adapter_repeat_v1/train_seed42'
    original.verified(source)
    c=rm.configuration();plan=rm.key.open_plan(c);rows=rm.previous.population(plan,c)
    rm.verified(rm.key.path(c,'metric_cache')/'train')
    return dict(stage=stage,files=rm.key.source_hashes(FILES),original_checkpoint=sha(source/'last.pt'),
        anchor_cache=sha(rm.key.path(c,'metric_cache')/'train/completed.json'),
        plan_sha256=sha(rm.key.path(c,'plan')/'plan.json'),
        protected_sources=rm.key.source_hashes(rm.key.PROTECTED)),c,plan,rows


def cache(args):
    from scripts.semantic_region_study import ImageRows
    from torch.utils.data import DataLoader,Subset
    contract,c,plan,rows=identity('channel_cache')
    out=CACHE/'channel'
    with owned_run(out,contract,args.resume):
        if (out/'completed.json').exists():verify(out);return
        models=original.build(original.configuration(ROOT/'config/spatial_adapter_repeat.yaml'),42)
        model=models['channel'];del models
        payload=torch.load(ROOT/'logs/spatial_adapter_repeat_v1/train_seed42/last.pt',map_location='cpu',weights_only=True)
        model.adapter.load_state_dict(payload['adapters']['channel'],strict=True);model.requires_grad_(False).eval()
        expected=read(ROOT/'logs/spatial_adapter_repeat_v1/train_seed42/summary.json')['adapter_sha256']['channel']
        if state_hash(model.adapter)!=expected:raise ValueError('Wrong channel weights')
        before=state_hash(model)
        ds=ImageRows(rm.key.path(c,'dataset_root'),plan['images'],322)
        todo=[]
        for i in range(len(rows)//64):
            p=out/f'{i:04d}.npz';ready=out/f'{i:04d}.ready.json'
            if ready.exists():
                if sha(p)!=read(ready)['sha256']:raise ValueError('Corrupt shard')
            else:todo.append(i)
        loader=DataLoader(Subset(ds,[r for i in todo for r in rows[i*64:(i+1)*64]]),batch_size=64,num_workers=2)
        with torch.inference_mode():
            for i,(images,ids) in zip(todo,loader):
                if ids.tolist()!=rows[i*64:(i+1)*64]:raise ValueError('Row order changed')
                z=torch.cat([model(x.to('cuda:0')) for x in images.split(2)]).cpu().numpy()
                rm.validate_vectors(z,64)
                save_npz(out/f'{i:04d}.npz',descriptors=z,row_ids=ids.numpy())
                write(out/f'{i:04d}.ready.json',dict(sha256=sha(out/f'{i:04d}.npz')))
                write(out/'progress.json',dict(shards=i+1,total=512));print('CHANNEL',i+1,'/512',flush=True)
        if state_hash(model)!=before or rm.key.source_hashes(rm.key.PROTECTED)!=contract['protected_sources']:raise ValueError('Frozen state changed')
        write(out/'summary.json',dict(images=len(rows),frozen_unchanged=True,adapter_sha256=expected));finish(out)


def retrieval(args):
    contract,c,plan,rows=identity('train_retrieval');verify(CACHE/'channel')
    contract['channel_seal']=sha(CACHE/'channel/completed.json')
    out=CACHE/'retrieval'
    with owned_run(out,contract,args.resume):
        if (out/'completed.json').exists():verify(out);return
        coords=np.deg2rad(np.array([plan['images'][r]['coordinates'] for r in rows]))
        qi=np.arange(0,len(rows),4);di=np.array([i for i in range(len(rows)) if i%4])
        saved=dict(query_rows=np.array(rows)[qi],coordinates=np.rad2deg(coords[qi]))
        for name in ('anchor','channel'):
            bank=rm.bank(c) if name=='anchor' else np.concatenate([np.load(CACHE/'channel'/f'{i:04d}.npz')['descriptors'] for i in range(512)])
            db=torch.tensor(bank[di],device='cuda:0');pred=[];scores=[]
            with torch.inference_mode():
                for start in range(0,len(qi),64):
                    q=qi[start:start+64];s=torch.tensor(bank[q],device='cuda:0')@db.T
                    x,y=coords[q,None,:],coords[None,di,:]
                    h=np.sin((x[:,:,0]-y[:,:,0])/2)**2+np.cos(x[:,:,0])*np.cos(y[:,:,0])*np.sin((x[:,:,1]-y[:,:,1])/2)**2
                    close=h<(np.sin(25/6371000/2)**2)
                    illegal=close & (q[:,None]//4!=di[None,:]//4)
                    s.masked_fill_(torch.tensor(illegal,device='cuda:0'),-torch.inf)
                    v,j=s.topk(20,dim=1);scores.append(v.cpu().numpy());pred.append(di[j.cpu().numpy()])
                    write(out/'progress.json',dict(branch=name,queries=start+len(q),total=len(qi)))
            saved[name+'_pred']=np.concatenate(pred);saved[name+'_scores']=np.concatenate(scores)
            del db,bank
        save_npz(out/'rankings.npz',**saved)
        write(out/'summary.json',dict(queries=8192,gallery=24576,gps_exclusion_m=25,training_only=True));finish(out)


def features(a,b,sa,sb):
    result=[]
    for s in (sa,sb):
        p=np.exp((s-s[:,:1])/.05);p/=p.sum(1,keepdims=True)
        result.extend([s[:,0],s[:,0]-s[:,1],s[:,0]-s[:,4],s[:,0]-s[:,19],-(p*np.log(p.clip(1e-12))).sum(1)])
    result.extend([np.array([len(set(x)&set(y))/len(set(x)|set(y)) for x,y in zip(a,b)]),
        (a[:,0]==b[:,0]).astype(float),
        np.array([1/(np.flatnonzero(y==x[0])[0]+1) if x[0] in y else 0 for x,y in zip(a,b)]),
        np.array([1/(np.flatnonzero(x==y[0])[0]+1) if y[0] in x else 0 for x,y in zip(a,b)])])
    f=np.stack(result,1)
    if not np.isfinite(f).all():raise ValueError('Invalid features')
    return f


def geo_split(coords):
    from scipy.spatial import cKDTree
    x=np.deg2rad(coords);xyz=np.c_[np.cos(x[:,0])*np.cos(x[:,1]),np.cos(x[:,0])*np.sin(x[:,1]),np.sin(x[:,0])]
    parent=np.arange(len(x))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for a,b in sorted(cKDTree(xyz).query_pairs(2*np.sin(50/6371000/2))):
        a,b=find(a),find(b);parent[max(a,b)]=min(a,b)
    groups=np.array([find(i) for i in range(len(x))]);unique=np.unique(groups)
    shuffled=np.random.default_rng(42).permutation(unique)
    train=np.isin(groups,shuffled[:int(.7*len(unique))])
    return train,groups


def analyze(args):
    from sklearn.linear_model import LogisticRegression
    from scripts.channel_rank_fusion import fuse
    contract,_,_,_=identity('cpu_admission');verify(CACHE/'retrieval')
    contract['retrieval_seal']=sha(CACHE/'retrieval/completed.json')
    with owned_run(REPORT,contract,args.resume):
        if (REPORT/'completed.json').exists():verify(REPORT);return
        with np.load(CACHE/'retrieval/rankings.npz') as f:d={n:f[n] for n in f.files}
        a,b=d['anchor_pred'],d['channel_pred'];labels=np.arange(8192)
        ha=a[:,0]//4==labels;hb=b[:,0]//4==labels
        x=features(a,b,d['anchor_scores'],d['channel_scores']);train,groups=geo_split(d['coordinates']);test=~train
        win=~ha&hb;lose=ha&~hb;discord=win|lose
        counts={n:dict(queries=int(m.sum()),channel_only=int((win&m).sum()),anchor_only=int((lose&m).sum())) for n,m in [('fit',train),('heldout',test)]}
        sufficient=all(counts[n]['channel_only']>=k and counts[n]['anchor_only']>=k for n,k in [('fit',10),('heldout',5)]) and int((discord&train).sum())>=30
        stats=dict(split_counts=counts,geo_components=len(set(groups)),training_populations_seen_by_frozen_adapters=True,
            no_benchmark_used=True,feature_dimensions=x.shape[1],sufficient_discordant_cases=sufficient)
        preds=dict(anchor=a[:,0],channel=b[:,0],fixed_rrf=fuse(a,b)[:,0],larger_margin=np.where(d['channel_scores'][:,0]-d['channel_scores'][:,1]>d['anchor_scores'][:,0]-d['anchor_scores'][:,1],b[:,0],a[:,0]))
        if sufficient:
            mu=x[train].mean(0);std=x[train].std(0).clip(1e-6);z=(x-mu)/std;fit=train&discord
            y=win[fit].astype(int)
            for name,target in [('logistic',y),('shuffled',np.random.default_rng(42).permutation(y))]:
                model=LogisticRegression(C=1,solver='lbfgs',max_iter=1000,random_state=42).fit(z[fit],target)
                if int(model.n_iter_.max())>=1000:raise ValueError('Logistic regression did not converge')
                p=model.predict_proba(z)[:,1];preds[name]=np.where(p>.5,b[:,0],a[:,0])
                save_npz(REPORT/(name+'.npz'),coefficient=model.coef_,intercept=model.intercept_,mean=mu,std=std,probability=p)
        results={}
        for split,mask in [('fit',train),('heldout',test)]:
            results[split]={}
            for name,pred in preds.items():
                hit=pred//4==labels;fix=~ha&hit&mask;harm=ha&~hit&mask
                results[split][name]=dict(correct=int((hit&mask).sum()),fixes=int(fix.sum()),harms=int(harm.sum()),net=int(fix.sum()-harm.sum()))
        gate=False
        if sufficient:
            r=results['heldout'];g=r['logistic']
            gate=g['fixes']>=5 and g['net']>=3 and g['harms']*2<=g['fixes'] and all(g['correct']>r[n]['correct'] for n in ('fixed_rrf','shuffled','larger_margin','channel'))
        save_npz(REPORT/'outcomes.npz',train=train,groups=groups,anchor_hit=ha,channel_hit=hb,**preds)
        write(REPORT/'summary.json',dict(**stats,results=results,admission=gate,stable_gain_proven=False,
            verdict='TRAIN_SIGNAL_REQUIRES_EXTERNAL_VALIDATION' if gate else 'STOP_NO_HELDOUT_RELIABILITY_SIGNAL'))
        finish(REPORT);print(read(REPORT/'summary.json'),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['cache','retrieval','analyze']);p.add_argument('--resume',action='store_true');p.add_argument('--allow-shared-gpu',action='store_true');a=p.parse_args()
    if a.stage=='analyze':analyze(a);return
    from scripts.single_dino_supervision import check_gpu_capacity
    import fcntl,subprocess
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':p.error('Physical GPU1 only')
    with (ROOT/'logs/query_relation_screen_v1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        s=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
        check_gpu_capacity(*map(int,s.strip().split(',')),a.allow_shared_gpu,.2)
        rm.key.gpu(rm.configuration());torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        {'cache':cache,'retrieval':retrieval}[a.stage](a)


if __name__=='__main__':main()
