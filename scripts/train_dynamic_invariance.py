"""Small frozen-RU LoRA pilot: CLIP appearance invariance versus spatial control."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.clip_token_drop import Cache,read,write,sha,seed,device,load_ru,image_tensor,save_torch,codes as old_codes
from scripts.dynamic_invariance_utils import SETTINGS,masks,views,make_split,rng_for

MODES = ['plain','random','semantic']


def codes():
    return {**old_codes(), **{p:hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
           for p in ['scripts/train_dynamic_invariance.py','scripts/dynamic_invariance_utils.py',
                     'src/models/dynamic_invariance.py']}}


def new_output(path):
    path.mkdir(parents=True,exist_ok=False)


def cached_plan(a):
    cached = Cache(a.cache)
    plan = read(a.plan/'plan.json')
    complete = read(a.plan/'completed.json')
    if (not complete['complete'] or sha(a.plan/'plan.json')!=complete['plan_sha256']
        or plan['cache_sha256']!=cached.hash or plan['code']!=codes() or plan['settings']!=SETTINGS):
        raise ValueError('Plan/cache/code mismatch')
    return cached,plan


def rgb(cached,root,index):
    tensor,_ = image_tensor(root,cached.records[index][0],cached.image_hashes[index])
    return np.clip(tensor.numpy()*np.array([.229,.224,.225],np.float32)[:,None,None]
                   +np.array([.485,.456,.406],np.float32)[:,None,None],0,1)


def normalized(x):
    import torch
    return torch.from_numpy(((x-np.array([.485,.456,.406],np.float32)[:,None,None])
                            /np.array([.229,.224,.225],np.float32)[:,None,None]).copy())


def batch(cached,root,ids,mode,epoch):
    import torch
    clean,one,two = [],[],[]
    active = 0
    for i in ids:
        path = cached.records[i][0]
        x = rgb(cached,root,i)
        aligned,random,_ = masks(cached.fractions[i],path)
        m = aligned if mode=='semantic' else random if mode=='random' else np.zeros_like(aligned)
        a,b = views(x,m,path,epoch)
        clean.append(normalized(x)); one.append(normalized(a)); two.append(normalized(b))
        active += int(m.any())
    return [torch.stack(v) for v in [clean,one,two]],active


def prepare(a):
    from PIL import Image
    cached = Cache(a.cache)
    if cached.contract['split']!='train': raise ValueError('Need GSV training cache')
    split = make_split(cached.records)
    new_output(a.output)
    stats = {}; previews = []
    for group in ['train','dev']:
        counts = {'images':0,'active':0,'empty_or_large':0,'unshiftable':0,'mean_area':0.,'mean_overlap':0.}
        for _,ids in split[group]:
            for i in ids:
                path = cached.records[i][0]
                m,r,reason = masks(cached.fractions[i],path)
                counts['images']+=1; counts[reason]+=1
                counts['mean_area']+=float(m.mean())
                counts['mean_overlap']+=float((m&r).sum()/max(int(m.sum()),1))
                if reason=='active' and len(previews)<12 and group=='train':
                    x = rgb(cached,a.gsv_root,i)
                    original=x.transpose(1,2,0)
                    panels=[original, views(x,m,path,0)[0].transpose(1,2,0),views(x,r,path,0)[0].transpose(1,2,0)]
                    for mask in [m,r]:
                        overlay=original.copy(); pix=np.repeat(np.repeat(mask,14,0),14,1)
                        overlay[pix]=.5*overlay[pix]+.5*np.array([1,0,0])
                        panels.append(overlay)
                    filename=f'preview_{len(previews):02d}.jpg'
                    Image.fromarray((np.concatenate(panels,axis=1)*255).astype('uint8')).save(a.output/filename)
                    previews.append({'path':path,'file':filename})
        counts['mean_area']/=counts['images']
        counts['mean_overlap']/=max(counts['active'],1)
        stats[group]=counts
    if stats['train']['active']==0 or stats['dev']['active']==0:
        raise ValueError('No usable semantic regions; do not train')
    plan={'settings':SETTINGS,'code':codes(),'cache_sha256':cached.hash,
          'teacher':cached.contract['teacher'],'split':split,'region_stats':stats,'previews':previews}
    write(a.output/'plan.json',plan)
    import html
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Automatic appearance perturbations</h1>'
        '<p>Left to right: original, semantic perturbation, translated-control perturbation, semantic mask, control mask.</p>'
        +''.join(f'<p>{html.escape(p["path"])}</p><img style="max-width:100%" src="{p["file"]}">' for p in previews),encoding='utf8')
    write(a.output/'completed.json',{'complete':True,'plan_sha256':sha(a.output/'plan.json')})
    print(json.dumps(stats,indent=2)); print('Prepared:',a.output)


def build(a,d):
    import torch
    from src.models.dynamic_invariance import install
    model,_,digest = load_ru(a,d)
    model.eval()
    with torch.no_grad(): before=model(torch.zeros(1,3,280,280,device=d))
    params=install(model)
    model.eval()
    with torch.no_grad(): after=model(torch.zeros(1,3,280,280,device=d))
    if not torch.equal(before,after): raise ValueError('Zero adapter changed RU')
    return model,params,digest


def preflight(a):
    import unittest
    import torch
    from src.models.dynamic_invariance import disabled,replay_backward
    from src.losses.vpr_losses import VPRLossFunction
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_dynamic_invariance*.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped: raise ValueError('All tests must pass without skips')
    cached,plan=cached_plan(a)
    seed(42); d=device(a.device); model,params,digest=build(a,d)
    groups=plan['split']['train']
    first=next(group for _,group in groups if any(masks(cached.fractions[i],cached.records[i][0])[0].any() for i in group))
    active=next(i for i in first if masks(cached.fractions[i],cached.records[i][0])[0].any())
    second=next(group for _,group in groups if group!=first)
    ids=[active,next(i for i in first if i!=active),*second[:2]]
    inputs,_=batch(cached,a.gsv_root,ids,'semantic',0); inputs=[x.to(d) for x in inputs]
    if torch.equal(inputs[0],inputs[1]): raise ValueError('Real preflight perturbation is empty')
    with torch.no_grad(),disabled(model): teacher=model(inputs[0])
    with torch.no_grad(): actual=model(inputs[0])
    if not torch.allclose(actual,teacher,atol=2e-6,rtol=2e-5): raise ValueError('Real-image identity failed')
    replay_backward(model,inputs,teacher,torch.tensor([0,0,1,1],device=d),VPRLossFunction(),a.microbatch,True)
    if not any(p.grad is not None and p.grad.abs().sum()>0 for p in params): raise ValueError('No adapter gradients')
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad): raise ValueError('Frozen gradients')
    new_output(a.output)
    write(a.output/'completed.json',{'complete':True,'tests':result.testsRun,'code':codes(),
          'init_sha256':digest,'adapter_parameters':sum(p.numel() for p in params),'real_gradient_check':True})
    print('Preflight passed:',a.output)


def train(a):
    import torch
    from src.models.dynamic_invariance import disabled,replay_backward,adapter_state,restore
    from src.losses.vpr_losses import VPRLossFunction
    cached,plan=cached_plan(a); seed(42); d=device(a.device)
    model,params,digest=build(a,d)
    contract={'mode':a.mode,'settings':SETTINGS,'code':codes(),'plan_sha256':sha(a.plan/'plan.json'),
              'init_sha256':digest,'microbatch':a.microbatch,'adapter_parameters':sum(p.numel() for p in params)}
    optimizer=torch.optim.AdamW(params,lr=SETTINGS['lr'],weight_decay=SETTINGS['weight_decay'])
    start=0
    if a.output.exists():
        if not a.resume or read(a.output/'contract.json')!=contract:
            raise ValueError('Existing output needs matching --resume')
        if (a.output/'completed.json').exists(): raise ValueError('Already complete')
        state=torch.load(a.output/'last.pt',map_location=d,weights_only=False)
        if state['contract']!=contract: raise ValueError('Resume contract differs')
        restore(model,state['adapter']); optimizer.load_state_dict(state['optimizer']); start=state['epoch']
    else:
        new_output(a.output); write(a.output/'contract.json',contract)
    frozen=[p for p in model.parameters() if not p.requires_grad]; versions=[p._version for p in frozen]
    loss_fn=VPRLossFunction()
    for epoch in range(start,SETTINGS['epochs']):
        seed(42+epoch)
        groups=plan['split']['train']
        order=rng_for(f'order:{epoch}').permutation(len(groups))
        logs=[]
        for step in range(0,len(order),SETTINGS['places']):
            ids=[]; labels=[]
            for j in order[step:step+SETTINGS['places']]:
                label,pool=groups[int(j)]
                chosen=rng_for(f'views:{epoch}:{label}').choice(pool,4,replace=False).tolist()
                ids.extend(chosen); labels.extend([label]*4)
            inputs,active=batch(cached,a.gsv_root,ids,a.mode,epoch)
            inputs=[x.to(d) for x in inputs]
            with torch.no_grad(),disabled(model):
                teacher=torch.cat([model(inputs[0][i:i+a.microbatch]) for i in range(0,len(ids),a.microbatch)])
            optimizer.zero_grad(set_to_none=True)
            values=replay_backward(model,inputs,teacher,torch.tensor(labels,device=d),loss_fn,a.microbatch,a.mode!='plain')
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            optimizer.step()
            logs.append(dict(values,active=active,images=len(ids),grad_norm=float(norm)))
            if len(logs)%8==0: print(f'{a.mode} epoch {epoch+1}/2 batch {len(logs)}/64 {values}',flush=True)
        if [p._version for p in frozen]!=versions or any(p.grad is not None for p in frozen):
            raise ValueError('Frozen RU changed')
        write(a.output/f'epoch_{epoch+1:02d}.json',{'batches':logs,'frozen_verified':True})
        save_torch(a.output/'last.pt',{'adapter':adapter_state(model),'optimizer':optimizer.state_dict(),
                    'contract':contract,'epoch':epoch+1})
    write(a.output/'completed.json',{'complete':True,'epochs':SETTINGS['epochs'],
          'checkpoint_sha256':sha(a.output/'last.pt'),'contract_sha256':sha(a.output/'contract.json')})
    print('Saved training:',a.output)


def checkpoints(a,model,digest):
    import torch
    states={}; contracts={}
    for mode in MODES:
        run=a.runs/mode; done=read(run/'completed.json')
        if (not done['complete'] or sha(run/'last.pt')!=done['checkpoint_sha256']
            or sha(run/'contract.json')!=done['contract_sha256']): raise ValueError('Incomplete or changed run')
        state=torch.load(run/'last.pt',map_location='cpu',weights_only=False); c=state['contract']
        if (c!=read(run/'contract.json') or c['code']!=codes() or c['settings']!=SETTINGS
            or c['init_sha256']!=digest or c['mode']!=mode or state['epoch']!=SETTINGS['epochs']):
            raise ValueError('Wrong checkpoint contract')
        states[mode]=state['adapter']; contracts[mode]=c
    shared=[{k:v for k,v in c.items() if k!='mode'} for c in contracts.values()]
    if not all(c==shared[0] for c in shared): raise ValueError('Unmatched arms')
    return states,contracts


def retrieval(arr,db_indices,q_indices,gt):
    database=np.asarray(arr[db_indices]); records=[]
    for q,idx in enumerate(q_indices):
        scores=np.asarray(arr[idx])@database.T
        order=np.argsort(-scores,kind='stable'); positives=set(gt[q])
        rank=next(k+1 for k,j in enumerate(order) if int(j) in positives)
        records.append({'query':q,'top1':int(order[0]),'best_gt_rank':rank})
    return records


def evaluate(a):
    import torch
    from src.models.dynamic_invariance import disabled,restore
    cached,plan=cached_plan(a); seed(42); d=device(a.device)
    model,_,digest=build(a,d); states,contracts=checkpoints(a,model,digest)
    if contracts['plain']['plan_sha256']!=sha(a.plan/'plan.json'): raise ValueError('Wrong development split')
    ids=[i for _,group in plan['split']['dev'] for i in group]
    dbidx=[i for i in range(len(ids)) if i%4!=3]; qidx=list(range(3,len(ids),4))
    gt=[list(range(q*3,q*3+3)) for q in range(len(qidx))]
    new_output(a.output); summaries={}; all_rows=[]
    for mode in ['frozen',*MODES]:
        if mode!='frozen': restore(model,states[mode])
        maps={k:np.lib.format.open_memmap(a.output/f'{mode}_{k}.npy',mode='w+',dtype='float32',
                                          shape=(len(ids),12288)) for k in ['clean','semantic','random']}
        from contextlib import nullcontext
        with torch.no_grad(), (disabled(model) if mode=='frozen' else nullcontext()):
            for start in range(0,len(ids),a.microbatch):
                chosen=ids[start:start+a.microbatch]
                sem,_=batch(cached,a.gsv_root,chosen,'semantic',999)
                rand,_=batch(cached,a.gsv_root,chosen,'random',999)
                for k,x in [('clean',sem[0]),('semantic',sem[1]),('random',rand[1])]:
                    z=model(x.to(d)).cpu().numpy()
                    if not np.isfinite(z).all() or not np.allclose(np.linalg.norm(z,axis=1),1,atol=2e-4):
                        raise ValueError('Invalid descriptors')
                    maps[k][start:start+len(chosen)]=z
                if (start+a.microbatch)%64==0: print(f'Dev {mode}: {start+len(chosen)}/{len(ids)}',flush=True)
        for v in maps.values(): v.flush()
        summaries[mode]={}
        for kind in maps:
            # Perturb queries only; all variants use clean reference descriptors.
            mixed=np.array(maps['clean']); mixed[qidx]=maps[kind][qidx]
            rows=retrieval(mixed,dbidx,qidx,gt)
            all_rows.extend(dict(r,mode=mode,condition=kind) for r in rows)
            active=np.array([masks(cached.fractions[i],cached.records[i][0])[0].any() for i in ids])
            sensitivity=np.square(np.asarray(maps[kind])-np.asarray(maps['clean'])).sum(1)
            summaries[mode][kind]={'correct':sum(r['best_gt_rank']==1 for r in rows),'queries':len(rows),
                'r_at_5':sum(r['best_gt_rank']<=5 for r in rows)/len(rows),
                'descriptor_squared_change_active_mean':float(sensitivity[active].mean()),
                'active_images':int(active.sum())}
    write(a.output/'summary.json',summaries); write(a.output/'outcomes.json',all_rows)
    write(a.output/'provenance.json',{'code':codes(),'training':contracts,'plan_sha256':sha(a.plan/'plan.json'),
          'warning':'Adaptation-held-out city, 3 references/place, place-label GT, not a standard VPR benchmark or unseen pretraining test.'})
    report=a.output/'report'; report.mkdir()
    for name in ['summary.json','outcomes.json','provenance.json']:
        (report/name).write_bytes((a.output/name).read_bytes())
    done={'complete':True,'files':{n:sha(report/n) for n in ['summary.json','outcomes.json','provenance.json']},
          'arrays':{p.name:sha(p) for p in a.output.glob('*.npy')}}
    write(a.output/'completed.json',done); write(report/'completed.json',done)
    print(json.dumps(summaries,indent=2)); print('Report:',report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','preflight','train','evaluate'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('.cache/clearclip_token_drop_train_v1'))
    p.add_argument('--plan',type=Path,default=Path('doc/dynamic_invariance_plan_v1'))
    p.add_argument('--runs',type=Path,default=Path('doc/dynamic_invariance_train_v1'))
    p.add_argument('--gsv-root',type=Path,default=Path('datasets/gsv_cities'))
    p.add_argument('--run',type=Path,default=Path('doc/dynamic_category_prior_screen_b0.5_full_db_condition_queries'))
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--microbatch',type=int,default=2)
    p.add_argument('--mode',choices=MODES,default='plain')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.microbatch<1: p.error('microbatch must be positive')
    globals()[a.stage](a)


if __name__=='__main__': main()
