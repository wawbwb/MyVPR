"""Matched auxiliary BoQ branches: whole image, random region, CLIP landmarks."""
import argparse
import copy
import hashlib
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.clip_token_drop import Cache,read,write,sha,seed,device,load_ru,image_tensor,save_torch,codes as old_codes
from src.models.static_evidence import LANDMARKS,selection,coordinates,make_batches
MODES=['full','random','static']


def codes():
    return {**old_codes(),**{p:hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for p in ['src/models/static_evidence.py','scripts/static_evidence.py']}}


def complete(out):
    write(out/'completed.json',{'complete':True,'files':{p.name:sha(p) for p in out.iterdir()
          if p.is_file() and p.name!='completed.json'}})


def verify(out):
    done=read(out/'completed.json')
    if not done['complete']: raise ValueError('Incomplete stage')
    for name,digest in done['files'].items():
        if sha(out/name)!=digest: raise ValueError('Changed file: '+name)
    c=read(out/'contract.json')
    if c['code']!=codes(): raise ValueError('Code contract changed')
    return c


def prepare(a):
    import torch
    from src.models.static_evidence import StaticTeacher
    from scripts.eval_dynamic_category_prior import extract_ru_feature_map
    cached=Cache(a.gsv_cache if a.split=='train' else a.msls_cache)
    root=a.gsv_root if a.split=='train' else a.msls_root
    if cached.contract['split']!=a.split: raise ValueError('Cache split mismatch')
    seed(42); d=device(a.device)
    visual,_,digest=load_ru(a,d)
    teacher=StaticTeacher(d)
    if a.split=='train':
        plan=read(a.plan/'plan.json'); done=read(a.plan/'completed.json')
        if not done['complete'] or sha(a.plan/'plan.json')!=done['plan_sha256'] or plan['cache_sha256']!=cached.hash:
            raise ValueError('Wrong original GSV adaptation split')
        groups=[[label,sorted(ids,key=lambda i:cached.records[i][0])[:4]] for label,ids in plan['split']['train']]
        ids=[i for _,g in groups for i in g]
        if len(groups)!=256 or len(ids)!=1024: raise ValueError('Expected 256 places x 4 views')
    else: ids=list(range(len(cached.records)));groups=None
    out=a.output;out.mkdir(parents=True,exist_ok=False)
    n=len(ids); fractions=np.lib.format.open_memmap(out/'fractions.npy',mode='w+',dtype='float16',shape=(n,20,20))
    features=None; desc=[]
    records=[cached.records[i] for i in ids]
    with torch.inference_mode():
        for start in range(0,n,4):
            selected=ids[start:start+4]
            x=torch.stack([image_tensor(root,cached.records[i][0],cached.image_hashes[i])[0] for i in selected]).to(d)
            fractions[start:start+len(selected)]=teacher.fractions(x)
            if a.split=='train':
                f=extract_ru_feature_map(visual,x)
                if not torch.isfinite(f).all(): raise ValueError('Nonfinite frozen features')
                if features is None:
                    features=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype='float32',shape=(n,*f.shape[1:]))
                features[start:start+len(selected)]=f.cpu().numpy()
                z=visual.aggregator(f)[0]
                if start==0 and not torch.allclose(z,visual(x),atol=2e-5,rtol=2e-4): raise ValueError('RU decomposition mismatch')
                desc.append(z.cpu().numpy())
            if (start+4)%64==0 or start+len(selected)==n: print(f'Static cache {a.split}: {start+len(selected)}/{n}',flush=True)
    fractions.flush()
    if features is not None:
        features.flush(); z=np.concatenate(desc).reshape(256,4,-1).mean(1);z/=np.linalg.norm(z,axis=1,keepdims=True)
        # Mean coordinate of all four selected views, not a cherry-picked view.
        coords=[np.mean([coordinates(cached.records[i][0]) for i in g],axis=0).tolist() for _,g in groups]
        write(out/'batches.json',make_batches(z,coords));np.save(out/'place_centroids.npy',z)
    available=[]; overlaps=[]
    for f,(path,_) in zip(fractions,records):
        m,ok=selection(f,path,'static');r,_=selection(f,path,'random');available.append(ok)
        if ok: overlaps.append(float(((~m)&(~r)).sum()/max(int((~m).sum()),1)))
    if not any(available): raise ValueError('No usable static evidence; stop')
    write(out/'contract.json',{'code':codes(),'split':a.split,'source_cache_sha256':cached.hash,'init_sha256':digest,
          'records':records,'teacher':teacher.identity,'landmarks':LANDMARKS,'threshold':.75,'minimum_tokens':16,
          'info':cached.contract['info'] if a.split=='msls' else {'train_plan_sha256':sha(a.plan/'plan.json')},
          'image_hashes':[cached.image_hashes[i] for i in ids],
          'available_images':sum(available),'images':n,'random_mean_overlap':float(np.mean(overlaps))})
    complete(out);print('Saved:',out)


def preflight(a):
    import unittest
    import torch
    from src.models.static_evidence import StaticTeacher
    from src.models.clip_token_drop import aggregate
    from scripts.eval_dynamic_category_prior import extract_ru_feature_map
    from src.losses.vpr_losses import VPRLossFunction
    prior=read(a.reference)
    if len(prior)!=740 or any(r['variant']!='frozen' for r in prior):raise ValueError('Missing original frozen outcomes')
    suite=unittest.TestSuite([unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=p)
                             for p in ['test_static_evidence.py','test_clip_token_drop.py']])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped: raise ValueError('All tests must pass without skips')
    seed(42);d=device(a.device);visual,_,digest=load_ru(a,d)
    cached=Cache(a.gsv_cache); plan=read(a.plan/'plan.json')
    ids=[i for _,g in plan['split']['train'][:2] for i in g[:2]]
    x=torch.stack([image_tensor(a.gsv_root,cached.records[i][0],cached.image_hashes[i])[0] for i in ids]).to(d)
    teacher=StaticTeacher(d);fractions=teacher.fractions(x)
    with torch.no_grad():
        f=extract_ru_feature_map(visual,x)
        expected=visual(x);full=aggregate(visual.aggregator,f,torch.zeros(4,20,20,dtype=torch.bool,device=d))
        if not torch.allclose(expected,full,atol=2e-5,rtol=2e-4): raise ValueError('Full branch differs')
    for p in visual.aggregator.parameters():p.requires_grad_(True)
    mask=torch.ones(4,20,20,dtype=torch.bool,device=d);mask[:,:5,:5]=False
    z=aggregate(visual.aggregator,f,mask)
    loss,_=VPRLossFunction()(z,torch.tensor([0,0,1,1],device=d))
    (loss+z[:,0].sum()).backward()
    if not any(p.grad is not None and p.grad.abs().sum()>0 for p in visual.aggregator.parameters()):raise ValueError('Missing branch gradients')
    if any(p.grad is not None for name,p in visual.named_parameters() if not name.startswith('aggregator.')):raise ValueError('Frozen encoder received gradients')
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'contract.json',{'code':codes(),'tests':result.testsRun,'ru_sha256':digest,'static_shape':list(fractions.shape)})
    complete(a.output)


def train(a):
    import torch
    from src.models.clip_token_drop import aggregate
    from src.losses.vpr_losses import VPRLossFunction
    c=verify(a.train_cache)
    if c['split']!='train': raise ValueError('Expected train features')
    seed(42);d=device(a.device);visual,_,digest=load_ru(a,d)
    if c['init_sha256']!=digest: raise ValueError('Wrong RU')
    agg=visual.aggregator
    for p in agg.parameters():p.requires_grad_(True)
    agg.train(); optimizer=torch.optim.AdamW(agg.parameters(),lr=1e-5,weight_decay=.001)
    frozen=[p for name,p in visual.named_parameters() if not name.startswith('aggregator.')];versions=[p._version for p in frozen]
    batches=read(a.train_cache/'batches.json');features=np.load(a.train_cache/'features.npy',mmap_mode='r');fractions=np.load(a.train_cache/'fractions.npy')
    contract={'code':codes(),'mode':a.mode,'cache_sha256':sha(a.train_cache/'completed.json'),'init_sha256':digest,
              'epochs':2,'batches_per_epoch':64,'P':16,'K':4,'lr':1e-5,'seed':42,'weight_decay':.001}
    start=0
    if a.output.exists():
        if not a.resume or read(a.output/'contract.json')!=contract: raise ValueError('Existing run needs matching resume')
        if (a.output/'completed.json').exists(): raise ValueError('Run complete')
        state=torch.load(a.output/'last.pt',map_location=d,weights_only=False)
        if state['contract']!=contract: raise ValueError('Checkpoint changed')
        agg.load_state_dict(state['aggregator']);optimizer.load_state_dict(state['optimizer']);start=state['epoch']
    else:
        a.output.mkdir(parents=True);write(a.output/'contract.json',contract)
    loss_fn=VPRLossFunction()
    for epoch in range(start,2):
        logs=[]
        for groups in batches[epoch]:
            ids=[g*4+j for g in groups for j in range(4)]
            f=torch.from_numpy(np.array(features[ids])).to(d)
            masks=[selection(fractions[i],c['records'][i][0],a.mode)[0] for i in ids]
            labels=torch.tensor([c['records'][i][1] for i in ids],device=d)
            optimizer.zero_grad(set_to_none=True)
            z=aggregate(agg,f,torch.from_numpy(np.stack(masks)).to(d));loss,uninformative=loss_fn(z,labels)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(agg.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            logs.append({'loss':float(loss),'uninformative_fraction':float(uninformative),'grad_norm':float(norm)})
            if len(logs)%8==0:print(f'{a.mode} epoch {epoch+1} batch {len(logs)}/64 loss {loss.item():.5f}',flush=True)
        if [p._version for p in frozen]!=versions or any(p.grad is not None for p in frozen):raise ValueError('Frozen encoder/gate changed')
        write(a.output/f'epoch_{epoch+1:02d}.json',{'batches':logs,'frozen_verified':True})
        save_torch(a.output/'last.pt',{'contract':contract,'aggregator':agg.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1})
    complete(a.output)


def evaluate(a):
    import torch
    from src.models.clip_token_drop import aggregate
    from scripts.eval_dynamic_category_prior import extract_ru_feature_map
    from scripts.clip_dynamic_screen import outcome
    c=verify(a.msls_static_cache); seed(42);d=device(a.device);visual,_,digest=load_ru(a,d)
    if c['split']!='msls' or c['init_sha256']!=digest or (len(c['records']),c['info']['ndb'])!=(19611,18871):raise ValueError('Wrong MSLS cache')
    heads={'frozen':visual.aggregator};contracts={}
    for mode in MODES:
        ct=verify(a.runs/mode);state=torch.load(a.runs/mode/'last.pt',map_location='cpu',weights_only=False)
        if state['contract']!=ct or ct['mode']!=mode or ct['init_sha256']!=digest or state['epoch']!=2:raise ValueError('Wrong branch checkpoint')
        agg=copy.deepcopy(visual.aggregator);agg.load_state_dict(state['aggregator']);heads[mode]=agg.to(d).eval();contracts[mode]=ct
    shared=[{k:v for k,v in ct.items() if k!='mode'} for ct in contracts.values()]
    if not all(ct==shared[0] for ct in shared):raise ValueError('Unmatched branch training')
    out=a.output;out.mkdir(parents=True,exist_ok=False);report=out/'report';report.mkdir()
    modes=['frozen',*MODES];fractions=np.load(a.msls_static_cache/'fractions.npy')
    arrays={m:np.lib.format.open_memmap(out/f'{m}_descriptors.npy',mode='w+',dtype='float32',shape=(19611,12288)) for m in modes}
    available=np.array([selection(f,p,'static')[1] for f,(p,_) in zip(fractions,c['records'])])
    with torch.inference_mode():
        for start in range(0,19611,4):
            end=min(start+4,19611)
            x=torch.stack([image_tensor(a.msls_root,c['records'][i][0],c['image_hashes'][i])[0] for i in range(start,end)]).to(d)
            f=extract_ru_feature_map(visual,x)
            for mode in modes:
                drop=[selection(fractions[i],c['records'][i][0],'full' if mode=='frozen' else mode)[0] for i in range(start,end)]
                z=aggregate(heads[mode],f,torch.from_numpy(np.stack(drop)).to(d)).cpu().numpy()
                if not np.isfinite(z).all() or not np.allclose(np.linalg.norm(z,axis=1),1,atol=2e-4):raise ValueError('Invalid descriptors')
                arrays[mode][start:end]=z
            if start//128!=end//128 or end==19611:print(f'Full branch features: {end}/19611',flush=True)
    rows=[]
    for mode,arr in arrays.items():
        arr.flush();db=torch.from_numpy(np.array(arr[:18871])).to(d)
        saved=np.lib.format.open_memmap(out/f'{mode}_scores.npy',mode='w+',dtype='float32',shape=(740,18871))
        with torch.inference_mode():
            for start in range(0,740,16):
                scores=(torch.from_numpy(np.array(arr[18871+start:18871+min(start+16,740)])).to(d)@db.T).cpu().numpy()
                saved[start:start+len(scores)]=scores
                for qi,s in enumerate(scores,start):rows.append({'query':qi,'mode':mode,'query_static_available':bool(available[18871+qi]),**outcome(s,c['info']['gt'][qi])})
        saved.flush();del db
    by={(r['query'],r['mode']):r for r in rows};results={}
    old=read(a.reference);old={r['query_index']:r for r in old if r['variant']=='frozen'}
    if len(old)!=740 or sum(by[q,'frozen']['top1_correct'] for q in range(740))!=675:raise ValueError('Frozen baseline mismatch')
    if any(by[q,'frozen']['top1_reference_index']!=old[q]['top1_reference_index'] for q in range(740)):raise ValueError('Frozen top1 IDs changed')
    for mode in modes:
        correct=[q for q in range(740) if by[q,mode]['top1_correct']]
        corrected=[q for q in correct if not by[q,'frozen']['top1_correct']]
        results[mode]={'correct':len(correct),'corrected_vs_ru':corrected,
            'regressed_vs_ru':[q for q in range(740) if by[q,'frozen']['top1_correct'] and not by[q,mode]['top1_correct']],
            'oracle_union_with_ru':675+len(corrected)}
    results['static_exclusive_corrections']=[q for q in results['static']['corrected_vs_ru'] if not by[q,'random']['top1_correct'] and not by[q,'full']['top1_correct']]
    results['static_exclusive_with_evidence_on_both_sides']=[q for q in results['static_exclusive_corrections']
        if available[18871+q] and available[by[q,'static']['top1_reference_index']]]
    write(report/'summary.json',results);write(report/'outcomes.json',rows)
    write(report/'training_logs.json',{m:{str(e):read(a.runs/m/f'epoch_{e:02d}.json') for e in [1,2]} for m in MODES})
    write(report/'mask_statistics.json',{k:c[k] for k in ['teacher','landmarks','threshold','minimum_tokens','available_images','images','random_mean_overlap']})
    write(report/'availability.json',{'query':available[18871:].tolist(),'database':available[:18871].tolist()})
    write(report/'contract.json',{'code':codes(),'training':contracts,'cache_sha256':sha(a.msls_static_cache/'completed.json'),
          'available_queries':int(available[18871:].sum()),'scope':'Separate auxiliary branches, no score fusion. MSLS historical exploratory test. Oracle union is an upper bound, not achieved recall.'})
    complete(report);write(out/'completed.json',{'complete':True,'report_sha256':sha(report/'completed.json'),
         'arrays':{p.name:sha(p) for p in out.glob('*.npy')}})
    print(results);print('Download:',report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','preflight','train','evaluate']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['train','msls'],default='train');p.add_argument('--mode',choices=MODES,default='full')
    p.add_argument('--gsv-cache',type=Path,default=Path('.cache/clearclip_token_drop_train_v1'))
    p.add_argument('--msls-cache',type=Path,default=Path('.cache/clearclip_token_drop_msls_v1'))
    p.add_argument('--train-cache',type=Path,default=Path('.cache/static_evidence_train_v1'))
    p.add_argument('--msls-static-cache',type=Path,default=Path('.cache/static_evidence_msls_v1'))
    p.add_argument('--plan',type=Path,default=Path('doc/dynamic_invariance_plan_v1'))
    p.add_argument('--runs',type=Path,default=Path('doc/static_evidence_train_v1'))
    p.add_argument('--gsv-root',type=Path,default=Path('datasets/gsv_cities'));p.add_argument('--msls-root',type=Path,default=Path('datasets/msls-val'))
    p.add_argument('--run',type=Path,default=Path('doc/dynamic_category_prior_screen_b0.5_full_db_condition_queries'))
    p.add_argument('--reference',type=Path,default=Path('doc/dynamic_invariance_msls_v1/report/frozen_outcomes.json'))
    p.add_argument('--checkpoint',type=Path);p.add_argument('--device',default='cuda:1');p.add_argument('--resume',action='store_true')
    a=p.parse_args();globals()[a.stage](a)


if __name__=='__main__':main()
