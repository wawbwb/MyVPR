"""Cache dense CLIP, train matched BoQ arms, and evaluate both sides of retrieval."""
import argparse
import csv
import hashlib
import io
import json
import random
import shutil
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.models.clip_token_drop import SETTINGS,DYNAMIC,STATIC,drop_mask


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,data):
    path=Path(path); tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(path)


def codes():
    paths=['scripts/clip_token_drop.py','src/models/clip_token_drop.py','src/models/clip_teacher.py',
           'src/models/aggregators/boq.py','src/dataloaders/train/gsv_cities.py',
           'scripts/eval_condition_robustness.py','scripts/eval_dynamic_category_prior.py',
           'src/losses/vpr_losses.py']
    return {p:hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in paths}


def seed(value):
    import torch
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def device(value):
    import torch
    d=torch.device(value)
    if d.type!='cuda' or d.index is None or d.index<1:
        raise ValueError('Use explicit cuda:1 or higher; GPU0 is faulty')
    return d


def dataset_records(a):
    if a.split=='train':
        from src.dataloaders.train.gsv_cities import GSVCitiesDataset
        root=a.gsv_root
        cities=sorted(f.stem for f in (root/'Dataframes').glob('*.csv'))
        if not cities: raise ValueError('No GSV Dataframes found')
        ds=GSVCitiesDataset(dataset_path=root,cities=cities,img_per_place=4)
        records=[['Images/'+str(row['city_id'])+'/'+ds.get_img_name(row),int(label)]
                 for label,row in ds.dataframe.iterrows()]
        info={'manifests':{str(f.relative_to(root)):sha(f) for f in sorted((root/'Dataframes').glob('*.csv'))}}
    else:
        from scripts.clip_dynamic_screen import load_dataset
        root=a.msls_root
        _,db,queries,gt,inputs=load_dataset(root,a.run)
        records=[[p,-1] for p in db+queries]
        info={'ndb':len(db),'gt':[v.tolist() for v in gt], 'source_inputs':inputs}
    if len({r[0] for r in records})!=len(records): raise ValueError('Duplicate image paths')
    return root,records,info


def image_tensor(root,path,expected=None,augment=False):
    from PIL import Image
    from scripts.eval_condition_robustness import build_transform
    target=(root/path).resolve()
    if not target.is_relative_to(root.resolve()): raise ValueError('Image outside dataset root')
    data=target.read_bytes(); digest=hashlib.sha256(data).hexdigest()
    if expected is not None and digest!=expected: raise ValueError('Image changed: '+path)
    with Image.open(io.BytesIO(data)) as im:
        im=im.convert('RGB')
        if augment:
            from torchvision.transforms import ColorJitter
            im=ColorJitter(.2,.2,.2,.05)(im)
        tensor=build_transform((280,280))(im)
    return tensor,digest


def cache(a):
    import torch
    from src.models.clip_token_drop import DenseCLIPTeacher
    seed(42); d=device(a.device); root,records,info=dataset_records(a)
    teacher=DenseCLIPTeacher(d)
    contract={'settings':SETTINGS,'dynamic':DYNAMIC,'static':STATIC,'code':codes(),
              'teacher':teacher.identity,'split':a.split,'records':records,'info':info,'chunk':128}
    out=a.output
    if out.exists():
        if not (out/'contract.json').is_file() or read(out/'contract.json')!=contract:
            raise ValueError('Cache contract changed; use new output')
    else:
        out.mkdir(parents=True); (out/'shards').mkdir(); write(out/'contract.json',contract)
    if (out/'completed.json').exists():
        Cache(out); print('Completed cache verified:',out); return
    hashes={}; total_drop=0; all_keep=0; fallback=0
    for start in range(0,len(records),128):
        end=min(start+128,len(records)); file=out/'shards'/f'{start:07d}.npz'
        if file.exists():
            with np.load(file,allow_pickle=False) as z:
                fractions=z['fractions'].copy(); image_hashes=z['image_hashes'].tolist()
            if fractions.shape!=(end-start,20,20) or len(image_hashes)!=end-start:
                raise ValueError('Invalid cache shard')
            for (path,_),h in zip(records[start:end],image_hashes):
                if sha(root/path)!=h: raise ValueError('Cached image changed: '+path)
        else:
            values=[]; image_hashes=[]
            for i in range(start,end,a.batch_size):
                batch=[image_tensor(root,p) for p,_ in records[i:min(i+a.batch_size,end)]]
                values.append(teacher.fractions(torch.stack([v[0] for v in batch])))
                image_hashes.extend(v[1] for v in batch)
            fractions=np.concatenate(values)
            tmp=file.with_suffix('.tmp')
            with tmp.open('wb') as f: np.savez_compressed(f,fractions=fractions,image_hashes=image_hashes)
            tmp.replace(file)
        for fraction,(path,_) in zip(fractions,records[start:end]):
            mask,fb=drop_mask(fraction,path,'aligned'); total_drop+=int(mask.sum()); all_keep+=int(not mask.any()); fallback+=int(fb)
        hashes[file.name]=sha(file)
        print(f'ClearCLIP {a.split}: {end}/{len(records)}',flush=True)
    write(out/'completed.json',{'complete':True,'contract_sha256':sha(out/'contract.json'),
          'shards':hashes,'images':len(records),'drop_fraction':total_drop/(len(records)*400),
          'all_keep_images':all_keep,'fallback_images':fallback})
    print('Saved cache:',out)


class Cache:
    def __init__(self,path):
        self.path=Path(path); self.contract=read(self.path/'contract.json'); done=read(self.path/'completed.json')
        if not done['complete'] or sha(self.path/'contract.json')!=done['contract_sha256']:
            raise ValueError('Incomplete or changed cache')
        if self.contract['settings']!=SETTINGS or self.contract['code']!=codes():
            raise ValueError('Cache code/config differs')
        self.records=self.contract['records']; self.fractions=np.empty((len(self.records),20,20),np.float16); self.image_hashes=[]
        expected_names=[f'{s:07d}.npz' for s in range(0,len(self.records),128)]
        if sorted(done['shards'])!=expected_names: raise ValueError('Missing cache shards')
        for start,name in zip(range(0,len(self.records),128),expected_names):
            file=self.path/'shards'/name
            if sha(file)!=done['shards'][name]: raise ValueError('Changed cache shard: '+name)
            with np.load(file,allow_pickle=False) as z:
                f=z['fractions']; n=min(128,len(self.records)-start)
                if f.shape!=(n,20,20) or not np.isfinite(f).all() or (f<0).any() or (f>1).any(): raise ValueError('Invalid fractions')
                self.fractions[start:start+n]=f; self.image_hashes.extend(z['image_hashes'].tolist())
        if len(self.image_hashes)!=len(self.records): raise ValueError('Invalid hash count')
        self.hash=sha(self.path/'completed.json')
        self.stats=done


class Places:
    def __init__(self,cached,root,epoch,mode):
        self.cache=cached; self.root=root; self.epoch=epoch; self.mode=mode
        groups={}
        for i,(_,label) in enumerate(cached.records): groups.setdefault(label,[]).append(i)
        self.groups=sorted(groups.items())
        if any(len(ids)<4 for _,ids in self.groups): raise ValueError('Place has fewer than four views')
    def __len__(self): return len(self.groups)
    def __getitem__(self,index):
        import torch
        label,ids=self.groups[index]; rng=np.random.default_rng(42+self.epoch*1000003+index)
        chosen=rng.choice(ids,4,replace=False); images=[]; masks=[]
        for i in chosen:
            path=self.cache.records[i][0]
            x,_=image_tensor(self.root,path,self.cache.image_hashes[i],augment=True)
            mask,_=drop_mask(self.cache.fractions[i],path,self.mode)
            images.append(x); masks.append(mask)
        return torch.stack(images),torch.from_numpy(np.stack(masks)),torch.full((4,),label,dtype=torch.long)


def load_ru(a,d):
    from scripts.eval_condition_robustness import load_inference_model_from_ckpt
    record=read(a.run/'run.json')['checkpoint']; path=a.checkpoint or Path(record['path'])
    if sha(path)!=record['sha256']: raise ValueError('Wrong RU checkpoint')
    model=load_inference_model_from_ckpt(path,d).eval()
    if model.semantic_region_gate is None or model.spatial_attn_head is not None or model.aggregator.semantic_num_classes is not None:
        raise ValueError('Requires original RU with plain BoQ')
    for p in model.parameters(): p.requires_grad_(False)
    return model,path,record['sha256']


def save_torch(path,data):
    import torch
    tmp=path.with_suffix('.tmp'); torch.save(data,tmp); tmp.replace(path)


def train(a):
    import torch
    from torch.utils.data import DataLoader
    from src.losses.vpr_losses import VPRLossFunction
    from src.models.clip_token_drop import aggregate
    from scripts.eval_dynamic_category_prior import extract_ru_feature_map
    seed(42); d=device(a.device); cached=Cache(a.cache)
    if cached.contract['split']!='train': raise ValueError('Need train cache')
    if a.mode!='none' and cached.stats['drop_fraction']==0:
        raise ValueError('Cache drops no tokens; semantic training would be identical to no-drop')
    visual,checkpoint,checkpoint_sha=load_ru(a,d)
    contract={'cache_sha256':cached.hash,'teacher':cached.contract['teacher'],'init_sha256':checkpoint_sha,'settings':SETTINGS,'code':codes(),
              'mode':a.mode,'epochs':a.epochs,'places_per_batch':a.places_per_batch,'images_per_place':4,
              'lr':1e-5,'weight_decay':.001,'seed':42,'smoke':a.smoke_test,'augmentation':'ColorJitter only',
              'microbatch':a.microbatch,'workers':a.workers}
    out=a.output
    if out.exists():
        if not a.resume or read(out/'contract.json')!=contract or (out/'completed.json').exists():
            raise ValueError('Existing run: require matching --resume and unfinished run')
    else:
        if a.resume: raise ValueError('Resume output missing')
        out.mkdir(parents=True); write(out/'contract.json',contract)
    for p in visual.aggregator.parameters(): p.requires_grad_(True)
    optimizer=torch.optim.AdamW(visual.aggregator.parameters(),lr=1e-5,weight_decay=.001)
    loss_fn=VPRLossFunction(); start_epoch=0; step=0
    if a.resume and (out/'last.pt').exists():
        state=torch.load(out/'last.pt',map_location=d,weights_only=False)
        if state['contract']!=contract: raise ValueError('Checkpoint contract differs')
        visual.aggregator.load_state_dict(state['aggregator']); optimizer.load_state_dict(state['optimizer'])
        start_epoch=state['epoch']; step=state['step']
    # Frozen components are never put into train mode, even during aggregation training.
    frozen=list(visual.backbone.parameters())+list(visual.semantic_region_gate.parameters())
    frozen_versions=[p._version for p in frozen]
    visual.aggregator.eval()
    with torch.no_grad():
        x=torch.zeros(1,3,280,280,device=d); f=extract_ru_feature_map(visual,x)
        baseline=visual.aggregator(f)[0]
        if not torch.allclose(baseline,visual(x),atol=2e-5,rtol=2e-4): raise ValueError('RU composition mismatch')
        if not torch.equal(baseline,aggregate(visual.aggregator,f,torch.zeros(1,20,20,dtype=torch.bool,device=d))):
            raise ValueError('All-keep path mismatch')
    completed_epochs=start_epoch
    for epoch in range(start_epoch,a.epochs):
        seed(42+epoch); visual.aggregator.train()
        ds=Places(cached,a.gsv_root,epoch,a.mode)
        loader=DataLoader(ds,batch_size=a.places_per_batch,shuffle=True,drop_last=True,num_workers=a.workers,
                          generator=torch.Generator().manual_seed(42+epoch))
        if not len(loader): raise ValueError('Too few places')
        losses=[]; removed=0; total=0
        for images,masks,labels in loader:
            images=images.flatten(0,1); masks=masks.flatten(0,1).to(d); labels=labels.flatten().to(d)
            fs=[]
            with torch.no_grad():
                for i in range(0,len(images),a.microbatch): fs.append(extract_ru_feature_map(visual,images[i:i+a.microbatch].to(d)))
            features=torch.cat(fs)
            optimizer.zero_grad(set_to_none=True)
            descriptors=aggregate(visual.aggregator,features,masks)
            loss,_=loss_fn(descriptors,labels)
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(visual.aggregator.parameters(),1.,error_if_nonfinite=True)
            optimizer.step(); step+=1; losses.append(loss.item()); removed+=int(masks.sum()); total+=masks.numel()
            if step%25==0: print(f'{a.mode} epoch {epoch+1} step {step}: loss={loss.item():.5f}',flush=True)
            if a.smoke_test and len(losses)>=2: break
        if [p._version for p in frozen]!=frozen_versions or any(p.grad is not None for p in frozen):
            raise ValueError('Frozen backbone/RU gate changed')
        save_torch(out/'last.pt',{'aggregator':visual.aggregator.state_dict(),'optimizer':optimizer.state_dict(),
                   'contract':contract,'epoch':epoch+1,'step':step})
        completed_epochs=epoch+1
        write(out/f'epoch_{epoch+1:02d}.json',{'loss_mean':float(np.mean(losses)),'steps':len(losses),
              'drop_fraction':removed/total,'frozen_verified':True})
        if a.smoke_test: break
    write(out/'completed.json',{'complete':True,'smoke':a.smoke_test,'checkpoint_sha256':sha(out/'last.pt'),
          'epochs':completed_epochs,'steps':step})
    print('Saved training:',out)


def preflight(a):
    """Run real-weight shape/gradient checks before expensive cache generation."""
    import torch
    import unittest
    from src.models.clip_token_drop import DenseCLIPTeacher,aggregate
    from scripts.eval_dynamic_category_prior import extract_ru_feature_map
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_clip_token_drop.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped: raise RuntimeError('All eight tests must pass without skips')
    if a.output.exists(): raise ValueError('Preflight output exists; use new directory')
    seed(42); d=device(a.device); visual,_,_=load_ru(a,d)
    record=read(a.run/'run.json'); split=next(s for s in record['datasets'] if s['name']=='msls-val')
    queries=np.load(a.msls_root/Path(split['manifests']['queries']['path']).name,allow_pickle=False).tolist()
    images=torch.stack([image_tensor(a.msls_root,p)[0] for p in queries[:2]]).to(d)
    teacher=DenseCLIPTeacher(d); fractions=teacher.fractions(images)
    del teacher
    with torch.no_grad():
        features=extract_ru_feature_map(visual,images)
        expected=visual(images)
        original=aggregate(visual.aggregator,features,torch.zeros(2,20,20,dtype=torch.bool,device=d))
        if not torch.allclose(expected,original,rtol=2e-4,atol=2e-5): raise ValueError('Real RU all-keep mismatch')
    mask=np.stack([drop_mask(f,p,'aligned')[0] for f,p in zip(fractions,queries)])
    # Synthetic deletion guarantees that the compact path is exercised even
    # when the two real images have no high-confidence dynamic region.
    mask[0,5:8,5:8]=True; mask[1,4:9,4:9]=True
    for p in visual.aggregator.parameters(): p.requires_grad_(True)
    visual.aggregator.train()
    desc=aggregate(visual.aggregator,features,torch.from_numpy(mask).to(d))
    desc[:,0].sum().backward()
    if not torch.isfinite(desc).all() or not any(p.grad is not None and p.grad.abs().sum()>0 for p in visual.aggregator.parameters()):
        raise ValueError('Real-weight compact gradient check failed')
    if any(p.grad is not None for p in visual.backbone.parameters()): raise ValueError('Backbone received gradients')
    a.output.mkdir(parents=True)
    write(a.output/'completed.json',{'complete':True,'tests':result.testsRun,'real_clip_shape':list(fractions.shape),
          'all_keep_max_error':float((expected-original).abs().max()),'compact_gradients':True,'code':codes()})
    print('PASS preflight:',a.output)


def evaluate(a):
    import copy
    import torch
    from src.models.clip_token_drop import aggregate
    from scripts.eval_dynamic_category_prior import extract_ru_feature_map
    from scripts.clip_dynamic_screen import outcome
    if a.output.exists(): raise ValueError('Evaluation output exists')
    seed(42); d=device(a.device); cached=Cache(a.cache)
    if cached.contract['split']!='msls': raise ValueError('Need MSLS cache')
    visual,_,ru_sha=load_ru(a,d); modes=['frozen','none','aligned','shuffled']; aggs={'frozen':visual.aggregator}
    contracts={}; checkpoint_hashes={}
    for mode in modes[1:]:
        run=a.runs/mode; done=read(run/'completed.json'); state=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        c=state['contract']
        if not done['complete'] or done['smoke'] or sha(run/'last.pt')!=done['checkpoint_sha256']:
            raise ValueError('Incomplete/smoke/changed training run')
        if (c['code']!=codes() or c['init_sha256']!=ru_sha or c['mode']!=mode or c['settings']!=SETTINGS
                or c['teacher']!=cached.contract['teacher'] or state['epoch']!=c['epochs']
                or done['epochs']!=c['epochs'] or done['steps']!=state['step']):
            raise ValueError('Training config differs')
        contracts[mode]=c; checkpoint_hashes[mode]=done['checkpoint_sha256']
        agg=copy.deepcopy(visual.aggregator); agg.load_state_dict(state['aggregator']); aggs[mode]=agg.to(d).eval()
    shared=[{k:v for k,v in c.items() if k!='mode'} for c in contracts.values()]
    if not all(c==shared[0] for c in shared): raise ValueError('Training arms not matched')
    out=a.output; out.mkdir(parents=True); n=len(cached.records); ndb=cached.contract['info']['ndb']
    gt=[np.asarray(v,dtype=np.int64) for v in cached.contract['info']['gt']]
    if (n,ndb,len(gt))!=(19611,18871,740): raise ValueError('Wrong MSLS universe')
    arrays={m:np.lib.format.open_memmap(out/f'{m}_descriptors.npy',mode='w+',dtype='float32',shape=(n,12288)) for m in modes}
    drop_counts={m:0 for m in modes}
    with torch.inference_mode():
        for start in range(0,n,a.batch_size):
            end=min(start+a.batch_size,n)
            batch=[image_tensor(a.msls_root,p,cached.image_hashes[i])[0] for i,(p,_) in enumerate(cached.records[start:end],start)]
            features=extract_ru_feature_map(visual,torch.stack(batch).to(d))
            for mode in modes:
                masks=[drop_mask(cached.fractions[i],cached.records[i][0],'none' if mode=='frozen' else mode)[0] for i in range(start,end)]
                drop=torch.from_numpy(np.stack(masks)).to(d); drop_counts[mode]+=int(drop.sum())
                vectors=aggregate(aggs[mode],features,drop).cpu().numpy()
                if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=2e-4): raise ValueError('Bad descriptors')
                arrays[mode][start:end]=vectors
            if end%128==0 or end==n: print(f'Full DB+query: {end}/{n}',flush=True)
    for arr in arrays.values(): arr.flush()
    rows=[]
    for mode in modes:
        database=torch.from_numpy(np.array(arrays[mode][:ndb])).to(d)
        scorefile=np.lib.format.open_memmap(out/f'{mode}_scores.npy',mode='w+',dtype='float32',shape=(740,ndb))
        for start in range(0,740,16):
            end=min(start+16,740)
            scores=(torch.from_numpy(np.array(arrays[mode][ndb+start:ndb+end])).to(d)@database.T).cpu().numpy()
            scorefile[start:end]=scores
            for q,s in enumerate(scores,start): rows.append({'query_index':q,'query_path':cached.records[ndb+q][0],'variant':mode,**outcome(s,gt[q])})
        scorefile.flush(); del database
    by={(r['query_index'],r['variant']):r for r in rows}
    if sum(by[q,'frozen']['top1_correct'] for q in range(740))!=675: raise ValueError('Frozen RU baseline must be 675/740')
    summary={}
    for mode in modes:
        summary[mode]={'correct':sum(by[q,mode]['top1_correct'] for q in range(740)),
                      'recall':{f'R@{k}':sum(by[q,mode]['best_gt_rank']<=k for q in range(740))/740 for k in [1,5,10,20]},
                      'drop_fraction':drop_counts[mode]/(n*400)}
        for ref in ['frozen','none']:
            summary[mode]['vs_'+ref]={'corrected':[q for q in range(740) if by[q,mode]['top1_correct'] and not by[q,ref]['top1_correct']],
                                    'regressed':[q for q in range(740) if not by[q,mode]['top1_correct'] and by[q,ref]['top1_correct']]}
    summary['aligned_vs_shuffled']={
        'aligned_only_correct':[q for q in range(740) if by[q,'aligned']['top1_correct'] and not by[q,'shuffled']['top1_correct']],
        'shuffled_only_correct':[q for q in range(740) if by[q,'shuffled']['top1_correct'] and not by[q,'aligned']['top1_correct']]}
    write(out/'query_outcomes.json',rows); write(out/'summary.json',summary)
    with (out/'query_outcomes.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=[k for k in rows[0] if k!='top20'],extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    write(out/'provenance.json',{'training':contracts,'training_checkpoint_hashes':checkpoint_hashes,
          'cache_sha256':cached.hash,'code':codes(),'scope':'Both query and complete DB transformed; final epochs only; no MSLS selection'})
    write(out/'completed.json',{'complete':True,'queries':740,'variants':modes,
          'outputs_sha256':{f.name:sha(f) for f in out.glob('*.npy')}})
    (out/'report').mkdir()
    for name in ['summary.json','query_outcomes.json','query_outcomes.csv','provenance.json','completed.json']:
        shutil.copyfile(out/name,out/'report'/name)
    print(json.dumps(summary,indent=2)); print('Saved evaluation:',out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['preflight','cache','train','evaluate'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['train','msls'],default='train')
    p.add_argument('--cache',type=Path)
    p.add_argument('--mode',choices=['none','aligned','shuffled'],default='none')
    p.add_argument('--runs',type=Path,default=Path('doc/clip_token_drop_train_v1'))
    p.add_argument('--gsv-root',type=Path,default=Path('datasets/gsv_cities'))
    p.add_argument('--msls-root',type=Path,default=Path('datasets/msls-val'))
    p.add_argument('--run',type=Path,default=Path('doc/dynamic_category_prior_screen_b0.5_full_db_condition_queries'))
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--places-per-batch',type=int,default=16)
    p.add_argument('--microbatch',type=int,default=8)
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke-test',action='store_true')
    a=p.parse_args()
    if min(a.batch_size,a.places_per_batch,a.microbatch,a.epochs)<1 or a.places_per_batch<2 or a.workers<0:
        p.error('Invalid batch/epoch/worker settings')
    if a.stage in ('train','evaluate') and a.cache is None: p.error('--cache is required')
    if a.resume and a.smoke_test: p.error('Smoke does not support resume')
    {'preflight':preflight,'cache':cache,'train':train,'evaluate':evaluate}[a.stage](a)


if __name__=='__main__': main()
