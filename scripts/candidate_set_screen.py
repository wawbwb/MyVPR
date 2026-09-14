"""First-screen candidate-conditioned reranking on frozen official Pair-VPR evidence."""
import argparse
from collections import OrderedDict,defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.candidate_set_utils import make_plan,summary,stable_key
MODES=['independent','set','density','consistency','competition']
EXPECTED_WEIGHT='18e7b95ba57d4d578fbf0a06a1846fc2dfb976c74da56bfd7d6a68afbcea8423'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def codes():
    return {n:hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in
        ['scripts/candidate_set_screen.py','src/candidate_set_utils.py','src/models/candidate_set.py']}


def complete(out):
    write(out/'completed.json',{'complete':True,'files':{p.relative_to(out).as_posix():sha(p)
        for p in out.rglob('*') if p.is_file() and p.name!='completed.json' and p.suffix!='.tmp'}})


def verify(out):
    done=read(out/'completed.json')
    if not done['complete']:raise ValueError('Incomplete '+str(out))
    for name,h in done['files'].items():
        if sha(out/name)!=h:raise ValueError('Changed file '+str(out/name))
    c=read(out/'contract.json')
    if c['code']!=codes():raise ValueError('Code changed; use matching code/cache')
    return c


def npz(path,**values):
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:np.savez(f,**values)
    temp.replace(path)
    write(path.with_suffix('.sha.json'),{'sha256':sha(path)})


def load_npz(path):
    if sha(path)!=read(path.with_suffix('.sha.json'))['sha256']:raise ValueError('Corrupt shard '+str(path))
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def seed():
    import torch
    torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def cuda():
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1' or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('Run with CUDA_VISIBLE_DEVICES=1 (physical GPU1, logical cuda:0)')


def official(a):
    import torch
    cuda();seed();repo=a.official_repo.resolve();prov=read(a.audit/'provenance.json')
    weight=repo/'trained_models/pairvpr-vitB.pth'
    if sha(weight)!=EXPECTED_WEIGHT:raise ValueError('Wrong official checkpoint')
    config=repo/'pairvpr/configs/pairvpr_speed_local.yaml'
    if sha(config)!=prov['config_sha256']:raise ValueError('Official config changed')
    for name,h in prov['official_python_sha256'].items():
        if sha(repo/name)!=h:raise ValueError('Official source changed: '+name)
    dino=Path(torch.hub.get_dir())/'facebookresearch_dinov2_main'
    if not (dino/'hubconf.py').exists():raise ValueError('Cached DINO source absent')
    sys.path.insert(0,str(repo))
    spec=importlib.util.spec_from_file_location('candidate_official_eval',repo/'pairvpr/eval/eval.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg=module.get_cfg_from_args_eval(SimpleNamespace(config_file_eval=str(config),trained_ckpt=str(weight),
        dsetroot=str(a.gsv_root.parent),val_datasets=['MSLS_val']))
    if cfg.augmentation.img_res!=322:raise ValueError('Expected 322 resolution')
    old=torch.hub.load
    def local_hub(repository,name,*args,**kw):
        if str(repository).split(':')[0]!='facebookresearch/dinov2':raise ValueError('Unexpected hub source')
        kw.pop('source',None);kw['pretrained']=False
        return old(str(dino),name,*args,source='local',**kw)
    torch.hub.load=local_hub
    try:model=module.PairVPRNet(cfg)
    finally:torch.hub.load=old
    state=torch.load(weight,map_location='cpu',weights_only=True)
    module.interpolate_pos_embed(cfg,model,state);model.load_state_dict(state,strict=True)
    model=model.cuda().eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,{'checkpoint_sha256':sha(weight),'config_sha256':sha(config),
        'official_source':prov['official_python_sha256'],'dino_source':{str(p.relative_to(dino)):sha(p) for p in dino.rglob('*.py')}}


def image(root,record,expected=None):
    import torch
    from PIL import Image
    from torchvision import transforms as T
    path=(root/record['path']).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Image outside dataset root')
    h=sha(path)
    if expected is not None and h!=expected:raise ValueError('Image changed: '+str(path))
    transform=T.Compose([T.Resize((322,322),interpolation=T.InterpolationMode.BILINEAR),T.ToTensor(),
        T.Normalize([.485,.456,.406],[.229,.224,.225])])
    with Image.open(path) as im:x=transform(im.convert('RGB'))
    if not torch.isfinite(x).all():raise ValueError('Invalid input')
    return x,h


def prepare(a):
    import pandas as pd
    groups=defaultdict(list);manifests={}
    for path in sorted((a.gsv_root/'Dataframes').glob('*.csv')):
        manifests[path.name]=sha(path)
        df=pd.read_csv(path)
        for _,r in df.iterrows():
            city=str(r['city_id']);label=city+':'+str(int(r['place_id']))
            name=f"{city}_{str(int(r['place_id'])%100000).zfill(7)}_{str(r['year']).zfill(4)}_{str(r['month']).zfill(2)}_{str(r['northdeg']).zfill(3)}_{r['lat']}_{r['lon']}_{r['panoid']}.jpg"
            groups[label].append({'path':f'Images/{city}/{name}','city':city,'panoid':str(r['panoid']),
                'date':[int(r['year']),int(r['month'])]})
    plan=make_plan(groups,a.train_queries,a.eval_queries,a.train_places)
    for split in ['train','dev','test']:
        part=plan[split]
        if len({r['path'] for r in part['database']+part['queries']})!=len(part['database'])+len(part['queries']):raise ValueError('Duplicate images')
        for r in part['database']+part['queries']:
            if not (a.gsv_root/r['path']).is_file():raise FileNotFoundError(a.gsv_root/r['path'])
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'contract.json',{'code':codes(),'manifests':manifests,'topk':20,'plan':plan})
    complete(a.output)
    print(plan['split_info']);print({s:(len(plan[s]['queries']),len(plan[s]['database'])) for s in ['train','dev','test']})


def preflight(a):
    import torch,unittest
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_candidate_set*.py'))
    if not result.wasSuccessful() or result.skipped:raise ValueError('Server tests must all pass, no skips')
    c=verify(a.plan);model,identity=official(a)
    r=c['plan']['train'];x=torch.stack([image(a.gsv_root,v)[0] for v in [r['queries'][0],r['database'][0]]]).cuda()
    with torch.inference_mode():
        dense,g=model(x,None,'global');saved=[]
        handle=model.classvprmodule.register_forward_pre_hook(lambda m,i:saved.append(i[0].detach().clone()))
        try:s=model(dense[:1],dense[1:],'pairvpr')
        finally:handle.remove()
        if len(saved)!=1 or saved[0].shape!=(1,768):raise ValueError('Wrong pair CLS interface')
        if not torch.allclose(s,model.classvprmodule(saved[0]),atol=1e-6,rtol=1e-6):raise ValueError('Cached pair embedding changed score')
        if dense.shape[1:]!=(529,768) or g.shape[1]!=512:raise ValueError('Wrong encoder')
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'contract.json',{'code':codes(),'official':identity,'tests':result.testsRun});complete(a.output)
    print('PASS official strict load, CLS reconstruction, model invariants and gradients')


def cache(a):
    import torch
    plan=verify(a.plan);model,identity=official(a);part=plan['plan'][a.split]
    records=part['database']+part['queries'];ndb=len(part['database']);nq=len(part['queries']);k=plan['topk']
    contract={'code':codes(),'official':identity,'plan_sha256':sha(a.plan/'completed.json'),'split':a.split,
        'ndb':ndb,'queries':nq,'topk':k,'scope':plan['plan']['split_info'],'transform':'PIL RGB, bilinear322, ToTensor, ImageNet'}
    out=a.output
    if out.exists():
        if read(out/'contract.json')!=contract:raise ValueError('Cache contract differs')
        if (out/'completed.json').exists():verify(out);print('Verified completed cache',out);return
    else:
        out.mkdir(parents=True);write(out/'contract.json',contract);(out/'global').mkdir();(out/'pairs').mkdir()
    globals_all=[];hashes=[]
    with torch.inference_mode():
        for start in range(0,len(records),128):
            stop=min(start+128,len(records));file=out/'global'/f'{start:07d}.npz'
            if file.exists() and file.with_suffix('.sha.json').exists():
                data=load_npz(file)
                for r,h in zip(records[start:stop],data['hashes']):
                    if sha(a.gsv_root/r['path'])!=str(h):raise ValueError('Image changed')
            else:
                vectors=[];hs=[]
                for j in range(start,stop,4):
                    inputs=[image(a.gsv_root,r) for r in records[j:min(j+4,stop)]]
                    _,z=model(torch.stack([v[0] for v in inputs]).cuda(),None,'global')
                    vectors.append(z.cpu().numpy());hs.extend([v[1] for v in inputs])
                data={'vectors':np.concatenate(vectors),'hashes':np.array(hs)};npz(file,**data)
            if (data['vectors'].shape!=(stop-start,512) or len(data['hashes'])!=stop-start
                or not np.isfinite(data['vectors']).all() or not np.allclose(np.linalg.norm(data['vectors'],axis=1),1,atol=2e-4)):
                raise ValueError('Invalid global shard')
            globals_all.append(data['vectors']);hashes.extend(data['hashes'].tolist())
            print(f'{a.split} global {stop}/{len(records)}',flush=True)
        global_z=np.concatenate(globals_all);memo=OrderedDict()
        def dense(i):
            if i in memo:memo.move_to_end(i);return memo[i].cuda()
            x,_=image(a.gsv_root,records[i],hashes[i]);f,_=model(x[None].cuda(),None,'global')
            memo[i]=f.cpu()
            if len(memo)>64:memo.popitem(last=False)
            return f
        captured=[]
        handle=model.classvprmodule.register_forward_pre_hook(lambda m,i:captured.append(i[0].detach().clone()))
        db=torch.from_numpy(global_z[:ndb]).cuda()
        try:
            for qi in range(nq):
                path=out/'pairs'/f'{qi:06d}.npz'
                if path.exists() and path.with_suffix('.sha.json').exists():load_npz(path);continue
                scores=(torch.from_numpy(global_z[ndb+qi]).cuda()@db.T).cpu().numpy()
                ids=np.argsort(-scores,kind='stable')[:k];qf=dense(ndb+qi);feats=[];base=[]
                for di in ids:
                    df=dense(int(di));captured.clear()
                    forward=model(qf,df,'pairvpr');reverse=model(df,qf,'pairvpr')
                    if len(captured)!=2:raise ValueError('Expected both directional CLS embeddings')
                    feats.append(torch.cat(captured,dim=-1)[0].cpu().numpy());base.append(float((forward+reverse).item()))
                values=dict(evidence=np.stack(feats),base=np.array(base,np.float32),db_vectors=global_z[ids],
                    candidates=ids,global_scores=scores[ids],labels=np.array([records[di]['label']==records[ndb+qi]['label'] for di in ids]))
                if not np.isfinite(values['evidence']).all() or not np.isfinite(values['base']).all():raise ValueError('Nonfinite pair representation')
                npz(path,**values)
                if (qi+1)%16==0 or qi+1==nq:print(f'{a.split} bidirectional pairs {qi+1}/{nq} x {k}',flush=True)
        finally:handle.remove()
    data=[load_npz(out/'pairs'/f'{i:06d}.npz') for i in range(nq)]
    scores=np.stack([r['base'] for r in data]);labels=np.stack([r['labels'] for r in data])
    write(out/'baseline.json',summary(scores,labels,scores))
    write(out/'global_baseline.json',summary(np.stack([r['global_scores'] for r in data]),labels,scores))
    complete(out);print('Saved',out)


def admission(a):
    c=verify(a.cache_root/'dev');base=read(a.cache_root/'dev'/'baseline.json')
    if c['split']!='dev':raise ValueError('Expected development cache')
    errors=base['reachable']-base['correct'];passed=errors>=10
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'contract.json',{'code':codes(),'cache_sha256':sha(a.cache_root/'dev'/'completed.json'),'scope':c['scope']})
    write(a.output/'decision.json',{'passed':passed,'pair':base,'global':read(a.cache_root/'dev'/'global_baseline.json'),
        'reachable_errors':errors,'minimum_reachable_errors':10,
        'note':'Feasibility threshold fixed before results, not statistical significance. Stop if saturated; do not change threshold to pass.'})
    complete(a.output)
    print('ADMISSION',passed,'reachable errors',errors,flush=True)
    if not passed:sys.exit(3)


def tensors(path):
    import torch
    c=verify(path);data=[load_npz(path/'pairs'/f'{i:06d}.npz') for i in range(c['queries'])]
    return {k:torch.from_numpy(np.stack([r[k] for r in data])).cuda() for k in ['evidence','base','db_vectors','labels']},c


def predict(model,data):
    import torch
    model.eval()
    with torch.no_grad():return torch.cat([model(data['evidence'][s:s+16],data['base'][s:s+16],data['db_vectors'][s:s+16]) for s in range(0,len(data['base']),16)])


def train(a):
    import torch
    from src.models.candidate_set import CandidateSet,list_loss,duplicate_consistency
    cuda();seed();tr,tc=tensors(a.cache_root/'train');dev,dc=tensors(a.cache_root/'dev')
    if tc['split']!='train' or dc['split']!='dev' or tc['plan_sha256']!=dc['plan_sha256'] or tc['official']!=dc['official']:raise ValueError('Unmatched caches')
    base=summary(dev['base'].cpu().numpy(),dev['labels'].cpu().numpy(),dev['base'].cpu().numpy())
    reachable_errors=base['reachable']-base['correct']
    contract={'code':codes(),'train_sha256':sha(a.cache_root/'train'/'completed.json'),
        'dev_sha256':sha(a.cache_root/'dev'/'completed.json'),'plan_sha256':tc['plan_sha256'],'official':tc['official'],
        'epochs':5,'seed':42,'lr':.0001,'modes':MODES,'scope':dc['scope']}
    if a.output.exists():
        if not a.resume or read(a.output/'contract.json')!=contract:raise ValueError('Existing run requires matching --resume')
        if (a.output/'completed.json').exists():
            verify(a.output)
            if not read(a.output/'gate.json')['passed']:sys.exit(3)
            print('Run already complete');return
    else:
        a.output.mkdir(parents=True);write(a.output/'contract.json',contract)
    if reachable_errors<10 or int((tr['labels'].any(1)&(~tr['labels']).any(1)).sum())<128:
        write(a.output/'gate.json',{'passed':False,'dev_baseline':base,'dev_reachable_errors':reachable_errors,
            'reason':'Insufficient non-saturated retrieval supervision. No training; do not tune this threshold to pass.'})
        complete(a.output);print('SCREEN STOP: download gate.json');sys.exit(3)
    write(a.output/'gate.json',{'passed':True,'dev_baseline':base,'dev_reachable_errors':reachable_errors})
    valid=torch.where(tr['labels'].any(1)&(~tr['labels']).any(1))[0].cpu().numpy()
    logs=read(a.output/'training_logs.json') if (a.output/'training_logs.json').exists() else {}
    selection=read(a.output/'selection.json') if (a.output/'selection.json').exists() else {}
    for mode in MODES:
        if len(logs.get(mode,[]))==5 and mode in selection and (a.output/f'{mode}.pt').exists():continue
        seed();model=CandidateSet(mode).cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.001)
        best=-1;logs[mode]=[]
        for epoch in range(5):
            model.train();order=np.random.default_rng(42+epoch).permutation(valid);epoch_logs=[]
            for start in range(0,len(order),8):
                ids=order[start:start+8];x=tr['evidence'][ids];s=tr['base'][ids];z=tr['db_vectors'][ids];y=tr['labels'][ids]
                original=model(x,s,z);loss,_=list_loss(original,y)
                # Same duplicate augmentation in all arms; no candidate labels enter the network.
                rng=np.random.default_rng(100000*epoch+start+42);extra=rng.integers(0,s.shape[1],size=5)
                ix=torch.tensor(list(range(s.shape[1]))+extra.tolist(),device='cuda')
                augmented=model(x[:,ix],s[:,ix],z[:,ix]);aug_loss,_=list_loss(augmented[:,:s.shape[1]],y)
                consistency=duplicate_consistency(original,augmented)
                total=.5*(loss+aug_loss)+(consistency if mode in ['consistency','competition'] else consistency*0)
                opt.zero_grad(set_to_none=True);total.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
                if not torch.isfinite(total):raise ValueError('Nonfinite training loss')
                epoch_logs.append({'loss':float(loss),'consistency':float(consistency),'grad_norm':float(norm)})
            scores=predict(model,dev).cpu().numpy();result=summary(scores,dev['labels'].cpu().numpy(),dev['base'].cpu().numpy())
            logs[mode].append({'epoch':epoch+1,'batches':epoch_logs,'dev':result})
            if result['correct']>best:
                best=result['correct'];temp=a.output/f'{mode}.tmp';torch.save(model.state_dict(),temp);temp.replace(a.output/f'{mode}.pt')
                selection[mode]={'epoch':epoch+1,'dev':result}
            print(f'{mode} epoch{epoch+1}: dev {result["correct"]}/{len(scores)}',flush=True)
        write(a.output/'training_logs.json',logs);write(a.output/'selection.json',selection)
    complete(a.output)


def evaluate(a):
    import torch
    from src.models.candidate_set import CandidateSet
    cuda();seed();runs=verify(a.runs)
    if not read(a.runs/'gate.json')['passed']:raise ValueError('Training gate failed; inspect report first')
    data,c=tensors(a.cache_root/'test')
    if c['split']!='test' or c['plan_sha256']!=runs['plan_sha256'] or c['official']!=runs['official']:raise ValueError('Wrong test cache')
    a.output.mkdir(parents=True,exist_ok=False)
    base=data['base'].cpu().numpy();labels=data['labels'].cpu().numpy();results={'frozen':summary(base,labels,base)};stress={}
    outputs={'labels':labels,'frozen':base}
    for mode in MODES:
        model=CandidateSet(mode).cuda();model.load_state_dict(torch.load(a.runs/f'{mode}.pt',map_location='cuda',weights_only=True));model.eval()
        scores=predict(model,data);outputs[mode]=scores.cpu().numpy();results[mode]=summary(outputs[mode],labels,base)
        perm=torch.arange(base.shape[1]-1,-1,-1,device='cuda');differences=[];dup=[]
        with torch.no_grad():
            for start in range(0,len(base),16):
                x=data['evidence'][start:start+16];s=data['base'][start:start+16];z=data['db_vectors'][start:start+16]
                original=scores[start:start+16]
                shuffled=model(x[:,perm],s[:,perm],z[:,perm])[:,perm]
                differences.append(float((original-shuffled).abs().max()))
                ix=torch.tensor(list(range(s.shape[1]))+[0]*5,device='cuda')
                repeated=model(x[:,ix],s[:,ix],z[:,ix])[:,:s.shape[1]]
                dup.append(repeated.cpu().numpy())
        if max(differences)>2e-4:raise ValueError('Candidate-order invariance failed')
        dup=np.concatenate(dup);stress[mode]={'permutation_max_error':max(differences),
            'duplicating_global_top1_five_times':summary(dup,labels,outputs[mode]),
            'unique_top1_changed':int((dup.argmax(1)!=outputs[mode].argmax(1)).sum()),
            'score_max_change':float(np.abs(dup-outputs[mode]).max())}
    npz(a.output/'scores.npz',**outputs)
    write(a.output/'summary.json',results);write(a.output/'stress.json',stress)
    write(a.output/'training_selection.json',read(a.runs/'selection.json'))
    write(a.output/'training_logs.json',read(a.runs/'training_logs.json'))
    write(a.output/'contract.json',{'code':codes(),'test_cache_sha256':sha(a.cache_root/'test'/'completed.json'),
        'training_sha256':sha(a.runs/'completed.json'),'scope':c['scope'],
        'note':'Frozen pair-CLS feasibility only, K20; not full Pair-VPR fine-tuning or MSLS R@1. Density uses global visual similarity, not known places.'})
    complete(a.output);print({m:r['correct'] for m,r in results.items()});print('Download:',a.output)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','preflight','cache','admission','train','evaluate'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=['train','dev','test'],default='train')
    p.add_argument('--official-repo',type=Path,default=Path('/home/wt/workspace/Pair-VPR-official'))
    p.add_argument('--audit',type=Path,default=Path('doc/pairvpr_official_paired_audit_v1'))
    p.add_argument('--gsv-root',type=Path,default=Path('datasets/gsv_cities'))
    p.add_argument('--plan',type=Path,default=Path('doc/candidate_set_plan_v1'))
    p.add_argument('--cache-root',type=Path,default=Path('.cache/candidate_set_v1'))
    p.add_argument('--runs',type=Path,default=Path('doc/candidate_set_train_v1'))
    p.add_argument('--train-queries',type=int,default=1024);p.add_argument('--eval-queries',type=int,default=512)
    p.add_argument('--train-places',type=int,default=4096)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();globals()[a.stage](a)


if __name__=='__main__':main()
