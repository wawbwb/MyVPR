"""Frozen-weight geometry mechanism audit; no optimizer or validation tuning."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from scripts import spatial_resample as source
from src.spatial_sampling_audit import VARIANTS,field,apply,intervention_fields
from src.models.spatial_resample import image_seeds
from src.semantic_region_protocol import read,write,sha,state_hash,owned_run,save_npz

OUT=ROOT/'logs/spatial_sampling_audit_v1'
FILES=('src/spatial_sampling_audit.py','scripts/audit_spatial_sampling.py',
    'scripts/launch_spatial_sampling_audit.py','tests/test_spatial_sampling_audit.py','doc/SPATIAL_SAMPLING_AUDIT_V1.md')


def identity(stage):
    c=source.configuration();run=ROOT/'logs/spatial_resample_v1'
    for s in ('train','eval'):
        old=source.verified(run/s)
        if old.get('new_sources')!=source.identity(c)['new_sources']:raise ValueError('Source implementation changed')
    return dict(stage=stage,source_identity=source.identity(c),
        train_sha256=sha(run/'train/completed.json'),eval_sha256=sha(run/'eval/completed.json'),
        checkpoint_sha256=sha(run/'train/last.pt'),code=source.old.key.source_hashes(FILES))


def finish(out):
    write(out/'completed.json',dict(complete=True,files={p.name:sha(p) for p in out.iterdir()
        if p.is_file() and p.name not in ('.lock','completed.json')}))


def verify(out):
    d=read(out/'completed.json')
    if not d['complete']:raise ValueError('Incomplete audit')
    for n,h in d['files'].items():
        if Path(n).name!=n or sha(out/n)!=h:raise ValueError('Audit files changed')
    return read(out/'contract.json')


def load():
    c=source.configuration();models=source.build(c)
    p=torch.load(ROOT/'logs/spatial_resample_v1/train/last.pt',map_location='cpu',weights_only=True)
    expected=read(ROOT/'logs/spatial_resample_v1/train/summary.json')['final_adapter_sha256']
    for a,m in models.items():
        m.adapter.load_state_dict(p['adapters'][a],strict=True)
        if state_hash(m.adapter)!=expected[a]:raise ValueError('Source weights differ')
        m.requires_grad_(False).eval()
    return c,models


def calibrate(args):
    from scripts.semantic_region_study import ImageRows
    c=source.configuration();plan=source.old.key.open_plan(c)
    places=sorted(plan['places'][:c['train_places']],key=lambda p:hashlib.sha256(
        ('sampling-audit-v1:'+plan['images'][p['rows'][0]]['place']).encode()).hexdigest())[:256]
    rows=[r for p in places for r in p['rows']]
    if len(rows)!=1024 or any(plan['images'][r]['split']!='train' for r in rows):raise ValueError('Wrong calibration population')
    out=OUT/'calibrate';contract=dict(identity('calibrate'),training_rows=rows)
    with owned_run(out,contract,args.resume):
        if (out/'completed.json').exists():verify(out);return
        _,models=load();anchor=models['global_shift'].anchor;before=state_hash(anchor)
        ds=ImageRows(source.old.key.path(c,'dataset_root'),plan['images'],322)
        loader=DataLoader(Subset(ds,rows),batch_size=8,num_workers=2,shuffle=False)
        means={a:[] for a in ('global','local')};spatial={a:[] for a in means}
        with torch.inference_mode():
            for batch,(images,ids) in enumerate(loader):
                if ids.tolist()!=rows[batch*8:batch*8+len(images)]:raise ValueError('Calibration order differs')
                for x in images.split(2):
                    f=models['global_shift'].encode(x.to('cuda:0'))
                    for name,arm in [('global','global_shift'),('local','local_shift')]:
                        off=field(models[arm].adapter,f[0]);means[name].append(off.mean((2,3)).cpu().numpy())
                        spatial[name].append(off.flatten(2).std(2,unbiased=False).cpu().numpy())
                        y,_=apply(f[0],off);direct=models[arm].adapter(f[0])[0]
                        if float((y-direct).abs().max())>1e-6:raise ValueError('Intervention wrapper changed source')
                write(out/'progress.json',dict(images=min((batch+1)*8,len(rows)),total=len(rows)))
        arrays={};stats={}
        for name in means:
            v=np.concatenate(means[name]);s=np.concatenate(spatial[name]);constant=v.astype(np.float64).mean(0).astype(np.float32)
            arrays[name+'_mean']=v;arrays[name+'_spatial_std']=s;arrays[name+'_constant']=constant
            stats[name]=dict(constant_xy=constant.tolist(),between_image_std_xy=v.std(0).tolist(),
                spatial_std_xy=s.mean(0).tolist(),positive_fraction_xy=(v>0).mean(0).tolist(),
                mean_displacement_norm=float(np.linalg.norm(v,axis=1).mean()))
        if state_hash(anchor)!=before:raise ValueError('Frozen anchor changed')
        save_npz(out/'calibration.npz',rows=np.array(rows),**arrays)
        write(out/'summary.json',dict(training_places=256,training_images=1024,statistics=stats,training_started=False))
        finish(out);print(read(out/'summary.json'),flush=True)


def evaluate(args):
    from scripts import eval_clip_key_study as metrics
    from scripts.eval_spatial_adapter_repeat import RGBImages
    verify(OUT/'calibrate');c=source.configuration();sets=metrics.populations(c,source.old.key.open_plan(c))
    contract=dict(identity('eval'),calibration_sha256=sha(OUT/'calibrate/completed.json'),
        populations={d['tag']:dict(references=d['references'],queries=d['queries'],order=metrics.order_digest(d['names']),gt=metrics.gt_digest(d['ground_truth'])) for d in sets})
    out=OUT/'eval'
    with owned_run(out,contract,args.resume):
        if (out/'completed.json').exists():verify(out);return
        _,models=load();first=models['global_shift'];before=state_hash(first.anchor)
        with np.load(OUT/'calibrate/calibration.npz') as z:
            constants={a:torch.tensor(z[a+'_constant'],device='cuda:0') for a in ('global','local')}
        results={}
        for d in sets:
            tag=d['tag'];target=out/(tag+'.json');ready=out/(tag+'.ready.json')
            if ready.exists():
                if sha(target)!=read(ready)['sha256']:raise ValueError('Dataset report changed')
                results[tag]=read(target)['comparison'];continue
            start=time.monotonic();arrays={a:np.empty((len(d['names']),12288),np.float32) for a in VARIANTS};contents=[]
            loader=DataLoader(RGBImages(d['root'],d['names'],d['hashes']),batch_size=16,num_workers=2,shuffle=False)
            stats={a:dict(images=0,relative_sum=0.,displacement_sum=0.,max_axis_offset=0.) for a in VARIANTS[1:]}
            with torch.inference_mode():
                for batch,(images,ids,hashes) in enumerate(loader):
                    if ids.tolist()!=list(range(batch*16,batch*16+len(ids))):raise ValueError('Eval order changed')
                    contents.extend(hashes)
                    for i in range(0,len(images),2):
                        f=first.encode(images[i:i+2].to('cuda:0'));seeds=image_seeds(hashes[i:i+2]);values=dict(innovation1=f[1])
                        g=field(models['global_shift'].adapter,f[0]);local=field(models['local_shift'].adapter,f[0])
                        for a,offset in intervention_fields(g,local,constants,seeds).items():
                            y,relative=apply(f[0],offset);values[a]=first.anchor.readout(y);v=stats[a];n=len(y)
                            v['images']+=n;v['relative_sum']+=float(relative.sum())
                            v['displacement_sum']+=n*float(offset.square().sum(1).sqrt().mean())
                            v['max_axis_offset']=max(v['max_axis_offset'],float(offset.abs().max()))
                        for a,z in values.items():
                            if not torch.isfinite(z).all() or float((z.norm(dim=1)-1).abs().max())>1e-4:raise ValueError('Invalid descriptor')
                            arrays[a][ids[i:i+2].numpy()]=z.cpu().numpy()
                    write(out/'progress.json',dict(dataset=tag,images=int(ids[-1])+1,total=len(d['names']),elapsed_seconds=time.monotonic()-start))
                reports={}
                for a in VARIANTS:
                    db=torch.tensor(arrays[a][:d['references']],device='cuda:0');queries=arrays[a][d['references']:]
                    ranks=np.concatenate([(torch.tensor(q,device='cuda:0')@db.T).topk(20,dim=1).indices.cpu().numpy()
                        for q in np.array_split(queries,max(1,(len(queries)+15)//16))]);del db
                    reports[a]=metrics.retrieval_report(ranks,d['ground_truth'],d['references'])
            old=read(ROOT/'logs/spatial_resample_v1/eval'/(tag+'.json'))
            for a,b in [('innovation1','innovation1'),('trained_global','global_shift'),('trained_local','local_shift')]:
                if reports[a]!=old['variants'][b]:raise ValueError('Full source Top20 not reproduced: '+a)
            comparison=dict(correct={a:r['correct'] for a,r in reports.items()},
                vs_innovation1={a:metrics.paired_report(reports['innovation1'],r) for a,r in reports.items() if a!='innovation1'},
                global_vs_constant=metrics.paired_report(reports['constant_global'],reports['trained_global']),
                local_vs_constant=metrics.paired_report(reports['constant_local'],reports['trained_local']),
                local_vs_mean=metrics.paired_report(reports['local_mean'],reports['trained_local']),
                centered_vs_shuffled=metrics.paired_report(reports['local_centered_shuffled'],reports['local_centered']))
            write(target,dict(population=contract['populations'][tag],variants=reports,comparison=comparison,sampling=stats,
                image_digest=metrics.order_digest([n+':'+h for n,h in zip(d['names'],contents)]),seconds=time.monotonic()-start))
            write(ready,dict(sha256=sha(target)));results[tag]=comparison;print(tag,comparison['correct'],flush=True);del arrays
        checks=dict(anchor_unchanged=state_hash(first.anchor)==before,
            checkpoint_unchanged=sha(ROOT/'logs/spatial_resample_v1/train/last.pt')==contract['checkpoint_sha256'])
        if not all(checks.values()):raise ValueError('Audit modified source')
        write(out/'summary.json',dict(checks=checks,results=results,training_started=False,
            diagnostic_only=True,stable_gain_proven=False,verdict='READ_ONLY_MECHANISM_AUDIT_COMPLETE'))
        finish(out)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['calibrate','eval']);p.add_argument('--resume',action='store_true');p.add_argument('--allow-shared-gpu',action='store_true');args=p.parse_args()
    import fcntl,subprocess
    from scripts.single_dino_supervision import check_gpu_capacity
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':p.error('Physical GPU1 only')
    with (ROOT/'logs/query_relation_screen_v1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        s=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
        check_gpu_capacity(*map(int,s.strip().split(',')),args.allow_shared_gpu,.2)
        source.old.key.gpu(source.configuration());torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        {'calibrate':calibrate,'eval':evaluate}[args.stage](args)


if __name__=='__main__':main()
