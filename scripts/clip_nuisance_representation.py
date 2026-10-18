"""Locate CLIP proxy information: random projection vs global vs pre-BoQ.

No VPR model training or benchmark evaluation. Frozen feature extraction and
small CPU probes on the existing GSV place-disjoint split only. Separate files
preserve every original failed-probe artifact, threshold, and innovation 1.
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

from scripts import clip_nuisance_study as old
from src.models.clip_nuisance import Regressor,center_four,projection
from src.models.official_boq import load_official
from src.semantic_region_protocol import owned_run,read,write,sha,save_npz,seed_all,state_hash

SCHEMA='clip_nuisance_representation_diagnostic_v1'
SOURCES=('scripts/clip_nuisance_representation.py','scripts/launch_clip_nuisance_representation.py',
    'config/clip_nuisance_representation.yaml','tests/test_clip_nuisance_representation.py')
DIMENSIONS={'global_full':12288,'patch_mean':768,'patch_grid2':3072}


def configuration(p):
    import yaml
    c=yaml.safe_load(p.read_text(encoding='utf-8'))
    if c['schema']!=SCHEMA or c['device']!='cuda:0' or not 0<c['cuda_memory_fraction']<=.2:
        raise ValueError('Expected isolated GPU1 diagnostic protocol')
    for k in ('seed','batch_size','flush_rows','workers','hidden','probe_epochs','probe_batch','bootstrap_samples'):
        if type(c[k])is not int or c[k]<1:raise ValueError('Positive integer required: '+k)
    for k in ('probe_lr','projection_tolerance','signal_min_r2','signal_min_aligned_shuffled_gap'):
        if not np.isfinite(c[k]) or c[k]<=0:raise ValueError('Positive finite setting required: '+k)
    if c['flush_rows']%c['batch_size']:raise ValueError('Flush rows must divide batches')
    return c


def path(c,k):
    p=Path(c[k]).expanduser()
    return p if p.is_absolute() else ROOT/p


def source(c):return old.configuration(path(c,'source_config'))


def identity(c):
    s=source(c)
    return dict(schema=SCHEMA,config=c,code=old.source_hashes(SOURCES),source=old.identity(s),
        prepared_completed_sha256=sha(old.path(s,'prepared')/'completed.json'),
        original_probe_completed_sha256=sha(old.path(s,'runs')/'preflight_probe/completed.json'),
        scope='GSV representation diagnosis only, no VPR model training, no MSLS/Pitts or blind test access')


def finish(output):
    files={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*'))
           if p.is_file() and p.name not in ('.lock','completed.json') and not p.name.endswith('.tmp')}
    write(output/'completed.json',dict(complete=True,schema=SCHEMA,files=files))


def verified(output):
    result=read(output/'completed.json')
    if result.get('schema')!=SCHEMA or result.get('complete')is not True:raise ValueError('Incomplete diagnostic artifact')
    for n,d in result['files'].items():
        p=(output/n).resolve()
        if not p.is_relative_to(output.resolve()) or not p.is_file() or sha(p)!=d:raise ValueError('Diagnostic checksum differs')
    return read(output/'contract.json')


def summarize_patches(features):
    if features.ndim!=4 or features.shape[1]!=768:raise ValueError('Need raw official B768HW pre-BoQ patch map')
    return dict(patch_mean=features.mean((2,3)),patch_grid2=F.adaptive_avg_pool2d(features,(2,2)).flatten(1))


def lifted_projection(values,matrix):
    """Dense minimum-norm lift, retaining only the ORIGINAL 256-D information.

    Gives the same 12288-input probe architecture as full descriptors. Not a new
    feature extractor, and does not recover information outside the row space.
    """
    r=np.asarray(matrix,np.float64)
    if values.ndim!=2 or values.shape[1]!=r.shape[1] or r.shape[0]<r.shape[1]:raise ValueError('Lift shapes differ')
    inverse=np.linalg.solve(r.T@r,r.T)
    return (np.asarray(values,np.float64)@inverse).astype(np.float32)


def extract(c,args):
    s=source(c);old.prepared(s);old.verified(old.path(s,'runs')/'preflight_probe')
    plan=old.open_plan(s);rows=plan['images'];count=len(rows)
    output=path(c,'cache');contract=dict(stage='frozen_feature_extraction',**identity(c))
    with owned_run(output,contract,args.resume):
        if (output/'completed.json').exists():verified(output);print('Feature extraction already complete',flush=True);return
        old.gpu(c)
        arrays={};cursor=read(output/'cursor.json') if (output/'cursor.json').exists() else dict(next=0,chunks=[],max_projection_error=0.)
        for name,dim in DIMENSIONS.items():
            p=output/(name+'.npy')
            if p.exists():
                a=np.load(p,mmap_mode='r+')
                if a.shape!=(count,dim) or a.dtype!=np.float32:raise ValueError('Cached feature shape differs')
            else:
                if cursor['next']:raise ValueError('Missing partial feature array')
                a=np.lib.format.open_memmap(p,mode='w+',dtype=np.float32,shape=(count,dim))
            arrays[name]=a
        end=0
        for chunk in cursor['chunks']:
            start,stop=chunk['start'],chunk['stop']
            if start!=end or not start<stop<=cursor['next']:raise ValueError('Invalid feature cursor')
            for name,a in arrays.items():
                if hashlib.sha256(a[start:stop].tobytes()).hexdigest()!=chunk[name]:raise ValueError('Corrupt feature block')
            end=stop
        if end!=cursor['next'] or not 0<=end<=count:raise ValueError('Cursor population differs')
        model=load_official(old.path(s,'official_cache'),old.path(s,'dino_source'),c['device'])
        before=state_hash(model)
        fixed=projection(12288,s['projection_dim'],s['seed']).to(c['device'])
        original=np.load(old.path(s,'prepared')/'baseline_projected.npy',mmap_mode='r')
        # This reader verifies original GSV image SHA and full-square geometry.
        dataset=old.DualImages(old.path(s,'dataset_root'),[r['path'] for r in rows],[r['image_sha256'] for r in rows])
        for start in tqdm(range(end,count,c['flush_rows']),desc='Frozen full descriptors + pre-BoQ summaries'):
            stop=min(start+c['flush_rows'],count)
            loader=DataLoader(Subset(dataset,range(start,stop)),batch_size=c['batch_size'],num_workers=c['workers'],shuffle=False)
            parts={n:[] for n in DIMENSIONS}
            with torch.inference_mode():
                for images,_,ids,_ in loader:
                    patches=model.backbone(images.to(c['device']))
                    descriptor=model.aggregator(patches)[0]
                    values=dict(global_full=descriptor,**summarize_patches(patches))
                    if float((descriptor.norm(dim=1)-1).abs().max())>1e-4:raise ValueError('Nonunit official descriptor')
                    error=float(np.abs((descriptor@fixed).cpu().numpy()-original[ids.numpy()]).max())
                    if error>c['projection_tolerance']:raise ValueError(f'Original projection cache mismatch: {error}')
                    cursor['max_projection_error']=max(cursor['max_projection_error'],error)
                    for name,value in values.items():
                        if not torch.isfinite(value).all():raise ValueError('Nonfinite extracted features')
                        parts[name].append(value.cpu().numpy())
            chunk=dict(start=start,stop=stop)
            for name in DIMENSIONS:
                value=np.concatenate(parts[name]);arrays[name][start:stop]=value;arrays[name].flush()
                chunk[name]=hashlib.sha256(value.tobytes()).hexdigest()
            cursor['chunks'].append(chunk);cursor['next']=stop;write(output/'cursor.json',cursor)
            write(output/'progress.json',dict(phase='extracting',images=stop,total=count,student_updates=0,
                max_original_projection_error=cursor['max_projection_error'],peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20))
        if state_hash(model)!=before:raise ValueError('Frozen model changed')
        write(output/'summary.json',dict(images=count,train_rows=plan['train_rows'],dimensions=DIMENSIONS,
            cache_gib=sum(p.stat().st_size for p in output.glob('*.npy'))/2**30,
            max_original_projection_error=cursor['max_projection_error'],official_unchanged=True,
            local_scope='Channel-complete spatial summaries (mean and 2x2), NOT full dense patch maps.',student_updates=0))
        finish(output);print('Frozen extraction complete',flush=True)


def bootstrap_scores(error,null,seed,samples):
    error=np.asarray(error,np.float64);null=np.asarray(null,np.float64)
    if error.ndim!=1 or error.shape!=null.shape or not len(error) or not null.sum()>0:raise ValueError('Invalid heldout-place energies')
    rng=np.random.default_rng(seed);scores=[]
    for _ in range(samples):
        ids=rng.integers(0,len(error),size=len(error))
        scores.append(1-error[ids].sum()/max(null[ids].sum(),1e-12))
    return [float(v) for v in np.quantile(scores,[.025,.975])]


def fit(c,features,targets,train_rows,seed):
    """Fixed eight epochs, own train normalization, no heldout-driven selection."""
    x=center_four(features)
    scale=x[:train_rows].std(0).clip(1e-4)
    x=torch.from_numpy(np.asarray(x/scale,np.float32));y=torch.from_numpy(np.asarray(targets,np.float32))
    seed_all(seed);head=Regressor(x.shape[1],c['hidden'],y.shape[1])
    optimizer=torch.optim.AdamW(head.parameters(),lr=c['probe_lr'],weight_decay=0.)
    history=[]
    for epoch in range(c['probe_epochs']):
        order=torch.randperm(train_rows,generator=torch.Generator().manual_seed(seed+epoch));loss_sum=0.
        for ids in order.split(c['probe_batch']):
            optimizer.zero_grad(set_to_none=True);loss=F.mse_loss(head(x[ids]),y[ids]);loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            loss_sum+=float(loss.detach())*len(ids)
        history.append(dict(epoch=epoch+1,train_online_mse=loss_sum/train_rows))
    with torch.inference_mode():
        errors=[]
        for start in range(0,len(x),512):
            errors.append((head(x[start:start+512])-y[start:start+512]).square().mean(1).numpy())
    errors=np.concatenate(errors);truth=y.numpy();null=((truth-truth[:train_rows].mean(0))**2).mean(1)
    hold_error=errors[train_rows:].reshape(-1,4).mean(1)
    hold_null=null[train_rows:].reshape(-1,4).mean(1)
    result=dict(inputs=x.shape[1],hidden=c['hidden'],parameters=sum(p.numel() for p in head.parameters()),
        train_r2=1-float(errors[:train_rows].sum()/null[:train_rows].sum()),
        holdout_r2=1-float(hold_error.sum()/hold_null.sum()),
        holdout_place_bootstrap_ci95=bootstrap_scores(hold_error,hold_null,c['seed'],c['bootstrap_samples']),
        history=history,epochs=c['probe_epochs'],train_rows=train_rows,holdout_rows=len(x)-train_rows,
        mean_input_train_scale=float(scale.mean()))
    return result,hold_error,hold_null


def paired_difference(left_error,right_error,null,seed,samples):
    # R2(left) - R2(right), resampling entire heldout places, not individual views.
    a,b,n=[np.asarray(v,np.float64) for v in (left_error,right_error,null)]
    if a.shape!=b.shape or a.shape!=n.shape or a.ndim!=1:raise ValueError('Paired energies differ')
    rng=np.random.default_rng(seed);values=[]
    for _ in range(samples):
        ids=rng.integers(0,len(a),size=len(a));values.append((b[ids].sum()-a[ids].sum())/max(n[ids].sum(),1e-12))
    return dict(delta_r2=float((b.sum()-a.sum())/n.sum()),place_bootstrap_ci95=[float(v) for v in np.quantile(values,[.025,.975])])


def probes(c,args):
    s=source(c);_,y,_,donors=old.prepared(s);plan=old.open_plan(s);n=plan['train_rows']
    cache=path(c,'cache')
    if verified(cache)!=dict(stage='frozen_feature_extraction',**identity(c)):raise ValueError('Feature extraction provenance differs')
    output=path(c,'output');contract=dict(stage='representation_probes',**identity(c),feature_complete_sha256=sha(cache/'completed.json'))
    with owned_run(output,contract,args.resume):
        if (output/'completed.json').exists():verified(output);print('Diagnostic already complete',flush=True);return
        original=np.load(old.path(s,'prepared')/'baseline_projected.npy',mmap_mode='r')
        names=('projected256','global_full','projected256_lifted','patch_mean','patch_grid2','clip_proxy_positive_control')
        results={};energies={}
        for name in names:
            report_path=output/(name+'.json');energy_path=output/(name+'.npz');ready=output/(name+'.ready.json')
            if ready.exists():
                saved=read(ready)
                if sha(report_path)!=saved['report_sha256'] or sha(energy_path)!=saved['energies_sha256']:raise ValueError('Corrupted finished probe')
                results[name]=read(report_path)
                with np.load(energy_path,allow_pickle=False) as z:energies[name]={k:z[k].copy() for k in z.files}
                continue
            if name=='projected256':features=original
            elif name=='projected256_lifted':features=lifted_projection(original,projection(12288,s['projection_dim'],s['seed']).numpy())
            elif name=='clip_proxy_positive_control':features=np.load(old.path(s,'prepared')/'scene.npy',mmap_mode='r')
            else:features=np.load(cache/(name+'.npy'),mmap_mode='r')
            reports={};scores={}
            for arm in ('aligned','shuffled'):
                write(output/'progress.json',dict(phase='cpu_probes',representation=name,target=arm,
                    student_updates=0,heldout_used_for_selection=False))
                print('PROBE',name,arm,flush=True)
                result,error,null=fit(c,features,y if arm=='aligned' else y[donors],n,s['seed']+8101)
                reports[arm]=result;scores[arm+'_error']=error;scores[arm+'_null']=null
            gap=paired_difference(scores['aligned_error'],scores['shuffled_error'],scores['aligned_null'],c['seed'],c['bootstrap_samples'])
            reports['aligned_shuffled_gap']=gap
            reports['diagnostic_signal']=(reports['aligned']['holdout_r2']>=c['signal_min_r2'] and gap['delta_r2']>=c['signal_min_aligned_shuffled_gap'])
            reports['scope']='Diagnostic only; multiple representations examined; no confirmatory significance or VPR benefit claim.'
            write(report_path,reports);save_npz(energy_path,**scores)
            write(ready,dict(report_sha256=sha(report_path),energies_sha256=sha(energy_path)))
            results[name]=reports;energies[name]=scores;del features
        contrasts={}
        for a,b in (('global_full','projected256'),('global_full','projected256_lifted'),
                    ('patch_mean','global_full'),('patch_grid2','global_full')):
            contrasts[a+'_minus_'+b]=paired_difference(energies[a]['aligned_error'],energies[b]['aligned_error'],
                energies[a]['aligned_null'],c['seed'],c['bootstrap_samples'])
        protected=old.source_hashes(old.PROTECTED)==contract['source']['base']['protected_sources']
        if not protected:raise ValueError('Innovation1 sources changed')
        summary=dict(results=results,contrasts=contrasts,innovation1_sources_unchanged=protected,
            original_projection_cache_reproduced=read(cache/'summary.json')['max_original_projection_error'],
            no_vpr_training=True,blind_test_used=False,
            limits=['Raw full global vs lifted uses identical probe parameter count, but different input rank/optimization geometry.',
                    'Pre-BoQ summaries are NOT full dense patch information. Failure does not establish absence in the backbone.',
                    'Proxy differences mix view/content/appearance; predictability is not pure-nuisance identification.',
                    'All heldout outcomes are diagnostic, no posthoc thresholds/hyperparameter selection or automatic training.'])
        write(output/'summary.json',summary);write(output/'progress.json',dict(phase='complete',no_vpr_training=True))
        finish(output);print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('extract','probe'))
    p.add_argument('--config',type=Path,default=ROOT/'config/clip_nuisance_representation.yaml')
    p.add_argument('--allow-probe-fit',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.command=='probe' and not a.allow_probe_fit:p.error('CPU probe fitting requires --allow-probe-fit; VPR never trains')
    c=configuration(a.config)
    {'extract':extract,'probe':probes}[a.command](c,a)


if __name__=='__main__':main()
