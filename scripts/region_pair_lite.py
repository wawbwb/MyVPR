"""Reuse existing visual-pair RU caches; SAM-B region verification, no training."""
import argparse
import json
import os
from pathlib import Path
import sys
import zipfile
import zlib
from uuid import uuid4
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src.region_pair_lite import pool, match_regions, conservative_order
from scripts.segvlad_official import sha


def write(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf8'); temp.replace(path)


LEGACY_SCRIPT_SHA = '2a24a8c8309860f40f466883cc1dd22dfd72de825029752246205f471019a101'


def compatible_legacy_contract(old, current):
    candidate = json.loads(json.dumps(old))
    code = candidate.get('code', {})
    key = 'scripts/region_pair_lite.py'
    if code.get(key) != LEGACY_SCRIPT_SHA:
        return False
    code[key] = current['code'][key]
    return candidate == current


def read_shard(path):
    """Read every member (including CRC) and validate the unchanged cache schema."""
    modes = ('sam', 'grid', 'shifted')
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {k for m in modes for k in (m, m+'_xy')}:
            raise ValueError('unexpected shard keys')
        arrays = {k: archive[k] for k in archive.files}
    if arrays['sam'].ndim != 2:
        raise ValueError('invalid SAM descriptor dimensions')
    n = len(arrays['sam'])
    if n != 0 and not 3 <= n <= 16:
        raise ValueError('invalid region count')
    for mode in modes:
        d, xy = arrays[mode], arrays[mode+'_xy']
        if d.shape != (n, 768) or xy.shape != (n, 2):
            raise ValueError('invalid region/centroid shape')
        if d.dtype != np.float16 or xy.dtype != np.float32:
            raise ValueError('invalid region/centroid dtype')
        if not np.isfinite(d).all() or not np.isfinite(xy).all():
            raise ValueError('non-finite shard values')
        if (xy < 0).any() or (xy > 1).any():
            raise ValueError('centroids outside normalized image')
    return arrays


def inspect_shard(path):
    if not path.exists():
        return 'missing'
    try:
        read_shard(path)
    except (EOFError, ValueError, zipfile.BadZipFile, zlib.error) as error:
        return f'{type(error).__name__}: {error}'
    return None


def save_shard(path, arrays):
    temp = path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    with temp.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    read_shard(temp)
    temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True,help='Existing visual_pair_msls_* output with local.npy/descriptors.npy/per_query.npz')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--sam-checkpoint',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,default=Path('datasets/msls-val'))
    p.add_argument('--output',type=Path,default=Path('doc/region_pair_lite_v1'))
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--limit-images',type=int,default=0)
    a=p.parse_args()
    for name in ('checkpoint', 'sam_checkpoint'):
        if not getattr(a, name).is_file(): p.error(f'{name} must point to an existing checkpoint file')
    if (a.output/'completed.json').exists(): p.error('Run completed; no overwrites')
    import torch
    from PIL import Image
    from tqdm import tqdm
    torch.backends.cuda.matmul.allow_tf32=False
    paths=[]
    for name in ('msls_val_dbImages.npy','msls_val_qImages.npy'):
        paths.append(np.load(a.dataset_root/name).astype(str).tolist())
    if list(map(len,paths))!=[18871,740]: p.error('Wrong split')
    images=paths[0]+paths[1]
    source_summary=json.loads((a.source/'summary.json').read_text())
    if source_summary['ru_sha256']!=sha(a.checkpoint): p.error('RU checkpoint mismatch')
    with np.load(a.source/'per_query.npz') as z:
        candidates=z['candidates'].copy(); scores=z['ru_scores'].copy()
    if candidates.shape!=(740,20) or scores.shape!=(740,20): p.error('Need original top20')
    if candidates.min()<0 or candidates.max()>=18871 or not np.isfinite(scores).all(): p.error('Invalid candidates')
    local=np.load(a.source/'local.npy',mmap_mode='r')
    desc=np.load(a.source/'descriptors.npy',mmap_mode='r')
    if local.shape!=(19611,400,768) or desc.shape[0]!=19611: p.error('Unexpected cache shape')
    for q,row in enumerate(candidates):
        if len(set(row.tolist()))!=20 or (np.diff(scores[q])>1e-6).any(): p.error('Invalid candidate order')
        actual=np.asarray(desc[row])@np.asarray(desc[18871+q])
        if not np.allclose(actual,scores[q],atol=2e-5,rtol=2e-4): p.error('Candidate scores do not match descriptors')
    required=np.unique(np.concatenate([candidates.ravel(),np.arange(18871,19611)]))
    contract={'source':str(a.source.resolve()),'ru':sha(a.checkpoint),'sam':sha(a.sam_checkpoint),
        'inputs':{n:sha(a.source/n) for n in ('local.npy','descriptors.npy','per_query.npz','summary.json')},
        'split':{n:sha(a.dataset_root/n) for n in ('msls_val_dbImages.npy','msls_val_qImages.npy','msls_val_gt_25m.npy')},
        'code':{n:sha(ROOT/n) for n in ('scripts/region_pair_lite.py','src/region_pair_lite.py','src/region_vlad.py')},
        'required_images':len(required),'max_regions':16,'scope':'Exploratory, fixed thresholds; no GT-based selection'}
    if (a.output/'contract.json').exists():
        old = json.loads((a.output/'contract.json').read_text())
        if old != contract:
            if not compatible_legacy_contract(old, contract): p.error('Resume contract changed')
            write(a.output/'contract_migration.json', {'previous': old, 'current': contract,
                'reason': 'Cache integrity/recovery only; extraction and scoring unchanged'})
            write(a.output/'contract.json', contract)
            print('Migrated known original cache contract; all data/model/scorer hashes unchanged.', flush=True)
    else:
        a.output.mkdir(parents=True,exist_ok=True); (a.output/'shards').mkdir(exist_ok=True)
        write(a.output/'contract.json',contract)
    print(f'Reusing native cache. New region arrays about {len(required)*3*16*768*2/1024**2:.1f} MiB plus memberships/metadata.',flush=True)
    pending=[]; repairs=[]
    (a.output/'shards').mkdir(exist_ok=True)
    for i in tqdm(required, desc='Validate existing region shards'):
        target=a.output/'shards'/f'{i:06d}.npz'
        reason=inspect_shard(target)
        if reason is None: continue
        pending.append(i)
        entry={'image_index': int(i), 'reason': reason}
        if reason != 'missing':
            quarantine=a.output/'quarantine'
            quarantine.mkdir(exist_ok=True)
            destination=quarantine/(target.name+'.'+uuid4().hex+'.bad')
            # Move only this exact shard, within this run; never delete caches.
            target.resolve().relative_to(a.output.resolve())
            destination.resolve().relative_to(a.output.resolve())
            target.rename(destination)
            entry['quarantined_to']=str(destination)
            print(f'Repair image {i}: {reason}; retained at {destination}', flush=True)
        repairs.append(entry)
    write(a.output/('cache_validation_'+uuid4().hex+'.json'), {
        'reused': len(required)-len(pending), 'to_generate': len(pending), 'repairs': repairs})
    print(f'Valid shards reused: {len(required)-len(pending)}; missing/damaged to generate: {len(pending)}', flush=True)
    # Audit legacy cache mapping by fresh deterministic checkpoint samples, not labels.
    if not (a.output/'cache_audit.json').exists():
        from scripts.eval_condition_robustness import load_inference_model_from_ckpt, build_transform
        from src.cc_lsa_features import extract_ru_descriptor_and_local
        model=load_inference_model_from_ckpt(a.checkpoint,torch.device(a.device)).eval()
        transform=build_transform((280,280)); checked=[]
        for i in [0,18870,18871,19610,*np.random.default_rng(42).choice(19611,8,replace=False).tolist()]:
            with Image.open(a.dataset_root/images[i]) as im: tensor=transform(im.convert('RGB')).unsqueeze(0).to(a.device)
            with torch.inference_mode(): d,t=extract_ru_descriptor_and_local(model,tensor,output_grid=(20,20))
            if not np.allclose(d.cpu().numpy()[0],desc[i],atol=3e-5,rtol=3e-4) or not np.allclose(t.cpu().numpy()[0],local[i],atol=8e-4,rtol=2e-3):
                raise ValueError(f'Legacy cache mapping/feature mismatch image {i}')
            checked.append(int(i))
        del model; torch.cuda.empty_cache()
        write(a.output/'cache_audit.json',{'checked_images':checked,'note':'Sampled mapping audit, not exhaustive checkpoint recomputation'})
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    from src.region_vlad import grid_masks
    generator=None; processed=0
    for i in tqdm(pending,desc='Compact candidate-region cache'):
        target=a.output/'shards'/f'{i:06d}.npz'
        if generator is None:
            sam=sam_model_registry['vit_b'](checkpoint=str(a.sam_checkpoint)).to(a.device).eval()
            generator=SamAutomaticMaskGenerator(sam,points_per_side=16,points_per_batch=32,
                pred_iou_thresh=.88,stability_score_thresh=.95,crop_n_layers=0)
        with Image.open(a.dataset_root/images[i]) as im: rgb=np.asarray(im.convert('RGB').resize((280,280),Image.Resampling.BICUBIC))
        masks=[]
        with torch.inference_mode(): raw=generator.generate(rgb)
        for item in sorted(raw,key=lambda x:-x['area']):
            m=item['segmentation'].reshape(20,14,20,14).mean((1,3))>=.5
            if not 4<=m.sum()<=240: continue
            if any((m&old).sum()/max(1,(m|old).sum())>.3 for old in masks): continue
            masks.append(m)
            if len(masks)==16: break
        if len(masks)<3:
            arrays={mode:np.empty((0,768),dtype='float16') for mode in ('sam','grid','shifted')}
            arrays.update({mode+'_xy':np.empty((0,2),dtype='float32') for mode in ('sam','grid','shifted')})
        else:
            masks=np.stack(masks)
            controls={'sam':masks,'grid':grid_masks(len(masks),size=20), 'shifted':np.roll(masks,(10,10),axis=(1,2))}
            arrays={}
            for mode,m in controls.items():
                d,xy=pool(np.asarray(local[i],dtype='float32'),m)
                arrays[mode]=d.astype('float16'); arrays[mode+'_xy']=xy.astype('float32')
        save_shard(target, arrays); processed+=1
        if a.limit_images and processed>=a.limit_images:
            print('SMOKE COMPLETE; rerun without --limit-images to continue'); return
    predictions={'ru':candidates.copy()}; all_scores={}; supports={}
    for mode in ('sam','grid','shifted'):
        regional=np.zeros((740,20)); support=np.zeros((740,20),dtype=int)
        prediction=[]
        for q,row in enumerate(tqdm(candidates,desc=f'Match {mode}')):
            with np.load(a.output/'shards'/f'{q+18871:06d}.npz') as z: qd=z[mode]; qxy=z[mode+'_xy']
            for j,i in enumerate(row):
                with np.load(a.output/'shards'/f'{i:06d}.npz') as z:
                    regional[q,j],support[q,j]=match_regions(qd,z[mode],qxy,z[mode+'_xy'])
            prediction.append(row[conservative_order(scores[q],regional[q],support[q])])
        predictions[mode]=np.asarray(prediction); all_scores[mode]=regional; supports[mode]=support
    gt=np.load(a.dataset_root/'msls_val_gt_25m.npy',allow_pickle=True)
    baseline=np.array([row[0] in g for row,g in zip(candidates,gt)])
    if len(gt)!=740 or baseline.sum()!=675: raise ValueError('RU/GT baseline not reproduced')
    report={}
    for mode,pred in predictions.items():
        hit=np.array([row[0] in g for row,g in zip(pred,gt)])
        report[mode]={'correct':int(hit.sum()),'corrections':np.flatnonzero(hit&~baseline).tolist(),
            'regressions':np.flatnonzero(~hit&baseline).tolist(),'top1_changed':int((pred[:,0]!=candidates[:,0]).sum())}
    np.savez_compressed(a.output/'predictions.npz',**predictions)
    np.savez_compressed(a.output/'region_scores.npz',**all_scores,**{k+'_supports':v for k,v in supports.items()})
    write(a.output/'summary.json',report)
    write(a.output/'completed.json',{'complete':True,'contract_sha256':sha(a.output/'contract.json')})
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
