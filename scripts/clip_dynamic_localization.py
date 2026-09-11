"""Fixed 2x2 localization diagnostic: coarse/fine crops x sigmoid/positive evidence."""
import argparse
import hashlib
import html
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import clip_dynamic_screen as base

FINE_BOXES=[(x,y,x+4,y+4) for y in range(0,17,2) for x in range(0,17,2)]
NAMES=('coarse_sigmoid','coarse_positive','fine_sigmoid','fine_positive')


def maps(cosines, boxes):
    c=np.asarray(cosines,dtype=np.float64)
    base.require(c.shape==(len(boxes),len(base.DYNAMIC)+len(base.EGO)+len(base.STATIC)) and np.isfinite(c).all(), 'Invalid cosine array')
    d=c[:,:len(base.DYNAMIC)].max(1)
    e=c[:,len(base.DYNAMIC):len(base.DYNAMIC)+len(base.EGO)].max(1)
    s=c[:,len(base.DYNAMIC)+len(base.EGO):].max(1)
    out={}
    for role,margin in [('external',d-np.maximum(e,s)),('ego',e-np.maximum(d,s))]:
        sigmoid=1/(1+np.exp(-np.clip(margin/base.TEMPERATURE,-60,60)))
        for method,values in [('sigmoid',sigmoid),('positive',np.maximum(2*sigmoid-1,0))]:
            total=np.zeros((20,20)); counts=np.zeros((20,20))
            for (x,y,x2,y2),value in zip(boxes,values):
                total[y:y2,x:x2]+=value; counts[y:y2,x:x2]+=1
            base.require((counts>0).all(),'Crop grid has uncovered locations')
            out[role+'_'+method]=(total/counts).astype(np.float32)
    return out


def statistics(masks):
    mean=masks.mean((1,2))
    centered=masks-mean[:,None,None]
    horizontal=masks-masks.mean(2,keepdims=True)
    variance=float((centered**2).sum())
    return {'mean_score':float(masks.mean()),'zero_fraction':float((masks==0).mean()),
            'all_zero_queries':np.flatnonzero(mean==0).tolist(),
            'score_quantiles':np.quantile(masks,[0,.1,.5,.9,1]).tolist(),
            'mean_row_score':masks.mean((0,2)).tolist(),
            'horizontal_variance_fraction':float((horizontal**2).sum())/variance if variance else None,
            'note':'Score mass/support, not dynamic pixel coverage or motion truth'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-cache',type=Path,default=Path('.cache/clip_dynamic_msls_v1'))
    p.add_argument('--output',type=Path,default=Path('doc/clip_dynamic_localization_v2'))
    p.add_argument('--dataset-root',type=Path,default=Path('datasets/msls-val'))
    p.add_argument('--run',type=Path,default=Path('doc/dynamic_category_prior_screen_b0.5_full_db_condition_queries'))
    p.add_argument('--audit-annotations',type=Path,default=Path('doc/dynamic_mechanism_cases_v1/annotations_validated.json'))
    p.add_argument('--device',default='cuda:1')
    a=p.parse_args(argv)
    _,_,queries,_,inputs=base.load_dataset(a.dataset_root,a.run)
    old=json.loads((a.old_cache/'contract.json').read_text(encoding='utf-8'))
    done=json.loads((a.old_cache/'completed.json').read_text())
    base.require(old['inputs']==inputs,'Old cache image/data identities differ')
    base.require(old['code']==base.code_hashes(),'Old CLIP implementation changed')
    base.require(done['complete'] and base.sha(a.old_cache/'masks.npz')==done['masks_sha256'] and
                 base.sha(a.old_cache/'teacher.json')==done['teacher_sha256'],'Old cache incomplete/changed')
    with np.load(a.old_cache/'masks.npz',allow_pickle=False) as z:
        old_masks=z['masks'].copy(); old_egos=z['ego_masks'].copy()
        base.require(z['query_paths'].tolist()==queries,'Old cache query order mismatch')
    coarse=[]; shard_hashes={}
    for q,path in enumerate(queries):
        file=a.old_cache/'shards'/f'{q:04d}.npz'
        base.shard_read(file,inputs[str(a.dataset_root/path)])
        shard_hashes[str(file)]=base.sha(file)
        with np.load(file,allow_pickle=False) as z: result=maps(z['cosines'],base.BOXES)
        base.require(np.array_equal(result['external_sigmoid'],old_masks[q]) and
                     np.array_equal(result['ego_sigmoid'],old_egos[q]),'Old aggregate does not reproduce shards')
        coarse.append(result)
    contract={'schema':'clip_localization_2x2_v2','inputs':inputs,'old_shards':shard_hashes,
        'old_teacher_sha':done['teacher_sha256'],'dependencies':base.code_hashes(),'script_sha':base.sha(__file__),
        'fine_boxes':FINE_BOXES,'mapping':'max(2*sigmoid(margin/0.05)-1,0) per crop then overlap average',
        'new_retrieval':False,'annotation_sha':base.sha(a.audit_annotations) if a.audit_annotations.is_file() else None}
    contract=json.loads(json.dumps(contract))
    if a.output.exists():
        base.require((a.output/'contract.json').is_file() and json.loads((a.output/'contract.json').read_text(encoding='utf-8'))==contract,
                     'Existing output contract differs; use a new directory')
    else:
        a.output.mkdir(parents=True); (a.output/'shards').mkdir(); base.write(a.output/'contract.json',contract)
    fine={}; missing=[]
    for q,path in enumerate(queries):
        file=a.output/'shards'/f'{q:04d}.npz'
        if file.exists():
            with np.load(file,allow_pickle=False) as z:
                base.require(str(z['image_sha'])==inputs[str(a.dataset_root/path)],'Fine shard image mismatch')
                fine[q]=maps(z['cosines'],FINE_BOXES)
        else: missing.append(q)
    if (a.output/'completed.json').exists():
        done2=json.loads((a.output/'completed.json').read_text())
        base.require(not missing and base.sha(a.output/'maps.npz')==done2['maps_sha256'],'Completed report changed')
        print('Completed localization report verified:',a.output); return
    if missing:
        import torch
        from PIL import Image
        from src.models.clip_teacher import CLIPTeacherEncoder
        from src.models.cc_lsa import module_state_sha256
        from scripts.eval_condition_robustness import build_transform
        device=base.choose_device(a.device)
        teacher=CLIPTeacherEncoder(dynamic_categories=base.DYNAMIC+base.EGO+base.STATIC).to(device).eval()
        teacher_state={'visual_sha256':module_state_sha256(teacher.visual),
            'text_sha256':hashlib.sha256(teacher.dynamic_text_feats.cpu().numpy().tobytes()).hexdigest()}
        base.require(teacher_state==json.loads((a.old_cache/'teacher.json').read_text()),'Teacher changed relative to coarse cache')
        transform=build_transform((280,280))
        with torch.inference_mode():
            for q in missing:
                with Image.open(base.safe_image(a.dataset_root,queries[q])) as im: image=transform(im.convert('RGB'))
                cosine=[]
                for start in range(0,len(FINE_BOXES),16):
                    crops=torch.stack([image[:,y*14:y2*14,x*14:x2*14] for x,y,x2,y2 in FINE_BOXES[start:start+16]]).to(device)
                    cls,_=teacher(crops)
                    cosine.append((cls.float()@teacher.dynamic_text_feats.float().T).cpu().numpy())
                c=np.concatenate(cosine); fine[q]=maps(c,FINE_BOXES)
                target=a.output/'shards'/f'{q:04d}.npz'; temp=target.with_suffix('.tmp')
                with temp.open('wb') as f: np.savez_compressed(f,cosines=c,image_sha=inputs[str(a.dataset_root/queries[q])])
                temp.replace(target)
                if (q+1)%25==0: print('Fine CLIP query',q+1,'/740',flush=True)
    arrays={}; ego_arrays={}
    for name in NAMES:
        scale,method=name.split('_')
        source=coarse if scale=='coarse' else fine
        arrays[name]=np.stack([source[q]['external_'+method] for q in range(len(queries))])
        ego_arrays[name]=np.stack([source[q]['ego_'+method] for q in range(len(queries))])
    np.savez_compressed(a.output/'maps.npz',query_paths=queries,**arrays,**{k+'_ego':v for k,v in ego_arrays.items()})
    report={name:{'external':statistics(arrays[name]),'ego':statistics(ego_arrays[name]),
        'existing_polygon_diagnostic':base.localization_diagnostics(arrays[name],ego_arrays[name],queries,a.audit_annotations)} for name in NAMES}
    base.write(a.output/'summary.json',{'variants':report,'retrieval_run':False,
        'scope':'Fixed 2x2 automatic localization diagnostic; no score tuning or new manual annotations; no accuracy claim'})
    # Same panels for all variants, independent of their outputs and retrieval outcomes.
    chosen=sorted(range(len(queries)),key=lambda q:hashlib.sha256(('42:'+queries[q]).encode()).hexdigest())[:12]
    if a.audit_annotations.is_file():
        labels=json.loads(a.audit_annotations.read_text(encoding='utf-8'))
        chosen=sorted(set(chosen)|{c['query_index'] for c in labels['cases'] if c['decision']=='include'})
    (a.output/'images').mkdir(exist_ok=True)
    pages=['<!doctype html><meta charset="utf-8"><h1>CLIP定位：窗口大小×评分映射</h1>',
        '<p>各列同一颜色尺度，无图内归一化。分数变小/零值增多不等于定位更好；结合物体内外差异和漏检查看。无需人工圈图。</p>']
    for q in chosen:
        pages.append('<h2>'+html.escape(f'q{q} {queries[q]}')+'</h2>')
        for name in NAMES:
            filename=f'images/{q:04d}_{name}.jpg'
            base.save_preview(a.dataset_root,queries[q],arrays[name][q],a.output/filename)
            pages.append('<p>'+name+f'</p><img src="{filename}">')
    (a.output/'index.html').write_text('\n'.join(pages),encoding='utf-8')
    base.write(a.output/'completed.json',{'complete':True,'queries':740,'maps_sha256':base.sha(a.output/'maps.npz'),
        'new_clip_crops':740*len(FINE_BOXES),'retrieval_run':False})
    for name in NAMES: print(name,report[name]['external'])
    print('Saved:',a.output)


if __name__=='__main__': main()
