"""Fixed external-dynamic CLIP screen: CPU remap, then query-only RU evaluation."""
import argparse
import csv
import hashlib
import html
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.clip_dynamic_screen import (DYNAMIC,EGO,STATIC,BOXES,TEMPERATURE,
    require,sha,write,code_hashes,shard_read,load_dataset,choose_device,controls,outcome,safe_image,save_preview)

VARIANTS=['baseline','aligned_input','shuffle_input']


def reference_code_matches(recorded):
    current=code_hashes()
    if set(current)!=set(recorded):
        return False
    root=Path(__file__).resolve().parents[1]
    return all(recorded[k] in (current[k],hashlib.sha256((root/k).read_bytes().replace(b'\r\n',b'\n')).hexdigest())
               for k in current)


def external_map(cosines):
    c=np.asarray(cosines,dtype=np.float64)
    require(c.shape==(16,len(DYNAMIC)+len(EGO)+len(STATIC)) and np.isfinite(c).all(),'Invalid cosine array')
    margin=c[:,:len(DYNAMIC)].max(1)-c[:,len(DYNAMIC)+len(EGO):].max(1)
    values=np.maximum(2/(1+np.exp(-np.clip(margin/TEMPERATURE,-60,60)))-1,0)
    total=np.zeros((20,20)); count=np.zeros((20,20))
    for (x,y,x2,y2),value in zip(BOXES,values):
        total[y:y2,x:x2]+=value; count[y:y2,x:x2]+=1
    return (total/count).astype(np.float32)


def prepare(a):
    require(not a.output.exists(),'Output exists; use a new directory')
    old=json.loads((a.old_cache/'contract.json').read_text(encoding='utf-8'))
    done=json.loads((a.old_cache/'completed.json').read_text(encoding='utf-8'))
    require(reference_code_matches(old['code']),'Old CLIP code differs')
    require(old['dynamic_text']==DYNAMIC and old['ego_text']==EGO and old['static_text']==STATIC and
            old['crop_boxes_grid20']==[list(b) for b in BOXES] and old['temperature']==TEMPERATURE,'Old cache configuration differs')
    require(done['complete'] and sha(a.old_cache/'masks.npz')==done['masks_sha256'] and
            sha(a.old_cache/'teacher.json')==done['teacher_sha256'],'Old cache incomplete/changed')
    with np.load(a.old_cache/'masks.npz',allow_pickle=False) as z:
        queries=z['query_paths'].tolist(); old_masks=z['masks'].copy(); old_egos=z['ego_masks'].copy()
    require(len(queries)==740 and len(set(queries))==740,'Invalid query identities')
    masks=[]; hashes={}
    for q,path in enumerate(queries):
        expected=[v for k,v in old['inputs'].items() if k.replace(chr(92),'/').endswith('/'+path)]
        require(len(expected)==1,'Missing/ambiguous image identity')
        file=a.old_cache/'shards'/f'{q:04d}.npz'
        m,e=shard_read(file,expected[0])
        require(np.array_equal(m,old_masks[q]) and np.array_equal(e,old_egos[q]),'Old shard aggregate mismatch')
        hashes[file.name]=sha(file)
        with np.load(file,allow_pickle=False) as z: masks.append(external_map(z['cosines']))
    masks=np.stack(masks)
    contract={'schema':'clip_external_positive_v1','inputs':old['inputs'],'old_contract':old,
              'old_maps_sha256':done['masks_sha256'],'teacher_sha256':done['teacher_sha256'],
              'old_shards_sha256':hashes,'script_sha256':sha(__file__),
              'mapping':'max(2*sigmoid((max dynamic-max static)/0.05)-1,0) per coarse crop then overlap average; ego ignored',
              'variants':VARIANTS,'input_strength':.5,'scope':'No annotations used; query only; fixed full DB'}
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'masks.npz',masks=masks,query_paths=queries)
    write(a.output/'contract.json',contract)
    write(a.output/'completed.json',{'complete':True,'masks_sha256':sha(a.output/'masks.npz'),
          'contract_sha256':sha(a.output/'contract.json'),'queries':740,
          'mean_score':float(masks.mean()),'zero_fraction':float((masks==0).mean())})
    print('Prepared without model inference:',a.output,flush=True)


def load_prepared(cache,inputs,queries):
    contract=json.loads((cache/'contract.json').read_text(encoding='utf-8'))
    done=json.loads((cache/'completed.json').read_text(encoding='utf-8'))
    require(done['complete'] and sha(cache/'masks.npz')==done['masks_sha256'] and
            sha(cache/'contract.json')==done['contract_sha256'],'Prepared cache changed')
    require(contract['schema']=='clip_external_positive_v1' and contract['inputs']==inputs and
            contract['script_sha256']==sha(__file__) and reference_code_matches(contract['old_contract']['code']) and
            contract['variants']==VARIANTS and contract['input_strength']==.5,'Prepared configuration differs')
    with np.load(cache/'masks.npz',allow_pickle=False) as z:
        require(z['query_paths'].tolist()==queries,'Query order mismatch')
        masks=z['masks'].copy()
    require(masks.shape==(740,20,20) and np.isfinite(masks).all() and (masks>=0).all() and (masks<=1).all(),'Invalid masks')
    return masks,contract


def evaluate(a):
    require(not a.output.exists(),'Evaluation output exists; use a new directory')
    run,db,queries,gt,inputs=load_dataset(a.dataset_root,a.run)
    masks,contract=load_prepared(a.cache,inputs,queries)
    checkpoint=a.checkpoint or Path(run['checkpoint']['path'])
    require(sha(checkpoint)==run['checkpoint']['sha256'],'Wrong RU checkpoint')
    source=json.loads((a.source/'summary.json').read_text())
    require(source['ru_sha256']==sha(checkpoint),'Source checkpoint mismatch')
    desc=np.load(a.source/'descriptors.npy',mmap_mode='r',allow_pickle=False)
    require(desc.shape==(19611,12288),'Wrong source descriptors')
    for start in range(0,len(desc),256):
        block=desc[start:start+256]
        require(np.isfinite(block).all() and np.allclose(np.linalg.norm(block,axis=1),1,atol=2e-4),'Invalid cached descriptors')
    with np.load(a.source/'per_query.npz',allow_pickle=False) as z:
        candidates=z['candidates'].copy(); saved_scores=z['ru_scores'].copy()
    require(candidates.shape==saved_scores.shape==(740,20) and np.issubdtype(candidates.dtype,np.integer)
            and (candidates>=0).all() and (candidates<18871).all(),'Invalid cached candidates')
    for q,ids in enumerate(candidates):
        require(len(set(ids.tolist()))==20 and (np.diff(saved_scores[q])<=1e-6).all(),'Invalid candidate ordering')
        require(np.allclose(desc[ids]@desc[18871+q],saved_scores[q],atol=2e-5,rtol=2e-4),'Cached scores mismatch')
    require(sum(int(candidates[q,0] in gt[q]) for q in range(740))==675,'Baseline is not 675/740')
    import torch
    from PIL import Image
    from scripts.eval_condition_robustness import build_transform,load_inference_model_from_ckpt
    from scripts.eval_dynamic_category_prior import extract_ru_feature_map,boq_descriptor
    device=choose_device(a.device)
    model=load_inference_model_from_ckpt(checkpoint,device).eval()
    transform=build_transform((280,280))
    # ~885 MiB DB tensor, one original reference space shared by all variants.
    database=torch.tensor(np.asarray(desc[:18871]),device=device)
    a.output.mkdir(parents=True)
    (a.output/'partial_queries').mkdir()
    fingerprint={str(p):sha(p) for p in (checkpoint,a.source/'descriptors.npy',a.source/'per_query.npz',
                 a.source/'summary.json',a.cache/'masks.npz',a.cache/'contract.json')}
    write(a.output/'provenance.json',{'inputs':inputs,'fingerprints':fingerprint,'code':code_hashes(),
          'clip_contract':contract,'evaluation_script_sha256':sha(__file__),'input_strength':.5,'device':str(device),
          'scope':'740 query-only interventions against unchanged full DB; developer MSLS, not independent validation'})
    scores_file=np.lib.format.open_memmap(a.output/'scores.npy',mode='w+',dtype='float32',shape=(740,len(VARIANTS),18871))
    vectors_file=np.lib.format.open_memmap(a.output/'descriptors.npy',mode='w+',dtype='float32',shape=(740,len(VARIANTS),12288))
    rows=[]
    with torch.inference_mode():
        # Fresh deterministic reference sentinels verify DB path/cache mapping without re-extracting the full DB.
        for i in (0,9435,18870):
            with Image.open(safe_image(a.dataset_root,db[i])) as im:
                x=transform(im.convert('RGB')).unsqueeze(0).to(device)
            d=torch.nn.functional.normalize(boq_descriptor(model.aggregator,extract_ru_feature_map(model,x),None).float(),dim=-1)
            require(np.allclose(d.cpu().numpy()[0],desc[i],atol=2e-5,rtol=2e-4),'Fresh DB sentinel mismatch')
        for q,path in enumerate(queries):
            with Image.open(safe_image(a.dataset_root,path)) as im:
                image=transform(im.convert('RGB')).unsqueeze(0).to(device)
            fm=extract_ru_feature_map(model,image)
            def encode(f,bias=None):
                return torch.nn.functional.normalize(boq_descriptor(model.aggregator,f,bias).float(),dim=-1)
            ds={'baseline':encode(fm)}
            require(np.allclose(ds['baseline'].cpu().numpy()[0],desc[18871+q],atol=2e-5,rtol=2e-4),'Fresh query/cache mismatch')
            for name in ('aligned','shuffle'):
                mask=controls(masks[q],path)[name]
                m=torch.tensor(mask[None],device=device)
                pixel=torch.nn.functional.interpolate(m[:,None],size=(280,280),mode='nearest')
                ds[name+'_input']=encode(extract_ru_feature_map(model,image*(1-.5*pixel)))
            vectors=torch.cat([ds[v] for v in VARIANTS])
            scores=(vectors@database.T).cpu().numpy()
            base=outcome(scores[0],gt[q])
            require(base['top1_reference_index']==int(candidates[q,0]),'Fresh full-DB baseline top1 mismatch')
            scores_file[q]=scores; vectors_file[q]=vectors.cpu().numpy()
            for i,v in enumerate(VARIANTS):
                result=outcome(scores[i],gt[q])
                rows.append({'query_index':q,'query_path':path,'variant':v,**result,
                    'baseline_correct':base['top1_correct'],
                    'correction':int(not base['top1_correct'] and result['top1_correct']),
                    'regression':int(base['top1_correct'] and not result['top1_correct']),
                    'margin_delta':result['positive_negative_margin']-base['positive_negative_margin'],
                    'mean_clip_score':float(masks[q].mean())})
            write(a.output/'partial_queries'/f'{q:04d}.json',rows[-len(VARIANTS):])
            if (q+1)%25==0: print('RU query',q+1,'/740',flush=True)
    scores_file.flush(); vectors_file.flush()
    write(a.output/'query_outcomes.json',rows)
    with (a.output/'query_outcomes.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=[k for k in rows[0] if k!='top20'],extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    summary={}
    for v in VARIANTS:
        selected=[r for r in rows if r['variant']==v]
        summary[v]={'n':740,'correct':sum(r['top1_correct'] for r in selected),
                    'corrected_query_ids':[r['query_index'] for r in selected if r['correction']],
                    'regressed_query_ids':[r['query_index'] for r in selected if r['regression']],
                    'baseline_wrong_n':65,'baseline_correct_n':675}
    paired={}
    for stage in ('input',):
        left=[r for r in rows if r['variant']=='aligned_'+stage]
        for control in ('shuffle',):
            right=[r for r in rows if r['variant']==control+'_'+stage]
            paired[stage+'_vs_'+control]={
                'aligned_only_correct':[l['query_index'] for l,r in zip(left,right) if l['top1_correct'] and not r['top1_correct']],
                'control_only_correct':[l['query_index'] for l,r in zip(left,right) if r['top1_correct'] and not l['top1_correct']]}
    write(a.output/'summary.json',{'variants':summary,'score_axis_order':VARIANTS,
          'aligned_vs_controls':paired,
          'scope':'Automatic crop-CLS CLIP, no manual labels used; query-only, unchanged full database',
          'limitations':'Uncalibrated dynamic-vs-static crop score; ego ignored; broad crops; query-only; no independent validation'})
    # Deterministic, bounded visual audit of original errors; no threshold or prompt tuning.
    errors=[r for r in rows if r['variant']=='baseline' and not r['top1_correct']]
    chosen=sorted(errors,key=lambda r:hashlib.sha256(('42:'+r['query_path']).encode()).hexdigest())[:12]
    (a.output/'images').mkdir()
    page=['<!doctype html><meta charset="utf-8"><h1>基线错误固定抽样（12/65）</h1><p>按路径哈希抽样；GT按原始分数最高正样本展示，不能当作独立评测。无需画图。</p>']
    for r in chosen:
        q=r['query_index']
        page.append(f'<h2>query {q}</h2>')
        save_preview(a.dataset_root,queries[q],masks[q],a.output/f'images/{q}_query.jpg')
        page.append(f'<img src="images/{q}_query.jpg">')
        for role,i in [('baseline_top1',r['top1_reference_index']),('baseline_best_GT',r['best_gt_reference_index'])]:
            with Image.open(safe_image(a.dataset_root,db[i])) as im:
                im.convert('RGB').resize((400,300)).save(a.output/f'images/{q}_{role}.jpg')
            page.append('<p>'+html.escape(role+': '+db[i])+f'</p><img src="images/{q}_{role}.jpg">')
    (a.output/'index.html').write_text('\n'.join(page),encoding='utf-8')
    write(a.output/'completed.json',{'complete':True,'queries':740,'variants':VARIANTS})
    print('Saved evaluation:',a.output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','evaluate'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--old-cache',type=Path,default=Path('.cache/clip_dynamic_msls_v1'))
    p.add_argument('--cache',type=Path,default=Path('.cache/clip_external_positive_v1'))
    p.add_argument('--source',type=Path,default=Path('doc/visual_pair_msls_hard_mix_v2'))
    p.add_argument('--run',type=Path,default=Path('doc/dynamic_category_prior_screen_b0.5_full_db_condition_queries'))
    p.add_argument('--dataset-root',type=Path,default=Path('datasets/msls-val'))
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--device',default='cuda:1')
    a=p.parse_args(); {'prepare':prepare,'evaluate':evaluate}[a.stage](a)

if __name__=='__main__':
    main()
