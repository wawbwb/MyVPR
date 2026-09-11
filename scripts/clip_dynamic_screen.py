"""Automatic frozen crop-CLS CLIP dynamic prior; no human annotation required."""
import argparse
import csv
import hashlib
import html
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_dynamic_coverage import require, sha, safe_image, save_preview, sample_queries

DYNAMIC = ['car on a street', 'truck on a street', 'bus on a street',
           'motorcycle on a street', 'bicycle on a street', 'pedestrian on a street']
EGO = ['car hood seen through a windshield', 'car dashboard inside a vehicle', 'windshield wipers inside a car']
STATIC = ['building facade', 'road surface', 'sidewalk', 'wall', 'tree', 'sky', 'bridge', 'street sign']
BOXES = [(x, y, x+8, y+8) for y in (0,4,8,12) for x in (0,4,8,12)]
TEMPERATURE = .05
VARIANTS = ['baseline', 'zero_bias', 'aligned_late', 'shuffle_late', 'rowmean_late',
            'aligned_input', 'shuffle_input', 'rowmean_input']


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def code_hashes():
    return {name: sha(ROOT/name) for name in ('scripts/clip_dynamic_screen.py',
        'src/models/clip_teacher.py', 'scripts/eval_condition_robustness.py',
        'scripts/eval_dynamic_category_prior.py')}


def load_dataset(root, run_dir):
    run_file = run_dir/'run.json'
    run = json.loads(run_file.read_text(encoding='utf-8'))
    require(run['schema_version'] == 3 and run['method'] == 'frozen_dynamic_category_negative_attention_prior', 'Need original RU screen v3')
    split = next(s for s in run['datasets'] if s['name']=='msls-val')
    hashes = {str(run_file): sha(run_file)}
    files = {}
    for role, info in split['manifests'].items():
        p = root/Path(info['path']).name
        hashes[str(p)] = sha(p)
        require(hashes[str(p)] == info['sha256'], 'Manifest changed: '+str(p))
        files[role] = p
    dbfile = root/'msls_val_dbImages.npy'
    hashes[str(dbfile)] = sha(dbfile)
    require(hashes[str(dbfile)] == run['descriptor_index']['database_manifest']['sha256'], 'DB manifest changed')
    db = [str(s).replace('\\','/') for s in np.load(dbfile, allow_pickle=False)]
    queries = [str(s).replace('\\','/') for s in np.load(files['queries'], allow_pickle=False)]
    gt = [np.asarray(g).reshape(-1) for g in np.load(files['ground_truth'], allow_pickle=True)]
    require(len(db)==18871 and len(queries)==len(gt)==740, 'Expected standard MSLS-val')
    require(len(set(db+queries))==19611, 'Duplicate image identities')
    for g in gt:
        require(len(g)>0 and np.issubdtype(g.dtype,np.integer) and (g>=0).all() and (g<len(db)).all(), 'Invalid GT')
    for q in queries:
        hashes[str(root/q)] = sha(safe_image(root,q))
    return run, db, queries, gt, hashes


def heatmap(cosines):
    c = np.asarray(cosines, dtype=np.float64)
    require(c.shape == (16,len(DYNAMIC)+len(EGO)+len(STATIC)) and np.isfinite(c).all(), 'Invalid CLIP cosine array')
    dynamic = c[:,:len(DYNAMIC)].max(1)
    ego = c[:,len(DYNAMIC):len(DYNAMIC)+len(EGO)].max(1)
    static = c[:,len(DYNAMIC)+len(EGO):].max(1)
    def aggregate(values):
        result, count = np.zeros((20,20)), np.zeros((20,20))
        for (x,y,x2,y2), value in zip(BOXES,values):
            result[y:y2,x:x2] += value
            count[y:y2,x:x2] += 1
        require((count>0).all(), 'Crop grid does not cover image')
        return (result/count).astype(np.float32)
    sigmoid = lambda x: 1/(1+np.exp(-np.clip(x/TEMPERATURE,-60,60)))
    return aggregate(sigmoid(dynamic-np.maximum(ego,static))), aggregate(sigmoid(ego-np.maximum(dynamic,static)))


def controls(mask, query_path):
    # Each row is permuted independently: exact values, mean area and vertical profile preserved.
    seed = int(hashlib.sha256(('42:'+query_path).encode()).hexdigest()[:16],16)
    rng = np.random.default_rng(seed)
    shuffled = np.stack([rng.permutation(row) for row in mask])
    rowmean = np.broadcast_to(mask.mean(axis=1,keepdims=True),mask.shape).copy()
    return {'aligned': mask, 'shuffle': shuffled, 'rowmean': rowmean}


def outcome(scores, gt):
    require(np.isfinite(scores).all(), 'Invalid search scores')
    order = np.argsort(-scores,kind='stable')
    positive = np.isin(order,gt)
    require(positive.any() and (~positive).any(), 'Invalid GT/search universe')
    return {'top1_reference_index': int(order[0]), 'top1_correct': int(positive[0]),
            'best_gt_reference_index': int(order[positive][0]), 'best_gt_rank': int(np.flatnonzero(positive)[0])+1,
            'positive_negative_margin': float(scores[order[positive][0]]-scores[order[~positive][0]]),
            'top20': order[:20].tolist()}


def choose_device(value):
    import torch
    device = torch.device(value)
    require(device.type=='cpu' or device.type=='cuda' and device.index is not None and device.index>0,
            'Use explicit cuda:1; GPU0 is faulty')
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return device


def shard_read(file, image_sha):
    with np.load(file,allow_pickle=False) as z:
        require(str(z['image_sha'])==image_sha, 'Shard image changed')
        cosine = z['cosines'].copy()
        expected, ego = heatmap(cosine)
        require(np.array_equal(expected,z['mask']) and np.array_equal(ego,z['ego_mask']), 'Shard scores/map mismatch')
        return expected, ego


def localization_diagnostics(masks, egos, queries, annotation_file):
    """Existing polygons are audit-only: no thresholds, prompts or masks are fitted."""
    if not annotation_file.is_file():
        return {'available':False,'reason':'Existing annotations not found; no new annotation requested'}
    from scripts.dynamic_mechanism_probe import polygon_mask
    data=json.loads(annotation_file.read_text(encoding='utf-8'))
    result=[]
    for case in data['cases']:
        if case['decision']!='include': continue
        q=case['query_index']
        require(queries[q]==case['query_path'],'Existing annotation image identity mismatch')
        for role,score in [('external',masks[q]),('ego',egos[q])]:
            target=polygon_mask(case[role]['polygons']).astype(bool)
            if not target.any(): continue
            pixels=np.repeat(np.repeat(score,14,axis=0),14,axis=1)
            inside=float(pixels[target].mean())
            outside=float(pixels[~target].mean()) if (~target).any() else None
            result.append({'query_index':q,'role':role,'mean_score_inside':inside,
                           'mean_score_outside':outside,'inside_minus_outside':inside-outside if outside is not None else None})
    return {'available':True,'annotation_sha256':sha(annotation_file),'regions':result,
            'scope':'Post-hoc existing coarse polygons only, not fitted labels or motion truth; no IoU threshold selected'}


def cache(a):
    run, db, queries, gt, inputs = load_dataset(a.dataset_root,a.run)
    contract = {'schema':1,'inputs':inputs,'code':code_hashes(),'model':'ViT-B-16','pretrained':'openai',
        'dynamic_text':DYNAMIC,'ego_text':EGO,'static_text':STATIC,'prompt_template':'a photo of a {category}',
        'crop_boxes_grid20':BOXES,'temperature':TEMPERATURE,'seed':42,
        'definition':'sigmoid((max external cosine-max(ego,static) cosine)/0.05), overlap average; not motion probability'}
    # JSON normalisation makes tuples stable across a resume.
    contract = json.loads(json.dumps(contract))
    if a.output.exists():
        require((a.output/'contract.json').is_file(), 'Existing output is not a CLIP cache')
        require(json.loads((a.output/'contract.json').read_text(encoding='utf-8'))==contract, 'Cache contract changed; use a new directory')
    else:
        a.output.mkdir(parents=True)
        write(a.output/'contract.json',contract)
        (a.output/'shards').mkdir()
    missing = []
    for q,path in enumerate(queries):
        file=a.output/'shards'/f'{q:04d}.npz'
        if file.exists():
            shard_read(file,inputs[str(a.dataset_root/path)])
        else:
            missing.append(q)
    require(not (a.output/'completed.json').exists() or not missing, 'Completed cache has missing shards')
    if (a.output/'completed.json').exists():
        done=json.loads((a.output/'completed.json').read_text())
        require(done['complete'] and sha(a.output/'masks.npz')==done['masks_sha256']
                and sha(a.output/'teacher.json')==done['teacher_sha256'],'Completed cache files changed')
        print('Existing CLIP cache verified:',a.output)
        return
    if missing:
        import torch
        from PIL import Image
        from src.models.clip_teacher import CLIPTeacherEncoder
        from src.models.cc_lsa import module_state_sha256
        from scripts.eval_condition_robustness import build_transform
        device=choose_device(a.device)
        teacher=CLIPTeacherEncoder(dynamic_categories=DYNAMIC+EGO+STATIC).to(device).eval()
        state = {'visual_sha256':module_state_sha256(teacher.visual),
                 'text_sha256':hashlib.sha256(teacher.dynamic_text_feats.cpu().numpy().tobytes()).hexdigest()}
        state_file=a.output/'teacher.json'
        if state_file.exists():
            require(json.loads(state_file.read_text())==state, 'Teacher weights changed during resume')
        else:
            write(state_file,state)
        transform=build_transform((280,280))
        with torch.inference_mode():
            for q in missing:
                path=queries[q]
                with Image.open(safe_image(a.dataset_root,path)) as im:
                    image=transform(im.convert('RGB'))
                crops=torch.stack([image[:,y*14:y2*14,x*14:x2*14] for x,y,x2,y2 in BOXES]).to(device)
                cls,_=teacher(crops)
                cosine=(cls.float()@teacher.dynamic_text_feats.float().T).cpu().numpy()
                mask,ego=heatmap(cosine)
                target=a.output/'shards'/f'{q:04d}.npz'
                temp=target.with_suffix('.tmp')
                with temp.open('wb') as f:
                    np.savez_compressed(f,cosines=cosine,mask=mask,ego_mask=ego,image_sha=inputs[str(a.dataset_root/path)])
                temp.replace(target)
                if (q+1)%25==0: print('CLIP query',q+1,'/740',flush=True)
    maps=[shard_read(a.output/'shards'/f'{q:04d}.npz',inputs[str(a.dataset_root/path)]) for q,path in enumerate(queries)]
    masks=np.stack([m[0] for m in maps]); egos=np.stack([m[1] for m in maps])
    np.savez_compressed(a.output/'masks.npz',masks=masks,ego_masks=egos,query_paths=queries)
    write(a.output/'localization_diagnostics.json',localization_diagnostics(masks,egos,queries,a.audit_annotations))
    (a.output/'images').mkdir(exist_ok=True)
    pages=['<!doctype html><meta charset="utf-8"><h1>CLIP自动区域评分</h1><p>红色为连续外部动态倾向，不是运动概率或像素真值。无人工标注要求。</p>']
    chosen=sample_queries(queries,masks.mean((1,2)),4)
    for label,ids in chosen.items():
        for q in ids:
            for role,m in [('external',masks[q]),('ego',egos[q])]:
                filename=f'images/{q:04d}_{role}.jpg'
                save_preview(a.dataset_root,queries[q],m,a.output/filename)
                pages.append('<h2>'+html.escape(f'q{q} {label} {role}')+f'</h2><img src="{filename}">')
    (a.output/'index.html').write_text('\n'.join(pages),encoding='utf-8')
    write(a.output/'completed.json',{'complete':True,'queries':740,'masks_sha256':sha(a.output/'masks.npz'),
          'teacher_sha256':sha(a.output/'teacher.json'),
          'mean_score':float(masks.mean()),'min_score':float(masks.min()),'max_score':float(masks.max())})
    print('Saved CLIP cache:',a.output)


def evaluate(a):
    require(not a.output.exists(),'Evaluation output exists; use a new directory')
    run,db,queries,gt,inputs=load_dataset(a.dataset_root,a.run)
    contract=json.loads((a.cache/'contract.json').read_text(encoding='utf-8'))
    complete=json.loads((a.cache/'completed.json').read_text())
    require(contract['inputs']==inputs and contract['code']==code_hashes(),'CLIP cache contract differs')
    require(complete['complete'] and sha(a.cache/'masks.npz')==complete['masks_sha256'],'Incomplete/changed CLIP cache')
    require(sha(a.cache/'teacher.json')==complete['teacher_sha256'],'CLIP teacher record changed')
    with np.load(a.cache/'masks.npz',allow_pickle=False) as z:
        masks=z['masks'].copy()
        require(z['query_paths'].tolist()==queries,'CLIP query order mismatch')
    require(masks.shape==(740,20,20) and np.isfinite(masks).all() and (masks>=0).all() and (masks<=1).all(),'Invalid CLIP maps')
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
                 a.source/'summary.json',a.cache/'masks.npz',a.cache/'teacher.json')}
    write(a.output/'provenance.json',{'inputs':inputs,'fingerprints':fingerprint,'code':code_hashes(),
          'clip_contract':contract,'beta':.5,'input_strength':.5,'device':str(device),
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
            ds={'baseline':encode(fm),'zero_bias':encode(fm,torch.zeros((1,20,20),device=device))}
            require(np.allclose(ds['baseline'].cpu().numpy()[0],desc[18871+q],atol=2e-5,rtol=2e-4),'Fresh query/cache mismatch')
            require(torch.allclose(ds['baseline'],ds['zero_bias'],atol=2e-5,rtol=2e-4),'Zero bias mismatch')
            for name,mask in controls(masks[q],path).items():
                m=torch.tensor(mask[None],device=device)
                ds[name+'_late']=encode(fm,-.5*m)
                # Soft blend toward ImageNet mean RGB; no generated or recovered background.
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
    for stage in ('late','input'):
        left=[r for r in rows if r['variant']=='aligned_'+stage]
        for control in ('shuffle','rowmean'):
            right=[r for r in rows if r['variant']==control+'_'+stage]
            paired[stage+'_vs_'+control]={
                'aligned_only_correct':[l['query_index'] for l,r in zip(left,right) if l['top1_correct'] and not r['top1_correct']],
                'control_only_correct':[l['query_index'] for l,r in zip(left,right) if r['top1_correct'] and not l['top1_correct']]}
    write(a.output/'summary.json',{'variants':summary,'score_axis_order':VARIANTS,
          'aligned_vs_controls':paired,
          'scope':'Automatic crop-CLS CLIP, no manual labels used; query-only, unchanged full database',
          'limitations':'Uncalibrated semantic score, broad crops, ego competition not verified self-vehicle segmentation; no independent validation'})
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
    p.add_argument('stage',choices=['cache','evaluate'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('.cache/clip_dynamic_msls_v1'))
    p.add_argument('--source',type=Path,default=Path('doc/visual_pair_msls_hard_mix_v2'))
    p.add_argument('--run',type=Path,default=Path('doc/dynamic_category_prior_screen_b0.5_full_db_condition_queries'))
    p.add_argument('--dataset-root',type=Path,default=Path('datasets/msls-val'))
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--audit-annotations',type=Path,default=Path('doc/dynamic_mechanism_cases_v1/annotations_validated.json'))
    p.add_argument('--device',default='cuda:1')
    a=p.parse_args(); {'cache':cache,'evaluate':evaluate}[a.stage](a)


if __name__=='__main__':
    main()
