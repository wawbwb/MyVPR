"""Fixed, manually reviewed query-only intervention; CPU preparation, bounded GPU probe."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_dynamic_coverage import require, sha, safe_image
from scripts.audit_dynamic_mask_regions import load_inputs

CASES = (295, 130, 664, 10, 581, 344, 498, 728)
ROLES = ('external', 'ego')


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def polygon_mask(polygons):
    from PIL import Image, ImageDraw
    image = Image.new('L', (280, 280))
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=float)
        require(points.ndim == 2 and points.shape[1] == 2 and len(points) >= 3 and
                np.isfinite(points).all() and (points >= 0).all() and (points <= 1).all(), 'Invalid polygon')
        draw.polygon([(round(x * 279), round(y * 279)) for x, y in points], fill=255)
    return np.asarray(image, dtype=np.float32) / 255


def pool(mask):
    return mask.reshape(20, 14, 20, 14).mean((1, 3))


def rank(scores, gt):
    require(np.asarray(scores).ndim == 1 and np.isfinite(scores).all(), 'Invalid retrieval scores')
    order = np.argsort(-scores, kind='stable')
    positive = np.isin(order, gt)
    require(positive.any() and (~positive).any(), 'Need positive and negative references')
    return {'top1_reference_index': int(order[0]), 'top1_correct': int(positive[0]),
            'best_gt_rank': int(np.flatnonzero(positive)[0]) + 1,
            'positive_negative_margin': float(scores[order[positive][0]] - scores[order[~positive][0]]),
            'top20_reference_indices': order[:20].tolist()}


def validate_annotations(template, annotations):
    require(annotations['probe_id'] == template['probe_id'] and
            len(annotations['cases']) == len(template['cases']), 'Annotation identity mismatch')
    accepted = []
    for expected, case in zip(template['cases'], annotations['cases']):
        require(case['query_index'] == expected['query_index'] and case['query_path'] == expected['query_path'],
                'Case identity/order changed')
        require(case['decision'] in ('include', 'skip'), 'Every case needs include/skip decision')
        if case['decision'] == 'skip':
            require(bool(case['notes'].strip()), 'Skipped case needs reason')
            continue
        require(case['static_overlap'] == 'yes', 'Included case needs confirmed visible static overlap')
        masks = {}
        for role in ROLES:
            region = case[role]
            require(region['status'] in ('reviewed', 'absent'), 'Each target needs reviewed polygons or explicit absent')
            mask = polygon_mask(region['polygons'])
            require((region['status'] == 'reviewed' and 0 < mask.mean() < .8) or
                    (region['status'] == 'absent' and not region['polygons']), 'Region status/area mismatch')
            masks[role] = mask
        require(not np.any(masks['external'] * masks['ego']), 'External and ego polygons overlap')
        accepted.append((expected, masks))
    require(accepted, 'No included cases')
    return accepted


EDITOR = r'''<!doctype html><meta charset="utf-8"><title>固定案例区域确认</title>
<style>body{font:16px sans-serif;max-width:1100px;margin:auto}canvas{width:560px;height:560px;border:1px solid;cursor:crosshair}img{width:420px}button,select{margin:8px;padding:8px}textarea{width:90%}article{border-bottom:2px solid;padding:20px}</style>
<h1>固定案例：外部动态物体与自车区域</h1><p>只标query。参考图用于确认共同可见静态结构，不编辑GT。
点击query图逐点画多边形，选择类别后点“完成多边形”。避免包含建筑、路面；自车与外部物体不可重叠。
无法确认重叠或区域时选择跳过并说明，不按预测好坏决定。遮挡物后的背景不可恢复。
所有案例必须决定纳入/跳过；每类明确已核验或不存在。页面不自动保存，关闭前导出。</p>
<button id="save">导出 annotations.json</button><input id="load" type="file" accept="application/json"><span id="message"></span><main></main>
<script id="payload" type="application/json">__DATA__</script><script>
let data=JSON.parse(document.getElementById('payload').textContent);
function add(tag,parent,text){let e=document.createElement(tag);if(text)e.textContent=text;parent.append(e);return e;}
function menu(parent,label,options,value,cb){add('span',parent,label);let s=add('select',parent);for(let x of options){let o=add('option',s,x);o.value=x;}s.value=value;s.onchange=()=>cb(s.value);return s;}
function render(){let main=document.querySelector('main');main.replaceChildren();for(let c of data.cases){
let a=add('article',main);add('h2',a,'q'+c.query_index+' '+c.query_path);let cv=add('canvas',a);cv.width=cv.height=560;
let ref=add('img',a);ref.src='images/'+c.query_index+'_gt.jpg';let ctx=cv.getContext('2d'),im=new Image(),pts=[],role='external';
function paint(){ctx.clearRect(0,0,560,560);ctx.drawImage(im,0,0,560,560);for(let r of ['external','ego'])for(let p of c[r].polygons){ctx.beginPath();p.forEach(([x,y],i)=>i?ctx.lineTo(x*560,y*560):ctx.moveTo(x*560,y*560));ctx.closePath();ctx.fillStyle=r==='external'?'#ff000055':'#0066ff55';ctx.fill();}ctx.beginPath();pts.forEach(([x,y],i)=>i?ctx.lineTo(x*560,y*560):ctx.moveTo(x*560,y*560));ctx.strokeStyle='#00dd00';ctx.lineWidth=3;ctx.stroke();}
im.onload=paint;im.src='images/'+c.query_index+'_query.jpg';cv.onclick=e=>{let b=cv.getBoundingClientRect();pts.push([(e.clientX-b.left)/b.width,(e.clientY-b.top)/b.height]);paint();};
add('br',a);menu(a,'绘制类别',['external','ego'],role,v=>{role=v;pts=[];paint();});
let finish=add('button',a,'完成多边形');finish.onclick=()=>{if(pts.length<3)return;c[role].polygons.push(pts);pts=[];paint();};
let undo=add('button',a,'撤销当前点/最后多边形');undo.onclick=()=>{if(pts.length)pts.pop();else c[role].polygons.pop();paint();};
add('br',a);menu(a,'案例决定',['pending','include','skip'],c.decision,v=>c.decision=v);
menu(a,'静态重叠',['unknown','yes','no'],c.static_overlap,v=>c.static_overlap=v);
for(let r of ['external','ego'])menu(a,r+'状态',['pending','reviewed','absent'],c[r].status,v=>c[r].status=v);
add('p',a,'备注：区域依据、视点差异、静态误标、跳过原因');let t=add('textarea',a);t.value=c.notes;t.oninput=()=>c.notes=t.value;}}
document.getElementById('save').onclick=()=>{let url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));let a=document.createElement('a');a.href=url;a.download='annotations.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
document.getElementById('load').onchange=async e=>{try{let v=JSON.parse(await e.target.files[0].text());if(v.probe_id!==data.probe_id||v.cases.length!==data.cases.length||v.cases.some((x,i)=>x.query_index!==data.cases[i].query_index||x.query_path!==data.cases[i].query_path))throw Error('身份不符');data=v;render();}catch(err){document.getElementById('message').textContent=String(err);}};render();</script>'''


def prepare(a):
    require(not a.output.exists(), 'Output already exists')
    paths, masks, ndb, splits, hashes = load_inputs(a.run, a.mask_cache, a.dataset_root)
    split = splits['msls-val']
    cases = []
    from PIL import Image
    for q in CASES:
        path = split['query_paths'][q]
        ref = paths[split['fixed_gt_indices'][q]]
        for rel in (path, ref):
            hashes[str(a.dataset_root / rel)] = sha(safe_image(a.dataset_root, rel))
        cases.append({'query_index': q, 'query_path': path, 'reference_path': ref,
                      'decision': 'pending', 'static_overlap': 'unknown', 'notes': '',
                      **{r: {'status': 'pending', 'polygons': []} for r in ROLES}})
    contract = {'schema': 1, 'cases': list(CASES),
                'case_identity': [{k: c[k] for k in ('query_index', 'query_path', 'reference_path')} for c in cases],
                'inputs': hashes, 'beta': .5,
                'shift_pixels': 140, 'image_size': 280, 'database_intervention': 'none',
                'scope': 'Post-hoc selected-case query-only sensitivity, not an independent benchmark',
                'code_sha256': sha(__file__)}
    data = {'probe_id': hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(), 'cases': cases}
    a.output.mkdir(parents=True)
    (a.output / 'images').mkdir()
    for c in cases:
        for role, rel in [('query', c['query_path']), ('gt', c['reference_path'])]:
            with Image.open(a.dataset_root / rel) as im:
                im.convert('RGB').resize((560, 560)).save(a.output / f"images/{c['query_index']}_{role}.jpg")
    write(a.output / 'contract.json', contract)
    write(a.output / 'template.json', data)
    (a.output / 'index.html').write_text(EDITOR.replace('__DATA__', json.dumps(data).replace('<', '\\u003c')), encoding='utf-8')
    print('Prepared:', a.output, '\nReview polygons and export annotations.json before run.')


def run(a):
    require(not a.output.exists(), 'Output already exists')
    contract = json.loads((a.prepared / 'contract.json').read_text())
    template = json.loads((a.prepared / 'template.json').read_text())
    require(contract['code_sha256'] == sha(__file__), 'Probe code changed; prepare again')
    require(template['probe_id'] == hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(), 'Contract changed')
    require([{k: c[k] for k in ('query_index','query_path','reference_path')} for c in template['cases']] == contract['case_identity'],
            'Template identities changed')
    for path, digest in contract['inputs'].items():
        require(sha(path) == digest, 'Prepared input changed: ' + path)
    annotations = json.loads(a.annotations.read_text(encoding='utf-8'))
    accepted = validate_annotations(template, annotations)
    old = json.loads((a.run / 'run.json').read_text())
    checkpoint = a.checkpoint or Path(old['checkpoint']['path'])
    require(sha(checkpoint) == old['checkpoint']['sha256'], 'Checkpoint differs from original RU')
    source = json.loads((a.source / 'summary.json').read_text())
    require(source['ru_sha256'] == sha(checkpoint), 'Descriptor source checkpoint mismatch')
    db_paths = np.load(a.dataset_root / 'msls_val_dbImages.npy', allow_pickle=False)
    qp = np.load(a.dataset_root / 'msls_val_qImages.npy', allow_pickle=False)
    gt = np.load(a.dataset_root / 'msls_val_gt_25m.npy', allow_pickle=True)
    require(len(db_paths) == 18871 and len(qp) == len(gt) == 740, 'Wrong MSLS split')
    desc = np.load(a.source / 'descriptors.npy', mmap_mode='r', allow_pickle=False)
    require(desc.shape == (19611, 12288), 'Unexpected descriptor cache shape')
    for start in range(0, len(desc), 256):
        block = desc[start:start+256]
        require(np.isfinite(block).all() and np.allclose(np.linalg.norm(block, axis=1), 1, atol=2e-4), 'Invalid descriptor norms')
    with np.load(a.source / 'per_query.npz', allow_pickle=False) as z:
        candidates, cached_scores = z['candidates'], z['ru_scores']
        require(candidates.shape == cached_scores.shape == (740, 20) and
                np.issubdtype(candidates.dtype, np.integer) and (candidates >= 0).all() and (candidates < 18871).all(), 'Invalid candidates')
        for q, ids in enumerate(candidates):
            require(len(set(ids.tolist())) == 20 and np.all(np.diff(cached_scores[q]) <= 1e-6), 'Candidate order invalid')
            require(np.allclose(desc[ids] @ desc[18871+q], cached_scores[q], atol=2e-5, rtol=2e-4), 'Cached scores mismatch')
    # Validate dataset manifests against the original experiment, not filenames alone.
    paths, old_masks, ndb, splits, current_hashes = load_inputs(a.run, a.mask_cache, a.dataset_root)
    require(all(contract['inputs'].get(p) == digest for p, digest in current_hashes.items()), 'Run input paths or hashes differ from preparation')
    import torch
    from PIL import Image
    from scripts.eval_condition_robustness import build_transform, load_inference_model_from_ckpt
    from scripts.eval_dynamic_category_prior import extract_ru_feature_map, boq_descriptor
    require(a.device == 'cpu' or a.device.startswith('cuda:') and a.device != 'cuda:0', 'Use explicit cuda:1 (GPU0 faulty), or cpu')
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(a.device)
    model = load_inference_model_from_ckpt(checkpoint, device).eval()
    transform = build_transform((280, 280))
    a.output.mkdir(parents=True)
    fingerprints = {str(p): sha(p) for p in (a.source/'descriptors.npy', a.source/'per_query.npz', a.source/'summary.json', a.annotations, checkpoint)}
    write(a.output/'provenance.json', {'contract': contract, 'input_sha256': fingerprints, 'device': str(device),
          'annotations': annotations, 'source_validation': 'checkpoint SHA, all norms and 740 cached top20 score rows; fresh selected query/GT descriptor checks'})
    all_rows, score_arrays, descriptors, mask_arrays = [], {}, {}, {}
    with torch.inference_mode():
        for case, regions in accepted:
            q = case['query_index']
            require(str(qp[q]).replace('\\', '/') == case['query_path'], 'Query path mismatch')
            ids = np.asarray(gt[q]).reshape(-1)
            require(ids.size and np.issubdtype(ids.dtype, np.integer) and (ids >= 0).all() and (ids < ndb).all(), 'Invalid GT')
            with Image.open(safe_image(a.dataset_root, case['query_path'])) as im:
                image = transform(im.convert('RGB')).unsqueeze(0).to(device)
            fm = extract_ru_feature_map(model, image)
            def encode(f, bias=None):
                return torch.nn.functional.normalize(boq_descriptor(model.aggregator, f, bias).float(), dim=-1)
            baseline = encode(fm)
            require(np.allclose(baseline.cpu().numpy()[0], desc[ndb+q], atol=2e-5, rtol=2e-4), 'Fresh query does not reproduce cache')
            ref_id = int(ids.min())
            with Image.open(safe_image(a.dataset_root, str(db_paths[ref_id]))) as im:
                ref_image = transform(im.convert('RGB')).unsqueeze(0).to(device)
            fresh_ref = encode(extract_ru_feature_map(model, ref_image)).cpu().numpy()[0]
            require(np.allclose(fresh_ref, desc[ref_id], atol=2e-5, rtol=2e-4), 'Fresh GT does not reproduce DB cache')
            query_mask = old_masks[splits['msls-val']['query_cache_indices'][q]]
            variants = {'baseline': baseline, 'zero_bias': encode(fm, torch.zeros((1,20,20),device=device)),
                        'old_late': encode(fm, -.5*torch.tensor(query_mask[None],device=device))}
            require(torch.allclose(variants['zero_bias'], baseline, atol=2e-5, rtol=2e-4), 'Zero-bias path does not reproduce baseline')
            mask_info = {}
            for role, mask in regions.items():
                mask_info[role] = {'area': float(mask.mean()), 'shift_overlap_fraction_of_image': float((mask*np.roll(mask,140,axis=1)).mean())}
                for control, m in [('aligned', mask), ('shift', np.roll(mask, 140, axis=1))]:
                    label = role+'_'+control
                    mask_arrays[f'q{q}_{label}'] = m
                    variants[label+'_late'] = encode(fm, -.5*torch.tensor(pool(m)[None],device=device))
                    # Zero in normalised image space is the fixed normalisation mean RGB, not recovered background.
                    altered = image * (1-torch.tensor(m[None,None],device=device))
                    variants[label+'_input'] = encode(extract_ru_feature_map(model, altered))
            vectors = np.concatenate([v.cpu().numpy() for v in variants.values()])
            scores = vectors @ np.asarray(desc[:ndb]).T
            baseline_result = rank(scores[0], ids)
            require(baseline_result['top1_reference_index'] == int(candidates[q, 0]),
                    'Fresh baseline top1 differs from saved RU; inspect cache/numerics before interpreting interventions')
            for i, name in enumerate(variants):
                result = rank(scores[i], ids)
                row = {'query_index': q, 'query_path': case['query_path'], 'variant': name, **result,
                       'correction': int(not baseline_result['top1_correct'] and result['top1_correct']),
                       'regression': int(baseline_result['top1_correct'] and not result['top1_correct']),
                       'margin_delta': result['positive_negative_margin']-baseline_result['positive_negative_margin'],
                       'descriptor_l2_from_baseline': float(np.linalg.norm(vectors[i]-vectors[0])), 'mask_info': mask_info}
                all_rows.append(row)
                score_arrays[f'q{q}_{name}'] = scores[i]
                descriptors[f'q{q}_{name}'] = vectors[i]
            print('query', q, 'baseline_correct', baseline_result['top1_correct'], flush=True)
    write(a.output/'query_outcomes.json', all_rows)
    with (a.output/'query_outcomes.csv').open('w', newline='', encoding='utf-8') as f:
        fields = ['query_index','query_path','variant','top1_reference_index','top1_correct','best_gt_rank','positive_negative_margin','margin_delta','correction','regression','descriptor_l2_from_baseline']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(all_rows)
    np.savez_compressed(a.output/'scores.npz', **score_arrays)
    np.savez_compressed(a.output/'descriptors.npz', **descriptors)
    np.savez_compressed(a.output/'intervention_masks.npz', **mask_arrays)
    summary = {}
    for name in dict.fromkeys(r['variant'] for r in all_rows):
        selected = [r for r in all_rows if r['variant'] == name]
        summary[name] = {'n': len(selected), 'correct': sum(r['top1_correct'] for r in selected),
                         'corrections': sum(r['correction'] for r in selected),
                         'regressions': sum(r['regression'] for r in selected)}
        for role in ROLES:
            if name.startswith(role+'_'):
                summary[name]['nonempty_target_n'] = sum(r['mask_info'][role]['area'] > 0 for r in selected)
    write(a.output/'summary.json', {'scope': contract['scope'], 'variants': summary,
          'limitations': 'Query-only, fixed original DB. Manual post-hoc cases; no significance or generalisation claim. Input fill is not recovered static background. Shift wraps and may overlap target.'})
    write(a.output/'completed.json', {'complete': True, 'queries': len(accepted), 'scope': contract['scope'], 'full_benchmark_complete': False})
    print('Saved:', a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'run'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepared', type=Path)
    p.add_argument('--annotations', type=Path)
    p.add_argument('--source', type=Path, default=Path('doc/visual_pair_msls_hard_mix_v2'))
    p.add_argument('--run', type=Path, default=Path('doc/dynamic_category_prior_screen_b0.5_full_db_condition_queries'))
    p.add_argument('--mask-cache', type=Path, default=Path('.cache/dynamic_prior/msls_val_full_db_condition_union_deeplabv3_mbv3_grid20.npz'))
    p.add_argument('--dataset-root', type=Path, default=Path('datasets/msls-val'))
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--device', default='cuda:1')
    a = p.parse_args()
    if a.stage == 'run' and (a.prepared is None or a.annotations is None):
        p.error('run requires --prepared and --annotations')
    {'prepare': prepare, 'run': run}[a.stage](a)


if __name__ == '__main__':
    main()
