"""CPU-only spatial audit of existing masks; never infer motion or rerun retrieval."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_dynamic_coverage import GROUPS, group, require, sample_queries, save_preview, sha


def spatial_stats(mask):
    """Fixed bands: top 50%, middle 25%, bottom 25%; all areas use full-image denominator."""
    mask = np.asarray(mask, dtype=np.float64)
    require(mask.shape == (20, 20), 'Expected a 20x20 mask')
    require(np.isfinite(mask).all() and (mask >= 0).all() and (mask <= 1).all(),
            'Invalid patch fractions')
    total = float(mask.mean())
    bands = {'top_half': float(mask[:10].sum() / 400),
             'middle_quarter': float(mask[10:15].sum() / 400),
             'bottom_quarter': float(mask[15:].sum() / 400)}
    return {'coverage': total, 'coverage_group': group(total),
            'row_mean_coverage': mask.mean(axis=1).tolist(),
            'band_area_fraction_of_image': bands,
            'bottom_share_of_mask': bands['bottom_quarter'] / total if total else None,
            'above_bottom_area_fraction_of_image': float(mask[:15].sum() / 400)}


def summarize(records):
    return {label: {'n': len(selected),
                   'mean_coverage': float(np.mean([r['coverage'] for r in selected])) if selected else None,
                   'mean_bottom_area_fraction_of_image': float(np.mean([
                       r['band_area_fraction_of_image']['bottom_quarter'] for r in selected])) if selected else None,
                   'mean_row_coverage': np.mean([r['row_mean_coverage'] for r in selected], axis=0).tolist() if selected else None,
                   'bottom_share_ge_half_count': sum(r['bottom_share_of_mask'] is not None
                                                    and r['bottom_share_of_mask'] >= .5 for r in selected)}
            for label in ('all', *GROUPS)
            for selected in [[r for r in records if label == 'all' or r['coverage_group'] == label]]}


def load_inputs(run_dir, cache, root):
    run_file = run_dir / 'run.json'
    run = json.loads(run_file.read_text(encoding='utf-8'))
    require(run['schema_version'] == 3 and
            run['method'] == 'frozen_dynamic_category_negative_attention_prior', 'Need dynamic-prior v3 run')
    hashes = {str(run_file): sha(run_file), str(cache): sha(cache)}
    require(hashes[str(cache)] == run['mask_cache']['sha256'], 'Mask cache SHA mismatch')
    with np.load(cache, allow_pickle=False) as z:
        paths = [str(s).replace('\\', '/') for s in z['image_paths']]
        masks = np.asarray(z['masks'], dtype=np.float32)
        ndb = int(z['num_references'])
    require(ndb == run['mask_cache']['num_references'] and 0 < ndb < len(paths), 'DB count mismatch')
    require(len(paths) == len(set(paths)), 'Duplicate cache paths')
    require(masks.shape == (len(paths), 20, 20), 'Mask shape mismatch')
    require(np.isfinite(masks).all() and (masks >= 0).all() and (masks <= 1).all(), 'Invalid masks')
    db_file = root / 'msls_val_dbImages.npy'
    hashes[str(db_file)] = sha(db_file)
    db = [str(s).replace('\\', '/') for s in np.load(db_file, allow_pickle=False)]
    require(paths[:ndb] == db, 'DB path/order mismatch')
    lookup = {s: i for i, s in enumerate(paths)}
    splits = {}
    for split in run['datasets']:
        name = split['name']
        require(name not in splits, 'Duplicate split')
        files = {}
        for role, info in split['manifests'].items():
            file = root / Path(info['path']).name
            hashes[str(file)] = sha(file)
            require(hashes[str(file)] == info['sha256'], 'Manifest SHA mismatch: ' + str(file))
            files[role] = file
        queries = [str(s).replace('\\', '/') for s in np.load(files['queries'], allow_pickle=False)]
        gt = np.load(files['ground_truth'], allow_pickle=True)  # Trusted local dataset GT only.
        require(len(queries) == len(gt) == split['num_queries'] and len(set(queries)) == len(queries),
                'Query count/identity mismatch')
        require(all(s in lookup and lookup[s] >= ndb for s in queries), 'Query missing from cache')
        refs = []
        for values in gt:
            ids = np.asarray(values).reshape(-1)
            require(ids.size > 0 and np.issubdtype(ids.dtype, np.integer) and
                    (ids >= 0).all() and (ids < ndb).all(), 'Invalid GT')
            refs.append(int(ids.min()))
        splits[name] = {'protocol': split['protocol'], 'query_paths': queries,
                        'query_cache_indices': [lookup[s] for s in queries], 'fixed_gt_indices': refs}
    return paths, masks, ndb, splits, hashes


PAGE = r'''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>动态掩码空间与语义核验</title>
<style>body{max-width:1100px;margin:24px auto;font:16px sans-serif;background:#fafafa;color:#222}
img{max-width:100%}article{background:white;padding:20px;margin:20px 0;border:1px solid #ccc}
label{display:block;margin:10px 0}select{margin-left:12px}textarea{width:95%;height:65px}
button{padding:12px}pre{white-space:pre-wrap}.warning{background:#fff1cd;padding:16px}</style>
<h1>动态掩码空间与语义核验</h1>
<p class="warning">仅旧掩码与图片，不含检索预测、准确率或因果结论。底部25%只是固定空间带，
不是自车真值，不会裁图或改掩码。原图与叠加图上下位置一致；蓝线标出75%高度。</p>
<p>按旧覆盖率分组、路径哈希固定抽样。每图左侧原图，右侧旧掩码。
GT固定选最小索引，未按外观或预测挑选，不保证可见重叠。
请将外部动态物体、自车区域、静态误标分开填写。无法判断时保留“未判定”。
20×20掩码不能作为像素分割真值。车辆不同也不自动证明物理消失。</p>
<p>填写后点击导出。刷新或关闭前务必导出；页面不自动保存。导入可继续已有标注。</p>
<button id="export">导出 review_annotations.json</button>
<input id="import" type="file" accept="application/json"><span id="status"></span>
<details><summary>空间统计与来源说明</summary><pre id="summary"></pre></details>
<main id="samples"></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent);
const imageFields={external_dynamic_visible:'可见外部车辆/行人等',external_dynamic_missed:'外部动态物体明显漏检',
 ego_region_selected:'自车车体/内饰被选中',static_region_selected:'静态结构疑似误标',
 boundary_mixing:'明显patch边界混合'};
const pairFields={visible_static_overlap:'存在可辨识的共同静态结构',
 apparent_external_dynamic_difference:'共同场景中外部动态物体外观/位置有差异',
 viewpoint_difference:'明显视点差异'};
const yn=[['unknown','未判定'],['yes','是'],['no','否']];
function el(tag,text,parent){let x=document.createElement(tag);if(text!==undefined)x.textContent=text;if(parent)parent.append(x);return x;}
function fields(parent,obj,labels){for(const [key,label] of Object.entries(labels)){
 const l=el('label',label,parent), s=el('select',undefined,l);
 for(const [v,t] of yn){let o=el('option',t,s);o.value=v;}s.value=obj[key];s.onchange=()=>obj[key]=s.value;
}const l=el('label','备注（指出位置、证据及不确定性）',parent),t=el('textarea',undefined,l);
t.value=obj.notes;t.oninput=()=>obj.notes=t.value;}
function render(){document.getElementById('samples').replaceChildren();
for(const r of data.reviews){const a=el('article',undefined,document.getElementById('samples'));
el('h2',r.dataset+' q'+r.query_index+' / '+r.group,a);
for(const im of r.images){el('h3',im.role+' · '+im.path,a);
let s=im.spatial;el('p','整图覆盖 '+(100*s.coverage).toFixed(2)+'%；底部掩码占整图 '+
(100*s.band_area_fraction_of_image.bottom_quarter).toFixed(2)+'%；底部占全部掩码 '+
(s.bottom_share_of_mask===null?'无掩码':(100*s.bottom_share_of_mask).toFixed(1)+'%'),a);
let img=el('img',undefined,a);img.src=im.preview;img.loading='lazy';fields(a,im.annotation,imageFields);}
fields(a,r.annotation,pairFields);}}
document.getElementById('summary').textContent=JSON.stringify(data.summary,null,2);
document.getElementById('export').onclick=()=>{let b=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});
let u=URL.createObjectURL(b),a=el('a');a.href=u;a.download='review_annotations.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);};
document.getElementById('import').onchange=async e=>{try{let incoming=JSON.parse(await e.target.files[0].text());
if(incoming.audit_id!==data.audit_id||incoming.reviews.length!==data.reviews.length)throw Error('来源或样本不匹配');
const assignments=[];
function validate(dst,src,labels){if(!src||Object.keys(labels).some(k=>!['unknown','yes','no'].includes(src[k]))||typeof src.notes!=='string')throw Error('标注格式不符');
assignments.push([dst,Object.fromEntries([...Object.keys(labels),'notes'].map(k=>[k,src[k]]))]);}
for(let i=0;i<data.reviews.length;i++){let r=data.reviews[i],v=incoming.reviews[i];
if(r.dataset!==v.dataset||r.query_index!==v.query_index||r.images.length!==v.images.length)throw Error('样本不匹配');
validate(r.annotation,v.annotation,pairFields);
for(let j=0;j<r.images.length;j++){if(r.images[j].path!==v.images[j].path)throw Error('图片不匹配');validate(r.images[j].annotation,v.images[j].annotation,imageFields);}}
for(const [dst,src] of assignments)Object.assign(dst,src);render();document.getElementById('status').textContent='已导入';
}catch(err){document.getElementById('status').textContent='导入失败：'+err.message;}};
render();
</script></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('doc/dynamic_category_prior_screen_b0.5_full_db_condition_queries'))
    parser.add_argument('--mask-cache', type=Path, default=Path('.cache/dynamic_prior/msls_val_full_db_condition_union_deeplabv3_mbv3_grid20.npz'))
    parser.add_argument('--dataset-root', type=Path, default=Path('datasets/msls-val'))
    parser.add_argument('--output', type=Path, default=Path('doc/dynamic_mask_regions_audit_v1'))
    parser.add_argument('--samples-per-group', type=int, default=4)
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'Choose a new output directory; reports are never overwritten')
    require(0 <= args.samples_per_group <= 10, 'samples-per-group must be 0..10')
    paths, masks, ndb, splits, hashes = load_inputs(args.run, args.mask_cache, args.dataset_root)
    stats = [dict(cache_index=i, path=p, **spatial_stats(masks[i])) for i, p in enumerate(paths)]
    summaries, reviews, per_query = {}, [], {}
    for name, split in splits.items():
        indices = split['query_cache_indices']
        records = [stats[i] for i in indices]
        summaries[name] = {'protocol': split['protocol'], 'strata': summarize(records)}
        per_query[name] = [dict(query_index=q, query_cache_index=i,
                               fixed_gt_reference_index=split['fixed_gt_indices'][q]) for q, i in enumerate(indices)]
        for label, chosen in sample_queries(split['query_paths'], [r['coverage'] for r in records], args.samples_per_group).items():
            for q in chosen:
                reviews.append({'dataset': name, 'query_index': q, 'group': label,
                    'annotation': dict.fromkeys(('visible_static_overlap', 'apparent_external_dynamic_difference', 'viewpoint_difference'), 'unknown'),
                    'images': [{'role': role, 'path': paths[i], 'cache_index': i,
                                'preview': f'images/{i:06d}.jpg', 'spatial': stats[i],
                                'annotation': dict.fromkeys(('external_dynamic_visible', 'external_dynamic_missed',
                                    'ego_region_selected', 'static_region_selected', 'boundary_mixing'), 'unknown')}
                               for role, i in [('query', indices[q]), ('fixed_GT_min_index', split['fixed_gt_indices'][q])]]})
    for r in reviews:
        r['annotation']['notes'] = ''
        for im in r['images']:
            im['annotation']['notes'] = ''
    # Check selected images before creating an output directory.
    from scripts.audit_dynamic_coverage import safe_image
    selected = sorted({im['cache_index'] for r in reviews for im in r['images']})
    for i in selected:
        safe_image(args.dataset_root, paths[i])
    union_indices = sorted({i for split in splits.values() for i in split['query_cache_indices']})
    summary = {'mode': 'spatial_mask_only', 'performance_audit_complete': False,
               'scope': 'Descriptive, post-hoc spatial audit. Bottom band is not ego ground truth. No masks or predictions changed.',
               'bands_rows': {'top_half': [0, 10], 'middle_quarter': [10, 15], 'bottom_quarter': [15, 20]},
               'interval_convention': 'start inclusive, end exclusive',
               'datasets': summaries, 'unique_query_union': summarize([stats[i] for i in union_indices]),
               'database': summarize(stats[:ndb]), 'review_pairs': len(reviews),
               'review_unique_query_paths': len({r['images'][0]['path'] for r in reviews}),
               'note': 'Condition queries overlap. Unique paths are not independent sequences. Zero-mask bottom share is null.'}
    provenance = {'input_sha256': hashes, 'script_sha256': sha(__file__),
                  'helper_sha256': sha(Path(__file__).with_name('audit_dynamic_coverage.py')),
                  'sampling': 'sha256(42:path), fixed coverage bins, no retrieval outcomes',
                  'samples_per_group': args.samples_per_group}
    import hashlib
    audit_id = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    data = {'audit_id': audit_id, 'summary': summary, 'reviews': reviews}
    args.output.mkdir(parents=True)
    (args.output / 'images').mkdir()
    from PIL import Image, ImageDraw
    for i in selected:
        dest = args.output / f'images/{i:06d}.jpg'
        save_preview(args.dataset_root, paths[i], masks[i], dest)
        with Image.open(dest) as im:
            canvas = im.copy()
        ImageDraw.Draw(canvas).line((0, 225, 799, 225), fill=(0, 140, 255), width=2)
        canvas.save(dest, quality=90)
    for filename, value in [('summary.json', summary), ('spatial_per_image.json', stats),
                            ('query_reference_map.json', per_query), ('review_samples.json', data),
                            ('provenance.json', provenance)]:
        (args.output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    embedded = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c').replace('&', '\\u0026')
    (args.output / 'index.html').write_text(PAGE.replace('__DATA__', embedded), encoding='utf-8')
    (args.output / 'completed.json').write_text(json.dumps({'mask_audit_complete': True,
        'performance_audit_complete': False, 'audit_id': audit_id}), encoding='utf-8')
    for name, value in summaries.items():
        print(name)
        for label, row in value['strata'].items():
            print(f"  {label}: n={row['n']}, bottom_share>=50%: {row['bottom_share_ge_half_count']}")
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
