"""CPU analysis of fixed retrieval anchors and automatic failure case pages."""
import argparse
import hashlib
import html
import json
import shutil
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.clip_dynamic_screen import require, sha, write, safe_image, save_preview


def anchor_changes(scores, positive, negative):
    s=np.asarray(scores)
    require(s.ndim==2 and s.shape[0]==3 and np.isfinite(s).all(), 'Invalid scores')
    require(0<=positive<s.shape[1] and 0<=negative<s.shape[1] and positive!=negative, 'Invalid anchors')
    return [{'positive_score':float(v[positive]), 'negative_score':float(v[negative]),
             'positive_delta':float(v[positive]-s[0,positive]),
             'negative_delta':float(v[negative]-s[0,negative]),
             'fixed_margin':float(v[positive]-v[negative]),
             'fixed_margin_delta':float((v[positive]-v[negative])-(s[0,positive]-s[0,negative]))}
            for v in s]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,default=Path('doc/clip_external_screen_v1'))
    p.add_argument('--cache',type=Path,default=Path('.cache/clip_external_positive_v1'))
    p.add_argument('--dataset-root',type=Path,help='Optional original MSLS images/manifests for complete case pages')
    p.add_argument('--output',type=Path,default=Path('doc/clip_failure_cases_v1'))
    a=p.parse_args(); require(not a.output.exists(),'Output exists; choose a new directory')
    read=lambda f:json.loads(f.read_text(encoding='utf-8'))
    done=read(a.evaluation/'completed.json'); provenance=read(a.evaluation/'provenance.json')
    variants=['baseline','aligned_input','shuffle_input']
    require(done['complete'] and done['queries']==740 and done['variants']==variants,'Wrong evaluation')
    cd=read(a.cache/'completed.json'); cc=read(a.cache/'contract.json')
    require(cd['complete'] and sha(a.cache/'masks.npz')==cd['masks_sha256'] and
            sha(a.cache/'contract.json')==cd['contract_sha256'],'Changed cache')
    require(cc['inputs']==provenance['clip_contract']['inputs'] and cc['mapping']==provenance['clip_contract']['mapping'] and
            cc['old_shards_sha256']==provenance['clip_contract']['old_shards_sha256'],'Different CLIP cache')
    with np.load(a.cache/'masks.npz',allow_pickle=False) as z:
        masks=z['masks'].copy(); paths=z['query_paths'].tolist()
    rows=read(a.evaluation/'query_outcomes.json'); by={(r['query_index'],r['variant']):r for r in rows}
    require(len(rows)==len(by)==2220 and len(paths)==740,'Incomplete outcomes')
    scores=np.load(a.evaluation/'scores.npy',mmap_mode='r',allow_pickle=False)
    require(scores.shape==(740,3,18871),'Wrong score shape')
    selected=[]
    for q in range(740):
        for i,v in enumerate(variants):
            r=by[q,v]
            require(r['query_path']==paths[q] and abs(r['mean_clip_score']-float(masks[q].mean()))<1e-7,'Query/mask mismatch')
            require(np.isfinite(scores[q,i]).all() and np.argsort(-scores[q,i],kind='stable')[:20].tolist()==r['top20'],'Score/order mismatch')
        if not by[q,'baseline']['top1_correct'] and masks[q].any(): selected.append(q)
    db=None
    if a.dataset_root:
        f=a.dataset_root/'msls_val_dbImages.npy'
        expected=[v for k,v in cc['inputs'].items() if k.replace(chr(92),'/').endswith('/msls_val_dbImages.npy')]
        require(expected==[sha(f)],'DB manifest changed')
        db=np.load(f,allow_pickle=False).tolist(); require(len(db)==18871,'Invalid DB manifest')
    a.output.mkdir(parents=True); (a.output/'images').mkdir()
    results=[]; page=['<!doctype html><meta charset="utf-8"><title>CLIP失败查询分析</title>',
        '<style>body{font:16px sans-serif;max-width:1250px;margin:auto}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:6px}section{margin:40px 0}</style>',
        '<h1>基线错误且实际施加抑制的全部查询</h1><p>按 query 编号排序，不按实验收益挑选。无需标注。比较固定的基线最佳 GT 和基线错误 top1，避免换候选造成分数解释混乱。GT 归属沿用原评测记录。</p>']
    for q in selected:
        b=by[q,'baseline']; pos=b['best_gt_reference_index']; neg=b['top1_reference_index']
        changes=anchor_changes(scores[q],pos,neg)
        result={'query_index':q,'query_path':paths[q],'fixed_positive':pos,'fixed_negative':neg,
                'mean_mask':float(masks[q].mean()),'nonzero_grid_fraction':float((masks[q]>0).mean()),
                'variants':{v:{**changes[i],'best_gt_rank':by[q,v]['best_gt_rank'],
                    'dynamic_margin':by[q,v]['positive_negative_margin'],'top1_reference_index':by[q,v]['top1_reference_index']}
                    for i,v in enumerate(variants)}}
        results.append(result)
        page.append(f'<section><h2>q{q}</h2><p>{html.escape(paths[q])}</p><p>mask均值 {result["mean_mask"]:.4f}；非零网格 {result["nonzero_grid_fraction"]:.1%}（不是动态物体面积）</p>')
        for role,idx in [('query',None),('baseline_best_GT',pos),('baseline_top1',neg)]:
            target=a.output/'images'/f'{q}_{role}.jpg'; existing=a.evaluation/'images'/target.name
            if a.dataset_root:
                if role=='query':
                    expected=[v for k,v in cc['inputs'].items() if k.replace(chr(92),'/').endswith('/'+paths[q])]
                    require(expected==[sha(safe_image(a.dataset_root,paths[q]))],'Query image changed')
                    save_preview(a.dataset_root,paths[q],masks[q],target)
                else:
                    from PIL import Image
                    with Image.open(safe_image(a.dataset_root,db[idx])) as im: im.convert('RGB').resize((400,300)).save(target)
            elif existing.is_file(): shutil.copyfile(existing,target)
            if target.exists(): page.append(f'<p>{role}</p><img src="images/{target.name}">')
            else: page.append(f'<p>{role}：本机缺少原图，训练机带 --dataset-root 可生成。</p>')
        page.append('<table><tr><th>方案</th><th>固定GT分数变化</th><th>固定错误分数变化</th><th>固定差值变化</th><th>最佳GT排名</th></tr>')
        for i,v in enumerate(variants):
            c=changes[i]; page.append(f'<tr><td>{v}</td><td>{c["positive_delta"]:+.6f}</td><td>{c["negative_delta"]:+.6f}</td><td>{c["fixed_margin_delta"]:+.6f}</td><td>{by[q,v]["best_gt_rank"]}</td></tr>')
        page.append('</table></section>')
    summary={}
    for v in variants[1:]:
        cs=[r['variants'][v] for r in results]
        summary[v]={'positive_up':sum(c['positive_delta']>0 for c in cs),
                    'positive_down':sum(c['positive_delta']<0 for c in cs),
                    'negative_up':sum(c['negative_delta']>0 for c in cs),
                    'negative_down':sum(c['negative_delta']<0 for c in cs),
                    'fixed_margin_improved':sum(c['fixed_margin_delta']>0 for c in cs),
                    'fixed_margin_worsened':sum(c['fixed_margin_delta']<0 for c in cs),
                    'beat_baseline_wrong_anchor':sum(c['fixed_margin']>0 for c in cs)}
    write(a.output/'report.json',{'selection':'all baseline R@1 failures with nonzero mask','n':len(results),
          'summary':summary,'cases':results,'script_sha256':sha(__file__),
          'inputs':{str(f):sha(f) for f in [a.evaluation/'scores.npy',a.evaluation/'query_outcomes.json',a.cache/'masks.npz']},
          'limitations':'Fixed anchors are not the best positive/negative after intervention. No pixel GT or visual correctness inferred from scores. No model inference.'})
    (a.output/'index.html').write_text('\n'.join(page),encoding='utf-8')
    write(a.output/'completed.json',{'complete':True,'cases':len(results),'image_files':len(list((a.output/'images').glob('*.jpg')))})
    print(json.dumps(summary,indent=2)); print('Saved:',a.output)


if __name__=='__main__': main()
