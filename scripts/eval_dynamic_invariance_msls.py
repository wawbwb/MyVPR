"""Full, clean MSLS-val regression of existing LoRA checkpoints; no training."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.clip_token_drop import Cache,read,write,sha,seed,device,image_tensor
from scripts.train_dynamic_invariance import build,checkpoints,codes,MODES
from scripts.clip_dynamic_screen import load_dataset,outcome


def summarize(rows,nqueries):
    modes=['frozen',*MODES]
    by={(r['query_index'],r['variant']):r for r in rows}
    if len(by)!=len(rows) or set(by)!={(q,m) for q in range(nqueries) for m in modes}:
        raise ValueError('Missing or duplicate query outcomes')
    results={}
    for mode in modes:
        results[mode]={'correct':sum(by[q,mode]['top1_correct'] for q in range(nqueries)),
            'recall':{f'R@{k}':sum(by[q,mode]['best_gt_rank']<=k for q in range(nqueries))/nqueries
                      for k in [1,5,10,20]}, 'comparisons':{}}
        for ref in modes:
            if ref==mode: continue
            corrected=[q for q in range(nqueries) if by[q,mode]['top1_correct'] and not by[q,ref]['top1_correct']]
            regressed=[q for q in range(nqueries) if not by[q,mode]['top1_correct'] and by[q,ref]['top1_correct']]
            results[mode]['comparisons'][ref]={'corrected':corrected,'regressed':regressed,
                                               'net':len(corrected)-len(regressed)}
    return {'queries':nqueries,'models':results,
            'scope':'All original query and reference images. Historical regression, not independent validation.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('doc/dynamic_invariance_msls_v1'))
    p.add_argument('--cache',type=Path,default=Path('.cache/clearclip_token_drop_msls_v1'))
    p.add_argument('--runs',type=Path,default=Path('doc/dynamic_invariance_train_v1'))
    p.add_argument('--plan',type=Path,default=Path('doc/dynamic_invariance_plan_v1'))
    p.add_argument('--run',type=Path,default=Path('doc/dynamic_category_prior_screen_b0.5_full_db_condition_queries'))
    p.add_argument('--baseline-reference',type=Path,
                   default=Path('doc/clip_token_drop_eval_v1/report/query_outcomes.json'))
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--msls-root',type=Path,default=Path('datasets/msls-val'))
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--batch-size',type=int,default=4)
    a=p.parse_args()
    if a.batch_size<1: p.error('batch-size must be positive')
    if a.output.exists(): raise ValueError('Output exists; use a new directory')
    import torch
    from src.models.dynamic_invariance import disabled,restore
    seed(42); d=device(a.device)
    cached=Cache(a.cache)
    if cached.contract['split']!='msls': raise ValueError('Need MSLS cache for identities/hashes only')
    _,db,queries,gt,inputs=load_dataset(a.msls_root,a.run)
    if [r[0] for r in cached.records]!=db+queries: raise ValueError('Cache/index order mismatch')
    info=cached.contract['info']
    if info['ndb']!=len(db) or info['gt']!=[g.tolist() for g in gt]: raise ValueError('Cached GT differs')
    model,_,digest=build(a,d); states,contracts=checkpoints(a,model,digest)
    plan=read(a.plan/'plan.json'); plan_done=read(a.plan/'completed.json')
    if (not plan_done['complete'] or sha(a.plan/'plan.json')!=plan_done['plan_sha256']
        or contracts['plain']['plan_sha256']!=plan_done['plan_sha256']
        or plan['code']!=codes()): raise ValueError('Training plan mismatch')
    previous=[r for r in read(a.baseline_reference) if r['variant']=='frozen']
    if len(previous)!=740 or {r['query_index'] for r in previous}!=set(range(740)):
        raise ValueError('Expected previous frozen RU outcomes')
    previous={r['query_index']:r for r in previous}
    if sum(r['top1_correct'] for r in previous.values())!=675: raise ValueError('Wrong baseline reference')
    out=a.output; out.mkdir(parents=True); report=out/'report'; report.mkdir()
    provenance={'training':contracts,'training_checkpoint_hashes':{m:sha(a.runs/m/'last.pt') for m in MODES},
        'init_sha256':digest,'training_code':codes(),'evaluation_sha256':sha(Path(__file__)),
        'cache_sha256':cached.hash,'baseline_reference_sha256':sha(a.baseline_reference),
        'inputs':inputs,'device':str(d),'gpu':torch.cuda.get_device_name(d),'torch':torch.__version__,
        'batch_size':a.batch_size,'scope':'Clean MSLS full database, fixed final checkpoints; no CLIP inference or masks.'}
    write(report/'provenance.json',provenance)
    rows=[]; arrays={}
    n=len(cached.records); ndb=len(db); nq=len(queries)
    for mode in ['frozen',*MODES]:
        if mode!='frozen': restore(model,states[mode])
        descriptor_file=out/f'{mode}_descriptors.npy'
        descriptors=np.lib.format.open_memmap(descriptor_file,mode='w+',dtype='float32',shape=(n,12288))
        with torch.inference_mode(), (disabled(model) if mode=='frozen' else nullcontext()):
            for start in range(0,n,a.batch_size):
                end=min(start+a.batch_size,n)
                tensors=[image_tensor(a.msls_root,cached.records[i][0],cached.image_hashes[i])[0]
                         for i in range(start,end)]
                vectors=model(torch.stack(tensors).to(d)).cpu().numpy()
                if (vectors.shape!=(end-start,12288) or not np.isfinite(vectors).all()
                    or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=2e-4)):
                    raise ValueError('Invalid descriptor')
                descriptors[start:end]=vectors
                if start//128!=(end)//128 or end==n: print(f'{mode}: {end}/{n}',flush=True)
        descriptors.flush()
        score_file=out/f'{mode}_scores.npy'
        scores_saved=np.lib.format.open_memmap(score_file,mode='w+',dtype='float32',shape=(nq,ndb))
        database=torch.from_numpy(np.array(descriptors[:ndb])).to(d)
        mode_rows=[]
        with torch.inference_mode():
            for start in range(0,nq,16):
                scores=(torch.from_numpy(np.array(descriptors[ndb+start:ndb+min(start+16,nq)])).to(d)@database.T).cpu().numpy()
                scores_saved[start:start+len(scores)]=scores
                for q,s in enumerate(scores,start):
                    mode_rows.append({'query_index':q,'query_path':queries[q],'variant':mode,**outcome(s,gt[q])})
        scores_saved.flush(); del database
        write(report/f'{mode}_outcomes.json',mode_rows)
        if mode=='frozen':
            if sum(r['top1_correct'] for r in mode_rows)!=675: raise ValueError('Frozen RU must reproduce 675/740')
            for r in mode_rows:
                old=previous[r['query_index']]
                if (r['query_path']!=old['query_path'] or r['top1_reference_index']!=old['top1_reference_index']
                    or r['top1_correct']!=old['top1_correct']):
                    raise ValueError('Frozen top1 differs from previous result; inspect before evaluating adapters')
            print('PASS frozen baseline: 675/740; all 740 top1 IDs match previous result',flush=True)
        rows.extend(mode_rows)
        arrays[descriptor_file.name]=sha(descriptor_file); arrays[score_file.name]=sha(score_file)
        del descriptors,scores_saved
    write(report/'outcomes.json',rows); summary=summarize(rows,nq); write(report/'summary.json',summary)
    completed={'complete':True,'queries':nq,'references':ndb,'variants':['frozen',*MODES],
               'frozen_top1_matches_previous':True,'arrays':arrays,
               'files':{f.name:sha(f) for f in report.glob('*.json')}}
    write(report/'completed.json',completed); write(out/'completed.json',completed)
    print(json.dumps(summary,indent=2)); print('Download only:',report)


if __name__=='__main__': main()
