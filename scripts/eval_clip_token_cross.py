"""Fixed-checkpoint 3x3 train/inference crossover; reuse verified diagonal outputs."""
import argparse
import copy
import csv
import json
import shutil
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.clip_token_drop import Cache,read,write,sha,codes,seed,device,load_ru,image_tensor
from src.models.clip_token_drop import SETTINGS,drop_mask

MODES=('none','aligned','shuffled')
CELLS=[(train,infer) for train in MODES for infer in MODES]


def name(train,infer): return train+'__'+infer


def summarize(rows,n):
    by={(r['query_index'],r['variant']):r for r in rows}
    expected=['frozen']+[name(t,i) for t,i in CELLS]
    if len(rows)!=len(by) or set(by)!={(q,v) for q in range(n) for v in expected}:
        raise ValueError('Incomplete or duplicate cross outcomes')
    matrix={}; comparisons={}
    for t,i in CELLS:
        v=name(t,i); baseline=name(t,'none')
        matrix[v]={'train':t,'inference':i,'correct':sum(by[q,v]['top1_correct'] for q in range(n)),
                   'recall':{f'R@{k}':sum(by[q,v]['best_gt_rank']<=k for q in range(n))/n for k in [1,5,10,20]}}
        comparisons[v]={'vs_same_checkpoint_all_keep':{
            'corrected':[q for q in range(n) if by[q,v]['top1_correct'] and not by[q,baseline]['top1_correct']],
            'regressed':[q for q in range(n) if not by[q,v]['top1_correct'] and by[q,baseline]['top1_correct']]}}
    return {'matrix':matrix,'within_checkpoint':comparisons,
            'frozen_correct':sum(by[q,'frozen']['top1_correct'] for q in range(n))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('doc/clip_token_drop_eval_v1'))
    p.add_argument('--cache',type=Path,default=Path('.cache/clearclip_token_drop_msls_v1'))
    p.add_argument('--runs',type=Path,default=Path('doc/clip_token_drop_train_v1'))
    p.add_argument('--output',type=Path,default=Path('doc/clip_token_cross_v1'))
    p.add_argument('--run',type=Path,default=Path('doc/dynamic_category_prior_screen_b0.5_full_db_condition_queries'))
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--msls-root',type=Path,default=Path('datasets/msls-val'))
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--batch-size',type=int,default=4)
    a=p.parse_args()
    if a.batch_size<1: p.error('batch-size must be positive')
    if a.output.exists(): raise ValueError('Output exists; use a new directory')
    import torch
    from src.models.clip_token_drop import aggregate
    from scripts.eval_dynamic_category_prior import extract_ru_feature_map
    from scripts.clip_dynamic_screen import outcome
    seed(42); d=device(a.device); cached=Cache(a.cache)
    if cached.contract['split']!='msls': raise ValueError('Need full MSLS cache')
    done=read(a.source/'completed.json'); provenance=read(a.source/'provenance.json')
    if (not done['complete'] or done['queries']!=740 or done['variants']!=['frozen',*MODES]
            or provenance['cache_sha256']!=cached.hash or provenance['code']!=codes()):
        raise ValueError('Source evaluation/cache/config differs')
    references={}
    for m in ['frozen',*MODES]:
        for kind in ['descriptors','scores']:
            file=a.source/f'{m}_{kind}.npy'
            if sha(file)!=done['outputs_sha256'][file.name]: raise ValueError('Changed source array: '+file.name)
        references[m]=np.load(a.source/f'{m}_descriptors.npy',mmap_mode='r',allow_pickle=False)
        if references[m].shape!=(19611,12288): raise ValueError('Wrong source descriptor dimensions')
    visual,_,ru_sha=load_ru(a,d); aggs={}; training={}
    for mode in MODES:
        run=a.runs/mode; complete=read(run/'completed.json')
        digest=sha(run/'last.pt')
        if not complete['complete'] or complete['smoke'] or digest!=complete['checkpoint_sha256'] or digest!=provenance['training_checkpoint_hashes'][mode]:
            raise ValueError('Checkpoint does not match source evaluation')
        state=torch.load(run/'last.pt',map_location='cpu',weights_only=False); c=state['contract']
        if (c!=provenance['training'][mode] or c['init_sha256']!=ru_sha or c['settings']!=SETTINGS
                or c['teacher']!=cached.contract['teacher'] or c['epochs']!=state['epoch']):
            raise ValueError('Training contract differs')
        agg=copy.deepcopy(visual.aggregator);agg.load_state_dict(state['aggregator']);aggs[mode]=agg.to(d).eval();training[mode]=c
    shared=[{k:v for k,v in c.items() if k!='mode'} for c in training.values()]
    if not all(c==shared[0] for c in shared): raise ValueError('Training arms unmatched')
    n=len(cached.records); ndb=cached.contract['info']['ndb']; gt=[np.asarray(v,dtype=np.int64) for v in cached.contract['info']['gt']]
    if (n,ndb,len(gt))!=(19611,18871,740): raise ValueError('Wrong MSLS universe')
    # Reconstruct old diagonal outcomes from hashed scores, not unverified JSON.
    rows=[]
    for mode in ['frozen',*MODES]:
        scores=np.load(a.source/f'{mode}_scores.npy',mmap_mode='r',allow_pickle=False)
        if scores.shape!=(740,ndb): raise ValueError('Wrong source scores')
        for q in range(740):
            rows.append({'query_index':q,'query_path':cached.records[ndb+q][0],
                'variant':'frozen' if mode=='frozen' else name(mode,mode),**outcome(scores[q],gt[q])})
    if sum(r['top1_correct'] for r in rows if r['variant']=='frozen')!=675: raise ValueError('Frozen baseline differs')
    newcells=[(t,i) for t,i in CELLS if t!=i]
    a.output.mkdir(parents=True)
    write(a.output/'provenance.json',{'source_completed_sha256':sha(a.source/'completed.json'),
        'source_provenance_sha256':sha(a.source/'provenance.json'),'training':training,'cache_sha256':cached.hash,
        'script_sha256':sha(__file__),'dependencies':codes(),'new_cells':newcells,
        'scope':'Post-hoc train/inference decomposition; both query and DB use same cell; no training or CLIP inference'})
    arrays={name(t,i):np.lib.format.open_memmap(a.output/f'{name(t,i)}_descriptors.npy',mode='w+',dtype='float32',shape=(n,12288)) for t,i in newcells}
    sentinels={0,9435,18870,18871,19610}; checked=set()
    with torch.inference_mode():
        for start in range(0,n,a.batch_size):
            end=min(start+a.batch_size,n)
            images=torch.stack([image_tensor(a.msls_root,cached.records[j][0],cached.image_hashes[j])[0] for j in range(start,end)]).to(d)
            features=extract_ru_feature_map(visual,images)
            masks={i:torch.from_numpy(np.stack([drop_mask(cached.fractions[j],cached.records[j][0],i)[0] for j in range(start,end)])).to(d) for i in MODES}
            active=sentinels.intersection(range(start,end))
            if active:
                for m in ['frozen',*MODES]:
                    values=aggregate(visual.aggregator if m=='frozen' else aggs[m],features,masks['none' if m=='frozen' else m]).cpu().numpy()
                    for j in active:
                        if not np.allclose(values[j-start],references[m][j],atol=2e-5,rtol=2e-4): raise ValueError('Fresh diagonal sentinel mismatch')
                checked.update(active)
            for t,i in newcells:
                values=aggregate(aggs[t],features,masks[i]).cpu().numpy()
                if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values,axis=1),1,atol=2e-4): raise ValueError('Invalid descriptors')
                arrays[name(t,i)][start:end]=values
            if end%128==0 or end==n:print(f'Cross full DB+query {end}/{n}',flush=True)
    if checked!=sentinels: raise ValueError('Missing reference sentinel checks')
    for arr in arrays.values():arr.flush()
    for v,arr in arrays.items():
        db=torch.from_numpy(np.array(arr[:ndb])).to(d)
        output=np.lib.format.open_memmap(a.output/f'{v}_scores.npy',mode='w+',dtype='float32',shape=(740,ndb))
        for start in range(0,740,16):
            end=min(start+16,740);scores=(torch.from_numpy(np.array(arr[ndb+start:ndb+end])).to(d)@db.T).cpu().numpy();output[start:end]=scores
            for q,s in enumerate(scores,start):rows.append({'query_index':q,'query_path':cached.records[ndb+q][0],'variant':v,**outcome(s,gt[q])})
        output.flush();del db
    summary=summarize(rows,740);write(a.output/'summary.json',summary);write(a.output/'query_outcomes.json',rows)
    with (a.output/'matrix.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['training','infer_none','infer_aligned','infer_shuffled'])
        for t in MODES:w.writerow([t]+[summary['matrix'][name(t,i)]['correct'] for i in MODES])
    write(a.output/'completed.json',{'complete':True,'queries':740,'new_cells':newcells,
        'checked_sentinels':sorted(checked),'outputs_sha256':{f.name:sha(f) for f in a.output.glob('*.npy')}})
    (a.output/'report').mkdir()
    for f in ['summary.json','query_outcomes.json','matrix.csv','provenance.json','completed.json']:shutil.copyfile(a.output/f,a.output/'report'/f)
    print(json.dumps(summary,indent=2));print('Saved:',a.output/'report')


if __name__=='__main__':main()
