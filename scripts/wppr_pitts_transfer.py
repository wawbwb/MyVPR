"""Frozen WPPR Pitts transfer and explicit no-dense-cache pipeline timing."""
import argparse
from collections import OrderedDict
import hashlib
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.candidate_set_screen import read,write,sha,load_npz,npz,complete,official,image
from scripts.adaptive_pair_budget import verified
from src.wppr_runtime import progressive,full
from src.top44_confirmation import cluster_interval

LEGACY_CODE = '4e8bef60cceceb299bfd9f987476c8190d13045d91eb0373b358c5dcd2a15599'
TIE_POLICY = dict(max_descriptor_abs=1e-6, max_score_gap=1e-6,
    action='same candidate set; cached descriptor must reproduce frozen order; retain frozen order and audit; otherwise stop')


def reconcile_candidates(expected, actual, scores, cached_scores, descriptor, cached_descriptor):
    """Only a validated near-tie permutation may use the original frozen order."""
    expected=np.asarray(expected);actual=np.asarray(actual)
    if np.array_equal(expected,actual):return expected,None
    arrays=[scores,cached_scores,descriptor,cached_descriptor]
    if not all(np.isfinite(v).all() for v in arrays):raise ValueError('Nonfinite retrieval values')
    if expected.shape!=(44,) or actual.shape!=(44,) or len(set(expected.tolist()))!=44 or set(expected.tolist())!=set(actual.tolist()):
        raise ValueError('Candidate set changed; not a tie permutation')
    if descriptor.shape!=cached_descriptor.shape:raise ValueError('Descriptor shape changed')
    drift=float(np.max(np.abs(descriptor-cached_descriptor)))
    if drift>TIE_POLICY['max_descriptor_abs']:raise ValueError('Query descriptor drift too large')
    if not np.array_equal(np.argsort(-cached_scores,kind='stable')[:44],expected):
        raise ValueError('Cached descriptor does not reproduce frozen order')
    changed=np.flatnonzero(expected!=actual)
    gaps=np.maximum(np.abs(scores[expected[changed]]-scores[actual[changed]]),
                    np.abs(cached_scores[expected[changed]]-cached_scores[actual[changed]]))
    if np.max(gaps)>TIE_POLICY['max_score_gap']:raise ValueError('Non-tied candidate order changed')
    return expected,dict(descriptor_max_abs=drift,max_score_gap=float(np.max(gaps)),
                         ranks_1based=(changed+1).tolist(),frozen_ids=expected.tolist(),recomputed_ids=actual.tolist())


def compatible_legacy(old,new):
    expected={k:v for k,v in new.items() if k!='tie_policy'}
    expected['code']=dict(new['code'])
    expected['code']['scripts/wppr_pitts_transfer.py']=LEGACY_CODE
    return old==expected


def paired(rows, baseline):
    delta=np.array([int(r['selected_correct'])-int(r[baseline]) for r in rows])
    return dict(queries=len(rows),baseline_correct=sum(r[baseline] for r in rows),
        selected_correct=sum(r['selected_correct'] for r in rows),
        corrections=[r['query'] for r,d in zip(rows,delta) if d==1],
        regressions=[r['query'] for r,d in zip(rows,delta) if d==-1],
        delta_r1_pp=float(100*delta.mean()),
        panorama_cluster_ci95_pp=cluster_interval(delta,[r['group'] for r in rows]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key,val in [('pilot','.cache/wppr_pilot_v1'),('work','.cache/pitts_top44_confirmation_v1'),
        ('cache','.cache/pitts_pair_admission_v1'),('previous-run','doc/candidate_top44_pitts_v1'),
        ('source-report','doc/pitts_top44_confirmation_v1'),('dataset','datasets/pitts30k-val'),
        ('official-repo','/home/wt/workspace/Pair-VPR-official'),('audit','doc/pairvpr_official_paired_audit_v1'),
        ('output','doc/wppr_pitts_transfer_v1')]:p.add_argument('--'+key,type=Path,default=Path(val))
    a=p.parse_args();a.gsv_root=a.dataset
    import fcntl
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with (a.output.parent/(a.output.name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print('Verify frozen sources...',flush=True)
        for root in [a.pilot,a.cache,a.previous_run,a.source_report]:verified(root)
        c=read(a.work/'contract.json');plan=read(a.work/'plan.json')
        if sha(a.work/'plan.json')!=c['plan_sha256']:raise ValueError('Plan changed')
        if sha(a.cache/'completed.json')!=c['old_cache_sha256'] or sha(a.previous_run/'completed.json')!=c['old_top44_sha256']:
            raise ValueError('Cache changed')
        if read(a.source_report/'contract.json')['work_contract_sha256']!=sha(a.work/'contract.json'):
            raise ValueError('Wrong source report')
        contract=dict(head=sha(a.pilot/'head.pt'),pilot=sha(a.pilot/'completed.json'),
            work=sha(a.work/'contract.json'),report=sha(a.source_report/'completed.json'),
            code={n:hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                  for n in ['scripts/wppr_pitts_transfer.py','src/wppr_runtime.py']},
            policy='All7608, fixed GSV head44->12, no tuning. First32 query IDs,3 rotated timing repeats. FP32 batch1.',
            timing='Online query encoding + global retrieval + candidate image read/encoding + decoder. NO persisted database dense cache. OS file cache uncontrolled; not disk-cold. Startup/hash audits excluded.')
        contract['tie_policy']=TIE_POLICY
        if a.output.exists():
            old_contract=read(a.output/'contract.json')
            if old_contract!=contract:
                if (a.output/'completed.json').exists() or not compatible_legacy(old_contract,contract):raise ValueError('Output differs')
                reused={}
                for file in sorted((a.output/'pairs').glob('*.npz')):
                    load_npz(file);reused[file.name]=sha(file)
                timing_file=a.output/'pipeline_timing.json'
                write(a.output/'legacy_migration.json',dict(original_contract=old_contract,
                    retained_shards=reused,timing_sha256=sha(timing_file) if timing_file.exists() else None,
                    note='Known338105b only; unchanged head/model/scoring; new near-tie audit for subsequent queries'))
                print('Verified legacy shards retained:',len(reused),flush=True)
            if (a.output/'completed.json').exists():verified(a.output);print('Already complete');return
        else:a.output.mkdir(parents=True)
        write(a.output/'contract.json',contract);(a.output/'pairs').mkdir(exist_ok=True)
        for name,h in plan['index_sha256'].items():
            if sha(a.dataset/name)!=h:raise ValueError('Index changed')
        import torch
        from tqdm import tqdm
        torch.set_num_threads(4)
        model,identity=official(a)
        if identity!=c['official']:raise ValueError('Model differs')
        head=torch.nn.Sequential(torch.nn.Linear(1536,128),torch.nn.ReLU(),torch.nn.Linear(128,1)).cuda().eval()
        head.load_state_dict(torch.load(a.pilot/'head.pt',map_location='cuda',weights_only=True))
        nd=len(plan['database']);vectors=[];hashes=[]
        for start in range(0,nd+1024,128):
            block=load_npz(a.cache/'global'/f'{start:07d}.npz');vectors.extend(block['vectors']);hashes.extend(block['hashes'].tolist())
        db=torch.tensor(np.array(vectors[:nd]),device='cuda')
        for i,path in enumerate(tqdm(plan['database'],desc='Verify reference bytes')):
            if sha(a.dataset/path)!=hashes[i]:raise ValueError('Database bytes changed')
        old_lookup={qid:i for i,qid in enumerate(plan['old_query_ids'])}
        source_hashes=read(a.source_report/'new_shard_hashes.json')
        def reference(r):
            qi=r['query_index']
            if r['prior_query']:path=a.previous_run/'pairs'/f'{old_lookup[qi]:06d}.npz'
            else:
                name=f'pairs/{qi:06d}.npz';path=a.work/name
                if sha(path)!=source_hashes[name]:raise ValueError('Teacher shard changed')
            z=load_npz(path)
            if not np.array_equal(z['labels'],np.isin(z['candidates'],r['gt'])):raise ValueError('GT mismatch')
            return z
        def query(r,expected=None):
            x,_=image(a.dataset,{'path':r['path']},r['image_sha256'])
            q,g=model(x[None].cuda(),None,'global')
            scores=(g[0]@db.T).cpu().numpy()
            ids=np.argsort(-scores,kind='stable')[:44]
            if expected is not None and not np.array_equal(ids,expected):
                qi=r['query_index']
                if qi not in old_lookup:raise ValueError('Candidate mismatch without cached query descriptor; stop')
                cached=np.asarray(vectors[nd+old_lookup[qi]])
                cached_scores=(torch.from_numpy(cached).cuda()@db.T).cpu().numpy()
                ids,event=reconcile_candidates(expected,ids,scores,cached_scores,g[0].cpu().numpy(),cached)
                event['query_index']=qi
                audit_dir=a.output/'retrieval_ties';audit_dir.mkdir(exist_ok=True)
                write(audit_dir/f'{qi:06d}.json',event)
                print('AUDITED NEAR TIE:',qi,event['ranks_1based'],flush=True)
            return q,ids
        def dense(i):
            x,_=image(a.dataset,{'path':plan['database'][i]},hashes[i])
            f,g=model(x[None].cuda(),None,'global')
            if not np.allclose(g[0].cpu().numpy(),vectors[i],atol=2e-5,rtol=2e-4):raise ValueError('Encoder changed')
            return f
        def pipeline(r,name):
            q,ids=query(r)
            count=20 if name=='full20' else 44
            features=torch.cat([dense(int(i)) for i in ids[:count]])
            if name=='progressive':
                keep,scores,_=progressive(model,head,q,features)
                return ids,int(keep[scores.argmax()])
            scores=full(model,q,features,count)
            return ids,int(scores.argmax())
        # Fixed first32; timing does not select head, K, policy, or subsequent queries.
        with torch.inference_mode():
            timing=a.output/'pipeline_timing.json'
            names=['full20','full44','progressive']
            if not timing.exists():
                measures=[]
                pipeline(plan['queries'][0],'progressive')
                for pos,r in enumerate(plan['queries'][:32]):
                    expected=reference(r);item=dict(query=r['query_index'],methods={})
                    for rep in range(3):
                        offset=(pos+rep)%3
                        for name in names[offset:]+names[:offset]:
                            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
                            ids,winner=pipeline(r,name);torch.cuda.synchronize();elapsed=time.perf_counter()-t
                            if not np.array_equal(ids,expected['candidates']):raise ValueError('Candidates changed')
                            item['methods'].setdefault(name,[]).append(dict(seconds=elapsed,peak_bytes=torch.cuda.max_memory_allocated(),winner=winner))
                    measures.append(item);print('pipeline timing',pos+1,'/32',flush=True)
                    write(a.output/'progress.json',dict(phase='pipeline_timing',done=pos+1,total=32))
                write(timing,measures)
            memo=OrderedDict();rows=[]
            for r in tqdm(plan['queries'],desc='Pitts frozen44->12'):
                qi=r['query_index'];path=a.output/'pairs'/f'{qi:06d}.npz'
                if path.exists() and path.with_suffix('.sha.json').exists():
                    saved=load_npz(path)
                else:
                    old=reference(r);q,ids=query(r,old['candidates'])
                    if not np.array_equal(ids,old['candidates']):raise ValueError('Candidates changed')
                    features=[]
                    for i in ids:
                        i=int(i)
                        if i not in memo:memo[i]=dense(i).cpu()
                        memo.move_to_end(i);features.append(memo[i].cuda())
                        if len(memo)>64:memo.popitem(last=False)
                    keep,scores,pred=progressive(model,head,q,torch.cat(features))
                    kk=keep.cpu().numpy();ss=scores.cpu().numpy()
                    if not np.allclose(ss,old['scores'][kk],atol=1e-4,rtol=1e-4):raise ValueError('Continuation scores differ')
                    npz(path,keep=kk,scores=ss,prediction=pred.cpu().numpy(),teacher=old['scores'],labels=old['labels'])
                    saved=load_npz(path)
                winner=int(saved['keep'][np.argmax(saved['scores'])]);tw=int(np.argmax(saved['teacher']));lab=saved['labels']
                rows.append(dict(query=qi,group=r['group'],selected_correct=bool(lab[winner]),
                    full44_correct=bool(lab[tw]),full20_correct=bool(lab[np.argmax(saved['teacher'][:20])]),
                    winner_retained=bool(tw in saved['keep']),tail=bool(tw>=20)))
                write(a.output/'progress.json',dict(phase='transfer',done=len(rows),total=len(plan['queries'])))
        if sha(a.pilot/'head.pt')!=contract['head']:raise ValueError('Head changed')
        times=read(timing);cost={name:float(np.mean([np.median([v['seconds'] for v in r['methods'][name]]) for r in times])) for name in names}
        tail=[r for r in rows if r['tail']]
        write(a.output/'per_query.json',rows)
        write(a.output/'summary.json',dict(vs20=paired(rows,'full20_correct'),vs44=paired(rows,'full44_correct'),
            winner_retained=sum(r['winner_retained'] for r in rows),tail_count=len(tail),tail_retained=sum(r['winner_retained'] for r in tail),
            pipeline_mean_seconds=cost,scope='Frozen GSV-trained head, Pitts historically exposed validation; no tuning, not untouched test. Timing no persistent DB dense cache; accuracy LRU runtime NOT a speed benchmark.'))
        write(a.output/'progress.json',dict(phase='complete'));complete(a.output)
        print(read(a.output/'summary.json'))


if __name__=='__main__':main()
