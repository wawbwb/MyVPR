"""Descriptive causal-prefix diagnostics; no threshold fitting or policy selection."""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import adaptive_pair_budget as source
from scripts.candidate_set_screen import read,write,sha,load_npz,complete
from src.adaptive_pair_budget import prefix_margin,outcomes


def features(prefix):
    x=np.asarray(prefix); margin=prefix_margin(x); ordered=np.sort(x,axis=1)
    spread=x.std(1)
    return dict(margin=margin,top_score=ordered[:,-1],
        normalized_margin=margin/np.maximum(spread,1e-8),
        winner_global_rank=np.argmax(x,axis=1)+1,
        tail5_deficit=ordered[:,-1]-x[:,15:20].max(1))


def auc(values,positive):
    # Probability a positive has a larger value; ties receive half credit.
    values=np.asarray(values);positive=np.asarray(positive,bool)
    a=values[positive];b=values[~positive]
    if not len(a) or not len(b):return None
    b=np.sort(b)
    return float(np.mean((np.searchsorted(b,a,'left')+np.searchsorted(b,a,'right'))/(2*len(b))))


def diagnose(scores,labels,ids,threshold):
    f=features(scores[:,:20]);n=len(scores)
    before,old=outcomes(scores,labels,np.full(n,20));after,new=outcomes(scores,labels,np.full(n,44))
    expand=f['margin']<=threshold;fix=~before & after;reg=before & ~after
    categories=dict(corrections=fix,missed_corrections=fix & ~expand,
        caught_corrections=fix & expand,regressions=reg,executed_regressions=reg & expand,
        unrepaired_errors=~before & ~after,stable_correct=before & after)
    top2=np.argsort(-scores[:,:20],axis=1,kind='stable')[:,:2]
    top2_positive=labels[np.arange(n)[:,None],top2].all(1)
    result=dict(queries=n,expanded=int(expand.sum()),
        expanded_with_two_gt_positive_leaders=int((expand & top2_positive).sum()),
        expanded_stable_correct=int((expand & before & after).sum()),groups={})
    for name,mask in categories.items():
        result['groups'][name]=dict(count=int(mask.sum()),query_ids=[ids[i] for i in np.flatnonzero(mask)],
            prefix_features={k:dict(median=float(np.median(v[mask])),p10=float(np.quantile(v[mask],.1)),p90=float(np.quantile(v[mask],.9)))
                             for k,v in f.items()} if mask.any() else {})
    errors=~before
    result['conditional_correction_auc_among_top20_errors']={k:auc(v[errors],fix[errors]) for k,v in f.items()}
    result['auc_note']='Descriptive ONLY. Values >0.5 mean larger feature associates with correction. No direction/threshold selected; rare positives and historic exposure.'
    cases=[]
    for i in np.flatnonzero(fix | reg):
        cases.append(dict(query_index=ids[i],kind='correction' if fix[i] else 'regression',
            expand=bool(expand[i]),prefix_features={k:float(v[i]) for k,v in f.items()},
            old_winner_rank=int(old[i]+1),new_winner_rank=int(new[i]+1),
            hindsight_new_minus_old_score=float(scores[i,new[i]]-scores[i,old[i]]),
            hindsight_old20_reachable=bool(labels[i,:20].any())))
    return result,cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key,default in [('gsv','doc/candidate_top44_eval_v1'),('pitts','doc/pitts_top44_confirmation_v1'),
        ('prior','doc/candidate_top44_pitts_v1'),('work','.cache/pitts_top44_confirmation_v1'),
        ('budget','doc/adaptive_pair_budget_v1'),('output','doc/pair_budget_diagnostic_v1')]:
        p.add_argument('--'+key,type=Path,default=Path(default))
    a=p.parse_args()
    if a.output.exists():raise ValueError('New output required; no overwrites')
    source.verified(a.budget);bc=read(a.budget/'contract.json')
    if bc['code']!=source.codes() or bc['policy']!=source.POLICY:raise ValueError('Budget source changed')
    if sha(a.gsv/'completed.json')!=bc['gsv_completed_sha256']:raise ValueError('GSV changed')
    gs,gy=source.gsv_rows(a)
    source.verified(a.pitts);source.verified(a.prior)
    pc,plan=source.confirmation.load_contract(argparse.Namespace(work=a.work))
    src=read(a.budget/'sources.json')
    if src['pitts_report_sha256']!=sha(a.pitts/'completed.json') or src['pitts_work_contract_sha256']!=sha(a.work/'contract.json'):
        raise ValueError('Pitts identity changed')
    if sha(a.prior/'completed.json')!=pc['old_top44_sha256']:raise ValueError('Old subset changed')
    lookup={q:i for i,q in enumerate(plan['old_query_ids'])};hashes=read(a.pitts/'new_shard_hashes.json');rows=[]
    for r in plan['queries']:
        qi=r['query_index'];file=a.prior/'pairs'/f'{lookup[qi]:06d}.npz' if r['prior_query'] else a.work/'pairs'/f'{qi:06d}.npz'
        if not r['prior_query'] and sha(file)!=hashes[f'pairs/{qi:06d}.npz']:raise ValueError('Shard changed')
        z=load_npz(file)
        if not np.array_equal(z['labels'],np.isin(z['candidates'],r['gt'])):raise ValueError('GT changed')
        rows.append(z)
    summary={};cases={};prior_summary=read(a.budget/'summary.json')
    for name,scores,labels,ids in [('gsv_dev',gs,gy,list(range(len(gs)))),
        ('pitts_all',np.stack([r['scores'] for r in rows]),np.stack([r['labels'] for r in rows]),[r['query_index'] for r in plan['queries']])]:
        for k in (20,44):
            hit,_=outcomes(scores,labels,np.full(len(scores),k))
            if int(hit.sum())!=prior_summary[name][f'fixed_{k}']['correct']:raise ValueError('Endpoint mismatch')
        summary[name],cases[name]=diagnose(scores,labels,ids,bc['thresholds']['0.5'])
    a.output.mkdir(parents=True)
    write(a.output/'summary.json',summary);write(a.output/'cases.json',cases)
    write(a.output/'contract.json',dict(budget_completed_sha256=sha(a.budget/'completed.json'),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        scope='Five prefix-only descriptive features. GT/future scores only label outcomes and hindsight cases; no learned policy or causal visual explanation.'))
    complete(a.output)
    for name,s in summary.items():print(name,{k:v['count'] for k,v in s['groups'].items()},flush=True)

if __name__=='__main__':main()
