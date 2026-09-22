"""Offline non-semantic budget screen. Lock GSV thresholds BEFORE reading Pitts scores."""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.candidate_set_screen import read, write, sha, complete, load_npz
from scripts import pitts_top44_confirmation as confirmation
from src.adaptive_pair_budget import prefix_margin, decide, outcomes, fixed_random
from src.top44_confirmation import cluster_interval

POLICY = dict(primary='margin quantile0.5 calibrated on GSV without GT', exploratory_quantiles=[.25,.5,.75],
    observation='only first20 pair scores; expand if top1 minus top2 <= frozen GSV threshold',
    comparisons='fixed K20..44 and query-ID hash-random expansion; no Pitts threshold fitting',
    scope='Exploratory historically exposed GSV/Pitts; not independent test; pair count is not wall-clock latency')


def codes():
    return {n: hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for n in ['scripts/adaptive_pair_budget.py','src/adaptive_pair_budget.py','src/top44_confirmation.py','scripts/candidate_set_screen.py']}


def verified(path):
    done=read(path/'completed.json')
    if not done['complete']: raise ValueError('Incomplete source')
    for n, h in done['files'].items():
        target=(path/n).resolve()
        if not target.is_relative_to(path.resolve()) or sha(target)!=h: raise ValueError('Source hash mismatch: '+n)
    return done


def gsv_rows(a):
    verified(a.gsv)
    result=[load_npz(a.gsv/'pairs'/f'{i:06d}.npz') for i in range(2048)]
    return np.stack([r['scores'] for r in result]), np.stack([r['labels'] for r in result])


def fit(a):
    if a.output.exists(): raise ValueError('Use a fresh output; locked rules are never overwritten')
    scores,_=gsv_rows(a)
    margins=prefix_margin(scores[:,:20])
    rules={str(q):float(np.quantile(margins,q)) for q in POLICY['exploratory_quantiles']}
    a.output.mkdir(parents=True)
    write(a.output/'contract.json',dict(code=codes(),policy=POLICY,thresholds=rules,gsv_completed_sha256=sha(a.gsv/'completed.json')))
    print('LOCKED GSV-only thresholds:',rules,flush=True)


def evaluate(a):
    c=read(a.output/'contract.json')
    if c['code']!=codes() or c['policy']!=POLICY or c['gsv_completed_sha256']!=sha(a.gsv/'completed.json'):
        raise ValueError('Locked contract changed')
    if (a.output/'completed.json').exists(): verified(a.output); print('Already complete'); return
    gs,gy=gsv_rows(a)
    verified(a.pitts)
    pc,plan=confirmation.load_contract(argparse.Namespace(work=a.work))
    if read(a.pitts/'contract.json')['work_contract_sha256']!=sha(a.work/'contract.json'): raise ValueError('Pitts contract mismatch')
    verified(a.prior)
    if sha(a.prior/'completed.json')!=pc['old_top44_sha256']: raise ValueError('Old Pitts mismatch')
    lookup={q:i for i,q in enumerate(plan['old_query_ids'])}; hashes=read(a.pitts/'new_shard_hashes.json')
    rows=[]
    for r in plan['queries']:
        qi=r['query_index']
        file=a.prior/'pairs'/f'{lookup[qi]:06d}.npz' if r['prior_query'] else a.work/'pairs'/f'{qi:06d}.npz'
        if not r['prior_query'] and sha(file)!=hashes[f'pairs/{qi:06d}.npz']: raise ValueError('Changed scored query')
        z=load_npz(file)
        if not r['prior_query']: confirmation.validate_new(z,r,len(plan['database']))
        rows.append(z)
    ps=np.stack([r['scores'] for r in rows]); py=np.stack([r['labels'] for r in rows])
    summary={}; all_records={}
    for name,scores,labels,ids,groups in [('gsv_dev',gs,gy,list(range(2048)),None),
        ('pitts_all',ps,py,[r['query_index'] for r in plan['queries']],[r['group'] for r in plan['queries']])]:
        variants={f'fixed_{k}':np.full(len(scores),k,int) for k in range(20,45)}
        for q,t in c['thresholds'].items():
            variants['margin_'+q]=decide(scores[:,:20],t)
            variants['hash_random_'+q]=fixed_random(ids,float(q))
        full,_=outcomes(scores,labels,variants['fixed_44']); base,_=outcomes(scores,labels,variants['fixed_20'])
        expected=read(a.gsv/'summary.json') if name=='gsv_dev' else read(a.pitts/'summary.json')['all']
        oldcount=expected['top20']['correct'] if name=='gsv_dev' else expected['top20_correct']
        newcount=expected['top44']['correct'] if name=='gsv_dev' else expected['top44_correct']
        if int(base.sum())!=oldcount or int(full.sum())!=newcount: raise ValueError('Source endpoints not reproduced')
        result={}; records={}
        for v,budget in variants.items():
            correct,rank=outcomes(scores,labels,budget)
            result[v]=dict(queries=len(scores),correct=int(correct.sum()),r1=float(correct.mean()),mean_pairs=float(budget.mean()),
                directional_forwards_mean=float(2*budget.mean()),pair_saving_vs44=float(1-budget.mean()/44),
                corrections_vs20=[ids[i] for i in np.flatnonzero(correct & ~base)],regressions_vs20=[ids[i] for i in np.flatnonzero(~correct & base)],
                net_vs44=int(correct.sum()-full.sum()))
            if v.startswith('margin_'):
                lo=int(np.floor(budget.mean())); hi=int(np.ceil(budget.mean()))
                result[v]['bracketing_fixed_budgets']=[lo,hi]
                if groups is not None:
                    result[v]['cluster_ci_pp_vs44']=cluster_interval(correct.astype(int)-full.astype(int),groups)
                    fixed,_=outcomes(scores,labels,variants[f'fixed_{hi}'])
                    result[v]['cluster_ci_pp_vs_ceil_fixed']=cluster_interval(correct.astype(int)-fixed.astype(int),groups)
                records[v]=[dict(query_index=ids[i],pairs=int(budget[i]),correct=bool(correct[i]),selected_global_rank=int(rank[i]+1)) for i in range(len(scores))]
        summary[name]=result;all_records[name]=records
    write(a.output/'summary.json',summary);write(a.output/'per_query.json',all_records)
    write(a.output/'sources.json',dict(pitts_report_sha256=sha(a.pitts/'completed.json'),pitts_work_contract_sha256=sha(a.work/'contract.json')))
    complete(a.output)
    for name,s in summary.items():
        print(name,{v:s[v] for v in ('fixed_20','margin_0.5','fixed_32','fixed_44')},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['fit','eval'])
    p.add_argument('--gsv',type=Path,default=Path('doc/candidate_top44_eval_v1'))
    p.add_argument('--pitts',type=Path,default=Path('doc/pitts_top44_confirmation_v1'))
    p.add_argument('--prior',type=Path,default=Path('doc/candidate_top44_pitts_v1'))
    p.add_argument('--work',type=Path,default=Path('.cache/pitts_top44_confirmation_v1'))
    p.add_argument('--output',type=Path,default=Path('doc/adaptive_pair_budget_v1'))
    a=p.parse_args();{'fit':fit,'eval':evaluate}[a.stage](a)

if __name__=='__main__':main()
