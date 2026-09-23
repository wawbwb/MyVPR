"""GSV-only uncertainty diagnostic; fixed calibration quantiles, no training."""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.candidate_set_screen import write,sha,load_npz,complete
from scripts.adaptive_pair_budget import verified
from scripts.wppr_budget_diagnostic import shortlist


def uncertainty(pred):
    if pred.ndim!=2 or pred.shape[1]!=44 or not np.isfinite(pred).all():raise ValueError('Invalid prediction')
    ordered=np.sort(pred,axis=1)[:,::-1]
    logits=pred-pred.max(1,keepdims=True);prob=np.exp(logits);prob/=prob.sum(1,keepdims=True)
    return {'boundary_gap':-(ordered[:,11]-ordered[:,12]),
            'winner_to_cutoff':-(ordered[:,0]-ordered[:,12]),
            'entropy':-(prob*np.log(np.maximum(prob,1e-30))).sum(1)}


def summarize(z,expand):
    pred,teacher,labels=[z[k] for k in ['prediction','teacher','labels']]
    n=len(pred);ix=np.arange(n);order=np.argsort(-pred,axis=1,kind='stable');winner=teacher.argmax(1)
    ranks=np.argmax(order==winner[:,None],axis=1)+1
    selected=[]
    for i in range(n):
        ids=np.sort(order[i,:20 if expand[i] else 12]);selected.append(ids[np.argmax(teacher[i,ids])])
    correct=labels[ix,selected];full=labels[ix,winner]
    need=ranks>12;fixable=need&(ranks<=20)
    return dict(queries=n,expanded=int(expand.sum()),mean_survivors=float(12+8*expand.mean()),
        theoretical_layers=float(208+80*expand.mean()),winner_misses_at12=int(need.sum()),
        fixable_by20=int(fixable.sum()),fixable_caught=int((fixable&expand).sum()),
        unfixable_by20=int((ranks>20).sum()),expanded_without_winner_miss=int((expand&~need).sum()),
        winner_retained=int((ranks<=np.where(expand,20,12)).sum()),correct=int(correct.sum()),
        corrections_vs44=int((correct&~full).sum()),regressions_vs44=int((~correct&full).sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',type=Path,default=Path('.cache/wppr_pilot_v1'))
    p.add_argument('--extension',type=Path,default=Path('doc/wppr_extension_v1'))
    p.add_argument('--output',type=Path,default=Path('doc/wppr_uncertainty_v1'))
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use new output directory')
    verified(a.pilot);verified(a.extension)
    data={'calibration128':load_npz(a.pilot/'calibration_predictions.npz'),
          'prior256':load_npz(a.pilot/'evaluation_predictions.npz'),
          'remaining1664':load_npz(a.extension/'predictions.npz')}
    features=uncertainty(data['calibration128']['prediction'])
    thresholds={name:{str(rate):float(np.quantile(v,1-rate)) for rate in [.25,.5,.75]} for name,v in features.items()}
    a.output.mkdir(parents=True)
    write(a.output/'contract.json',dict(sources={str(d):sha(d/'completed.json') for d in [a.pilot,a.extension]},
        code={name:hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
              for name in ['scripts/wppr_uncertainty_diagnostic.py']},thresholds=thresholds,
        policy='GSV only; expand12->20; calibration prediction quantiles25/50/75%; expand>=threshold; no chosen winner; no Pitts access',
        scope='Historically exposed GSV; descriptive diagnostic, not independent validation. Tiny numbers of missed winners may prevent reliable identification.'))
    result={}
    for split,z in data.items():
        n=len(z['prediction']);f=uncertainty(z['prediction'])
        report={'fixed12':summarize(z,np.zeros(n,bool)),'fixed20':summarize(z,np.ones(n,bool)),'rules':[]}
        # Random controls have identical actual expansion counts, 100 fixed seeds.
        for name,rates in thresholds.items():
            for rate,t in rates.items():
                expand=f[name]>=t;stats=summarize(z,expand);random_hits=[]
                predorder=np.argsort(-z['prediction'],axis=1,kind='stable');w=z['teacher'].argmax(1)
                ranks=np.argmax(predorder==w[:,None],axis=1)+1;fixable=(ranks>12)&(ranks<=20)
                for seed in range(100):
                    chosen=np.random.default_rng(seed).permutation(n)[:int(expand.sum())]
                    random_hits.append(int(fixable[chosen].sum()))
                report['rules'].append(dict(feature=name,calibration_rate=rate,threshold=t,**stats,
                    random_fixable_caught_mean=float(np.mean(random_hits)),
                    random_fixable_caught_range=[min(random_hits),max(random_hits)]))
        result[split]=report
    write(a.output/'summary.json',result);complete(a.output)
    for split,r in result.items():
        print(split,'fixed12',r['fixed12'],flush=True)
        for v in r['rules']:print(v,flush=True)


if __name__=='__main__':main()
