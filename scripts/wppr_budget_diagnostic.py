"""CPU-only exploratory shortlist budgets. No fitting or deployment selection."""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.candidate_set_screen import read,write,sha,load_npz,complete
from scripts.adaptive_pair_budget import verified

POLICIES=[(k,p) for k in (12,16,20,24) for p in (0,2,4)]


def shortlist(prediction,k,protect):
    pred=np.asarray(prediction)
    if pred.ndim!=2 or pred.shape[1]!=44 or not np.isfinite(pred).all() or not 0<=protect<=k<=44:
        raise ValueError('Invalid shortlist inputs')
    order=np.argsort(-pred,axis=1,kind='stable')
    # Protected prefix is INCLUDED in the fixed budget, not added on top.
    return np.array([list(range(protect))+[int(i) for i in row if i>=protect][:k-protect] for row in order])


def evaluate(pred,teacher,labels,k,protect):
    if teacher.shape!=pred.shape or labels.shape!=pred.shape or labels.dtype!=np.bool_ or not np.isfinite(teacher).all():
        raise ValueError('Invalid teacher/labels')
    keep=shortlist(pred,k,protect);ix=np.arange(len(pred));winner=teacher.argmax(1)
    # Resolve full-score ties by frozen global rank, independent of shortlist order.
    ordered=np.sort(keep,axis=1)
    selected=ordered[ix,np.take_along_axis(teacher,ordered,1).argmax(1)]
    correct=labels[ix,selected];full=labels[ix,winner];base=labels[ix,teacher[:,:20].argmax(1)]
    retained=(keep==winner[:,None]).any(1);tail=winner>=20
    return dict(k=k,protect=protect,queries=len(pred),correct=int(correct.sum()),
        full44_correct=int(full.sum()),full20_correct=int(base.sum()),
        corrections_vs20=int((correct&~base).sum()),regressions_vs20=int((~correct&base).sum()),
        corrections_vs44=int((correct&~full).sum()),regressions_vs44=int((~correct&full).sum()),
        winner_retained=int(retained.sum()),tail_count=int(tail.sum()),tail_retained=int((retained&tail).sum()),
        layers_per_direction=88+10*k,
        note='Layer count is theoretical; no latency estimate or significance claim')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gsv',type=Path,default=Path('doc/wppr_extension_v1'))
    p.add_argument('--pitts',type=Path,default=Path('doc/wppr_pitts_transfer_v1'))
    p.add_argument('--output',type=Path,default=Path('doc/wppr_budget_diagnostic_v1'))
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new output directory')
    verified(a.gsv);verified(a.pitts)
    contract=dict(code=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        sources={str(d):sha(d/'completed.json') for d in [a.gsv,a.pitts]},policies=POLICIES,
        scope='Post-hoc descriptive analysis. Pitts exposed; do not choose a deployment rule from this sweep. No training/GPU/image extraction.')
    a.output.mkdir(parents=True);write(a.output/'contract.json',contract)
    g=load_npz(a.gsv/'predictions.npz');result={}
    result['gsv_remaining1664']=[evaluate(g['prediction'],g['teacher'],g['labels'],k,s) for k,s in POLICIES]
    # GSV is evaluated and written before loading Pitts outcomes. No automatic winner selection.
    write(a.output/'gsv.json',result['gsv_remaining1664'])
    records=read(a.pitts/'per_query.json');ids=[r['query'] for r in records]
    if len(ids)!=7608 or len(set(ids))!=7608:raise ValueError('Incomplete Pitts report')
    rows=[load_npz(a.pitts/'pairs'/f'{i:06d}.npz') for i in ids]
    pred,teacher,labels=[np.stack([r[key] for r in rows]) for key in ['prediction','teacher','labels']]
    result['pitts7608_exploratory']=[evaluate(pred,teacher,labels,k,s) for k,s in POLICIES]
    baseline=result['pitts7608_exploratory'][0];expected=read(a.pitts/'summary.json')
    if baseline['correct']!=expected['vs44']['selected_correct'] or baseline['regressions_vs44']!=len(expected['vs44']['regressions']):
        raise ValueError('Original policy not reproduced')
    write(a.output/'summary.json',result);complete(a.output)
    for dataset,values in result.items():
        print(dataset,flush=True)
        for r in values:print(r,flush=True)


if __name__=='__main__':main()
