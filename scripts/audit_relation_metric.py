"""CPU-only posthoc fixed-training-pair diagnosis; no fitting or image access."""
import argparse
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch


def distribution(values):
    x=np.asarray(values,dtype=np.float64)
    if not len(x):return dict(n=0)
    if not np.isfinite(x).all():raise ValueError('Nonfinite diagnostic')
    return dict(n=len(x),mean=float(x.mean()),std=float(x.std()),
                quantiles=dict(zip(('min','p10','median','p90','max'),map(float,np.quantile(x,[0,.1,.5,.9,1])))))


def compare_margin(before,after,tol=1e-6):
    before=np.asarray(before);after=np.asarray(after)
    if before.shape!=after.shape:raise ValueError('Mismatched queries')
    delta=after-before
    result=dict(before=distribution(before),after=distribution(after),delta=distribution(delta),
        improved=int((delta>tol).sum()),worsened=int((delta< -tol).sum()),
        unchanged_within_tolerance=int((np.abs(delta)<=tol).sum()),
        corrections=int(((before<=0)&(after>0)).sum()),regressions=int(((before>0)&(after<=0)).sum()),
        robust_corrections=int(((before< -tol)&(after>tol)).sum()),
        robust_regressions=int(((before>tol)&(after< -tol)).sum()),tolerance=tol)
    result['strata']={name:dict(delta=distribution(delta[mask]),
        improved=int((delta[mask]>tol).sum()),worsened=int((delta[mask]< -tol).sum()))
        for name,mask in [('baseline_error',before<=0),('low_positive_margin',(before>0)&(before<=.05)),('high_margin',before>.05)]}
    return result


def fixed_scores(values,batches,negative_ids):
    """Same best of 3 positives, paired negative and GPS-filtered batch negatives."""
    n=len(values);positive=np.empty(n,np.float32);extra=np.empty(n,np.float32);combined=np.empty(n,np.float32)
    for ids,pos,neg in batches:
        x=values[ids];scores=x@x.T
        best=scores.masked_fill(~pos,-torch.inf).amax(1)
        paired=(x*values[negative_ids[ids]]).sum(1)
        worst=torch.maximum(paired,scores.masked_fill(~neg,-torch.inf).amax(1))
        positive[ids]=best.numpy();extra[ids]=paired.numpy();combined[ids]=worst.numpy()
    return dict(positive=positive,negative=extra,paired_margin=positive-extra,combined_margin=positive-combined)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'doc/relation_metric_train_audit_v1')
    a=p.parse_args()
    from scripts import relation_metric as rm
    from src.channel_rank_preservation import valid_pairs
    from src.models.relation_metric import calibrated
    from src.semantic_region_protocol import read,write,sha,state_hash,owned_run
    torch.set_num_threads(4)
    c=rm.configuration();train=rm.key.path(c,'runs')/'train';base=rm.key.path(c,'metric_cache')
    tc=rm.verified(train);rm.verified(base/'basis');rm.verified(base/'train')
    if rm.key.source_hashes(rm.FILES)!=tc['code']:raise ValueError('Experiment sources changed')
    if not all(read(train/'summary.json')['checks'].values()):raise ValueError('Training mechanics failed')
    seals={str(path.relative_to(ROOT)):sha(path/'completed.json') for path in (train,base/'basis',base/'train')}
    contract=dict(schema='relation_metric_train_audit_v1',sources=seals,script_sha256=sha(Path(__file__)),
        scope='Posthoc on ALL 32768 training images, original512 batches and fixed old selected negatives. Not full-bank retrieval or heldout evidence.',
        device='cpu',epsilon=1e-6,mean_control='Final conditional coefficients averaged over all training images; no optimization.')
    with owned_run(a.output,contract,False),torch.inference_mode():
        plan=rm.key.open_plan(c);pairs=rm.previous.pairs_ready(rm.previous.configuration())
        if pairs['row_ids'].tolist()!=rm.previous.population(plan,c):raise ValueError('Training row mismatch')
        z=torch.from_numpy(rm.bank(c));n=len(z);schedule=plan['schedule'][:c['pilot_steps']]
        flat=[4*p+v for batch in schedule for p in batch for v in range(4)]
        if sorted(flat)!=list(range(n)):raise ValueError('Schedule must cover each training image exactly once')
        coordinates=torch.tensor([plan['images'][int(i)]['coordinates'] for i in pairs['row_ids']],dtype=torch.float64)
        negative=torch.from_numpy(pairs['hard'].astype(np.int64));batches=[]
        for batch in schedule:
            ids=torch.tensor([4*p+v for p in batch for v in range(4)])
            pos,neg=valid_pairs(ids//4,coordinates[ids]);batches.append((ids,pos,neg))
            rm.paired_contract(z[torch.cat((ids,negative[ids]))],torch.cat((ids,negative[ids]))//4,
                              coordinates[torch.cat((ids,negative[ids]))],c['rank_cap'])
        bases=torch.load(base/'basis/basis.pt',map_location='cpu',weights_only=True)
        models=rm.build(dict(c,device='cpu'),bases)
        payload=torch.load(train/'last.pt',map_location='cpu',weights_only=True)
        if payload['state']['step']!=512 or payload['state']['contract_sha256']!=sha(train/'contract.json'):
            raise ValueError('Final checkpoint contract mismatch')
        for arm,m in models.items():
            m.load_state_dict(payload['models'][arm]);m.eval().requires_grad_(False)
            if state_hash(m)!=read(train/'summary.json')['final_sha256'][arm]:raise ValueError('Model state mismatch')
        m=models['relation_conditional']
        coefficients=torch.cat([m.cap*torch.tanh(m.gate(x@m.basis)) for x in z.split(256)])
        average=coefficients.mean(0);centered=coefficients-average
        gate_stats=dict(mean_coefficient_rms=float(average.square().mean().sqrt()),
            image_varying_coefficient_rms=float(centered.square().mean().sqrt()),
            varying_fraction_squared_energy=float(centered.square().sum()/coefficients.square().sum().clamp_min(1e-30)),
            per_channel_std=distribution(coefficients.std(0,unbiased=False).numpy()),
            saturation_fraction=float((coefficients.abs()>.99*c['metric_cap']).float().mean()))
        scores={};diagnostics={};mean_vectors=None
        # Conditional last so its output can be compared to the fixed mean control.
        arms=['innovation1','random_basis','pca_basis','relation_static','bounded_whitening','conditional_mean','relation_conditional']
        for arm in arms:
            print('Audit',arm,flush=True)
            if arm=='innovation1':x=z
            else:
                x=torch.empty_like(z)
                for start in range(0,n,256):
                    chunk=z[start:start+256]
                    if arm=='bounded_whitening':out=calibrated(chunk,bases['whitening_basis'],bases['whitening_coefficients'])[0]
                    elif arm=='conditional_mean':out=calibrated(chunk,m.basis,average)[0]
                    else:out=models[arm](chunk)
                    x[start:start+len(chunk)]=out
            scores[arm]=fixed_scores(x,batches,negative)
            diagnostics[arm]=dict(descriptor_drift=distribution((x-z).norm(dim=1).numpy()),
                paired_softplus_mean=float(np.logaddexp(0,(c['pair_margin']-scores[arm]['paired_margin'])/c['pair_temperature']).mean()))
            if arm=='conditional_mean':mean_vectors=x
            if arm=='relation_conditional':diagnostics[arm]['distance_from_mean_control']=distribution((x-mean_vectors).norm(dim=1).numpy())
        comparisons={arm:{kind:compare_margin(scores['innovation1'][kind],s[kind]) for kind in ('paired_margin','combined_margin')}
                     for arm,s in scores.items() if arm!='innovation1'}
        conditional_vs_mean={kind:compare_margin(scores['conditional_mean'][kind],scores['relation_conditional'][kind])
                             for kind in ('paired_margin','combined_margin')}
        with (a.output/'per_query.csv').open('w',newline='',encoding='utf8') as f:
            fields=['training_row','source_row']+[arm+'_'+k for arm in arms for k in scores[arm]]
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for i in range(n):writer.writerow(dict(training_row=i,source_row=int(pairs['row_ids'][i]),
                **{arm+'_'+k:float(v[i]) for arm in arms for k,v in scores[arm].items()}))
        for rel,h in seals.items():
            if sha(ROOT/rel/'completed.json')!=h:raise ValueError('Source seal changed during audit')
        summary=dict(queries=n,steps=len(batches),gate=gate_stats,diagnostics=diagnostics,
            vs_innovation1=comparisons,conditional_vs_mean=conditional_vs_mean,
            note='Training-only posthoc diagnostic, fixed selected negatives; no new training, no image feature extraction, no full-bank re-mining.')
        write(a.output/'summary.json',summary)
        write(a.output/'completed.json',dict(complete=True,files={f.name:sha(f) for f in a.output.iterdir() if f.is_file() and f.name!='.lock'}))
        print('Saved',a.output,flush=True)


if __name__=='__main__':main()
