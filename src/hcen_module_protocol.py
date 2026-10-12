"""Equal-step, explicitly NOT equal-parameter/equal-FLOP module necessity screen."""
import copy
import statistics
import numpy as np
from src.models.hcen_ablation import VARIANTS,COUNTS
from src.hcen_plain_replication import SEEDS,effective_policy,comparable_contract

POLICY=dict(revision=1,seeds=list(SEEDS),variants=list(VARIANTS),smoke_steps=8,screen_steps=512,
    trainable_parameters=COUNTS,
    linear='Keep depth LN/projections, two pointwise low/high linear fusions, final up and zero-weight auxiliary heads; remove all spatial/SSM paths',
    conv='Replace only selective scan with local 2D gated convolution; retain shallow/symmetric/SASF paths, share initialized in/out projections',
    no_sasf='Only replace both SASF blocks by identity; keep selective scan and all other paths',
    initialization='Construct the exact historical full neck first; preserve all untouched/shared parameters; independently seeded replacement draws',
    reference='Reuse sealed full-HCEN three-seed results; never reuse trained weights for new arms',
    fairness='Equal images/order/512 updates/optimizer, NOT equal parameter count, FLOPs or measured acceleration. Conv is about 0.12% larger than full.',
    scope='Exploratory module-necessity ablation on exposed fixed queries; not originality, independent generalization or pure parameter-count control')


def matched_reference_contract(contract):
    result=comparable_contract(contract)
    if result['mode']!='hierarchy':raise ValueError('All module ablations use cross-depth inputs')
    return result


def summarize(outcomes,baseline,groups,cutoff):
    expected={'full','last_only',*VARIANTS}
    if set(outcomes)!=expected:raise ValueError('Missing full, last-only or ablation arm')
    baseline=np.asarray(baseline,bool);groups=np.asarray(groups);values={k:np.asarray(v,bool) for k,v in outcomes.items()}
    if baseline.ndim!=1 or groups.shape!=baseline.shape or any(v.shape!=baseline.shape for v in values.values()):raise ValueError('Unmatched query identities')
    result={}
    for role,mask in [('calibration',groups<cutoff),('evaluation',groups>=cutoff),('all',np.ones(len(groups),bool))]:
        variants={}
        for name,v in dict(baseline=baseline,**values).items():
            variants[name]=dict(correct=int(v[mask].sum()),r1=float(v[mask].mean()) if mask.any() else None,
                correction_ids=np.flatnonzero(mask&~baseline&v).tolist(),regression_ids=np.flatnonzero(mask&baseline&~v).tolist())
        contrasts={}
        for a,b in [('full','linear'),('full','conv'),('full','no_sasf'),('full','last_only')]:
            x,y=values[a],values[b]
            fixes=np.flatnonzero(mask&~y&x).tolist();damage=np.flatnonzero(mask&y&~x).tolist()
            contrasts[a+'_vs_'+b]=dict(corrections=fixes,regressions=damage,net=len(fixes)-len(damage))
        result[role]=dict(queries=int(mask.sum()),variants=variants,comparisons=contrasts)
    return result


def aggregate_reports(reports):
    if set(reports)!=set(SEEDS):raise ValueError('All preregistered seeds required')
    result={}
    for name in ('gsv_calibration','gsv_evaluation','gsv_all','msls-val','pitts30k-val'):
        parts={s:reports[s][name] for s in SEEDS};n=parts[42]['queries'];base=parts[42]['variants']['baseline']['correct']
        if n<=0 or any(v['queries']!=n or v['variants']['baseline']['correct']!=base for v in parts.values()):raise ValueError('Different fixed query population/baseline')
        variants={}
        for v in ('full','last_only',*VARIANTS):
            counts=[parts[s]['variants'][v]['correct'] for s in SEEDS];deltas=[x-base for x in counts]
            variants[v]=dict(correct_by_seed=dict(zip(SEEDS,counts)),mean_delta_correct=statistics.mean(deltas),
                sample_std_delta_correct=statistics.stdev(deltas),mean_delta_pp=100*statistics.mean(deltas)/n)
        contrasts={}
        for key in parts[42]['comparisons']:
            nets=[parts[s]['comparisons'][key]['net'] for s in SEEDS]
            contrasts[key]=dict(net_by_seed=dict(zip(SEEDS,nets)),mean_net=statistics.mean(nets),
                sample_std_net=statistics.stdev(nets),mean_delta_pp=100*statistics.mean(nets)/n,
                positive_every_seed=all(x>0 for x in nets),negative_every_seed=all(x<0 for x in nets))
        result[name]=dict(queries=n,baseline_correct=base,variants=variants,comparisons=contrasts)
    return result
