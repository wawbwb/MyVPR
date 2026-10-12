"""Three-initialization plain-BoQ replication without changing sealed seed42."""
import copy
import statistics
from src.hcen_plain_protocol import POLICY as BASE_POLICY,MODES

SEEDS=(42,43,44)
NEW_SEEDS=(43,44)
POLICY=dict(revision=1,seeds=list(SEEDS),new_seeds=list(NEW_SEEDS),steps=512,
    varied='New neck initialization only; same plain-BoQ checkpoint, training images/views and schedule',
    selection='Fixed last512; all seeds and both modes reported; no weight search or automatic extension',
    scope='Initialization-only exploratory replication on previously exposed fixed queries. Historical RU-mined indices retained; no RU module or RU descriptor input.')


def effective_policy(s):
    if s not in SEEDS:raise ValueError('Only preregistered seeds allowed')
    return {**BASE_POLICY,'seed':s}


def comparable_contract(contract):
    keys=('mode','stage','policy','checkpoint_sha256','source_completed_sha256','cache_completed_sha256',
          'metadata','train_places','original_code','benchmark_metadata','versions','schedule','baseline_completed_sha256')
    result={k:copy.deepcopy(contract[k]) for k in keys}
    s=result['policy'].pop('seed')
    if s not in SEEDS or contract.get('initialization_seed',s)!=s:raise ValueError('Seed identity differs')
    return result


def aggregate_reports(reports):
    if set(reports)!=set(SEEDS):raise ValueError('All three seeds required')
    result={}
    for name in ('gsv_calibration','gsv_evaluation','gsv_all','msls-val','pitts30k-val'):
        parts={s:reports[s][name] for s in SEEDS};n=parts[42]['queries'];baseline=parts[42]['variants']['baseline']['correct']
        if n<=0 or any(p['queries']!=n or p['variants']['baseline']['correct']!=baseline for p in parts.values()):
            raise ValueError('Query population or plain baseline differs')
        variants={}
        for mode in MODES:
            counts=[parts[s]['variants'][mode]['correct'] for s in SEEDS];deltas=[x-baseline for x in counts]
            variants[mode]=dict(correct_by_seed=dict(zip(SEEDS,counts)),delta_correct_by_seed=dict(zip(SEEDS,deltas)),
                mean_delta_correct=statistics.mean(deltas),sample_std_delta_correct=statistics.stdev(deltas),
                mean_delta_pp=100*statistics.mean(deltas)/n,positive_every_seed=all(x>0 for x in deltas),negative_every_seed=all(x<0 for x in deltas))
        nets=[parts[s]['hierarchy_vs_last_only']['net'] for s in SEEDS]
        result[name]=dict(queries=n,baseline_correct=baseline,variants=variants,
            hierarchy_vs_last_only=dict(net_by_seed=dict(zip(SEEDS,nets)),mean_net=statistics.mean(nets),
                sample_std_net=statistics.stdev(nets),mean_delta_pp=100*statistics.mean(nets)/n,
                positive_every_seed=all(x>0 for x in nets),negative_every_seed=all(x<0 for x in nets)))
    return result
