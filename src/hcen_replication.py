"""Initialization-only replication of the sealed HCEN 512-step factorial."""
import copy
import statistics

from src.hcen_aux_ablation import ARMS, POLICY as BASE_POLICY

SEEDS = (42, 43, 44)
NEW_SEEDS = (43, 44)
POLICY = dict(revision=1, seeds=list(SEEDS), new_seeds=list(NEW_SEEDS), steps=512,
    varied='HCEN neck initialization only; fixed data seed42 and per-place view seed99',
    fixed='Same four arms, hard-negative cache, 512-step schedule, frozen RU, optimizer and preprocessing',
    selection='All preregistered seeds/arms reported; fixed last; no benchmark selection or automatic extension',
    scope='Three-initialization exploratory replication on fixed previously exposed queries; not independent final validation')


def effective_policy(initialization_seed):
    if initialization_seed not in SEEDS:
        raise ValueError('Only preregistered initialization seeds allowed')
    return {**BASE_POLICY, 'seed': initialization_seed}


def comparable_contract(contract):
    """Explicit shared scientific protocol; code provenance checked separately."""
    names = ('arm', 'mode', 'auxiliary_weight', 'stage', 'policy', 'ru_sha256',
             'cache_completed_sha256', 'source_completed_sha256', 'previous',
             'metadata', 'benchmark_metadata', 'train_places', 'schedule', 'original_code', 'versions')
    result = {name: copy.deepcopy(contract[name]) for name in names}
    seed = result['policy'].pop('seed')
    if seed not in SEEDS or contract.get('initialization_seed', seed) != seed:
        raise ValueError('Initialization seed identity differs')
    return result


def aggregate_reports(reports):
    if set(reports) != set(SEEDS):
        raise ValueError('All three preregistered seeds required')
    out = {}
    for dataset in ('gsv_calibration', 'gsv_evaluation', 'gsv_all', 'msls-val', 'pitts30k-val'):
        parts = {s: reports[s][dataset] for s in SEEDS}
        n = parts[42]['queries']
        if n <= 0 or any(p['queries'] != n for p in parts.values()):
            raise ValueError('Different query populations across seeds')
        ru = parts[42]['variants']['ru']['correct']
        if any(p['variants']['ru']['correct'] != ru for p in parts.values()):
            raise ValueError('Different RU baseline counts')
        variants = {}
        for arm in ARMS:
            counts = [parts[s]['variants'][arm]['correct'] for s in SEEDS]
            deltas = [x-ru for x in counts]
            variants[arm] = dict(correct_by_seed=dict(zip(SEEDS, counts)),
                delta_correct_by_seed=dict(zip(SEEDS, deltas)), mean_delta_correct=statistics.mean(deltas),
                sample_std_delta_correct=statistics.stdev(deltas), mean_delta_pp=100*statistics.mean(deltas)/n,
                positive_seeds=sum(x > 0 for x in deltas), negative_seeds=sum(x < 0 for x in deltas))
        keys = set(parts[42]['comparisons'])
        if any(set(p['comparisons']) != keys for p in parts.values()):
            raise ValueError('Different matched contrasts')
        contrasts = {}
        for key in sorted(keys):
            deltas = [parts[s]['comparisons'][key]['net'] for s in SEEDS]
            contrasts[key] = dict(net_by_seed=dict(zip(SEEDS, deltas)),
                mean_net=statistics.mean(deltas), sample_std_net=statistics.stdev(deltas),
                mean_delta_pp=100*statistics.mean(deltas)/n, positive_every_seed=all(x > 0 for x in deltas),
                negative_every_seed=all(x < 0 for x in deltas))
        out[dataset] = dict(queries=n, ru_correct=ru, variants=variants, comparisons=contrasts)
    return out
