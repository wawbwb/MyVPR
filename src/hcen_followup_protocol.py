"""Fixed 2048-update follow-up, not full GSV training or fresh benchmark validation."""
import copy
import statistics
import numpy as np
from src.models.hcen_followup import ARMS, COUNTS
from src.hcen_aux_ablation import training_schedule as original_schedule
from src.hcen_plain_replication import SEEDS, effective_policy as original_policy

CONTRASTS = (('scan', 'last_only'), ('scan', 'linear'), ('scan', 'conv'))
POLICY = dict(
    revision=1, seeds=list(SEEDS), arms=list(ARMS), smoke_steps=8, screen_steps=2048,
    trainable_parameters=COUNTS, auxiliary_weight=0., rank_weight=0.,
    schedule='Four exact passes over sealed 512-step schedule; same fixed epoch99 views, no new places/augmentation/mining',
    initialization='Fresh plain-BoQ for every seed/arm; all common outer weights match historical initialization; no trained warm start',
    last_only='Same no-SASF scan neck and initialization as scan; use L12/L12/L12 instead of L8/L10/L12',
    selection='Fixed last2048; report every arm/seed/dataset; no best checkpoint/seed or adaptive extension',
    interpretation='Pitts positive directions AND MSLS nondegradation are separate checks, not statistical significance',
    fairness='Equal updates/images/order/optimizer. Scan and last_only exactly same capacity; conv +1088 (~0.13%); linear smaller. Not equal FLOPs.',
    scope='Exploratory longer-training follow-up on exposed fixed queries; not independent generalization, stable gain or Mamba novelty proof')


def effective_policy(seed):
    return {**original_policy(seed), 'screen_steps': POLICY['screen_steps']}


def training_schedule(triplets, stage):
    if stage not in ('smoke', 'screen'):
        raise ValueError('Unknown follow-up stage')
    original = original_schedule(triplets, 'screen')
    schedule = [list(batch) for _ in range(4) for batch in original]
    return schedule[:POLICY[stage + '_steps']]


def comparable_contract(contract):
    keys = ('stage', 'policy', 'checkpoint_sha256', 'source_completed_sha256',
            'cache_completed_sha256', 'metadata', 'train_places', 'original_code',
            'benchmark_metadata', 'versions', 'schedule', 'baseline_completed_sha256',
            'followup_policy', 'reference_evidence', 'code')
    result = {k: copy.deepcopy(contract[k]) for k in keys}
    seed = result['policy'].pop('seed')
    if seed not in SEEDS or contract['initialization_seed'] != seed:
        raise ValueError('Seed identity differs')
    return result


def summarize(outcomes, baseline, groups, cutoff):
    if set(outcomes) != set(ARMS):
        raise ValueError('All four matched arms required')
    baseline = np.asarray(baseline, bool)
    groups = np.asarray(groups)
    values = {k: np.asarray(v, bool) for k, v in outcomes.items()}
    if baseline.ndim != 1 or groups.shape != baseline.shape or any(v.shape != baseline.shape for v in values.values()):
        raise ValueError('Unmatched paired query identities')
    report = {}
    for role, mask in [('calibration', groups < cutoff), ('evaluation', groups >= cutoff),
                       ('all', np.ones(len(baseline), bool))]:
        variants = {}
        for name, ok in dict(baseline=baseline, **values).items():
            variants[name] = dict(correct=int(ok[mask].sum()), r1=float(ok[mask].mean()) if mask.any() else None,
                corrections=np.flatnonzero(mask & ~baseline & ok).tolist(),
                regressions=np.flatnonzero(mask & baseline & ~ok).tolist())
        contrasts = {}
        for candidate, reference in CONTRASTS:
            c, r = values[candidate], values[reference]
            fixes = np.flatnonzero(mask & ~r & c).tolist()
            damage = np.flatnonzero(mask & r & ~c).tolist()
            contrasts[candidate + '_vs_' + reference] = dict(corrections=fixes, regressions=damage, net=len(fixes)-len(damage))
        report[role] = dict(queries=int(mask.sum()), variants=variants, comparisons=contrasts)
    return report


def aggregate_reports(reports):
    if set(reports) != set(SEEDS):
        raise ValueError('All preregistered seeds required')
    result = {}
    for name in ('gsv_calibration', 'gsv_evaluation', 'gsv_all', 'msls-val', 'pitts30k-val'):
        parts = {s: reports[s][name] for s in SEEDS}
        n = parts[42]['queries']; base = parts[42]['variants']['baseline']['correct']
        if n <= 0 or any(p['queries'] != n or p['variants']['baseline']['correct'] != base for p in parts.values()):
            raise ValueError('Unmatched fixed query population/baseline')
        variants = {}
        for arm in ARMS:
            counts = [parts[s]['variants'][arm]['correct'] for s in SEEDS]
            deltas = [x-base for x in counts]
            variants[arm] = dict(correct_by_seed=dict(zip(SEEDS, counts)), mean_delta_correct=statistics.mean(deltas),
                sample_std_delta_correct=statistics.stdev(deltas), mean_delta_pp=100*statistics.mean(deltas)/n)
        contrasts = {}
        for candidate, reference in CONTRASTS:
            key = candidate + '_vs_' + reference
            nets = [parts[s]['comparisons'][key]['net'] for s in SEEDS]
            contrasts[key] = dict(net_by_seed=dict(zip(SEEDS, nets)), mean_net=statistics.mean(nets),
                sample_std_net=statistics.stdev(nets), mean_delta_pp=100*statistics.mean(nets)/n,
                positive_every_seed=all(x > 0 for x in nets))
        result[name] = dict(queries=n, baseline_correct=base, variants=variants, comparisons=contrasts)
    scan = result['msls-val']['variants']['scan']
    checks = dict(
        pitts_beats_last_only_every_seed=result['pitts30k-val']['comparisons']['scan_vs_last_only']['positive_every_seed'],
        pitts_beats_conv_every_seed=result['pitts30k-val']['comparisons']['scan_vs_conv']['positive_every_seed'],
        msls_no_regression_vs_plain_every_seed=all(x >= result['msls-val']['baseline_correct'] for x in scan['correct_by_seed'].values()),
        gsv_evaluation_no_regression_vs_plain_every_seed=all(x >= result['gsv_evaluation']['baseline_correct'] for x in result['gsv_evaluation']['variants']['scan']['correct_by_seed'].values()))
    return dict(metrics=result, directional_checks=checks)
