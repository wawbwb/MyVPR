"""Preregistered three-arm residual mechanism experiment; no outcome tuning."""
import copy
import statistics

import numpy as np

from src.models.conv_residual_ablation import ARMS, RHO, PARAMETERS
from src.hcen_plain_replication import SEEDS
from src.hcen_followup_protocol import effective_policy, training_schedule
from src.conv_fusion_protocol import matched_contract


CONTRASTS = (('rms_tanh', 'fixed_tanh'), ('fixed_tanh', 'linear_residual'),
             ('rms_tanh', 'linear_residual'))
POLICY = dict(
    revision=1, arms=list(ARMS), seeds=list(SEEDS), rho=RHO,
    smoke_steps=8, screen_steps=2048, trainable_registered_parameters=PARAMETERS,
    effective_parameters=PARAMETERS - 65536,
    formulas=dict(linear_residual='R = 0.1 * V', fixed_tanh='R = 0.1 * tanh(V)',
                  rms_tanh='R = 0.1 * stopgrad(RMS_channel(X12)) * tanh(V)'),
    limits=dict(linear_residual='No feature-amplitude bound; no artificial rho-relative rejection',
                fixed_tanh='Elementwise absolute bound 0.1, NOT a 10% relative bound',
                rms_tanh='Per-patch and per-image relative L2 <= 0.1; element bound 0.1*RMS(X12)'),
    fairness='Same no-SASF L8/L10/L12 conv, exact historical initial weights and parameter names, zero up, frozen plain DINO/BoQ, same rho/data/order/FP32/AdamW/lr/clip/microbatch/budget',
    sampling='Historical 512-step deterministic schedule repeated four times; no new images/mining/augmentation',
    auxiliary_weight=0., rank_weight=0., zero_initialization='Identical in all arms; NOT a zero-init ablation',
    selection='Fresh nine runs, fixed last2048 and all three seeds; smoke never warm-starts screen; no best model/weight/seed',
    diagnostic='All training-forward images summarized per step before its update; all eval gallery/query images have CSV rows and exact image-distribution quantiles. Patch p95 is per image, not pooled patch p95.',
    interpretation='rms-vs-fixed isolates reference-amplitude scaling, fixed-vs-linear isolates tanh; amplitude and initial effective gradient change by design, not a matched-perturbation experiment',
    scope='Exploratory residual-mechanism ablation on exposed benchmarks, not independent validation, universal stability, novelty or patent authorization proof',
)


def arm_contract(contract):
    """Remove only declared arm identity; never ignore formula/recipe/code drift."""
    result = copy.deepcopy(contract)
    result.pop('arm')
    return result


def diagnostic_summary(rows):
    if not rows:
        raise ValueError('No diagnostic images')
    keys = set(rows[0])
    if any(set(row) != keys for row in rows):
        raise ValueError('Diagnostic fields differ')
    metrics = {}
    for key in sorted(keys):
        values = np.asarray([row[key] for row in rows if row[key] is not None], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite diagnostic values')
        metrics[key] = dict(n=len(values), missing=len(rows) - len(values),
            mean=float(values.mean()) if len(values) else None,
            p50=float(np.quantile(values, .5)) if len(values) else None,
            p95=float(np.quantile(values, .95)) if len(values) else None,
            max=float(values.max()) if len(values) else None)
    valid = sum(row['valid_reference_patch_count'] for row in rows)
    return dict(images=len(rows), metrics=metrics,
        relative_patch_above_rho_fraction=(sum(row['patch_relative_above_rho_count'] for row in rows) / valid if valid else None),
        zero_reference_nonzero_residual_patches=sum(row['zero_reference_nonzero_residual_patch_count'] for row in rows),
        note=POLICY['diagnostic'])


def summarize(outcomes, baseline, groups, cutoff):
    if set(outcomes) != set(ARMS):
        raise ValueError('All three arms required')
    baseline = np.asarray(baseline, bool); groups = np.asarray(groups)
    values = {arm: np.asarray(row, bool) for arm, row in outcomes.items()}
    if baseline.ndim != 1 or groups.shape != baseline.shape or any(row.shape != baseline.shape for row in values.values()):
        raise ValueError('Unmatched query outcomes/groups')
    result = {}
    for role, mask in [('calibration', groups < cutoff), ('evaluation', groups >= cutoff),
                       ('all', np.ones(len(baseline), bool))]:
        variants = {}
        for arm, row in dict(baseline=baseline, **values).items():
            variants[arm] = dict(correct=int(row[mask].sum()), r1=float(row[mask].mean()) if mask.any() else None,
                corrections=np.flatnonzero(mask & ~baseline & row).tolist(),
                regressions=np.flatnonzero(mask & baseline & ~row).tolist())
        comparisons = {}
        for candidate, reference in CONTRASTS:
            fixes = np.flatnonzero(mask & ~values[reference] & values[candidate]).tolist()
            damage = np.flatnonzero(mask & values[reference] & ~values[candidate]).tolist()
            comparisons[candidate + '_vs_' + reference] = dict(corrections=fixes, regressions=damage, net=len(fixes) - len(damage))
        result[role] = dict(queries=int(mask.sum()), variants=variants, comparisons=comparisons)
    return result


def aggregate_reports(reports):
    if set(reports) != set(SEEDS):
        raise ValueError('All three fixed seeds required')
    result = {}
    for dataset in ('gsv_calibration', 'gsv_evaluation', 'gsv_all', 'msls-val', 'pitts30k-val'):
        parts = {s: reports[s][dataset] for s in SEEDS}
        n = parts[42]['queries']; base = parts[42]['variants']['baseline']['correct']
        if n <= 0 or any(p['queries'] != n or p['variants']['baseline']['correct'] != base for p in parts.values()):
            raise ValueError('Fixed baseline/population differs')
        variants = {}
        for arm in ARMS:
            counts = [parts[s]['variants'][arm]['correct'] for s in SEEDS]
            variants[arm] = dict(correct_by_seed=dict(zip(SEEDS, counts)), mean_r1=statistics.mean(counts) / n,
                mean_delta_pp=100 * (statistics.mean(counts) - base) / n,
                sample_std_correct=statistics.stdev(counts))
        contrasts = {}
        for candidate, reference in CONTRASTS:
            key = candidate + '_vs_' + reference
            nets = [parts[s]['comparisons'][key]['net'] for s in SEEDS]
            contrasts[key] = dict(net_by_seed=dict(zip(SEEDS, nets)), mean_net=statistics.mean(nets),
                sample_std_net=statistics.stdev(nets), mean_delta_pp=100 * statistics.mean(nets) / n,
                positive_every_seed=all(value > 0 for value in nets), negative_every_seed=all(value < 0 for value in nets))
        result[dataset] = dict(queries=n, baseline_correct=base, baseline_r1=base / n,
                               variants=variants, comparisons=contrasts)
    return result
