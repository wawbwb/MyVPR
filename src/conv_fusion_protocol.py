"""Two fresh fusion controls vs sealed conv; no benchmark model selection."""
import statistics
import numpy as np
from src.models.conv_fusion_controls import ARMS, COUNTS
from src.hcen_plain_replication import SEEDS
from src.hcen_followup_protocol import training_schedule, effective_policy, comparable_contract as original_comparable

REFERENCE_SHA = 'c6fddae11b80f6c7db564340b85d3e6bb80f41c6db191778fc4379d081e5a6f2'
CONTRASTS = (('conv', 'conv_last'), ('conv', 'concat_token'))
POLICY = dict(revision=1, seeds=list(SEEDS), new_arms=list(ARMS), smoke_steps=8, screen_steps=2048,
    reference_aggregate_sha256=REFERENCE_SHA, trainable_parameters=COUNTS,
    mechanism='Progressive local gated conv vs concatenate/raw-layer LN/1x1-ReLU/two global token MLPs; same BoQ residual wrapper',
    concat_token=dict(input_layers=[8, 10, 12], channels=768, width=128, grid=[20, 20],
        token_blocks=2, token_ratio=1, layernorm='shared interface LN before concat; per-block LN128 before spatial mixing'),
    paper='SciceVPR arXiv:2502.20676v1 Fig5/equations4-9 and official Tokenmixen; adapted fusion, not complete reproduction',
    official_code='https://github.com/shuimushan/SciceVPR/blob/main/scicevpr.py',
    fairness='conv_last exactly same parameters/initialization as conv. concat_token is larger, not matched capacity or FLOPs. Same layers/budget/optimizer/readout/cap.',
    selection='Reuse all three sealed conv references; fresh new arms from plain checkpoint; fixed last2048, no best seed/checkpoint',
    scope='Exploratory mechanism ablation on exposed queries; neither beating official SciceVPR nor novelty/stable generalization proof')


def matched_contract(contract):
    value = original_comparable(contract)
    value.pop('code')  # Implementations differ; their actual hashes are checked independently.
    return value


def summarize(outcomes, baseline, groups, cutoff):
    if set(outcomes) != {'conv', *ARMS}:
        raise ValueError('Both new controls and sealed conv required')
    baseline = np.asarray(baseline, bool); groups = np.asarray(groups)
    values = {k: np.asarray(v, bool) for k, v in outcomes.items()}
    if baseline.ndim != 1 or groups.shape != baseline.shape or any(v.shape != baseline.shape for v in values.values()):
        raise ValueError('Unmatched query identities')
    result = {}
    for role, mask in [('calibration', groups < cutoff), ('evaluation', groups >= cutoff), ('all', np.ones(len(baseline), bool))]:
        variants = {}
        for name, v in dict(baseline=baseline, **values).items():
            variants[name] = dict(correct=int(v[mask].sum()), r1=float(v[mask].mean()) if mask.any() else None,
                corrections=np.flatnonzero(mask & ~baseline & v).tolist(), regressions=np.flatnonzero(mask & baseline & ~v).tolist())
        contrasts = {}
        for a, b in CONTRASTS:
            x, y = values[a], values[b]
            fixes = np.flatnonzero(mask & ~y & x).tolist(); damage = np.flatnonzero(mask & y & ~x).tolist()
            contrasts[a+'_vs_'+b] = dict(corrections=fixes, regressions=damage, net=len(fixes)-len(damage))
        result[role] = dict(queries=int(mask.sum()), variants=variants, comparisons=contrasts)
    return result


def aggregate_reports(reports):
    if set(reports) != set(SEEDS):
        raise ValueError('All three seeds required')
    result = {}
    for name in ('gsv_calibration', 'gsv_evaluation', 'gsv_all', 'msls-val', 'pitts30k-val'):
        parts = {s: reports[s][name] for s in SEEDS}
        n = parts[42]['queries']; base = parts[42]['variants']['baseline']['correct']
        if n <= 0 or any(p['queries'] != n or p['variants']['baseline']['correct'] != base for p in parts.values()):
            raise ValueError('Different fixed population/baseline')
        variants = {}
        for arm in ('conv', *ARMS):
            counts = [parts[s]['variants'][arm]['correct'] for s in SEEDS]; delta = [c-base for c in counts]
            variants[arm] = dict(correct_by_seed=dict(zip(SEEDS, counts)), mean_r1=statistics.mean(counts)/n,
                mean_delta_correct=statistics.mean(delta), sample_std_delta_correct=statistics.stdev(delta),
                mean_delta_pp=100*statistics.mean(delta)/n)
        contrasts = {}
        for a, b in CONTRASTS:
            key = a+'_vs_'+b; nets = [parts[s]['comparisons'][key]['net'] for s in SEEDS]
            contrasts[key] = dict(net_by_seed=dict(zip(SEEDS, nets)), mean_net=statistics.mean(nets),
                sample_std_net=statistics.stdev(nets), mean_delta_pp=100*statistics.mean(nets)/n,
                positive_every_seed=all(x > 0 for x in nets))
        result[name] = dict(queries=n, baseline_correct=base, baseline_r1=base/n, variants=variants, comparisons=contrasts)
    checks = {dataset+'_'+a+'_vs_'+b: result[dataset]['comparisons'][a+'_vs_'+b]['positive_every_seed']
              for dataset in ('msls-val', 'pitts30k-val') for a, b in CONTRASTS}
    return dict(metrics=result, directional_checks=checks)
