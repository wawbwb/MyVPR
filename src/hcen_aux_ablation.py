"""Predeclared 2x2 HCEN depth/auxiliary ablation; no ranking regularizer."""
import numpy as np
import torch

ARMS = {
    'last_noaux': dict(mode='last_only', auxiliary_weight=0.),
    'hierarchy_noaux': dict(mode='hierarchy', auxiliary_weight=0.),
    'last_lowaux': dict(mode='last_only', auxiliary_weight=.001),
    'hierarchy_lowaux': dict(mode='hierarchy', auxiliary_weight=.001),
}
POLICY = dict(revision=1, seed=42, smoke_steps=8, screen_steps=512,
    anchor_places_per_step=8, views=4, lr=1e-4, weight_decay=0., clip=1.,
    precision='fp32', feature_relative_cap=.1, rank_weight=0.,
    initialization='Fresh identical RU + zero-start HCEN; never resume old pilot weights',
    auxiliary='Both heads computed in all arms; mean GPS-filtered MultiSimilarity times 0 or 0.001',
    smoke='First eight batches of the formal schedule, from the full train-only cache; mechanics only',
    selection='Fixed last at 512 updates; no checkpoint, weight or benchmark selection',
    scope='Exploratory single-seed 2x2 ablation on previously exposed benchmarks; not independent final validation')


def training_schedule(triplets, stage):
    if stage not in ('smoke', 'screen'):
        raise ValueError('Unknown ablation stage')
    rows = np.asarray(triplets)
    if rows.ndim != 2 or rows.shape[1] != 3 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError('Invalid training triplets')
    if (rows < 0).any() or (rows >= len(rows)*4).any():
        raise ValueError('Training row index outside the original gallery')
    if len(np.unique(rows[:, 0]//4)) != len(rows):
        raise ValueError('Training anchors must be unique places')
    if not np.all(rows[:, 0]//4 == rows[:, 1]//4) or np.any(rows[:, 0]//4 == rows[:, 2]//4):
        raise ValueError('Invalid positive/negative place identities')
    count = POLICY['screen_steps']*POLICY['anchor_places_per_step']
    if len(rows) < count:
        raise ValueError('Insufficient train-only anchors for 512 steps')
    selected = rows[::max(1, len(rows)//count)][:count]
    batches = [list(dict.fromkeys(np.concatenate((b[:, 0]//4, b[:, 2]//4)).tolist()))
               for b in np.split(selected, POLICY['screen_steps'])]
    return batches[:POLICY[stage+'_steps']]


def objective(primary, auxiliary, arm):
    if arm not in ARMS:
        raise ValueError('Unknown HCEN ablation arm')
    return primary + ARMS[arm]['auxiliary_weight']*auxiliary


def gradient_diagnostic(primary, weighted_auxiliary, named_parameters):
    """Unclipped gradients at the SAME current training batch; not causal proof."""
    shared = [(n, p) for n, p in named_parameters
              if p.requires_grad and n.startswith('neck.') and not n.startswith('neck.aux.')]
    if not shared:
        raise ValueError('No shared neck parameters')
    result = {}
    for name, loss in [('primary', primary), ('weighted_auxiliary', weighted_auxiliary)]:
        grads = torch.autograd.grad(loss, [p for _, p in shared], retain_graph=True, allow_unused=True)
        squared = [g.detach().float().square().sum() for g in grads if g is not None]
        up = [g.detach().float().square().sum() for (n, _), g in zip(shared, grads)
              if n.startswith('neck.up.') and g is not None]
        result[name+'_shared_norm'] = float(torch.stack(squared).sum().sqrt()) if squared else 0.
        result[name+'_up_norm'] = float(torch.stack(up).sum().sqrt()) if up else 0.
    return result


def paired_report(outcomes, ru, groups, hard, cutoff):
    if set(outcomes) != set(ARMS):
        raise ValueError('All four matched arms required')
    ru = np.asarray(ru, bool); groups = np.asarray(groups); hard = np.asarray(hard, bool)
    values = {name: np.asarray(v, bool) for name, v in outcomes.items()}
    if ru.ndim != 1 or any(x.shape != ru.shape for x in [groups, hard, *values.values()]):
        raise ValueError('Unmatched paired query identities/shapes')
    report = {}
    for role, mask in [('calibration', groups < cutoff), ('evaluation', groups >= cutoff),
                       ('all', np.ones(len(ru), bool))]:
        variants = {}
        for name, ok in dict(ru=ru, **values).items():
            variants[name] = dict(correct=int(ok[mask].sum()),
                r1=float(ok[mask].mean()) if mask.any() else None,
                hard_correct=int(ok[mask & hard].sum()),
                correction_ids=np.flatnonzero(mask & ~ru & ok).tolist(),
                regression_ids=np.flatnonzero(mask & ru & ~ok).tolist())
        comparisons = {}
        for candidate, reference in [('hierarchy_noaux', 'last_noaux'),
                                    ('hierarchy_lowaux', 'last_lowaux'),
                                    ('last_lowaux', 'last_noaux'),
                                    ('hierarchy_lowaux', 'hierarchy_noaux')]:
            c, r = values[candidate], values[reference]
            corrections = int((mask & ~r & c).sum()); regressions = int((mask & r & ~c).sum())
            comparisons[candidate+'_vs_'+reference] = dict(corrections=corrections,
                regressions=regressions, net=corrections-regressions)
        interaction = (values['hierarchy_lowaux'].astype(int)-values['last_lowaux'].astype(int)
                       -values['hierarchy_noaux'].astype(int)+values['last_noaux'].astype(int))
        report[role] = dict(queries=int(mask.sum()), hard_queries=int((mask & hard).sum()),
            variants=variants, comparisons=comparisons,
            depth_auxiliary_interaction_count=int(interaction[mask].sum()))
    return report
