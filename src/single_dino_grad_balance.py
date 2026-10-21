"""Per-parameter-block auxiliary norm cap. Not loss-weight adaptation/PCGrad.

For block b: s_b=min(1, beta*||g_main,b||/||g_aux,b||).
Use g_main,b + s_b*g_aux,b + g_keep,b, then ONE common global clip.
g_aux and g_keep already include their registered 0.1 scalar weights.
When main gradient is zero and aux is nonzero, aux is suppressed (no floor).
"""
import math
import torch


def combine(main, auxiliary, keep, max_ratio=.5):
    if not math.isfinite(max_ratio) or not 0 < max_ratio <= 1:
        raise ValueError('Auxiliary cap must be finite and in (0,1]')
    if not main or set(main) != set(auxiliary) or set(main) != set(keep):
        raise ValueError('Gradient parameter inventories differ')
    groups = {}
    for name, g in main.items():
        if any(x.shape != g.shape or x.device != g.device or not torch.isfinite(x).all()
               for x in (g, auxiliary[name], keep[name])):
            raise ValueError('Invalid component gradient: '+name)
        block = name.split('.')[0]
        groups.setdefault(block, []).append(name)
    combined, report = {}, {}
    for block, names in groups.items():
        norms = [sum(float(values[n].double().square().sum()) for n in names)**.5
                 for values in (main, auxiliary, keep)]
        m, a, k = norms
        scale = min(1., max_ratio*m/a) if a > 0 else 1.
        for name in names:
            combined[name] = main[name] + scale*auxiliary[name] + keep[name]
        report[block] = dict(main_l2=m, weighted_auxiliary_l2=a, auxiliary_scale=scale,
            capped_auxiliary_l2=scale*a, weighted_keep_l2=k,
            cap_satisfied=scale*a <= max_ratio*m + 1e-12)
    return combined, report
