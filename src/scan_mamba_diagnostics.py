"""Matched optimizer clock and read-only objective gradient diagnostics."""
import math
import torch


def fixed_step(optimizer, parameters):
    # Every trainable parameter has the same Adam clock, even on zero-loss batches.
    for parameter in parameters:
        if parameter.grad is None: parameter.grad = torch.zeros_like(parameter)
    optimizer.step()


def gradient_comparison(vpr, consistency, named_parameters, weight=.1):
    items=list(named_parameters)
    parameters=[p for _,p in items]
    gv=torch.autograd.grad(vpr,parameters,retain_graph=True,allow_unused=True)
    gc=torch.autograd.grad(weight*consistency,parameters,allow_unused=True)
    result={}
    for group in ('all','aggregator','mixer'):
        v2=c2=dot=0.
        for (name,_),v,c in zip(items,gv,gc):
            if group=='aggregator' and not name.startswith('base.aggregator.'):continue
            if group=='mixer' and not name.startswith('mixer.'):continue
            if v is not None:v2+=float(v.detach().double().square().sum())
            if c is not None:c2+=float(c.detach().double().square().sum())
            if v is not None and c is not None:dot+=float((v.detach().double()*c.detach().double()).sum())
        nv,nc=math.sqrt(v2),math.sqrt(c2)
        if not all(math.isfinite(x) for x in (nv,nc,dot)):raise ValueError('Nonfinite objective gradient')
        result[group]=dict(vpr_norm=nv,weighted_consistency_norm=nc,
            consistency_to_vpr=nc/nv if nv>0 else None,
            cosine=dot/(nv*nc) if nv>0 and nc>0 else None)
    return result
