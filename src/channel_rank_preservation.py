"""Train-only, GPS-filtered preservation of reliable teacher decisions.

Batch correctness is a proxy, not knowledge of validation correctness. Cross-
sample ranking distillation is prior art; this is a constrained repair study.
"""
import torch

ARMS=('cosine_only','all_margin','correct_margin')


def valid_pairs(labels,coordinates):
    if labels.ndim!=1 or coordinates.shape!=(len(labels),2):raise ValueError('Invalid training batch')
    if not torch.isfinite(coordinates).all():raise ValueError('Nonfinite GPS')
    xy=torch.deg2rad(coordinates.detach().double().to(labels.device))
    lat,lon=xy[:,0],xy[:,1]
    a=torch.sin((lat[:,None]-lat[None,:])/2).square()
    a+=torch.cos(lat[:,None])*torch.cos(lat[None,:])*torch.sin((lon[:,None]-lon[None,:])/2).square()
    metres=6371000.*2*torch.asin(a.clamp(0,1).sqrt())
    same=labels[:,None]==labels[None,:]
    eye=torch.eye(len(labels),device=labels.device,dtype=torch.bool)
    return same & ~eye,~same & (metres>=25.)


def teacher_contract(reference,labels,coordinates,cap=.05):
    if reference.ndim!=2 or len(reference)!=len(labels) or not 0<cap<=1:raise ValueError('Invalid reference/cap')
    reference=reference.detach()
    if not torch.isfinite(reference).all():raise ValueError('Nonfinite reference')
    positive,negative=valid_pairs(labels.to(reference.device),coordinates)
    valid=positive.any(1)&negative.any(1)
    scores=reference@reference.T
    best_positive=scores.masked_fill(~positive,-torch.inf).amax(1)
    best_negative=scores.masked_fill(~negative,-torch.inf).amax(1)
    # No artificial target for rows with no genuine positive or safe negative.
    gap=torch.where(valid,best_positive-best_negative,torch.zeros_like(best_positive))
    correct=valid & (gap>0)
    unambiguous=valid & (gap!=0)
    return dict(positive=positive,negative=negative,valid=valid,correct=correct,
        unambiguous=unambiguous,sign=gap.sign(),target=gap.abs().clamp(max=cap),scale=float(cap))


def ranking_penalty(descriptors,teacher,arm):
    if arm not in ARMS:raise ValueError('Unknown preservation arm')
    if descriptors.ndim!=2 or descriptors.shape[0]!=len(teacher['valid']):raise ValueError('Population mismatch')
    scores=descriptors@descriptors.T
    pos=scores.masked_fill(~teacher['positive'],-torch.inf).amax(1)
    neg=scores.masked_fill(~teacher['negative'],-torch.inf).amax(1)
    # Safe masked branch avoids infinities propagating through invalid rows.
    gap=torch.where(teacher['valid'],pos,torch.zeros_like(pos))-torch.where(teacher['valid'],neg,torch.zeros_like(neg))
    mask=(teacher['correct'] if arm=='correct_margin' else teacher['unambiguous'])
    if arm=='cosine_only':mask=torch.zeros_like(mask)
    violation=(teacher['target']-teacher['sign']*gap).clamp_min(0)/teacher['scale']
    # Same full-batch denominator in all groups; not mean over variable subsets.
    penalty=torch.where(mask,violation,torch.zeros_like(violation)).mean()
    stats=dict(valid=int(teacher['valid'].sum()),teacher_correct=int(teacher['correct'].sum()),
        protected=int(mask.sum()),violating=int((mask & (violation>1e-6)).sum()),
        batch_queries=len(descriptors))
    return penalty,stats
