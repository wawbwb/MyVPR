"""Image-level global negative mining; cached vectors NEVER replace students."""
import numpy as np
import torch
from torch.nn import functional as F

from src.channel_rank_preservation import valid_pairs

ARMS=('random_ms','random_protect','hard_ms','hard_protect')


def negative_mask(query_labels,bank_labels,query_coordinates,bank_coordinates):
    q=torch.deg2rad(query_coordinates.double());b=torch.deg2rad(bank_coordinates.double())
    a=torch.sin((q[:,0,None]-b[None,:,0])/2).square()
    a+=torch.cos(q[:,0,None])*torch.cos(b[None,:,0])*torch.sin((q[:,1,None]-b[None,:,1])/2).square()
    distance=6371000.*2*torch.asin(a.clamp(0,1).sqrt())
    return (query_labels[:,None]!=bank_labels[None,:]) & (distance>=25.)


@torch.no_grad()
def full_train_negatives(vectors,labels,coordinates,chunk=128):
    if vectors.ndim!=2 or labels.shape!=(len(vectors),) or coordinates.shape!=(len(vectors),2) or chunk<1:
        raise ValueError('Invalid training bank')
    if not torch.isfinite(vectors).all() or not torch.isfinite(coordinates).all():raise ValueError('Nonfinite bank')
    if float((vectors.norm(dim=1)-1).abs().max())>.003:raise ValueError('Unit descriptors required')
    ids=[];margins=[];n=len(vectors)
    for start in range(0,n,chunk):
        stop=min(start+chunk,n);scores=vectors[start:stop]@vectors.T
        negative=negative_mask(labels[start:stop],labels,coordinates[start:stop],coordinates)
        positive=labels[start:stop,None]==labels[None,:]
        positive[torch.arange(stop-start,device=vectors.device),torch.arange(start,stop,device=vectors.device)]=False
        if not negative.any(1).all() or not positive.any(1).all():raise ValueError('Training query lacks positive/safe negative')
        best_negative,index=scores.masked_fill(~negative,-torch.inf).max(1)
        best_positive=scores.masked_fill(~positive,-torch.inf).amax(1)
        ids.append(index.cpu().numpy());margins.append((best_positive-best_negative).cpu().numpy())
    return np.concatenate(ids),np.concatenate(margins)


def random_train_negatives(labels,coordinates,seed):
    labels=np.asarray(labels);coordinates=np.asarray(coordinates,dtype=np.float64)
    if labels.ndim!=1 or coordinates.shape!=(len(labels),2) or not np.isfinite(coordinates).all():raise ValueError('Invalid random bank')
    rng=np.random.default_rng(seed);result=np.full(len(labels),-1,np.int64);pending=np.arange(len(labels));rad=np.deg2rad(coordinates)
    for _ in range(100):
        if not len(pending):return result
        candidate=rng.integers(len(labels),size=len(pending))
        q,b=rad[pending],rad[candidate]
        a=np.sin((q[:,0]-b[:,0])/2)**2+np.cos(q[:,0])*np.cos(b[:,0])*np.sin((q[:,1]-b[:,1])/2)**2
        valid=(labels[pending]!=labels[candidate]) & (6371000.*2*np.arcsin(np.sqrt(np.clip(a,0,1)))>=25.)
        result[pending[valid]]=candidate[valid];pending=pending[~valid]
    raise ValueError('Cannot draw safe random negatives; no silent fallback')


def paired_contract(reference,labels,coordinates,cap=.05):
    if reference.ndim!=2 or len(reference)%2 or labels.shape!=(len(reference),) or not 0<cap<=1:
        raise ValueError('Expected [base queries,paired negatives]')
    reference=reference.detach();n=len(reference)//2;labels=labels.to(reference.device)
    positive,negative=valid_pairs(labels,coordinates)
    idx=torch.arange(n,device=reference.device)
    if not negative[idx,idx+n].all():raise ValueError('Paired negative is same-place or inside25m')
    positive=positive[:n,:n];negative=negative[:n,:n]
    if not positive.any(1).all():raise ValueError('Base batch lacks same-place positives')
    scores=reference[:n]@reference[:n].T
    pos=scores.masked_fill(~positive,-torch.inf).amax(1)
    neg=scores.masked_fill(~negative,-torch.inf).amax(1)
    extra=(reference[:n]*reference[n:]).sum(1)
    gap=pos-torch.maximum(neg,extra)
    return dict(positive=positive,negative=negative,correct=gap>0,target=gap.clamp(0,cap),margin=gap,scale=float(cap),n=n)


def paired_losses(descriptors,teacher,protect,margin=.05,temperature=.05):
    n=teacher['n']
    if descriptors.ndim!=2 or len(descriptors)!=2*n or margin<0 or temperature<=0:raise ValueError('Invalid paired loss')
    scores=descriptors[:n]@descriptors[:n].T
    pos=scores.masked_fill(~teacher['positive'],-torch.inf).amax(1)
    base_neg=scores.masked_fill(~teacher['negative'],-torch.inf).amax(1)
    extra=(descriptors[:n]*descriptors[n:]).sum(1)
    gap=pos-torch.maximum(base_neg,extra)
    pair=F.softplus((extra-pos+margin)/temperature).mean()
    mask=teacher['correct'] if protect else torch.zeros_like(teacher['correct'])
    violation=F.relu(teacher['target']-gap)/teacher['scale']
    ranking=torch.where(mask,violation,torch.zeros_like(violation)).mean()
    stats=dict(queries=n,teacher_errors=int((~teacher['correct']).sum()),
        low_margin=int((teacher['margin']<=teacher['scale']).sum()),protected=int(mask.sum()),
        violating=int((mask & (violation>1e-6)).sum()))
    return pair,ranking,stats
