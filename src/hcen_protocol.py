"""Label/GPS-safe soft ranking decorrelation adapted from uploaded ReID draft."""
import numpy as np
import torch
from torch.nn import functional as F
from src.slgd_hard_protocol import distance_metres

POLICY=dict(revision=2,seed=42,layers=[8,10,12],dim=128,inner=32,state=4,
    microbatch=4,anchor_places_per_step=8,views=4,smoke_steps=4,pilot_steps=128,
    lr=1e-4,weight_decay=0.,clip=1.,precision='fp32',auxiliary_weight=.1,
    rank_weight=.02,rank_temperature=.02,rank_upper=.5,rank_reference_gap=.02,
    feature_relative_cap=.1,negative_exclusion_metres=25.,
    benchmark_preprocessing='torchvision v2: ToImage -> uint8 Tensor Resize280 bicubic antialias -> ToDtype float32 scale -> ImageNet normalize',
    frozen='All original RU tensors: DINO, spatial gate and BoQ',
    sampling='Uniform stride across fixed training-only hard triplets; include 4 views of anchor and hard-negative places',
    selection='Fixed last only, no benchmark/weight selection; smoke mechanical gate then 128-step pilots',
    scope='ReID-inspired exploratory cross-depth VPR adaptation; not literal paper reproduction, independent final validation or a novelty claim')


def benchmark_transform():
    """Match historical RU MSLS/Pitts pixels; GSV retains its original PIL policy."""
    from torchvision.transforms import v2 as T
    return T.Compose([T.ToImage(),T.Resize((280,280),interpolation=T.InterpolationMode.BICUBIC,antialias=True),
                      T.ToDtype(torch.float32,scale=True),T.Normalize([.485,.456,.406],[.229,.224,.225])])


def soft_tau(left,right,mask,temperature=.02):
    if left.shape!=right.shape or mask.shape!=left.shape or temperature<=0:
        raise ValueError('Unmatched soft Kendall inputs')
    d=left.shape[1]
    a,b=torch.triu_indices(d,d,offset=1,device=left.device)
    valid=mask[:,a]&mask[:,b]
    count=valid.sum(1)
    values=torch.tanh((left[:,a]-left[:,b])/temperature)*torch.tanh((right[:,a]-right[:,b])/temperature)
    tau=(values*valid).sum(1)/count.clamp_min(1)
    return tau,count>0


def ranking_regularizer(descriptor,reference,labels,coordinates):
    """Never decorrelate positive-vs-negative order; do not push tau negative."""
    reference=reference.detach()
    bs=reference@reference.T;ss=descriptor@descriptor.T
    same=labels[:,None]==labels[None,:]
    eye=torch.eye(len(labels),dtype=torch.bool,device=labels.device)
    positive=same & ~eye
    safe=torch.as_tensor(distance_metres(np.asarray(coordinates),np.asarray(coordinates)),device=labels.device)>=25
    negative=~same & safe
    gap=bs.masked_fill(~positive,torch.inf).min(1).values-bs.masked_fill(~negative,-torch.inf).max(1).values
    reliable=torch.isfinite(gap) & (gap>POLICY['rank_reference_gap'])
    ids=labels.unique(sorted=True)
    rc=F.normalize(torch.stack([reference[labels==p].mean(0) for p in ids]),dim=-1)
    dc=F.normalize(torch.stack([descriptor[labels==p].mean(0) for p in ids]),dim=-1)
    class_mask=torch.stack([(labels!=p)&safe[:,labels==p].all(1) for p in ids],dim=1)
    intra,vi=soft_tau(bs,ss,positive,POLICY['rank_temperature'])
    inter,ve=soft_tau(reference@rc.T,descriptor@dc.T,class_mask,POLICY['rank_temperature'])
    pieces=[];stats={}
    for name,tau,valid in [('intra',intra,vi),('inter',inter,ve)]:
        valid=valid & reliable
        loss=F.relu(tau-POLICY['rank_upper']).square()
        pieces.append(loss[valid].mean() if valid.any() else descriptor.sum()*0)
        stats[name]=dict(coverage=float(valid.float().mean()),
            active_fraction=float((valid & (tau>POLICY['rank_upper'])).float().mean()),
            tau_mean=float(tau[valid].detach().mean()) if valid.any() else None)
    return sum(pieces),stats


def training_schedule(triplets,stage):
    rows=np.asarray(triplets,np.int64)
    steps=POLICY[stage+'_steps'];anchors=POLICY['anchor_places_per_step']
    stride=max(1,len(rows)//(steps*anchors))
    selected=rows[::stride][:steps*anchors]
    if len(selected)!=steps*anchors:raise ValueError('Insufficient fixed training triplets')
    return [list(dict.fromkeys(np.concatenate((batch[:,0]//4,batch[:,2]//4)).tolist()))
            for batch in np.split(selected,steps)]


def paired_partitions(outcomes,ru,groups,hard,cutoff):
    ru=np.asarray(ru,bool);groups=np.asarray(groups);hard=np.asarray(hard,bool)
    report={}
    for role,mask in [('calibration',groups<cutoff),('evaluation',groups>=cutoff),('all',np.ones(len(ru),bool))]:
        variants={}
        for mode,values in dict(ru=ru,**outcomes).items():
            ok=np.asarray(values,bool)
            if ok.shape!=ru.shape:raise ValueError('Unmatched HCEN outcomes')
            variants[mode]=dict(correct=int(ok[mask].sum()),r1=float(ok[mask].mean()),
                hard_correct=int(ok[mask & hard].sum()),corrections=int((mask & ~ru & ok).sum()),
                regressions=int((mask & ru & ~ok).sum()))
        pairs={}
        for candidate,reference in [('hierarchy','last_only'),('hierarchy_rank','hierarchy')]:
            c=np.asarray(outcomes[candidate],bool);r=np.asarray(outcomes[reference],bool)
            pairs[candidate+'_vs_'+reference]=dict(corrections=int((mask & ~r & c).sum()),
                regressions=int((mask & r & ~c).sum()),net=int(c[mask].sum())-int(r[mask].sum()))
        report[role]=dict(queries=int(mask.sum()),hard_queries=int((mask & hard).sum()),variants=variants,comparisons=pairs)
    return report
