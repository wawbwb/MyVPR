"""Training-only native-patch relational guidance; unchanged shared student."""
import numpy as np
import torch
from torch.nn import functional as F

from src.models.slgd import mutual_score
from src.slgd_hard_protocol import distance_metres


MID_POLICY = dict(revision=1, seed=42, sampling_epoch=99, views=4,
    smoke_places=64, pilot_places=8192, batch_triplets=16,
    smoke_steps=4, pilot_steps=512, microbatch=4, teacher_grid=20,
    modes=['retrieval','last12','mid10'], beta=.2, slots=16, local_dim=128,
    auxiliary_weight=.1, target_min=.01, target_max=.05,
    boq_lr=1e-5, local_lr=1e-4, clip=1., weight_decay=0.,
    negative_exclusion_metres=25., min_matches=3,
    selection='Fixed final step only; no epoch/weight/layer selection on heldout data',
    training='Clean deterministic views; weakest other same-place positive and closest GPS-safe negative across entire train-only gallery',
    teacher='Frozen raw RU-DINO L12 or L10, 400 native patches; no learned teacher or CLIP',
    student='Identical L12 RU-global + L12 OT local slots, same pretrained projector initialization in all arms',
    loss='MultiSimilarity with GPS-filtered negative pairs + 0.1 one-sided teacher-margin hinge',
    scope='Exploratory single-seed fixed-last GSV training/holdout screen; not semantic or stable benefit',
    min_net_vs_last=5, min_net_vs_retrieval=0)


@torch.no_grad()
def native_teacher_margin(features, anchors, positives, negatives, chunk=4):
    """MNN pseudomatches, not geometric ground truth; teacher is stop-gradient."""
    if features.ndim!=4 or tuple(features.shape[-2:])!=(20,20):
        raise ValueError('Teacher needs native 20x20 patches')
    tokens=F.normalize(features.detach().flatten(2).transpose(1,2).float(),dim=-1)
    margins,flags=[],[]
    for start in range(0,len(anchors),chunk):
        a,p,n=anchors[start:start+chunk],positives[start:start+chunk],negatives[start:start+chunk]
        ps,pc=mutual_score(tokens[a],tokens[p])
        ns,nc=mutual_score(tokens[a],tokens[n])
        margins.append(ps-ns)
        flags.append((pc>=MID_POLICY['min_matches']) & (nc>=MID_POLICY['min_matches']))
    margin=torch.cat(margins)
    reliable=torch.cat(flags) & (margin>0)
    return margin,reliable


def guidance_loss(descriptors, anchors, positives, negatives, margin, reliable):
    """Lower-bound guidance never shrinks an already superior student margin."""
    target=margin.detach().clamp(MID_POLICY['target_min'],MID_POLICY['target_max'])
    student=((descriptors[anchors]*descriptors[positives]).sum(1)
             -(descriptors[anchors]*descriptors[negatives]).sum(1))
    if not reliable.any():
        return descriptors.sum()*0,torch.zeros_like(reliable)
    violation=F.relu(target-student)
    return violation[reliable].mean(), reliable & (violation.detach()>0)


def gps_filtered_retrieval(loss_function, descriptors, labels, coordinates):
    """Drop near different-ID negative pairs, retaining ordinary MS positives."""
    mined=loss_function.miner(descriptors,labels)
    if len(mined)!=4:
        raise ValueError('Expected MultiSimilarity pair miner')
    ap,p,an,n=mined
    distances=distance_metres(np.asarray(coordinates),np.asarray(coordinates))
    keep=torch.as_tensor(distances,device=descriptors.device)[an,n]>=MID_POLICY['negative_exclusion_metres']
    return loss_function.loss_fn(descriptors,labels,(ap,p,an[keep],n[keep]))


def mine_training_rows(global_scores, positive_indices, anchor_indices, coordinates):
    """Small CPU reference used in regression tests (runtime uses GPU chunks)."""
    scores=np.asarray(global_scores)
    anchors=np.asarray(anchor_indices)
    positives=np.asarray(positive_indices)
    if scores.shape!=(len(anchors),len(coordinates)) or positives.shape!=(len(anchors),3):
        raise ValueError('Training mining shapes differ')
    pos=positives[np.arange(len(anchors)),scores[np.arange(len(anchors))[:,None],positives].argmin(1)]
    allowed=distance_metres(np.asarray(coordinates)[anchors],coordinates)>=MID_POLICY['negative_exclusion_metres']
    allowed &= anchors[:,None]//4 != np.arange(len(coordinates))[None]//4
    if not allowed.any(1).all():
        raise ValueError('No GPS-safe train negative')
    neg=np.where(allowed,scores,-np.inf).argmax(1)
    return np.stack((anchors,pos,neg),axis=1)


def paired_report(finals, groups, ru, hard, cutoff):
    groups,ru,hard=map(np.asarray,(groups,ru,hard))
    results={}
    for name,mask in [('calibration',groups<cutoff),('evaluation',groups>=cutoff)]:
        variants={}
        for mode,outcome in finals.items():
            ok=np.asarray(outcome,bool)
            if ok.shape!=ru.shape:
                raise ValueError('Paired query outcomes differ')
            variants[mode]=dict(correct=int(ok[mask].sum()),r1=float(ok[mask].mean()),
                corrections_vs_ru=int((mask & ~ru & ok).sum()),regressions_vs_ru=int((mask & ru & ~ok).sum()),
                hard_correct=int(ok[mask & hard].sum()))
        comparisons={}
        mid=np.asarray(finals['mid10'],bool)
        for reference in ('retrieval','last12'):
            baseline=np.asarray(finals[reference],bool)
            comparisons['mid10_vs_'+reference]=dict(
                corrections=int((mask & ~baseline & mid).sum()),regressions=int((mask & baseline & ~mid).sum()),
                net=int(mid[mask].sum())-int(baseline[mask].sum()),
                hard_net=int(mid[mask & hard].sum())-int(baseline[mask & hard].sum()))
        results[name]=dict(queries=int(mask.sum()),hard_queries=int((mask & hard).sum()),
                          frozen_ru_correct=int(ru[mask].sum()),variants=variants,comparisons=comparisons)
    return results
