"""Training-only preservation of a frozen author's retrieval relations."""
import math

import numpy as np
import torch
from torch.nn import functional as F

from src.slgd_hard_protocol import distance_metres


ARMS = {
    'metric_control': dict(rank_weight=0., relation_weight=0.),
    'rank_preserve': dict(rank_weight=1., relation_weight=0.),
    'rank_and_relation': dict(rank_weight=1., relation_weight=.1),
}
POLICY = dict(revision=1, arms=ARMS, seed=42, max_training_runs=3, steps=1024,
    rho=.025, lr=1e-5, weight_decay=0., clip=1., microbatch=4,
    train_size=280, eval_sizes=[280,322], primary_eval_size=322,
    layers=[8,10,12], margin_cap=.05, temperature=.07, exclusion_metres=25.,
    initialization='Same trained official seed42 neck as E2, residual scale 0.25; NEW Adam in every arm',
    schedule='Same first1024 rows of sealed original2048 schedule; fixed epoch99 four views',
    teacher='Frozen author BoQ without the neck; descriptors already computed by the unchanged training forward',
    preservation='Teacher-best positive versus student-hardest GPS-safe negative; protect only teacher-correct in-batch anchors, target min(teacher gap,0.05)',
    relation='KL(teacher||student) over same-place positives and GPS-safe negatives, temperature0.07, no temperature-squared multiplier',
    inference='Same frozen DINO+conv/RMS neck+frozen BoQ; single global descriptor and dot product, no teacher/gate/reranking',
    selection='Fixed last1024 for ALL arms; no validation-driven coefficient/layer/epoch/seed selection or extra training',
    success='Candidate only if both primary datasets strictly beat official AND metric-control; separately report280 and regressions vs unchanged E2',
    scope='Warm-start single-seed adaptation on previously exposed validation; not independent test, stable-gain or novelty proof')


def preservation_losses(student, teacher, labels, coordinates):
    """Rank lower bound and soft retrieval-distribution distillation.

    The rank constraint is a soft TRAINING loss, not a guarantee on unseen
    full-gallery rankings. Teacher errors are not protected by the rank term.
    The optional KL still carries teacher mistakes, so its arm is separate.
    """
    if student.ndim!=2 or student.shape!=teacher.shape or len(student)<2:
        raise ValueError('Unmatched descriptor matrices')
    if labels.ndim!=1 or len(labels)!=len(student) or labels.device!=student.device:
        raise ValueError('Unmatched labels/device')
    if student.device!=teacher.device:
        raise ValueError('Unmatched descriptor devices')
    coords=np.asarray(coordinates,dtype=np.float64)
    if coords.shape!=(len(student),2) or not np.isfinite(coords).all():
        raise ValueError('Invalid GPS population')
    if not bool(torch.isfinite(student).all() and torch.isfinite(teacher).all()):
        raise ValueError('Nonfinite descriptors')
    if any(float((x.detach().norm(dim=1)-1).abs().max())>1e-3 for x in (student,teacher)):
        raise ValueError('Expected unit descriptors')
    distances=distance_metres(coords,coords)
    same=labels[:,None]==labels[None,:]
    positive=same & ~torch.eye(len(labels),device=labels.device,dtype=torch.bool)
    negative=~same & (torch.as_tensor(distances,device=labels.device)>=POLICY['exclusion_metres'])
    valid=positive.any(1) & negative.any(1)
    candidate=positive|negative
    # Use finite sentinels. Invalid rows are excluded from every reduction;
    # this avoids 0*inf/NaN in both loss and its backward.
    sentinel=-1e4
    scores=student.float()@student.float().T
    with torch.no_grad():
        teacher_scores=teacher.detach().float()@teacher.detach().float().T
        positive_index=teacher_scores.masked_fill(~positive,sentinel).argmax(1)
        rows=torch.arange(len(labels),device=labels.device)
        teacher_margin=(teacher_scores[rows,positive_index]
                        -teacher_scores.masked_fill(~negative,sentinel).max(1).values)
        protected=valid & (teacher_margin>0)
        target=teacher_margin.clamp(max=POLICY['margin_cap'])
        teacher_logp=F.log_softmax((teacher_scores/POLICY['temperature']).masked_fill(~candidate,sentinel),dim=1)
        teacher_prob=teacher_logp.exp()
    student_margin=(scores[rows,positive_index]-scores.masked_fill(~negative,sentinel).max(1).values)
    violation=F.relu(target-student_margin)
    zero=student.sum()*0.
    rank=violation[protected].mean() if bool(protected.any()) else zero
    student_logp=F.log_softmax((scores/POLICY['temperature']).masked_fill(~candidate,sentinel),dim=1)
    row_kl=(teacher_prob*(teacher_logp-student_logp)).sum(1)
    relation=row_kl[valid].mean().clamp_min(0.) if bool(valid.any()) else zero
    with torch.no_grad():
        actual_student_margin=(scores.masked_fill(~positive,sentinel).max(1).values
                               -scores.masked_fill(~negative,sentinel).max(1).values)
        student_correct=valid & (actual_student_margin>0)
        teacher_correct=valid & (teacher_margin>0)
        stats=dict(valid_anchors=int(valid.sum()),teacher_correct=int(teacher_correct.sum()),
            student_correct=int(student_correct.sum()),
            corrections=int((valid & ~teacher_correct & student_correct).sum()),
            regressions=int((teacher_correct & ~student_correct).sum()),
            active_preservation=int((protected & (violation>0)).sum()),
            allowed_pairs=int(candidate.sum()),near_negative_pairs_excluded=int((~same & ~negative).sum()),
            descriptor_drift=float((1-(student.detach()*teacher.detach()).sum(1)).clamp_min(0.).mean()))
    return rank,relation,stats


def exact_paired(reference, candidate):
    """Two-sided exact discordant-pair binomial test, descriptive only."""
    from src.official_transfer_diagnosis import paired_summary
    result=paired_summary(reference,candidate)
    b,c=len(result['corrections']),len(result['regressions']);n=b+c
    result['exact_two_sided_p']=min(1.,2*sum(math.comb(n,i) for i in range(min(b,c)+1))/(2**n)) if n else 1.
    return result
