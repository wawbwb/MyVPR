import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.official_adaptation import ARMS,POLICY,preservation_losses,exact_paired


def batch():
    teacher=F.normalize(torch.tensor([[1.,.1,0.],[1.,-.1,0.],[0.,.1,1.],[0.,-.1,1.]]),dim=1)
    labels=torch.tensor([0,0,1,1])
    coords=np.array([[0.,0.],[0.,0.],[1.,0.],[1.,0.]])
    return teacher,labels,coords


def test_three_matched_arms_only():
    assert len(ARMS)==POLICY['max_training_runs']==3
    assert ARMS['metric_control']==dict(rank_weight=0.,relation_weight=0.)
    assert ARMS['rank_preserve']==dict(rank_weight=1.,relation_weight=0.)
    assert ARMS['rank_and_relation']==dict(rank_weight=1.,relation_weight=.1)
    assert POLICY['steps']==1024 and POLICY['rho']==.025


def test_identical_descriptors_have_zero_preservation_and_relation_losses():
    t,y,c=batch();s=t.clone().requires_grad_()
    rank,relation,stats=preservation_losses(s,t,y,c)
    assert rank.item()==0. and relation.item()<1e-7
    assert stats['valid_anchors']==stats['teacher_correct']==4
    assert stats['active_preservation']==stats['regressions']==0
    (rank+relation).backward();assert torch.isfinite(s.grad).all()


def test_destroyed_rank_is_penalized_and_teacher_has_no_gradient():
    t,y,c=batch();t.requires_grad_()
    s=t.detach()[[0,2,1,3]].clone().requires_grad_()
    rank,relation,stats=preservation_losses(s,t,y,c)
    assert rank.item()>0 and relation.item()>0 and stats['regressions']>0
    (rank+.1*relation).backward()
    assert t.grad is None and s.grad is not None and bool((s.grad!=0).any())


def test_teacher_errors_are_not_rank_protected():
    good,y,c=batch();bad=good[[0,2,1,3]]
    rank,_,stats=preservation_losses(bad,bad,y,c)
    assert stats['teacher_correct']==0 and rank.item()==0


def test_near_different_place_excluded_and_no_negatives_is_finite():
    t,y,c=batch();c[:]=0
    s=t.clone().requires_grad_();rank,relation,stats=preservation_losses(s,t,y,c)
    assert stats['valid_anchors']==0 and stats['near_negative_pairs_excluded']==8
    assert rank.item()==relation.item()==0
    (rank+relation).backward();assert torch.isfinite(s.grad).all()


def test_no_positives_or_no_negatives_returns_differentiable_zero():
    t,_,c=batch()
    for labels in (torch.arange(4),torch.zeros(4,dtype=torch.long)):
        s=t.clone().requires_grad_();rank,relation,stats=preservation_losses(s,t,labels,c)
        assert stats['valid_anchors']==0 and rank.item()==relation.item()==0
        (rank+relation).backward();assert torch.isfinite(s.grad).all()


def test_joint_permutation_does_not_change_objectives():
    t,y,c=batch();s=F.normalize(t+torch.tensor([.05,-.1,.1]),dim=1)
    a,b,stats=preservation_losses(s,t,y,c);p=torch.tensor([3,1,0,2])
    x,z,permuted=preservation_losses(s[p],t[p],y[p],c[p.numpy()])
    assert torch.allclose(a,x,atol=1e-7) and torch.allclose(b,z,atol=1e-7)
    assert stats['teacher_correct']==permuted['teacher_correct']


@pytest.mark.parametrize('kind',['shape','nan','norm','gps'])
def test_bad_populations_rejected(kind):
    t,y,c=batch();s=t.clone()
    if kind=='shape':s=s[:3]
    if kind=='nan':s[0,0]=float('nan')
    if kind=='norm':s*=2
    if kind=='gps':c[0,0]=float('nan')
    with pytest.raises(ValueError):preservation_losses(s,t,y,c)


def test_exact_paired_keeps_both_directions():
    result=exact_paired([True,False,True,False],[False,True,True,True])
    assert result['net']==1 and result['exact_two_sided_p']==1.
