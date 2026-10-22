import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.image_negative_pairs import (ARMS,negative_mask,full_train_negatives,
    random_train_negatives,paired_contract,paired_losses)
from scripts.image_negative import configuration,population


def bank():
    torch.manual_seed(31)
    z=F.normalize(torch.randn(12,7),dim=1)
    labels=torch.arange(3).repeat_interleave(4)
    coords=torch.stack((labels.double(),labels.double()),dim=1)
    return z,labels,coords


def pair_fixture():
    z,labels,coords=bank()
    ids=np.roll(np.arange(len(z)),4)
    return torch.cat((z,z[ids])),torch.cat((labels,labels[ids])),torch.cat((coords,coords[ids]))


def test_global_image_mining_matches_brute_force_and_chunk_invariant():
    z,labels,coords=bank()
    mask=negative_mask(labels,labels,coords,coords)
    scores=z@z.T
    positive=(labels[:,None]==labels[None,:]) & ~torch.eye(len(z),dtype=torch.bool)
    expected=scores.masked_fill(~mask,-torch.inf).max(1)
    gap=scores.masked_fill(~positive,-torch.inf).amax(1)-expected.values
    for chunk in (1,5,30):
        ids,margins=full_train_negatives(z,labels,coords,chunk)
        np.testing.assert_array_equal(ids,expected.indices.numpy())
        np.testing.assert_allclose(margins,gap.numpy(),atol=2e-7)


def test_nearby_different_labels_are_excluded():
    labels=torch.tensor([0,1,2]);coords=torch.tensor([[0.,0.],[0.,.00001],[1.,1.]])
    valid=negative_mask(labels,labels,coords,coords)
    assert not valid[0,1] and valid[0,2] and not valid.diag().any()


def test_random_pairs_are_safe_reproducible_and_not_place_averages():
    _,labels,coords=bank()
    a=random_train_negatives(labels.numpy(),coords.numpy(),42)
    b=random_train_negatives(labels.numpy(),coords.numpy(),42)
    np.testing.assert_array_equal(a,b)
    assert negative_mask(labels,labels,coords,coords)[torch.arange(len(labels)),a].all()


def test_random_impossible_population_rejected():
    with pytest.raises(ValueError):random_train_negatives(np.zeros(4),np.zeros((4,2)),42)


@pytest.mark.parametrize('bad',['nonfinite','unnormalized','shape','no_positive','no_negative'])
def test_invalid_mining_bank_rejected(bad):
    z,labels,coords=bank()
    if bad=='nonfinite':z[0,0]=torch.nan
    elif bad=='unnormalized':z*=2
    elif bad=='shape':coords=coords[:2]
    elif bad=='no_positive':labels=torch.arange(len(z))
    else:coords.zero_()
    with pytest.raises(ValueError):full_train_negatives(z,labels,coords)


def test_pair_loss_has_gradient_on_queries_and_fresh_negative_end():
    z,labels,coords=pair_fixture();teacher=paired_contract(z,labels,coords)
    leaf=z.clone().requires_grad_();pair,rank,stats=paired_losses(leaf,teacher,False)
    (pair+rank).backward();n=len(z)//2
    assert torch.isfinite(pair) and stats['queries']==n
    assert torch.isfinite(leaf.grad).all() and leaf.grad[:n].norm()>0 and leaf.grad[n:].norm()>0
    assert rank==0


def test_unchanged_teacher_has_zero_ranking_gradient_including_boundary():
    z,labels,coords=pair_fixture();teacher=paired_contract(z,labels,coords)
    leaf=z.clone().requires_grad_();_,rank,_=paired_losses(leaf,teacher,True)
    rank.backward()
    assert rank==0 and torch.count_nonzero(leaf.grad)==0


def test_protection_frees_teacher_errors_and_penalizes_correct_margin_loss():
    z,labels,coords=pair_fixture();teacher=paired_contract(z,labels,coords)
    assert teacher['correct'].any() and (~teacher['correct']).any()
    student=z.clone()
    # Make each paired negative identical to its query: destroy positive margins.
    n=len(z)//2;student[n:]=student[:n]
    leaf=student.requires_grad_();_,rank,stats=paired_losses(leaf,teacher,True)
    rank.backward()
    assert rank>0 and stats['violating']>0 and leaf.grad.norm()>0
    all_wrong=dict(teacher,correct=torch.zeros(n,dtype=torch.bool))
    _,zero,_=paired_losses(leaf,all_wrong,True)
    assert zero==0


def test_pairs_reject_same_place_and_close_gps():
    z,labels,coords=pair_fixture();n=len(z)//2
    bad=labels.clone();bad[n:]=bad[:n]
    with pytest.raises(ValueError):paired_contract(z,bad,coords)
    with pytest.raises(ValueError):paired_contract(z,labels,coords*0)


def test_config_keeps_fixed_warm_budget_and_capacity():
    c=configuration()
    assert c['warm_arm']=='structure_gate' and c['warm_run']=='logs/structural_channel_v1/train'
    assert c['pilot_steps']==512 and c['smoke_steps']==8 and c['pair_weight']==.1
    assert c['rank_weight']==.1 and c['rank_cap']==.05 and c['cuda_memory_fraction']==.2
    assert len(ARMS)==4 and c['train_places']*4==32768


def test_population_excludes_holdout_and_duplicate_rows():
    c=dict(train_places=1)
    plan=dict(places=[dict(rows=[0,1,2,3])],images=[dict(split='train') for _ in range(4)])
    assert population(plan,c)==[0,1,2,3]
    plan['images'][0]['split']='holdout'
    with pytest.raises(ValueError):population(plan,c)
