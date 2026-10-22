from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from src.channel_rank_preservation import ARMS,valid_pairs,teacher_contract,ranking_penalty
from src.models.structural_channel_adapter import StructuralChannelVPR
from scripts.structural_channel import replay
from scripts.channel_preservation import configuration


@pytest.fixture(autouse=True)
def deterministic():torch.set_num_threads(2);torch.manual_seed(42)


def batch():
    z=F.normalize(torch.tensor([[1.,.01],[1.,.02],[.01,1.],[.02,1.]]),dim=1)
    labels=torch.tensor([0,0,1,1])
    gps=torch.tensor([[0.,0.],[0.,.000001],[1.,1.],[1.,1.000001]],dtype=torch.float64)
    return z,labels,gps


def test_gps_excludes_nearby_different_id_but_not_same_id_positive():
    labels=torch.tensor([0,0,1,2]);gps=torch.tensor([[0.,0.],[0.,0.],[0.,.00001],[1.,1.]])
    pos,neg=valid_pairs(labels,gps)
    assert pos[0,1] and not pos[0,0] and not neg[0,2] and neg[0,3]
    assert not (pos&neg).any()


def test_teacher_correct_mask_and_clipped_target():
    z,y,gps=batch();t=teacher_contract(z,y,gps)
    assert t['valid'].all() and t['correct'].all()
    torch.testing.assert_close(t['target'],torch.full((4,),.05))
    loss,s=ranking_penalty(z,t,'correct_margin')
    assert loss==0 and s['protected']==4 and s['violating']==0


def test_zero_violation_boundary_has_zero_gradient_not_clamp_subgradient():
    z,y,gps=batch()
    t=teacher_contract(z,y,gps)
    scores=z@z.T
    # Force an exact boundary for each valid row, below clipping concerns.
    t['target']=scores.masked_fill(~t['positive'],-torch.inf).amax(1)-scores.masked_fill(~t['negative'],-torch.inf).amax(1)
    for arm in ARMS:
        student=z.clone().requires_grad_()
        penalty,_=ranking_penalty(student,t,arm)
        penalty.backward()
        assert penalty==0 and torch.count_nonzero(student.grad)==0


def test_wrong_teacher_not_protected_by_selective_but_all_margin_preserves_it():
    good,y,gps=batch();bad=good[[0,2,1,3]]
    t=teacher_contract(bad,y,gps)
    assert not t['correct'].any()
    all_loss,_=ranking_penalty(good,t,'all_margin')
    selective,s=ranking_penalty(good,t,'correct_margin')
    assert all_loss>0 and selective==0 and s['protected']==0


def test_worsening_correct_order_has_nonzero_finite_gradient():
    good,y,gps=batch();t=teacher_contract(good,y,gps)
    student=good[[0,2,1,3]].clone().requires_grad_()
    loss,s=ranking_penalty(student,t,'correct_margin');loss.backward()
    assert loss>0 and s['violating']>0
    assert torch.isfinite(student.grad).all() and student.grad.norm()>0


def test_loss_not_elementwise_descriptor_distillation():
    z,y,gps=batch();t=teacher_contract(z,y,gps)
    rotation=torch.tensor([[0.,-1.],[1.,0.]])
    student=z@rotation
    loss,_=ranking_penalty(student,t,'correct_margin')
    assert loss==0 and not torch.allclose(z,student)


def test_teacher_is_detached():
    z,y,gps=batch();ref=z.clone().requires_grad_();t=teacher_contract(ref,y,gps)
    student=z[[0,2,1,3]].clone().requires_grad_()
    ranking_penalty(student,t,'correct_margin')[0].backward()
    assert ref.grad is None and student.grad is not None


@pytest.mark.parametrize('arm',ARMS)
def test_no_valid_negative_returns_connected_finite_zero(arm):
    z,y,gps=batch();gps=torch.zeros_like(gps)
    t=teacher_contract(z,y,gps);student=z.clone().requires_grad_()
    loss,s=ranking_penalty(student,t,arm);loss.backward()
    assert torch.isfinite(loss) and loss==0 and s['valid']==0
    assert torch.isfinite(student.grad).all() and torch.count_nonzero(student.grad)==0


def test_no_positive_or_single_sample_safe():
    for n in (1,4):
        z=F.normalize(torch.randn(n,5),dim=1);y=torch.arange(n);gps=torch.ones(n,2)
        t=teacher_contract(z,y,gps);v=z.clone().requires_grad_()
        loss,_=ranking_penalty(v,t,'all_margin');loss.backward()
        assert loss==0 and torch.isfinite(v.grad).all()


def test_masks_use_training_labels_and_have_same_batch_denominator():
    z,y,gps=batch();t=teacher_contract(z,y,gps)
    student=z[[0,2,1,3]]
    full,_=ranking_penalty(student,t,'correct_margin')
    one=deepcopy(t);one['correct'][1:]=False
    subset,_=ranking_penalty(student,one,'correct_margin')
    scores=student@student.T
    margin=scores[0,1]-scores[0,2:].max()
    expected=(.05-margin).clamp_min(0)/.05/4
    torch.testing.assert_close(subset,expected)
    assert subset<=full


class Anchor(torch.nn.Module):
    def __init__(self):super().__init__();self.fc=torch.nn.Linear(12,16)
    def readout(self,x):return F.normalize(self.fc(x.mean((2,3))),dim=1)


@pytest.mark.parametrize('arm',ARMS)
def test_full_batch_rank_gradient_replay_matches_monolithic(arm):
    a=StructuralChannelVPR(Anchor(),'structure_gate',channels=12,hidden=8,rank=4)
    torch.nn.init.normal_(a.adapter.up.weight,std=.1)
    b=deepcopy(a)
    features=(torch.randn(4,12,5,5),F.normalize(torch.randn(4,16),dim=1))
    _,y,gps=batch();t=teacher_contract(features[1],y,gps)
    z,_=a.describe(features,[1,2,3,4]);loss=z.square().sum()+ranking_penalty(z,t,arm)[0]
    loss.backward()
    with torch.no_grad():cached=b.describe(features,[1,2,3,4])[0].detach().requires_grad_()
    cached_loss=cached.square().sum()+ranking_penalty(cached,t,arm)[0]
    grad=torch.autograd.grad(cached_loss,cached)[0]
    replay(b,[tuple(x[:2] for x in features),tuple(x[2:] for x in features)],[1,2,3,4],grad)
    for p,q in zip(a.adapter.parameters(),b.adapter.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=3e-4)


def test_permutation_equivariance_of_full_batch_objective():
    z,y,gps=batch();v=z[[0,2,1,3]];p=torch.tensor([2,0,3,1])
    t=teacher_contract(z,y,gps);tp=teacher_contract(z[p],y[p],gps[p])
    for arm in ARMS:torch.testing.assert_close(ranking_penalty(v,t,arm)[0],ranking_penalty(v[p],tp,arm)[0])


@pytest.mark.parametrize('bad',['arm','gps','cap','shape'])
def test_invalid_input_rejected(bad):
    z,y,gps=batch()
    with pytest.raises(ValueError):
        if bad=='arm':ranking_penalty(z,teacher_contract(z,y,gps),'bad')
        elif bad=='gps':valid_pairs(y,gps[:2])
        elif bad=='cap':teacher_contract(z,y,gps,0)
        else:teacher_contract(z[:2],y,gps)


def test_locked_short_repair_budget():
    c=configuration()
    assert c['pilot_steps']==512 and c['warm_arm']=='structure_gate' and c['rank_weight']==.1
    assert c['rank_cap']==.05 and c['cuda_memory_fraction']==.2 and c['keep_weight']==.1
