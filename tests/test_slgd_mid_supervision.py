"""Run on training machine only: layer source, reliability and paired controls."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.slgd_mid_supervision import (MID_POLICY,native_teacher_margin,guidance_loss,
                                     gps_filtered_retrieval,mine_training_rows,paired_report)


def test_teacher_native_matches_are_detached_and_positive_when_views_match():
    torch.manual_seed(1)
    anchor=torch.randn(1,32,20,20)
    features=torch.cat((anchor,anchor,torch.randn_like(anchor))).requires_grad_()
    margin,valid=native_teacher_margin(features,torch.tensor([0]),torch.tensor([1]),torch.tensor([2]))
    assert valid.tolist()==[True] and float(margin[0])>0
    assert not margin.requires_grad


def test_teacher_rejects_pooled_grid():
    with pytest.raises(ValueError):
        native_teacher_margin(torch.randn(3,8,10,10),torch.tensor([0]),torch.tensor([1]),torch.tensor([2]))


def test_negative_teacher_margin_abstains():
    torch.manual_seed(2)
    a=torch.randn(1,32,20,20)
    features=torch.cat((a,torch.randn_like(a),a))
    margin,valid=native_teacher_margin(features,torch.tensor([0]),torch.tensor([1]),torch.tensor([2]))
    assert float(margin[0])<0 and not valid.any()


def test_guidance_does_not_backprop_teacher_and_moves_wrong_student_relation():
    d=torch.tensor([[1.,0],[0.,1],[1.,0]],requires_grad=True)
    teacher=torch.tensor([.03],requires_grad=True)
    loss,active=guidance_loss(d,torch.tensor([0]),torch.tensor([1]),torch.tensor([2]),teacher,torch.tensor([True]))
    assert active.tolist()==[True] and float(loss.detach())>0
    loss.backward()
    assert teacher.grad is None and d.grad.abs().sum()>0


def test_guidance_never_shrinks_already_better_student_margin():
    d=torch.tensor([[1.,0],[1.,0],[0.,1]],requires_grad=True)
    loss,active=guidance_loss(d,torch.tensor([0]),torch.tensor([1]),torch.tensor([2]),torch.tensor([.03]),torch.tensor([True]))
    assert float(loss.detach())==0 and not active.any()
    loss.backward()
    assert torch.equal(d.grad,torch.zeros_like(d))


def test_no_reliable_target_has_zero_finite_loss_and_gradient():
    d=torch.randn(3,8,requires_grad=True)
    loss,active=guidance_loss(d,torch.tensor([0]),torch.tensor([1]),torch.tensor([2]),torch.tensor([-.01]),torch.tensor([False]))
    assert torch.isfinite(loss) and float(loss.detach())==0 and not active.any()
    loss.backward()
    assert torch.equal(d.grad,torch.zeros_like(d))


def test_training_mining_uses_weakest_positive_and_gps_safe_hardest_negative():
    coordinates=np.array([[0.,0.]]*4+[[0.,.00001]]*4+[[0.,.01]]*4)
    scores=np.zeros((1,12))
    scores[0]=[1,.7,.4,.8,.99,.98,.97,.96,.6,.5,.4,.3]
    triplets=mine_training_rows(scores,[[1,2,3]],[0],coordinates)
    assert triplets.tolist()==[[0,2,8]]
    with pytest.raises(ValueError):
        mine_training_rows(scores,[[1,2,3]],[0],np.zeros((12,2)))


def test_ms_loss_filters_near_other_place_negatives_only():
    device='cpu'
    def miner(d,labels):
        return tuple(torch.tensor(x) for x in ([0],[1],[0,0],[2,3]))
    captured={}
    def loss(d,labels,indices):
        captured['indices']=indices
        return d.sum()*0
    fn=SimpleNamespace(miner=miner,loss_fn=loss)
    gps_filtered_retrieval(fn,torch.randn(4,8),torch.tensor([0,0,1,2]),
                           [[0,0],[0,0],[0,.00001],[0,.01]])
    assert captured['indices'][1].tolist()==[1]
    assert captured['indices'][2].tolist()==[0] and captured['indices'][3].tolist()==[3]


def test_paired_outcomes_report_both_references_and_keep_ru_hard_fixed():
    finals=dict(retrieval=[True,False,True,False],last12=[True,False,False,True],mid10=[True,True,True,True])
    result=paired_report(finals,np.arange(4),[True,False,True,False],[False,True,False,True],2)
    assert result['calibration']['comparisons']['mid10_vs_last12']['net']==1
    assert result['evaluation']['comparisons']['mid10_vs_retrieval']['net']==1
    assert result['evaluation']['variants']['mid10']['corrections_vs_ru']==1


def test_schedule_budget_is_exact_and_inference_architecture_shared():
    assert MID_POLICY['pilot_steps']*MID_POLICY['batch_triplets']==MID_POLICY['pilot_places']
    assert MID_POLICY['smoke_steps']*MID_POLICY['batch_triplets']==MID_POLICY['smoke_places']
    assert MID_POLICY['modes']==['retrieval','last12','mid10']
    assert MID_POLICY['teacher_grid']==20
    root=Path(__file__).resolve().parents[1]
    launcher=(root/'scripts/run_slgd_mid_screen.sh').read_bytes()
    assert b'\r' not in launcher and b'flock -n' in launcher
    assert b'sed -i' not in launcher and b'CUDA_VISIBLE_DEVICES=1' in launcher


def test_triplet_rows_reconstruct_same_views_without_resampling():
    from scripts.slgd_mid_screen import RowTriplets
    obj=object.__new__(RowTriplets)
    obj.triplets=np.array([[0,2,9],[4,7,15]])
    obj.place_images=lambda place:torch.arange(4)+10*place
    images,rows=obj[0]
    assert images.tolist()==[0,2,21] and rows.tolist()==[0,2,9]
