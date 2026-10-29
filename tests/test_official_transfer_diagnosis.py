import numpy as np
import pytest
import torch
from torch import nn

from src.official_transfer_diagnosis import POLICY,EXPERIMENTS,ScaledRMSNeck,choose_negative_batch,paired_summary,gradient_comparison
from src.models.conv_residual_ablation import ResidualRuleNeck


def toy_neck():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__();self.norms=nn.ModuleList([nn.LayerNorm(4) for _ in range(3)])
            self.projections=nn.ModuleList([nn.Linear(4,4,bias=False) for _ in range(3)])
            class Stage(nn.Module):
                def forward(self,a,b):return a+b
            self.stages=nn.ModuleList([Stage(),Stage()]);self.aux=nn.ModuleList([nn.Linear(4,2),nn.Linear(4,2)])
            self.up=nn.Conv2d(4,4,1,bias=False)
    return Toy()


def test_exactly_five_items_and_only_three_new_training_runs():
    assert len(EXPERIMENTS)==POLICY['max_experiments']==5
    assert sum(x['kind']=='train' for x in EXPERIMENTS.values())==3
    assert POLICY['steps']==2048 and POLICY['seed']==42


def test_scaling_changes_no_parameter_and_preserves_gradients():
    original=toy_neck();full=ResidualRuleNeck(original,'rms_tanh');small=ScaledRMSNeck(full,.025)
    x=tuple(torch.randn(2,4,3,3) for _ in range(3))
    f=full(x)[-1];s=small(x)[-1]
    assert torch.equal(s,.25*f)
    assert set(full.state_dict())==set(small.state_dict())
    s.sum().backward();assert original.up.weight.grad is not None
    assert bool((original.up.weight.grad!=0).any())


def test_scaling_bound_and_zero_reference():
    neck=ScaledRMSNeck(toy_neck(),.025)
    x=tuple(torch.randn(2,4,3,3) for _ in range(3))
    x[-1][0,:,0,0]=0
    residual=neck(x)[-1]
    assert torch.equal(residual[0,:,0,0],torch.zeros(4))
    assert float((residual.flatten(1).norm(dim=1)/x[-1].flatten(1).norm(dim=1)).max())<=.025+1e-7


@pytest.mark.parametrize('rho',[0,-.01,.11])
def test_invalid_extra_scale_rejected(rho):
    with pytest.raises(ValueError):ScaledRMSNeck(toy_neck(),rho)


def mining_inputs():
    coords=np.column_stack((np.arange(20)*.01,np.zeros(20)))
    scores=np.ones((8,20));scores[:,0]=100
    return list(range(8)),scores,coords


@pytest.mark.parametrize('count',[8,12,16])
def test_native_negatives_keep_anchors_counts_and_deterministic_ties(count):
    anchors,scores,coords=mining_inputs()
    batch,selected=choose_negative_batch(anchors,scores,coords,count)
    assert batch==list(range(count)) and batch[:8]==anchors
    assert len(selected)==count-8 and len(set(batch))==count


def test_nearby_different_place_is_not_a_negative():
    anchors,scores,coords=mining_inputs();coords[8]=coords[0]
    scores[0,8]=1000
    batch,selected=choose_negative_batch(anchors,scores,coords,9)
    assert batch[-1]==9 and selected[0]['negative']==9


def test_mining_never_silently_falls_back():
    anchors,scores,coords=mining_inputs();coords[:]=0
    with pytest.raises(ValueError,match='no silent fallback'):choose_negative_batch(anchors,scores,coords,16)


def test_nonfinite_mining_rejected():
    anchors,scores,coords=mining_inputs();scores[0,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):choose_negative_batch(anchors,scores,coords,16)


def test_paired_outcomes_keep_corrections_and_regressions():
    p=paired_summary([True,False,True,False],[False,True,True,True])
    assert p['corrections']==[1,3] and p['regressions']==[0] and p['net']==1
    assert p['delta_r1_pp']==25


def test_training_controls_change_one_registered_factor():
    assert EXPERIMENTS['e3_low_rho']==dict(kind='train',rho=.025,lr=1e-4,negatives='historical')
    assert EXPERIMENTS['e4_low_lr']==dict(kind='train',rho=.1,lr=1e-5,negatives='historical')
    assert EXPERIMENTS['e5_native_negatives']==dict(kind='train',rho=.1,lr=1e-4,negatives='official')


def test_gradient_comparison_keeps_strict_limit():
    a=torch.ones(4,dtype=torch.float64)
    assert gradient_comparison(a,a,1e-10)['passed']
    assert not gradient_comparison(a,a+.0002,1e-4)['passed']


@pytest.mark.parametrize('candidate',[torch.zeros(4),torch.full((4,),float('nan'))])
def test_zero_or_nonfinite_gradients_cannot_pass(candidate):
    assert not gradient_comparison(torch.zeros(4),candidate,1e-4)['passed']
