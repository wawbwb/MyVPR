"""Run only on the training machine; no downloads or new environment."""
from pathlib import Path
import hashlib

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.hcen_protocol import POLICY,soft_tau,ranking_regularizer,training_schedule,paired_partitions
from src.models.hcen_vpr import HierarchicalNeck,HCENVPR,SASF,MODES


class Readout(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(8,12,bias=False)

    def forward(self,x):return F.normalize(self.linear(x.mean((2,3))),dim=-1)


class Visual(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=nn.Conv2d(3,8,1);self.aggregator=Readout();self.semantic_region_gate=None


def model(mode='hierarchy'):
    torch.manual_seed(42)
    return HCENVPR(Visual(),mode,channels=8,dim=8,inner=8,state=2)


def maps():
    torch.manual_seed(21)
    return tuple(torch.randn(4,8,4,4) for _ in range(3))


def rank_example():
    torch.manual_seed(15)
    labels=torch.arange(4).repeat_interleave(4)
    ref=F.normalize(torch.eye(4)[labels]+torch.randn(16,4)*.05,dim=-1).requires_grad_()
    desc=ref.detach().clone().requires_grad_()
    coords=np.column_stack((labels.numpy()*.01,np.zeros(16)))
    return desc,ref,labels,coords


def test_soft_tau_same_and_reverse_order():
    x=torch.tensor([[0.,.1,.2]])
    mask=torch.ones_like(x,dtype=torch.bool)
    a,valid=soft_tau(x,x,mask);b,_=soft_tau(x,-x,mask)
    assert valid.all() and float(a)> .99 and float(b)<-.99


@pytest.mark.parametrize('count',[0,1])
def test_soft_tau_no_pairs_is_finite(count):
    x=torch.zeros(2,count)
    tau,valid=soft_tau(x,x,torch.ones_like(x,dtype=torch.bool))
    assert torch.equal(tau,torch.zeros(2)) and not valid.any()


def test_soft_tau_mask_hides_wrong_stratum():
    x=torch.tensor([[0.,.1,99.]])
    y=torch.tensor([[0.,.1,-99.]])
    tau,_=soft_tau(x,y,torch.tensor([[True,True,False]]))
    assert float(tau)> .99


def test_soft_tau_rejects_shapes():
    with pytest.raises(ValueError):soft_tau(torch.zeros(1,2),torch.zeros(1,3),torch.ones(1,2,dtype=torch.bool))


def test_ranking_regularizer_reference_is_detached():
    d,r,y,c=rank_example()
    loss,stats=ranking_regularizer(d,r,y,c)
    assert torch.isfinite(loss) and float(loss.detach())>0
    loss.backward()
    assert r.grad is None and d.grad is not None and torch.isfinite(d.grad).all()
    assert stats['intra']['coverage']>0 and stats['inter']['coverage']>0


def test_ranking_gps_nearby_negative_places_abstain():
    d,r,y,c=rank_example();c[:]=0
    loss,stats=ranking_regularizer(d,r,y,c);loss.backward()
    assert float(loss.detach())==0 and torch.equal(d.grad,torch.zeros_like(d))
    assert stats['intra']['coverage']==stats['inter']['coverage']==0


def test_ranking_no_positive_stratum_abstains():
    d,r,y,c=rank_example();y=torch.arange(16)
    loss,stats=ranking_regularizer(d,r,y,c);loss.backward()
    assert float(loss.detach())==0 and all(s['coverage']==0 for s in stats.values())


def test_regularizer_does_not_reward_anticorrelation():
    values=torch.tensor([-.8,0.,.5,.8])
    penalty=F.relu(values-POLICY['rank_upper']).square()
    assert penalty[:3].tolist()==[0.,0.,0.] and float(penalty[3])>0
    assert POLICY['rank_weight']==.02 and POLICY['rank_reference_gap']>0


def test_sasf_both_paths_receive_finite_gradients():
    block=SASF(8)
    block(torch.randn(2,8,4,4)).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
    assert all(path[0].weight.grad.abs().sum()>0 for path in block.paths)


@pytest.mark.parametrize('mode',MODES)
def test_zero_start_reproduces_frozen_readout(mode):
    m=model(mode);layers=maps()
    d,*_=m.aggregate(*layers,auxiliary=True)
    assert torch.equal(d,m.readout(layers[-1]))
    assert all(not p.requires_grad for p in m.visual.parameters())
    m.train();assert not m.visual.training and m.neck.training


def test_feature_residual_is_bounded_not_a_descriptor_guarantee():
    m=model()
    with torch.no_grad():m.neck.up.weight.normal_(std=10)
    mixed,aux,residual=m.neck(maps(),True)
    cap=residual.flatten(1).norm(dim=1)/maps()[-1].flatten(1).norm(dim=1)
    assert float(cap.detach().max())<=.100001 and len(aux)==2
    assert torch.allclose(aux[0].norm(dim=-1),torch.ones(4),atol=1e-6)


def test_auxiliary_gradient_reaches_selective_parameters_after_warm_start():
    m=model();optimizer=torch.optim.AdamW(m.neck.parameters(),lr=1e-3)
    layers=maps()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        d,a,b,_=m.aggregate(*layers,auxiliary=True)
        loss=d[:,0].mean()+a[:,0].mean()+b[:,1].mean();loss.backward()
        optimizer.step()
    selected=[p.grad for n,p in m.neck.named_parameters() if 'ssm.select' in n]
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in selected)
    assert all(p.grad is None for p in m.visual.parameters())


def test_last_only_control_has_same_parameters_but_no_cross_depth_input():
    models=[model(mode) for mode in MODES]
    hashes=[]
    for m in models:
        h=hashlib.sha256()
        for name,p in m.neck.named_parameters():h.update(name.encode());h.update(p.detach().numpy().tobytes())
        hashes.append(h.hexdigest())
        with torch.no_grad():m.neck.up.weight.normal_(std=.1)
    assert len(set(hashes))==1
    l8,l10,l12=maps()
    m=models[0]
    assert torch.equal(m.aggregate(l8,l10,l12)[0],m.aggregate(l8*5,l10*3,l12)[0])
    m=models[1]
    assert not torch.equal(m.aggregate(l8,l10,l12)[0],m.aggregate(-l8,-l10,l12)[0])


def test_fixed_schedule_uniformly_covers_training_anchor_ids():
    rows=[[4*i,4*i+1,4*((i+1)%8192)] for i in range(8192)]
    batches=training_schedule(rows,'pilot')
    assert len(batches)==128 and max(map(len,batches))<=16
    assert all(0<=p<8192 for batch in batches for p in batch)
    assert batches==training_schedule(rows,'pilot')
    assert 0 in batches[0] and 8184 in batches[-1]


def test_schedule_rejects_insufficient_rows():
    with pytest.raises(ValueError):training_schedule([[0,1,4]],'pilot')


def test_pair_report_counts_cross_arm_not_just_ru():
    report=paired_partitions(dict(last_only=[True,False],hierarchy=[False,True],hierarchy_rank=[True,True]),
        [True,False],[0,1],[False,True],1)
    assert report['all']['comparisons']['hierarchy_vs_last_only']==dict(corrections=1,regressions=1,net=0)
    assert report['evaluation']['variants']['hierarchy']['hard_correct']==1


def test_launcher_is_lf_only_and_uses_existing_environment_lock():
    root=Path(__file__).resolve().parents[1];raw=(root/'scripts/run_hcen_vpr.sh').read_bytes()
    assert b'\r' not in raw
    text=raw.decode()
    assert 'CUDA_VISIBLE_DEVICES=1' in text and 'query_relation_screen_v1.lock' in text
    assert 'pip install' not in text and 'conda create' not in text
    assert 'for MODE in last_only hierarchy hierarchy_rank' in text
    assert POLICY['selection'].startswith('Fixed last')
