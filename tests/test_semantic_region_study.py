"""CPU-only unit tests; supplied for the training machine, not run locally."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.models.semantic_region_pool import (
    SemanticRegionBoQ, SemanticSpatialPool, guidance_loss, permute_target,
    pooled_context, spatial_basis)
from src.models.sclip_region_teacher import correlative_attention, CLASSES
from src.semantic_region_protocol import (
    owned_run, paired, read, restore_rng, rng_state, schedule, seal, seed_all,
    state_hash, verify, write)


def test_basis_partition():
    basis=spatial_basis(7,9,'cpu',torch.float32)
    assert basis.shape==(4,63)
    assert (basis>0).all()
    assert torch.allclose(basis.sum(0),torch.ones(63))
    assert not torch.allclose(basis[0],basis[3])


def test_constant_visual_features_are_preserved_by_pooling():
    features=torch.ones(2,5,7,9)*3.5
    probs=torch.randn(2,18,7,9).softmax(1)
    context,mass=pooled_context(features,probs)
    assert mass.shape==(2,72)
    assert torch.allclose(context,features,atol=1e-5)
    assert torch.allclose(mass.sum(1),torch.full((2,),63.))


def test_class_renaming_does_not_change_visual_pooling():
    x=torch.randn(2,6,5,7)
    p=torch.randn(2,18,5,7).softmax(1)
    a,_=pooled_context(x,p)
    b,_=pooled_context(x,p[:,torch.randperm(18)])
    assert torch.allclose(a,b,atol=2e-6)


@pytest.mark.parametrize('shape',[(1,8,7,7),(3,8,9,6)])
def test_zero_start_and_bounded_update(shape):
    pool=SemanticSpatialPool(channels=8,hidden=4,classes=3)
    x=torch.randn(*shape)
    value,_,_=pool(x)
    assert torch.equal(value,x)
    with torch.no_grad():
        pool.up.weight.normal_(0,20)
    value,_,_=pool(x)
    scale=x.square().mean(1,keepdim=True).sqrt().clamp_min(1e-6)
    assert ((value-x).abs()<=.1*scale+1e-6).all()


def test_teacher_target_is_stop_gradient_but_router_receives_gradient():
    logits=torch.randn(2,4,9,9,requires_grad=True)
    target=torch.randn(2,4,7,7).softmax(1).requires_grad_()
    loss,confidence=guidance_loss(logits,target)
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum()>0
    assert target.grad is None
    assert 0<=confidence<=1


def test_uniform_teacher_does_not_create_nonfinite_loss():
    logits=torch.zeros(2,4,9,9,requires_grad=True)
    loss,_=guidance_loss(logits,torch.full((2,4,7,7),.25))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_permutation_is_deterministic_image_keyed_derangement():
    value=torch.arange(2*3*7*7).reshape(2,3,7,7).float()
    result=permute_target(value,['a.jpg','b.jpg'])
    assert torch.equal(result,permute_target(value,['a.jpg','b.jpg']))
    assert torch.equal(result[1],permute_target(value[1:],['b.jpg'])[0])
    assert (result!=value).all()
    assert torch.equal(result.flatten(2).sort(-1).values,value.flatten(2).sort(-1).values)
    assert torch.equal(result.sum((2,3)),value.sum((2,3)))


def test_pool_and_router_receive_retrieval_gradient_after_zero_start():
    pool=SemanticSpatialPool(channels=8,hidden=4,classes=3)
    with torch.no_grad():
        pool.up.weight.normal_(0,.1)
    x=torch.randn(2,8,7,7)
    value,_,_=pool(x)
    (value*torch.randn_like(value)).sum().backward()
    for module in [pool.router,pool.up,pool.down,pool.norm]:
        assert any(p.grad is not None and bool((p.grad!=0).any()) for p in module.parameters())


class FakeOfficial(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=nn.Conv2d(3,768,1)
        self.aggregator=FakeAggregator()

    def forward(self,x):
        return self.aggregator(self.backbone(x))[0]


class FakeAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(768,16)

    def forward(self,x):
        return F.normalize(self.fc(x.mean((2,3))),dim=-1),None


def test_student_is_teacher_free_and_keeps_backbone_frozen():
    official=FakeOfficial()
    model=SemanticRegionBoQ(official,hidden=4)
    model.train()
    assert not model.visual.backbone.training
    assert all(not p.requires_grad for p in model.visual.backbone.parameters())
    assert all(p.requires_grad for p in model.visual.aggregator.parameters())
    x=torch.randn(2,3,5,5)
    model.eval()
    assert torch.allclose(model(x),official(x),atol=1e-6)
    model(x).square().sum().backward()
    assert all(p.grad is None for p in model.visual.backbone.parameters())
    assert not any('teacher' in n or 'clip' in n for n in model.state_dict())


def test_three_arm_initialization_and_capacity_match():
    models=[]
    for _ in range(3):
        seed_all(42)
        models.append(SemanticRegionBoQ(FakeOfficial(),hidden=4))
    assert len({state_hash(m) for m in models})==1
    assert len({sum(p.numel() for p in m.parameters() if p.requires_grad) for m in models})==1


def test_csa_matches_per_head_equation_not_qk_attention():
    attn=nn.MultiheadAttention(8,2,batch_first=True)
    x=torch.randn(2,5,8)
    q,k,v=F.linear(x,attn.in_proj_weight,attn.in_proj_bias).chunk(3,-1)
    outputs=[]
    for h in range(2):
        qq,kk,vv=[a[:,:,h*4:(h+1)*4] for a in (q,k,v)]
        weights=(qq@qq.transpose(1,2)/2).softmax(-1)+(kk@kk.transpose(1,2)/2).softmax(-1)
        outputs.append(weights@vv)
    reference=attn.out_proj(torch.cat(outputs,-1))
    assert torch.allclose(correlative_attention(attn,x),reference,atol=1e-6)


def test_schedule_each_place_exactly_once_per_epoch():
    rows=schedule(32,3,8,42)
    assert rows==schedule(32,3,8,42)
    for i in range(3):
        assert sorted(x for row in rows[4*i:4*i+4] for x in row)==list(range(32))
    assert rows[:4]!=rows[4:8]


def test_atomic_contract_resume_and_corruption_detection(tmp_path):
    path=tmp_path/'run'
    with owned_run(path,{'schema':'test'}):
        write(path/'report.json',{'value':1})
        seal(path)
    assert verify(path)=={'schema':'test'}
    with pytest.raises(FileExistsError):
        with owned_run(path,{'schema':'test'}): pass
    with pytest.raises(ValueError):
        with owned_run(path,{'schema':'changed'},True): pass
    (path/'report.json').write_text('{}')
    with pytest.raises(ValueError): verify(path)


def test_rng_roundtrip():
    seed_all(42)
    value=rng_state()
    expected=torch.randn(4),np.random.rand(4)
    restore_rng(value)
    assert torch.equal(torch.randn(4),expected[0])
    assert np.array_equal(np.random.rand(4),expected[1])


def test_paired_counts():
    result=paired([True,True,False,False],[True,False,True,True])
    assert result['corrections']==2 and result['regressions']==1 and result['net']==1


def test_all_training_modes_require_explicit_authorization():
    script=Path(__file__).resolve().parents[1]/'scripts/semantic_region_study.py'
    for command in ('smoke','train'):
        result=subprocess.run([sys.executable,str(script),command],capture_output=True,text=True)
        assert result.returncode!=0 and '--allow-training' in result.stderr


def test_config_preserves_blind_test_and_five_epoch_schedule():
    from scripts.semantic_region_study import configuration
    root=Path(__file__).resolve().parents[1]
    c=configuration(root/'config/semantic_region_study.yaml')
    assert c['epochs']==5 and c['student_size']==322 and len(CLASSES)==18
    assert c['train_places']//c['places_per_batch']*c['epochs']==2560
    assert c['pitts_val_root'].endswith('pitts30k-val')
    assert 'pitts30k-test' not in json.dumps(c)
