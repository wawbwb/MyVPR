"""CPU synthetic tests only; never invokes real-image smoke or training."""
import copy
import inspect
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from src.models.single_dino_interaction import SingleDinoInteraction, ARMS
from src.models.single_dino_supervision import PostInteractionFusion, PostInteractionVPR
from src.single_dino_grad_balance import combine


def head(cls,arm='aligned'):
    torch.manual_seed(42)
    return cls(arm,channels=12,dim=8,rank=4)


@pytest.mark.parametrize('arm',ARMS)
def test_v1_v2_exact_inference_path_and_parameter_inventory(arm):
    old=head(SingleDinoInteraction,arm);new=head(PostInteractionFusion,arm)
    assert set(old.state_dict())==set(new.state_dict())
    assert all(torch.equal(v,new.state_dict()[n]) for n,v in old.state_dict().items())
    with torch.no_grad():
        old.output.weight.normal_(std=.05);new.load_state_dict(old.state_dict())
    x=torch.randn(3,12,3,4)
    a,aux0,s0=old(x);b,aux1,s1=new(x)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    torch.testing.assert_close(s0['relative_residual'],s1['relative_residual'],atol=0,rtol=0)
    assert not torch.allclose(aux0[0],aux1[0],atol=1e-7,rtol=1e-7)
    assert float(s1['relative_residual'].max())<=.025+1e-6


@pytest.mark.parametrize('arm',ARMS)
def test_post_exchange_auxiliary_directly_reaches_exchange_scan(arm):
    m=head(PostInteractionFusion,arm);x=torch.randn(4,12,3,4)
    _,aux,_=m(x)
    loss=sum((z*torch.randn_like(z)).sum() for z in aux)
    grads=torch.autograd.grad(loss,list(m.exchange_scan.parameters()),allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert all(bool(g.abs().sum()>0) for g in grads)


def test_auxiliary_is_post_exchange_not_after_output_zero_projection():
    m=head(PostInteractionFusion);x=torch.randn(3,12,2,3)
    mixed,aux,_=m(x)
    assert torch.equal(mixed,x)  # precise zero-start, but aux still supervises exchange
    sum(z[:,0].sum() for z in aux).backward()
    assert m.exchange_scan.out.weight.grad.abs().sum()>0
    assert m.output.weight.grad is None  # intentional, main task supervises final projection


def test_gradient_cap_is_block_norm_not_individual_tensor_and_keep_unchanged():
    main={'p.w':torch.tensor([3.,0.]),'p.b':torch.tensor([4.])}
    aux={'p.w':torch.tensor([0.,20.]),'p.b':torch.tensor([0.])}
    keep={'p.w':torch.tensor([1.,1.]),'p.b':torch.tensor([2.])}
    before={n:v.clone() for n,v in main.items()}
    result,r=combine(main,aux,keep,.5)
    assert r['p']['auxiliary_scale']==pytest.approx(.125)
    assert r['p']['capped_auxiliary_l2']==pytest.approx(2.5)
    torch.testing.assert_close(result['p.w'],torch.tensor([4.,3.5]))
    torch.testing.assert_close(result['p.b'],torch.tensor([6.]))
    assert all(torch.equal(main[n],v) for n,v in before.items())


def test_zero_main_suppresses_auxiliary_without_floor():
    z={'x.w':torch.zeros(2)};a={'x.w':torch.ones(2)}
    result,r=combine(z,a,z)
    assert r['x']['auxiliary_scale']==0 and result['x.w'].count_nonzero()==0


def test_zero_auxiliary_keeps_main_and_keep():
    m={'x.w':torch.ones(2)};z={'x.w':torch.zeros(2)}
    result,r=combine(m,z,m)
    assert r['x']['auxiliary_scale']==1
    torch.testing.assert_close(result['x.w'],torch.full((2,),2.))


@pytest.mark.parametrize('bad',[0,-1,2,float('nan')])
def test_bad_cap_rejected(bad):
    z={'x.w':torch.zeros(2)}
    with pytest.raises(ValueError):combine(z,z,z,bad)


def test_bad_gradient_inventory_shape_and_nan_rejected():
    z={'x.w':torch.zeros(2)}
    for a in ({},{'x.w':torch.zeros(3)},{'x.w':torch.full((2,),float('nan'))}):
        with pytest.raises(ValueError):combine(z,a,z)


class TinyAnchor(nn.Module):
    def readout(self,x):return F.normalize(x.flatten(1),dim=1)


def test_each_component_micro_replay_equals_direct_full_batch_gradient():
    from scripts.single_dino_supervision import component_gradients
    model=PostInteractionVPR(TinyAnchor(),'aligned',channels=12,dim=8,rank=4)
    with torch.no_grad():model.fusion.output.weight.normal_(std=.04)
    x=torch.randn(4,12,2,3);features=(x,)*6
    direct=copy.deepcopy(model)
    values=direct.describe(features)[:3]
    targets=[torch.randn_like(v) for v in values]
    direct_losses=dict(main=(values[0]*targets[0]).sum(),
        auxiliary=.1*sum((v*t).sum() for v,t in zip(values[1:],targets[1:])),
        keep=.1*values[0].square().sum())
    expected={}
    for name,loss in direct_losses.items():
        gs=torch.autograd.grad(loss,list(direct.fusion.parameters()),allow_unused=True,retain_graph=True)
        expected[name]={n:torch.zeros_like(p) if g is None else g for (n,p),g in zip(direct.fusion.named_parameters(),gs)}
    with torch.no_grad():leaves=[v.detach().requires_grad_() for v in model.describe(features)[:3]]
    losses=dict(main=(leaves[0]*targets[0]).sum(),
        auxiliary=.1*sum((v*t).sum() for v,t in zip(leaves[1:],targets[1:])),keep=.1*leaves[0].square().sum())
    actual=component_gradients(model,features,leaves,losses,2)
    for component in expected:
        for name in expected[component]:
            torch.testing.assert_close(actual[component][name],expected[component][name],rtol=8e-4,atol=3e-5,msg=component+'/'+name)
    _,r=combine(actual['main'],actual['auxiliary'],actual['keep'])
    assert all(v['cap_satisfied'] for v in r.values())


def test_new_evaluator_private_namespace_preserves_v1():
    from scripts import single_dino_interaction as v1
    from scripts import single_dino_supervision as v2
    before=(v1.identity,v1.contract_for,v1.build,v1.bound_engine,v1.SCHEMA)
    evaluate=v2.evaluator()
    assert evaluate.__globals__ is not v1.evaluate.__globals__
    assert evaluate.__globals__['build'] is v2.build
    assert before==(v1.identity,v1.contract_for,v1.build,v1.bound_engine,v1.SCHEMA)


@pytest.mark.parametrize('command',['smoke','train'])
def test_cli_without_explicit_consent_cannot_touch_identity_model_or_output(monkeypatch,command):
    from scripts import single_dino_supervision as v2
    def forbidden(*a,**k):raise AssertionError('No new work permitted')
    monkeypatch.setattr(v2,'identity',forbidden);monkeypatch.setattr(v2,'build',forbidden)
    monkeypatch.setattr('sys.argv',['single_dino_supervision.py',command])
    with pytest.raises(SystemExit) as error:v2.main()
    assert error.value.code==2


def test_configuration_keeps_old_budget_and_new_isolated_output():
    from scripts import single_dino_supervision as v2
    from scripts import single_dino_interaction as v1
    c=v2.configuration();old=v1.configuration()
    assert c.pop('aux_gradient_max_ratio')==.5
    assert c.pop('schema')=='single_dino_supervision_v2'
    assert c.pop('runs')=='logs/single_dino_supervision_v2'
    old.pop('schema');old.pop('runs')
    assert c==old


def test_three_heads_matched_capacity_and_legacy_initialization():
    models=[head(PostInteractionFusion,a) for a in ARMS]
    legacy=head(SingleDinoInteraction)
    assert {sum(p.numel() for p in m.parameters()) for m in models}=={sum(p.numel() for p in legacy.parameters())}
    assert all(all(torch.equal(v,m.state_dict()[n]) for n,v in legacy.state_dict().items()) for m in models)
    assert not set(map(id,models[0].parameters()))&set(map(id,models[1].parameters()))


def test_training_uses_one_combined_clip_and_no_scalar_total_loss_backward():
    from scripts import single_dino_supervision as v2
    text=inspect.getsource(v2.train)
    assert text.index('combine(')<text.index('clip_grad_norm_(')<text.index('optimizers[arm].step()')
    assert text.count('clip_grad_norm_(')==1
    assert 'loss.backward()' not in text
