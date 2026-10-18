"""CPU tests for the training server. No local model execution is needed."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.models.clip_nuisance import (InvariantBoQ,ResidualPredictor,center_four,
    donor_indices,projection,reverse,whitening,within_place)


def test_gradient_reversal_changes_only_backward_not_forward():
    x=torch.tensor([1.,2.,3.],requires_grad=True)
    y=reverse(x,.3)
    assert torch.equal(y,x)
    y.square().sum().backward()
    assert torch.allclose(x.grad,-.6*x.detach())


def test_conditional_centering_removes_place_offsets_and_respects_permutation():
    x=torch.randn(12,6,requires_grad=True);labels=torch.tensor([0]*4+[3]*4+[8]*4)
    y=within_place(x,labels)
    for p in labels.unique():assert torch.allclose(y[labels==p].sum(0),torch.zeros(6),atol=1e-6)
    offsets=torch.randn(9,6)
    assert torch.allclose(y,within_place(x+offsets[labels],labels),atol=1e-6)
    order=torch.randperm(12)
    assert torch.allclose(y[order],within_place(x[order],labels[order]),atol=1e-6)
    (y*torch.randn_like(y)).sum().backward()
    for p in labels.unique():assert torch.allclose(x.grad[labels==p].sum(0),torch.zeros(6),atol=1e-6)


def test_single_view_cannot_define_within_place_proxy():
    with pytest.raises(ValueError):within_place(torch.randn(2,3),torch.tensor([0,1]))


def test_numpy_and_torch_conditional_residuals_match():
    x=np.random.default_rng(0).normal(size=(20,7)).astype(np.float32)
    a=center_four(x)
    b=within_place(torch.from_numpy(x),torch.arange(5).repeat_interleave(4)).numpy()
    assert np.allclose(a,b,atol=1e-6)


def test_predictor_reversal_preserves_head_gradient_but_reverses_student_gradient():
    torch.manual_seed(4)
    predictor=ResidualPredictor(torch.ones(16),descriptor_dim=8,hidden=12,targets=3)
    x=torch.randn(8,8,requires_grad=True);labels=torch.arange(2).repeat_interleave(4);target=torch.randn(8,3)
    head=list(predictor.parameters())
    normal=predictor(x,labels,False)
    a=torch.autograd.grad(F.mse_loss(normal,target),[x,*head])
    reversed_value=predictor(x,labels,True)
    b=torch.autograd.grad(F.mse_loss(reversed_value,target),[x,*head])
    assert torch.equal(normal,reversed_value)
    assert torch.allclose(a[0],-b[0]) and a[0].abs().sum()>0
    assert all(torch.allclose(u,v) for u,v in zip(a[1:],b[1:]))


def test_detached_predictor_step_does_not_update_student():
    predictor=ResidualPredictor(torch.ones(16),descriptor_dim=8,hidden=12,targets=3)
    x=torch.randn(8,8,requires_grad=True);labels=torch.arange(2).repeat_interleave(4)
    F.mse_loss(predictor(x.detach(),labels),torch.randn(8,3)).backward()
    assert x.grad is None and any(p.grad is not None for p in predictor.parameters())


def test_frozen_predictor_still_backpropagates_into_student():
    predictor=ResidualPredictor(torch.ones(16),descriptor_dim=8,hidden=12,targets=3).requires_grad_(False)
    x=torch.randn(8,8,requires_grad=True);labels=torch.arange(2).repeat_interleave(4)
    F.mse_loss(predictor(x,labels,True),torch.randn(8,3)).backward()
    assert x.grad.abs().sum()>0 and all(p.grad is None for p in predictor.parameters())


def test_view_corruption_preserves_place_distribution_and_is_not_spatial_shuffle():
    names=['a','b','c'];ids=donor_indices(names)
    assert np.array_equal(ids,donor_indices(names))
    assert (ids!=np.arange(12)).all()
    for i in range(3):assert set(ids[i*4:i*4+4])==set(range(i*4,i*4+4))
    assert np.array_equal(ids[4:8]-4,donor_indices(['b']))
    y=center_four(np.random.default_rng(4).normal(size=(12,7)))
    assert np.allclose(center_four(y[ids]),y[ids],atol=1e-6)
    # Spatial permutation would leave a pooled vector unchanged, so is invalid.
    patches=np.random.default_rng(2).normal(size=(2,512,196))
    assert np.allclose(patches.mean(2),patches[:,:,::-1].mean(2))


def test_whitening_fits_training_rows_only_and_keeps_zero_place_means():
    scene=np.random.default_rng(8).normal(size=(80,16)).astype(np.float32)
    matrix,y,stats=whitening(scene,64,4)
    changed=scene.copy();changed[64:]*=100
    second,_,_=whitening(changed,64,4)
    assert np.array_equal(matrix,second)
    assert np.allclose(y.reshape(-1,4,4).mean(1),0,atol=1e-6)
    assert stats['target_train_variance']>0 and np.isfinite(y).all()
    with pytest.raises(ValueError):whitening(np.ones_like(scene),64,4)


def test_projection_is_fixed_reproducible_and_independent_of_global_rng():
    a=projection(8,16,42)
    torch.randn(100)
    assert torch.equal(a,projection(8,16,42))
    assert not torch.equal(a,projection(8,16,43))


def fake_official():
    p=Path(__file__).with_name('test_clip_key_boq.py')
    spec=importlib.util.spec_from_file_location('frozen_clip_key_test_helpers',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.Official()


def test_inference_accepts_only_rgb_and_reproduces_official_at_zero_start():
    model=InvariantBoQ(fake_official(),hidden=4)
    x=torch.randn(2,3,4,5)
    with torch.inference_mode():assert torch.equal(model(x),model.visual(x))
    with pytest.raises(TypeError):model(x,torch.randn(2,512,14,14))
    model.train()
    assert not model.visual.training
    assert all(n.startswith('routers.') for n,p in model.named_parameters() if p.requires_grad)
    assert not any('predictor' in n or 'teacher' in n for n in model.state_dict())


def test_full_student_adversary_path_has_gate_gradient_without_changing_official():
    model=InvariantBoQ(fake_official(),hidden=4)
    predictor=ResidualPredictor(torch.ones(16),descriptor_dim=32,hidden=12,targets=3).requires_grad_(False)
    x=torch.randn(8,3,4,5);labels=torch.arange(2).repeat_interleave(4)
    loss=F.mse_loss(predictor(model(x),labels,True),torch.randn(8,3))
    loss.backward()
    assert any(r.beta.grad is not None and r.beta.grad.abs().sum()>0 for r in model.routers)
    assert all(p.grad is None for p in model.visual.parameters())


def test_probe_statistics_use_train_only_feature_scaling():
    from scripts.clip_nuisance_study import fit_probe
    c=dict(projection_dim=4,predictor_hidden=8,target_dim=2,probe_epochs=1,probe_lr=.001,probe_batch=16)
    rng=np.random.default_rng(42)
    x=rng.normal(size=(48,4)).astype(np.float32)
    y=center_four(rng.normal(size=(48,2)))
    result=fit_probe(c,x,y,32,42)
    assert result['train_rows']==32 and result['holdout_rows']==16
    assert np.isfinite(result['holdout_r2'])


def test_artifact_checksums_refuse_corruption(tmp_path):
    from scripts.clip_nuisance_study import finish,verified
    from src.semantic_region_protocol import owned_run,write
    p=tmp_path/'run'
    with owned_run(p,{'schema':'within_place_clip_residual_v1'}):
        write(p/'summary.json',{'x':1});finish(p)
    assert verified(p)['schema']=='within_place_clip_residual_v1'
    (p/'summary.json').write_text('{}')
    with pytest.raises(ValueError):verified(p)


def test_updates_and_evaluation_require_explicit_authorization():
    script=Path(__file__).resolve().parents[1]/'scripts/clip_nuisance_study.py'
    for command in ('probe','smoke','train','eval'):
        result=subprocess.run([sys.executable,str(script),command],capture_output=True,text=True)
        assert result.returncode!=0
        assert ('--evaluate' if command=='eval' else '--allow-training') in result.stderr


def test_config_keeps_fixed_budget_and_original_artifacts_out_of_write_roots():
    from scripts.clip_nuisance_study import configuration
    c=configuration(Path(__file__).resolve().parents[1]/'config/clip_nuisance.yaml')
    assert c['epochs']*c['train_places']//c['places_per_batch']==2560
    assert c['runs']=='logs/clip_nuisance_v1' and c['prepared']=='.cache/clip_nuisance_v1/prepared'
    assert c['adversarial_weight']==.02 and c['probe_min_r2']==.01
    assert c['cuda_memory_fraction']<=.2
