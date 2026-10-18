"""CPU diagnostics tests; run on server, not on the local Windows workstation."""
import inspect
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from scripts.clip_nuisance_representation import (bootstrap_scores,paired_difference,
    summarize_patches,lifted_projection,fit,finish,verified)
from src.models.clip_nuisance import projection,Regressor,center_four
from src.semantic_region_protocol import owned_run,write


def test_channel_complete_summaries_keep_spatial_order():
    x=torch.arange(2*768*4*6).reshape(2,768,4,6).float()
    value=summarize_patches(x)
    assert value['patch_mean'].shape==(2,768)
    assert value['patch_grid2'].shape==(2,3072)
    assert torch.equal(value['patch_mean'],x.mean((2,3)))
    assert torch.equal(value['patch_grid2'][:,::4],x[:,:,:2,:3].mean((2,3)))


def test_invalid_patch_map_is_rejected():
    with pytest.raises(ValueError):summarize_patches(torch.randn(2,512,14,14))


def test_lift_retains_projection_but_cannot_recover_lost_subspace():
    rng=np.random.default_rng(0)
    x=rng.normal(size=(20,32)).astype(np.float32)
    matrix=projection(32,6,42).numpy()
    y=x@matrix
    lifted=lifted_projection(y,matrix)
    assert lifted.shape==x.shape
    assert np.allclose(lifted@matrix,y,atol=2e-5)
    assert not np.allclose(lifted,x)
    assert np.linalg.matrix_rank(lifted,tol=1e-4)<=6


def test_lifted_and_full_probes_have_identical_trainable_capacity():
    full=Regressor(32,8,4);lifted=Regressor(32,8,4)
    assert sum(p.numel() for p in full.parameters())==sum(p.numel() for p in lifted.parameters())
    with pytest.raises(ValueError):lifted_projection(np.zeros((2,7)),np.zeros((32,6)))


def test_bootstrap_is_place_level_deterministic_and_exact_for_constant_scores():
    null=np.ones(8)
    assert bootstrap_scores(np.zeros(8),null,42,100)==[1.,1.]
    assert bootstrap_scores(np.ones(8),null,42,100)==[0.,0.]
    a=bootstrap_scores(np.arange(8)/8,null,42,100)
    assert a==bootstrap_scores(np.arange(8)/8,null,42,100)


def test_paired_difference_orientation_and_population_checks():
    result=paired_difference(np.ones(8)*.2,np.ones(8)*.7,np.ones(8),42,100)
    assert result['delta_r2']==pytest.approx(.5)
    assert result['place_bootstrap_ci95']==pytest.approx([.5,.5])
    with pytest.raises(ValueError):paired_difference(np.ones(4),np.ones(5),np.ones(4),42,10)


def test_fixed_probe_reports_train_and_holdout_without_selecting_epoch():
    c=dict(hidden=8,probe_epochs=2,probe_batch=16,probe_lr=.001,seed=42,bootstrap_samples=30)
    rng=np.random.default_rng(42)
    x=rng.normal(size=(48,6)).astype(np.float32)
    y=center_four(rng.normal(size=(48,3)))
    report,error,null=fit(c,x,y,32,42)
    assert report['epochs']==2 and len(report['history'])==2
    assert report['train_rows']==32 and report['holdout_rows']==16
    assert error.shape==null.shape==(4,)
    assert np.isfinite(report['train_r2']) and np.isfinite(report['holdout_r2'])


def test_artifact_corruption_detected(tmp_path):
    p=tmp_path/'run'
    with owned_run(p,{'schema':'clip_nuisance_representation_diagnostic_v1'}):
        write(p/'summary.json',{'ok':True});finish(p)
    assert verified(p)['schema']=='clip_nuisance_representation_diagnostic_v1'
    (p/'summary.json').write_text('{}')
    with pytest.raises(ValueError):verified(p)


def test_probe_fit_is_explicit_and_extraction_has_no_optimization():
    from scripts.clip_nuisance_representation import extract
    assert '.backward(' not in inspect.getsource(extract) and 'optimizer' not in inspect.getsource(extract)
    p=Path(__file__).resolve().parents[1]/'scripts/clip_nuisance_representation.py'
    result=subprocess.run([sys.executable,str(p),'probe'],capture_output=True,text=True)
    assert result.returncode!=0 and '--allow-probe-fit' in result.stderr


def test_cache_budget_and_protocol_do_not_include_benchmark_tuning():
    from scripts.clip_nuisance_representation import DIMENSIONS,configuration
    p=Path(__file__).resolve().parents[1]/'config/clip_nuisance_representation.yaml'
    c=configuration(p)
    assert c['probe_epochs']==8 and c['hidden']==128 and c['bootstrap_samples']==500
    assert sum(DIMENSIONS.values())*36864*4/2**30 < 2.3
    assert c['cache'].startswith('.cache/clip_nuisance_representation_v1/')
