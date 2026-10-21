"""Synthetic CPU decomposition / full-DB protocol tests; no real-image run."""
import inspect
from pathlib import Path
import pytest
import torch
from src.single_dino_centering import decompose,VARIANTS


def test_exact_spatial_decomposition_and_unchanged_original_reference():
    torch.manual_seed(42);a=torch.randn(2,12,3,4);b=a+.01*torch.randn_like(a)
    maps,s=decompose(a,b)
    assert tuple(maps)==VARIANTS
    assert maps['innovation1'] is a and maps['aligned'] is b
    torch.testing.assert_close((maps['centered']-a)+(maps['mean_only']-a),b-a,atol=3e-7,rtol=1e-4)
    torch.testing.assert_close((maps['centered']-a).mean((2,3)),torch.zeros(2,12),atol=1e-7,rtol=0)
    assert bool((s['centered_relative']<=s['residual_relative']+1e-6).all())


def test_uniform_spatial_residual_disappears_and_is_image_specific():
    a=torch.zeros(2,4,3,3);b=torch.tensor([1.,2.])[:,None,None,None].expand_as(a)
    maps,s=decompose(a,b)
    assert maps['centered'].count_nonzero()==0
    torch.testing.assert_close(maps['mean_only'],b)
    torch.testing.assert_close(s['dc_energy_fraction'],torch.ones(2))


def test_zero_dc_residual_is_retained():
    a=torch.zeros(1,1,2,2);b=torch.tensor([[[[1.,-1.],[-1.,1.]]]])
    maps,s=decompose(a,b)
    torch.testing.assert_close(maps['centered'],b)
    assert maps['mean_only'].count_nonzero()==0
    assert s['dc_energy_fraction'].item()==0


def test_zero_residual_has_finite_defined_statistics():
    a=torch.zeros(2,4,3,3);_,s=decompose(a,a)
    assert all(torch.isfinite(v).all() for v in s.values())
    assert s['dc_energy_fraction'].count_nonzero()==0
    assert s['residual_nonzero'].count_nonzero()==0


@pytest.mark.parametrize('bad',['shape','nan'])
def test_invalid_maps_rejected(bad):
    a=torch.ones(1,4,3,3);b=torch.ones_like(a)
    if bad=='shape':b=b[:,:,:2]
    else:b[0,0,0,0]=float('nan')
    with pytest.raises(ValueError):decompose(a,b)


def test_centering_preserves_no_parameter_and_does_not_change_inputs():
    a=torch.randn(2,3,4,4);b=a+.1;before=(a.clone(),b.clone())
    decompose(a,b)
    assert torch.equal(a,before[0]) and torch.equal(b,before[1])


def test_full_db_protocol_has_no_training_or_subset_selector():
    from scripts import eval_single_dino_centered as e
    text=inspect.getsource(e)
    assert 'torch.optim' not in text and '.backward(' not in text
    assert "choices=('plan','run')" in text
    assert 'candidates' not in inspect.getsource(e.run)
    assert "('msls-val',MapillarySLSDataset)" in inspect.getsource(e.datasets)
    assert "('pitts30k-val',PittsburghDataset)" in inspect.getsource(e.datasets)


def test_exact_original_top20_required_not_just_same_recall():
    from scripts import eval_single_dino_centered as e
    text=inspect.getsource(e.validate_dataset)
    assert "report['variants'][v]!=original['variants'][v]" in text
    assert "('innovation1','aligned')" in text


def test_old_v2_globals_remain_unmodified_when_evaluator_imported():
    from scripts import single_dino_supervision as v2
    before=(v2.SCHEMA,v2.SOURCES,v2.build,v2.train)
    from scripts import eval_single_dino_centered
    assert before==(v2.SCHEMA,v2.SOURCES,v2.build,v2.train)
