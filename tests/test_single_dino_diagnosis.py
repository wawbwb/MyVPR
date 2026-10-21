import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
from src.single_dino_diagnosis import stats,selected_queries,candidate_union,margin_report
from src.models.single_dino_interaction import SingleDinoInteraction


def report():
    v=dict(outcomes=[True,False,True,True],predictions=[[0,1],[1,2],[2,3],[3,0]])
    return dict(variants={a:dict(v) for a in ('innovation1','separate','aligned','permuted')})


def test_selection_keeps_errors_and_disagreements():
    r=report();r['variants']['aligned']['outcomes']=[True,False,False,True]
    assert selected_queries(r,2)==[1,2]


def test_union_includes_all_gt():
    r=report();assert candidate_union(r,[1],[[0],[9],[2],[3]])==[1,2,9]


def test_margin_not_just_positive_vs_original_winner():
    r=margin_report([[.5,.9,.8]],[0,1,2],[0],[[2]],[0])
    assert r['positive_negative_margin'][0]==pytest.approx(-.1)
    assert r['positive_minus_baseline_top1'][0]==pytest.approx(.3)
    assert r['correct']==[False]


def test_stats_reject_nonfinite():
    with pytest.raises(ValueError):stats([float('nan')])


def test_stats_empty():
    assert stats([])['mean'] is None


def test_auxiliary_cannot_directly_supervise_exchange():
    torch.manual_seed(1)
    m=SingleDinoInteraction('aligned',channels=8,dim=8,rank=2)
    x=torch.randn(4,8,3,3)
    _,aux,_=m(x)
    loss=sum((z*torch.randn_like(z)).sum() for z in aux)
    g=torch.autograd.grad(loss,list(m.exchange_scan.parameters()),allow_unused=True)
    assert all(v is None for v in g)
    g=torch.autograd.grad(sum((z*torch.randn_like(z)).sum() for z in m(x)[1]),list(m.context_scan.parameters()),allow_unused=True)
    assert any(v is not None and bool(v.abs().sum()>0) for v in g)


def test_interventions_restore_hooks_and_route_on_failure():
    spec=importlib.util.spec_from_file_location('diag',ROOT/'scripts/diagnose_single_dino_interaction.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    m=SingleDinoInteraction('aligned',channels=8,dim=8,rank=2)
    original={k:v.clone() for k,v in m.state_dict().items()}
    with pytest.raises(RuntimeError):
        with mod.intervention(m,'permuted',('exchange_scan',)):
            assert m.arm=='permuted'
            assert m.exchange_scan(torch.randn(1,9,8)).count_nonzero()==0
            raise RuntimeError('test')
    assert m.arm=='aligned' and not m.exchange_scan._forward_hooks
    assert all(torch.equal(original[k],v) for k,v in m.state_dict().items())
