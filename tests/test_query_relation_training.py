import ast
from pathlib import Path
import pytest
import torch
from src.query_relation_training import score_development


def test_expanded_gallery_self_exclusion_and_offset():
    x=torch.eye(3).repeat_interleave(4,0)
    r=score_development(x,8,chunk=2)
    assert r['correct']==r['queries']==4 and r['gallery']==12
    assert all(pred!=i+8 for i,pred in enumerate(r['predictions']))
    assert all(pred//4==2 for pred in r['predictions'])


def test_external_distractor_can_win():
    x=torch.tensor([[1.,0.]]*4+[[1.,0.],[0.,1.],[0.,1.],[0.,1.]])
    r=score_development(x,4)
    assert r['correct']==3 and r['predictions'][0]<4
    with pytest.raises(ValueError):score_development(x,3)


def test_trainer_fixed_last_no_best_selection():
    path=Path(__file__).resolve().parents[1]/'scripts/train_query_relation.py'
    source=path.read_text();ast.parse(source)
    assert 'best_epoch' not in source and 'best.pt' not in source and 'better(' not in source
    assert "benchmark(state['epoch'])" in source
