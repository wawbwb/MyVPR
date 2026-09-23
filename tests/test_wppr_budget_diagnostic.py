import numpy as np
import pytest
from scripts.wppr_budget_diagnostic import shortlist,evaluate


def test_protection_consumes_budget():
    pred=np.arange(44,dtype=float)[None,:]
    ids=shortlist(pred,12,4)[0]
    assert len(set(ids))==12
    assert ids.tolist()==[0,1,2,3,43,42,41,40,39,38,37,36]


def test_full_budget_and_teacher_tie():
    pred=np.arange(44,dtype=float)[None,:];teacher=np.zeros((1,44));labels=np.zeros((1,44),bool);labels[0,0]=True
    assert evaluate(pred,teacher,labels,44,0)['correct']==1
    assert evaluate(pred,teacher,labels,12,2)['correct']==1


def test_bad_input():
    with pytest.raises(ValueError):shortlist(np.zeros((1,44)),12,13)
    with pytest.raises(ValueError):shortlist(np.full((1,44),np.nan),12,0)
