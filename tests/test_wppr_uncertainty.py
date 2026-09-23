import numpy as np
import pytest
from scripts.wppr_uncertainty_diagnostic import uncertainty,summarize


def test_shift_invariance_and_bad_values():
    x=np.arange(88,dtype=float).reshape(2,44)
    for k,v in uncertainty(x).items():assert np.allclose(v,uncertainty(x+100)[k])
    with pytest.raises(ValueError):uncertainty(np.full((2,44),np.nan))


def test_expand_only_recovers_rank13_to20():
    pred=np.tile(-np.arange(44,dtype=float),(3,1));teacher=np.zeros((3,44));labels=np.zeros((3,44),bool)
    for i,j in enumerate([0,15,25]):teacher[i,j]=1;labels[i,j]=True
    z=dict(prediction=pred,teacher=teacher,labels=labels)
    r=summarize(z,np.array([False,True,True]))
    assert r['fixable_by20']==r['fixable_caught']==1
    assert r['unfixable_by20']==1 and r['correct']==2
