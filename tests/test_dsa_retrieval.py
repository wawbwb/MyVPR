import numpy as np
import pytest
import torch
from src.dsa_retrieval import cross_view, paired


def test_cross_view_no_self_image_and_exact_places():
    x=torch.eye(6)[:,None,:].repeat(1,4,1)
    hits,ranks=cross_view(x)
    assert hits.shape==(6,12) and hits.all() and (ranks==1).all()


def test_ties_not_all_counted_correct():
    hits,ranks=cross_view(torch.ones(5,4,3))
    assert hits.sum()==12
    assert (ranks==np.arange(1,6)[:,None]).all()


def test_clustered_paired_identical():
    a=np.array([[True,False],[False,True]])
    s=paired(a,a)
    assert s['delta_r1_pp']==0 and s['place_bootstrap_95ci_pp']==[0,0]
    assert s['corrections']==s['regressions']==0
    with pytest.raises(ValueError): cross_view(torch.zeros(3,4,5))
