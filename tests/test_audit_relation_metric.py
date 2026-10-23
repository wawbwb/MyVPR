import numpy as np
import torch
from scripts.audit_relation_metric import compare_margin,fixed_scores


def test_flips_and_numerical_tolerance():
    r=compare_margin(np.array([-.1,.2,.01]),np.array([.1,-.2,.01000001]))
    assert r['corrections']==1 and r['regressions']==1
    assert r['improved']==1 and r['worsened']==1 and r['unchanged_within_tolerance']==1
    assert r['strata']['baseline_error']['delta']['n']==1


def test_fixed_scores_include_batch_negative_and_exclude_self():
    x=torch.tensor([[1.,0.],[.8,.6],[.9,.1],[0.,1.]])
    ids=torch.arange(4)
    pos=torch.tensor([[False,True,False,False],[True,False,False,False],
                      [False,False,False,True],[False,False,True,False]])
    neg=~pos & ~torch.eye(4,dtype=torch.bool)
    pairs=torch.tensor([3,3,0,0])
    r=fixed_scores(x,[(ids,pos,neg)],pairs)
    np.testing.assert_allclose(r['paired_margin'][0],.8,atol=1e-7)
    np.testing.assert_allclose(r['combined_margin'][0],-.1,atol=1e-7)
    assert np.all(r['combined_margin']<=r['paired_margin']+1e-7)
