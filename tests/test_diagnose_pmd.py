import numpy as np
import pytest
import torch
from scripts.diagnose_pmd import ranking,normalize_mass


def test_reachability_and_margin():
    assert ranking([1.,2.],[True,False])['margin']==-1
    r=ranking([1.,2.],[False,False])
    assert not r['reachable'] and not r['correct'] and r['margin'] is None
    with pytest.raises(ValueError):ranking([np.nan],[True])


def test_mass_intervention_preserves_relative_assignment():
    p=torch.tensor([[[.1,.3],[.2,.2]]]);q=normalize_mass(p)
    assert torch.allclose(q.sum(-1),torch.ones(1,2))
    assert torch.allclose(q[0,0],torch.tensor([.25,.75]))
    assert torch.isfinite(normalize_mass(torch.zeros_like(p))).all()
