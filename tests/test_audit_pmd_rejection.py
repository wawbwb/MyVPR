import torch
from scripts.audit_pmd_rejection import metrics,explicit_assignment
from src.pmd_correspondence import CorrespondenceHead


def test_forced_residual_is_not_rejection():
    p=torch.tensor([[[.1,.1],[.1,.1]]]);d=torch.zeros(1,2);t=torch.tensor([0,-1])
    r=metrics(p,d,t,False)
    assert r['categorical_false_rejected']==r['categorical_unmatched_rejected']==0
    assert r['row_residual_max']>.7


def test_location_and_rejection_are_separate():
    r=metrics(torch.tensor([[[.2,.1]]]),torch.tensor([[.4]]),torch.tensor([0]),True)
    assert r['location_correct']==1 and r['categorical_correct']==0
    assert r['binary_half_false_rejected']==1


def test_explicit_dustbin_reproduces_transport():
    for mode in ['independent_dustbin','forced','partial']:
        h=CorrespondenceHead(mode);a=torch.randn(1,4,768);b=torch.randn(1,5,768)
        native=h(a,b);explicit=explicit_assignment(h,a,b,20)
        for old,(new,dust) in zip(native,explicit):
            assert torch.allclose(old,new,atol=1e-6)
            assert (dust>=0).all()
            if mode=='forced':assert (dust==0).all()
