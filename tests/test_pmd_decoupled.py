import torch
from src.pmd_decoupled import DecoupledHead,supervised_loss,counts


def test_rejection_gradient_does_not_reach_matcher():
    h=DecoupledHead('decoupled');a=torch.randn(1,4,768);b=torch.randn(1,4,768)
    plans=h(a,b);sum(d.sum() for p,d in plans).backward()
    assert h.match.key.weight.grad is None and h.match.norm.weight.grad is None
    assert h.reject[-1].weight.grad is not None


def test_location_gradient_exists_and_forced_cannot_reject():
    for mode in ['forced','partial','decoupled']:
        torch.manual_seed(42);h=DecoupledHead(mode)
        plans=h(torch.randn(1,4,768),torch.randn(1,4,768));t=[torch.tensor([0,1,-1,-1])]*2
        loss=supervised_loss(plans,t,mode!='forced');loss.backward()
        assert torch.isfinite(loss) and h.match.key.weight.grad.abs().sum()>0
        if mode=='forced':assert counts(plans,t)['false_rejected_matched']==0
