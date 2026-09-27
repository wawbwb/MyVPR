import pytest
import torch
from src.models.consensus_matching import CorrelationConsensus,ConsensusMatching


def test_identity_and_context_gradient_after_update():
    torch.manual_seed(42);m=ConsensusMatching(dim=16,width=8)
    x=torch.randn(1,10,16);y=torch.randn(1,9,16)
    xx,yy,_=m(x,y);assert torch.equal(xx,x) and torch.equal(yy,y)
    opt=torch.optim.SGD(m.parameters(),lr=.1)
    target=torch.randn_like(x)
    for _ in range(2):
        opt.zero_grad();xx,yy,_=m(x,y);loss=(xx-target).square().mean()+yy.square().mean();loss.backward();opt.step()
    assert m.consensus.source.weight.grad.abs().sum()>0
    assert torch.equal(xx[:,:1],x[:,:1])


def test_spatial_filter_uses_neighbor_structure():
    torch.manual_seed(42);m=ConsensusMatching(dim=16,width=8)
    s=torch.randn(1,9,9);a=m.contextual_bias(s);m.mode='shuffled';b=m.contextual_bias(s)
    assert not torch.allclose(a,b)
    assert torch.equal(b,m.contextual_bias(s))


def test_pointwise_filter_is_permutation_equivariant():
    m=CorrelationConsensus(pointwise=True);s=torch.randn(1,9,9);p=torch.randperm(9)
    assert torch.allclose(m(s[:,p][:,:,p]),m(s)[:,p][:,:,p],atol=1e-6)
    with pytest.raises(ValueError):m(torch.randn(1,5,9))
