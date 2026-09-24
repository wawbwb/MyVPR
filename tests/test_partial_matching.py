import pytest
import torch
from src.models.partial_matching import PartialMatching


@pytest.mark.parametrize('mode',PartialMatching.MODES)
def test_zero_start_and_gradient(mode):
    torch.manual_seed(42)
    m=PartialMatching(16,8,mode);x=torch.randn(2,6,16);y=torch.randn(2,5,16)
    ox,oy,_=m(x,y)
    assert torch.equal(ox,x) and torch.equal(oy,y)
    assert torch.equal(ox[:,:1],x[:,:1])
    opt=torch.optim.Adam(m.parameters(),lr=.01)
    for _ in range(3):
        ox,oy,_=m(x,y);loss=(ox.square().mean()+oy.square().mean());opt.zero_grad();loss.backward();opt.step()
    assert m.key.weight.grad is not None and m.key.weight.grad.abs().sum()>0
    assert torch.isfinite(ox).all()


def test_dustbin_rejects_and_forced_matches():
    z=torch.zeros(1,5,5)
    m=PartialMatching(16,8,'partial',iterations=100)
    with torch.no_grad():m.bin_score.fill_(20)
    p,q=m.assignment(z)
    assert p.sum(-1).max()<.05 and q.sum(-1).max()<.05
    m.mode='forced';p,q=m.assignment(z)
    assert torch.allclose(p.sum(-1),torch.ones(1,5),atol=1e-5)


def test_shared_transport_is_bidirectional():
    torch.manual_seed(42);m=PartialMatching(16,8,'partial',iterations=100)
    p,q=m.assignment(torch.randn(2,5,5))
    assert torch.allclose(p,q.transpose(1,2),atol=1e-5)
    assert p.sum(-1).max()<=1.0001
