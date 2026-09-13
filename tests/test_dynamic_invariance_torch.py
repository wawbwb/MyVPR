"""Required torch tests on training server, not skipped if dependencies are absent."""
import copy
from pathlib import Path
import sys
import unittest
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.models.dynamic_invariance import LowRank,disabled,replay_backward,losses


class Toy(nn.Module):
    def __init__(self):
        super().__init__(); self.layer=LowRank(nn.Linear(5,7),rank=2,alpha=2)
    def forward(self,x): return nn.functional.normalize(self.layer(x),dim=-1)


class Objective(nn.Module):
    def forward(self,z,labels):
        similarity=z@z.T
        same=(labels[:,None]==labels[None,:]).float()
        return (similarity-same).square().mean(),0


class GradientChecks(unittest.TestCase):
    def test_identity_and_disabled_teacher(self):
        torch.manual_seed(4); m=Toy().eval(); x=torch.randn(4,5)
        original=nn.functional.normalize(m.layer.base(x),dim=-1)
        torch.testing.assert_close(m(x),original,rtol=0,atol=0)
        with torch.no_grad(): m.layer.up.weight.fill_(.2)
        self.assertFalse(torch.equal(m(x),original))
        with disabled(m): torch.testing.assert_close(m(x),original,rtol=0,atol=0)
        self.assertTrue(m.layer.enabled)

    def test_replay_matches_full_graph(self):
        for consistency in [False,True]:
            torch.manual_seed(9); one=Toy().eval(); two=copy.deepcopy(one)
            with torch.no_grad():
                one.layer.up.weight.normal_(0,.03)
                two.load_state_dict(one.state_dict())
            xs=[torch.randn(8,5) for _ in range(3)]; labels=torch.tensor([0,0,1,1,2,2,3,3])
            with torch.no_grad(),disabled(one): teacher=one(xs[0])
            total,_=losses([one(x) for x in xs],teacher,labels,Objective(),consistency)
            total.backward()
            replay_backward(two,xs,teacher,labels,Objective(),2,consistency)
            for p,q in zip(one.parameters(),two.parameters()):
                if p.requires_grad:
                    torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-5)
                else:
                    self.assertIsNone(q.grad)


if __name__=='__main__': unittest.main()
