import importlib.util
import unittest

HAS_TORCH=importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_TORCH,'torch only on training server')
class ModelTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t=torch;torch.manual_seed(7);torch.set_num_threads(1)
        self.x=torch.randn(2,6,12);self.s=torch.randn(2,6);self.z=torch.nn.functional.normalize(torch.randn(2,6,8),dim=-1)

    def model(self,mode,zero=False):
        from src.models.candidate_set import CandidateSet
        m=CandidateSet(mode,12,16,db_dim=8)
        if not zero:self.t.nn.init.normal_(m.out.weight,std=.1)
        return m

    def test_initial_scores_equal_frozen(self):
        from src.models.candidate_set import MODES
        for mode in MODES:
            self.t.testing.assert_close(self.model(mode,True)(self.x,self.s,self.z),self.s,rtol=0,atol=0)

    def test_permutation_equivariance(self):
        from src.models.candidate_set import MODES
        p=self.t.tensor([4,1,5,0,2,3])
        for mode in MODES:
            m=self.model(mode)
            self.t.testing.assert_close(m(self.x[:,p],self.s[:,p],self.z[:,p]),m(self.x,self.s,self.z)[:,p],atol=1e-5,rtol=1e-5)

    def test_independent_has_no_candidate_context(self):
        m=self.model('independent');original=m(self.x,self.s,self.z)
        x=self.x.clone();x[:,1:]=100
        self.t.testing.assert_close(m(x,self.s,self.z)[:,0],original[:,0])

    def test_set_has_candidate_context(self):
        m=self.model('set');original=m(self.x,self.s,self.z)
        x=self.x.clone();x[:,1:]=self.t.randn_like(x[:,1:])
        self.assertGreater(float((m(x,self.s,self.z)[:,0]-original[:,0]).abs().max()),1e-5)

    def test_density_uniform_replication(self):
        # Uniform replication of every candidate must not alter original scores.
        m=self.model('density');ix=self.t.arange(6).repeat(2)
        self.t.testing.assert_close(m(self.x[:,ix],self.s[:,ix],self.z[:,ix])[:,:6],m(self.x,self.s,self.z),atol=1e-5,rtol=1e-5)

    def test_loss_and_gradients(self):
        from src.models.candidate_set import list_loss,duplicate_consistency
        m=self.model('competition');s=m(self.x,self.s,self.z)
        y=self.t.tensor([[True,True,False,False,False,False],[False]*6])
        loss,n=list_loss(s,y);self.assertEqual(n,1)
        ix=self.t.tensor(list(range(6))+[0,0,0]);aug=m(self.x[:,ix],self.s[:,ix],self.z[:,ix])
        total=loss+duplicate_consistency(s,aug);total.backward()
        self.assertTrue(self.t.isfinite(total));self.assertGreater(float(m.encode[1].weight.grad.abs().sum()),0)
        zero,n=list_loss(s,self.t.zeros_like(y));self.assertEqual(n,0);self.assertEqual(float(zero),0)


if __name__=='__main__':unittest.main()
