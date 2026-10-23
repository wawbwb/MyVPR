import numpy as np
import pytest
import torch
from torch.nn import functional as F
from src.models.full_gallery_supervision import FixedGallery,positive_without_self,GalleryVPR,distance,ARMS
from scripts.full_gallery_supervision import configuration


@pytest.fixture(autouse=True)
def seed():np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(2)


def data():
    bank=np.random.randn(12,16).astype(np.float32);bank/=np.linalg.norm(bank,axis=1,keepdims=True)
    coords=np.repeat([[0,0],[0,.00001],[0,1]],4,axis=0)
    return bank,coords


def test_positive_does_not_include_query():
    bank,_=data();s=bank.reshape(3,4,16).sum(1);i=np.array([0])
    p=positive_without_self(bank,s,i)
    b=bank.copy();b[0]=-b[0];q=positive_without_self(b,b.reshape(3,4,16).sum(1),i)
    np.testing.assert_allclose(p,q,rtol=1e-6,atol=1e-7)


def test_full_gallery_mask_and_gradient():
    bank,coords=data();g=FixedGallery(bank,list(range(100,112)),coords,'cpu')
    z=torch.from_numpy(bank[:1].copy()).requires_grad_();loss,s=g.loss(z,[100],torch.tensor([0]),torch.tensor(coords[:1]))
    pos=torch.from_numpy(positive_without_self(bank,g.sums,np.array([0])))
    scores=torch.cat([(z*pos).sum(1,keepdim=True),z@g.proxies[2:3].T],1)*20
    torch.testing.assert_close(loss,F.cross_entropy(scores,torch.tensor([0])))
    assert s['safe_negative_mean']==1 and s['positive_excludes_self']
    loss.backward();assert torch.isfinite(z.grad).all() and z.grad.abs().max()>0
    assert not g.proxies.requires_grad


def test_loss_excludes_self_even_after_prototype_change():
    bank,coords=data();z=torch.from_numpy(bank[:1].copy())
    a=FixedGallery(bank,list(range(12)),coords,'cpu').loss(z,[0],torch.tensor([0]),torch.tensor(coords[:1]))[0]
    b=bank.copy();b[0]=-b[0]
    c=FixedGallery(b,list(range(12)),coords,'cpu').loss(z,[0],torch.tensor([0]),torch.tensor(coords[:1]))[0]
    torch.testing.assert_close(a,c,atol=1e-5,rtol=1e-5)


def test_wrong_row_labels_rejected():
    bank,coords=data();g=FixedGallery(bank,list(range(12)),coords,'cpu')
    with pytest.raises(ValueError):g.loss(torch.tensor(bank[:1]),[0],torch.tensor([1]),torch.tensor(coords[:1]))


def test_spherical_distance():
    a=torch.tensor([[0.,0.]],dtype=torch.float64);b=torch.tensor([[0.,np.pi/180]],dtype=torch.float64)
    assert abs(float(distance(a,b))-111194.9266)<.01


def test_fixed_budget():
    c=configuration();assert c['pilot_steps']==1024 and c['neck_learning_rate']==1e-5 and c['learning_rate']==1e-4
