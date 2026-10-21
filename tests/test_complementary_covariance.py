"""CPU-only math/replay checks; never instantiate/download a real backbone."""
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from src.models.complementary_covariance import (ARMS, ComplementaryCovariance,
    ComplementaryVPR, combine_descriptors, cross_statistic, image_seeds, permute_positions)
from scripts.complementary_covariance import configuration, replay


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2)
    torch.manual_seed(42)


def test_covariance_hand_computation_and_dc_invariance():
    a = torch.randn(2, 11, 4, dtype=torch.float64)
    b = torch.randn_like(a)
    expected = torch.stack([sum(torch.outer(x-a[i].mean(0),y-b[i].mean(0)) for x,y in zip(a[i],b[i]))/10 for i in range(2)])
    actual = cross_statistic(a,b,'aligned_cov')
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(actual,cross_statistic(a+3,b-4,'aligned_cov'))


def test_simultaneous_permutation_invariant_but_single_permutation_changes():
    a,b = torch.randn(2,17,5),torch.randn(2,17,5)
    p = torch.randperm(17)
    z = cross_statistic(a,b,'aligned_cov')
    torch.testing.assert_close(z,cross_statistic(a[:,p],b[:,p],'aligned_cov'))
    assert not torch.allclose(z,cross_statistic(a,b,'permuted_cov',[5,9]))


def test_mean_control_discards_position_pairing():
    a,b = torch.randn(2,9,4),torch.randn(2,9,4)
    z = cross_statistic(a,b,'mean_outer')
    torch.testing.assert_close(z,cross_statistic(a[:,torch.randperm(9)],b,'mean_outer'))
    assert not torch.allclose(z,cross_statistic(a,b,'aligned_cov'))


def test_content_permutation_independent_of_batch_order_and_chunks():
    a = torch.randn(3,19,5)
    digests = ['a'*64,'b'*64,'c'*64]
    seeds = image_seeds(digests)
    z = permute_positions(a,seeds)
    torch.testing.assert_close(z,torch.cat([permute_positions(a[i:i+1],seeds[i:i+1]) for i in range(3)]),rtol=0,atol=0)
    torch.testing.assert_close(z[[2,0]],permute_positions(a[[2,0]],[seeds[2],seeds[0]]),rtol=0,atol=0)
    assert not torch.equal(permute_positions(a[:1],[seeds[0]]),permute_positions(a[:1],[seeds[1]]))


@pytest.mark.parametrize('arm',ARMS)
def test_all_active_parameters_finite_gradients_and_norm(arm):
    m = ComplementaryCovariance(arm,channels=12,dim=8,rank=4,descriptor_dim=16)
    z = m(torch.randn(4,12,5,5),[1,2,3,4])
    assert z.shape == (4,16)
    torch.testing.assert_close(z.norm(dim=1),torch.ones(4))
    (z*torch.randn_like(z)).sum().backward()
    for name,p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and (p.grad!=0).any(),name


def test_same_parameter_inventory_and_initialization():
    heads=[]
    for arm in ARMS:
        torch.manual_seed(42)
        heads.append(ComplementaryCovariance(arm,channels=12,dim=8,rank=4,descriptor_dim=16))
    for m in heads[1:]:
        assert list(m.state_dict()) == list(heads[0].state_dict())
        for k,v in m.state_dict().items(): assert torch.equal(v,heads[0].state_dict()[k])
    assert not set(map(id,heads[0].parameters())) & set(map(id,heads[1].parameters()))


def test_fusion_weight_formula_and_exact_bypass():
    a = F.normalize(torch.randn(5,12),dim=1)
    h = F.normalize(torch.randn(5,4),dim=1)
    z = combine_descriptors(a,h,.25)
    torch.testing.assert_close(z.norm(dim=1),torch.ones(5))
    torch.testing.assert_close(z@z.T,(a@a.T+.25**2*(h@h.T))/(1+.25**2))
    zero = combine_descriptors(a,h,0)
    assert torch.equal(zero[:,:12],a) and torch.count_nonzero(zero[:,12:])==0


@pytest.mark.parametrize('arm',ARMS)
def test_direct_full_batch_gradient_replay_matches_monolithic(arm):
    a = ComplementaryVPR(torch.nn.Identity(),arm,channels=12,dim=8,rank=4,descriptor_dim=16)
    b = deepcopy(a)
    features = (torch.randn(5,12,5,5),F.normalize(torch.randn(5,12),dim=1))
    seeds=[9,10,11,12,13]
    outputs = a.describe(features,seeds,.25)
    g=[torch.randn_like(v) for v in outputs]
    torch.autograd.backward(outputs,g)
    chunks=[tuple(t[i:i+2] for t in features) for i in range(0,5,2)]
    replay(b,chunks,seeds,g,.25)
    for (n,p),(n2,q) in zip(a.head.named_parameters(),b.head.named_parameters()):
        assert n==n2
        torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=2e-4)


def test_branch_loss_has_gradients_when_final_weight_zero():
    m=ComplementaryVPR(torch.nn.Identity(),'aligned_cov',channels=12,dim=8,rank=4,descriptor_dim=16)
    base=F.normalize(torch.randn(3,12),dim=1)
    z,h=m.describe((torch.randn(3,12,5,5),base),[1,2,3],0)
    (h*torch.randn_like(h)).sum().backward()
    assert all(p.grad is not None and (p.grad!=0).any() for p in m.head.parameters())
    assert torch.equal(z[:,:12],base)


def test_single_encoder_frozen_in_train_mode():
    anchor=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.Dropout())
    m=ComplementaryVPR(anchor,'aligned_cov',channels=12,dim=8,rank=4,descriptor_dim=16).train()
    assert not m.anchor.training and all(not p.requires_grad for p in anchor.parameters())
    assert all(p.requires_grad for p in m.head.parameters())
    assert not hasattr(m,'resnet')


@pytest.mark.parametrize('case',['arm','shape','seed','hash','weight'])
def test_invalid_inputs_rejected(case):
    with pytest.raises(ValueError):
        x=torch.randn(2,5,4)
        if case=='arm': cross_statistic(x,x,'bad')
        elif case=='shape': cross_statistic(x,x[:1],'aligned_cov')
        elif case=='seed': cross_statistic(x,x,'permuted_cov')
        elif case=='hash': image_seeds(['a'*63])
        else: combine_descriptors(torch.ones(2,4),torch.ones(2,4),float('nan'))


def test_registered_budget_and_no_test_selection():
    c=configuration()
    assert c['pilot_steps']==1024 and c['complement_weight']==.25 and c['descriptor_dim']==512
    assert c['cuda_memory_fraction']==.2 and c['device']=='cuda:0'
    assert c['train_places']==8192 and c['views']==4
