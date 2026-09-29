import copy
import torch
from src.models.aggregators.boq import BoQ
from scripts.train_local_value_repeat import seeded_install
from scripts.summarize_local_value_repeat import comparable


def build(mode, seed):
    torch.manual_seed(8)
    agg=BoQ(in_channels=12, proj_channels=64, num_queries=4, num_layers=2, row_dim=2)
    before=torch.get_rng_state().clone()
    adapters=seeded_install(agg, mode, seed)
    assert torch.equal(before,torch.get_rng_state())
    return {k:p.detach().clone() for k,p in agg.named_parameters() if p.requires_grad}, adapters


def test_seed_changes_only_initializer_not_mode_matching_or_shuffle():
    a,aa=build('local_contrast',43)
    b,bb=build('shuffled_contrast',43)
    c,cc=build('local_contrast',44)
    assert a.keys()==b.keys()==c.keys()
    assert all(torch.equal(a[k],b[k]) for k in a)
    assert any(not torch.equal(a[k],c[k]) for k in a)
    assert all(torch.equal(x.permutation,y.permutation) for x,y in zip(aa,cc))
    assert all(torch.count_nonzero(x.output.weight)==0 for x in aa+bb+cc)


def test_contract_normalization_does_not_hide_training_changes():
    c=dict(mode='local_contrast',policy=dict(seed=42,lr=.0001),code={'scripts/train_local_value.py':'old','model':'same'},plan_sha256='plan')
    d=copy.deepcopy(c)
    d['mode']='shuffled_contrast';d['policy']['initialization_seed']=43
    d['code']={'scripts/train_local_value_repeat.py':'new','model':'same'}
    assert comparable(c)==comparable(d)
    d['policy']['lr']=.001
    assert comparable(c)!=comparable(d)
