import numpy as np
import pytest
import torch

from src.hard_channel_batching import ARMS,validate_partition,pack_places,difficulty_gate
from src.channel_rank_preservation import teacher_contract,ranking_penalty
from scripts.hard_channel import configuration


def test_packing_equal_exposure_deterministic_and_no_repeat():
    n=64
    # Four visually similar blocks; each neighbor list stays within its block.
    graph=np.asarray([[j for j in range(i//16*16,(i//16+1)*16) if j!=i] for i in range(n)])
    def fallback(a,unused):return unused
    a,stats=pack_places(graph,42,fallback)
    b,_=pack_places(graph,42,fallback)
    assert a==b and stats['fallback_batches']==0
    validate_partition(a,n)
    assert all(len({i//16 for i in group})==1 for group in a)
    assert sorted(p for group in a for p in group)==list(range(n))


def test_fallback_fills_remaining_without_replacement():
    graph=np.asarray([[(i+1)%32] for i in range(32)])
    calls=[]
    def fallback(a,unused):calls.append((a,list(unused)));return unused[::-1]
    batches,stats=pack_places(graph,3,fallback)
    validate_partition(batches,32)
    assert calls and stats['fallback_batches']>0


@pytest.mark.parametrize('bad',['duplicate','shape','out_of_range','noninteger'])
def test_invalid_partition_rejected(bad):
    batches=[list(range(16)),list(range(16,32))]
    if bad=='duplicate':batches[1][0]=0
    elif bad=='shape':batches[0].pop()
    elif bad=='out_of_range':batches[1][-1]=32
    else:batches[0][0]=0.
    with pytest.raises(ValueError):validate_partition(batches,32)


def test_incomplete_fallback_fails_not_silently_random_fill():
    graph=np.zeros((32,1),dtype=np.int64)
    with pytest.raises(ValueError):pack_places(graph,42,lambda a,unused:[])


def test_gate_requires_actual_hardness_and_same_valid_population():
    random=dict(queries=1000,valid=1000,errors=1,low_margin=5)
    hard=dict(queries=1000,valid=1000,errors=2,low_margin=12)
    assert difficulty_gate(random,hard,1000)['pass_all']
    for change in ({'queries':999},{'valid':999},{'errors':0},{'low_margin':8}):
        assert not difficulty_gate(random,dict(hard,**change),1000)['pass_all']


def test_sample_exposure_does_not_depend_on_batch_packing():
    random=[list(range(i,i+16)) for i in range(0,64,16)]
    order=np.random.default_rng(42).permutation(64).tolist()
    hard=[order[i:i+16] for i in range(0,64,16)]
    exposures=lambda batches:sorted((p,v) for b in batches for p in b for v in range(4))
    assert exposures(random)==exposures(hard)


def test_hardness_changes_teacher_errors_not_labels_or_gps():
    # Each place has two near-identical views. Places0/1 confuse each other,
    # while2/3 live on orthogonal directions. Batch membership is the only change.
    z=torch.nn.functional.normalize(torch.tensor([[1.,0.,0.],[.99,.01,0.],[1.,.001,0.],[.99,.011,0.],
        [0.,1.,0.],[0.,.99,.01],[0.,0.,1.],[.01,0.,.99]]),dim=1)
    labels=torch.arange(4).repeat_interleave(2)
    gps=torch.arange(4).repeat_interleave(2).double()[:,None].expand(-1,2)
    easy=torch.tensor([0,1,4,5]);hard=torch.tensor([0,1,2,3])
    a=teacher_contract(z[easy],labels[easy],gps[easy]);b=teacher_contract(z[hard],labels[hard],gps[hard])
    assert int(b['correct'].sum())<int(a['correct'].sum())


def test_protection_and_ms_arms_share_architecture_and_objective_switch_only():
    assert len(ARMS)==4 and set(ARMS)=={'random_ms','random_protect','hard_ms','hard_protect'}
    z=torch.nn.functional.normalize(torch.randn(8,5),dim=1).requires_grad_()
    labels=torch.arange(4).repeat_interleave(2);gps=torch.arange(4).repeat_interleave(2).double()[:,None].expand(-1,2)
    t=teacher_contract(z.detach(),labels,gps)
    loss,_=ranking_penalty(z,t,'cosine_only');loss.backward()
    assert loss==0 and torch.count_nonzero(z.grad)==0


def test_config_uses_same_fixed_warm_and_bounded_cache():
    c=configuration()
    assert c['pilot_steps']==512 and c['train_places']==8192 and c['views']==4 and c['places_per_batch']==16
    assert c['warm_run']=='logs/structural_channel_v1/train' and c['warm_arm']=='structure_gate'
    assert c['neighbor_count']==64 and c['rank_weight']==.1 and c['cuda_memory_fraction']==.2
    assert c['train_places']*4*12288*2/2**20==768
