import numpy as np
import pytest
from scripts.channel_rank_fusion import fuse,oracle


def test_identity_and_consensus():
    a=np.arange(20)[None,:]
    assert np.array_equal(fuse(a,a),a)
    b=np.r_[1,20,np.arange(2,20)][None,:]
    f=fuse(a,b)[0]
    assert f[0]==1 and len(set(f))==20


def test_disjoint_ties_use_baseline_then_reference_id():
    a=np.arange(20)[None,:];b=np.arange(20,40)[None,:]
    f=fuse(a,b)[0]
    assert f[:4].tolist()==[0,20,1,21]
    assert set(f)<=set(a[0])|set(b[0])


def test_batch_independent_and_oracle_not_in_ranking():
    a=np.array([np.arange(20),np.arange(1,21)])
    b=np.array([np.arange(20,40),np.arange(21,41)])
    assert np.array_equal(fuse(a,b),np.concatenate([fuse(a[i:i+1],b[i:i+1]) for i in range(2)]))
    info=oracle(a,b,[[20],[1]])
    assert info['top1_oracle_correct']==2 and info['branch_only_correct_ids']==[0]
    assert info['baseline_only_correct_ids']==[1] and info['added_top20_reachable_ids']==[0]


@pytest.mark.parametrize('bad',['duplicate','shape','k','float','negative'])
def test_invalid(bad):
    a=np.arange(20)[None,:];b=a.copy();k=60
    if bad=='duplicate':b[0,1]=0
    if bad=='shape':b=b[:,:19]
    if bad=='k':k=10
    if bad=='float':b=b.astype(float)
    if bad=='negative':b[0,0]=-1
    with pytest.raises(ValueError):fuse(a,b,k)
