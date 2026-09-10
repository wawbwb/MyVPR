import numpy as np
from src.region_pair_lite import match_regions, conservative_order, pool


def test_no_regions_abstain():
    assert match_regions(np.empty((0,4)),np.eye(4),np.empty((0,2)),np.zeros((4,2)))==(0.,0)


def test_mutual_matches_translation_consensus():
    d=np.eye(4); xy=np.array([[0,0],[0,1],[1,0],[1,1]],float)
    score,count=match_regions(d,d,xy,xy+.1)
    assert count==4 and score==1.


def test_large_ru_margin_preserved():
    assert conservative_order(np.array([.9,.7]),np.array([0.,1.]),np.array([0,4])).tolist()==[0,1]


def test_small_margin_and_strong_support_promotes():
    assert conservative_order(np.array([.9,.89]),np.array([.5,.8]),np.array([3,4])).tolist()==[1,0]


def test_pool_shape():
    d,xy=pool(np.ones((400,8)),np.ones((3,20,20),bool))
    assert d.shape==(3,8) and np.allclose(xy,.5)
