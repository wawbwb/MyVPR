import numpy as np
from scripts.channel_reliability import features,geo_split


def test_features_no_labels_and_batch_independent():
    a=np.array([np.arange(20),np.arange(1,21)])
    b=np.array([np.arange(5,25),np.arange(20)])
    sa=np.tile(np.linspace(.9,.3,20),(2,1));sb=sa-.1
    x=features(a,b,sa,sb)
    assert x.shape==(2,14) and np.isfinite(x).all()
    np.testing.assert_array_equal(x,np.concatenate([features(a[i:i+1],b[i:i+1],sa[i:i+1],sb[i:i+1]) for i in range(2)]))
    assert x[0,10]==15/25 and x[0,11]==0 and x[0,12]==0 and x[0,13]==1/6


def test_equal_scores_finite_entropy():
    a=np.arange(20)[None,:];s=np.ones((1,20))
    x=features(a,a,s,s)
    np.testing.assert_allclose(x[0,4],np.log(20));assert x[0,11]==1


def test_geographic_components_are_not_split():
    coords=np.array([[0,0],[0,.0002],[0,.0004],[0,1],[0,2],[0,3],[0,4]])
    train,groups=geo_split(coords)
    assert groups[0]==groups[1]==groups[2]
    for g in set(groups):assert len(set(train[groups==g]))==1
    assert train.any() and (~train).any()
    np.testing.assert_array_equal(train,geo_split(coords)[0])
