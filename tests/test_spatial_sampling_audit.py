import pytest
import torch
from src.models.spatial_resample import SpatialResample
from src.spatial_sampling_audit import field,centered,shuffle,apply,intervention_fields,VARIANTS


@pytest.fixture(autouse=True)
def seed():torch.manual_seed(42);torch.set_num_threads(2)


@pytest.mark.parametrize('arm',['global_shift','local_shift'])
def test_source_forward_reproduced(arm):
    m=SpatialResample(arm,12,8);torch.nn.init.normal_(m.output.weight,std=.2)
    x=torch.randn(3,12,3,5);offset=field(m,x)
    y,r=apply(x,offset);old,s=m(x)
    assert torch.equal(y,old);torch.testing.assert_close(r,s['relative_delta'])


def test_centering_bound_and_permutation_marginals():
    offset=torch.rand(3,2,3,5)*2-1
    offset[0,0,0,0]=1;offset[0,0,1:]=-1
    c=centered(offset)
    torch.testing.assert_close(c.mean((2,3)),torch.zeros(3,2),atol=1e-7,rtol=0)
    assert c.abs().max()<=1
    d=shuffle(c,[1,2,3]);assert not torch.equal(d,c)
    torch.testing.assert_close(c.flatten(2).sort(2).values,d.flatten(2).sort(2).values)
    # Entire xy vectors must move together, including their cross-product.
    torch.testing.assert_close((c[:,0]*c[:,1]).flatten(1).sort(1).values,(d[:,0]*d[:,1]).flatten(1).sort(1).values)


def test_constant_fields_and_interventions_batch_independent():
    g=torch.rand(3,2,1,1).expand(3,2,3,5);local=torch.rand(3,2,3,5)*2-1
    constants={'global':torch.tensor([.5,-.2]),'local':torch.tensor([-.3,.7])}
    fields=intervention_fields(g,local,constants,[1,2,3])
    assert set(fields)==set(VARIANTS[1:])
    for name,x in fields.items():
        one=torch.cat([intervention_fields(g[i:i+1],local[i:i+1],constants,[i+1])[name] for i in range(3)])
        assert torch.equal(x,one)
    assert torch.equal(fields['constant_global'][0],fields['constant_global'][2])


def test_zero_offset_identity_and_bound():
    f=torch.randn(2,12,3,5);z=torch.zeros(2,2,3,5)
    assert torch.equal(apply(f,z)[0],f)
    y,r=apply(f,torch.ones_like(z));assert r.max()<=.025001
    with pytest.raises(ValueError):apply(f,torch.ones_like(z)*1.1)
