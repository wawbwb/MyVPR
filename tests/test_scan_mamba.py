import pytest
import torch
from src.models.scan_mamba import selective_scan, SpatialMixer, descriptor_consistency, scan_order


def sequential(u,dt,a,b,c,d):
    h=torch.zeros(len(u),u.shape[-1],a.shape[-1],dtype=u.dtype,device=u.device)
    out=[]
    for i in range(u.shape[1]):
        h=(dt[:,i,:,None]*a).exp()*h + dt[:,i,:,None]*b[:,i,None,:]*u[:,i,:,None]
        out.append((h*c[:,i,None,:]).sum(-1)+d*u[:,i])
    return torch.stack(out,1)


@pytest.mark.parametrize('length',[1,7,20,31])
def test_scan_forward_and_backward(length):
    torch.manual_seed(4)
    values=[torch.randn(2,length,3,dtype=torch.double),torch.rand(2,length,3,dtype=torch.double)*.1,
            -torch.rand(3,4,dtype=torch.double),torch.randn(2,length,4,dtype=torch.double),
            torch.randn(2,length,4,dtype=torch.double),torch.ones(3,dtype=torch.double)]
    values=[v.requires_grad_() for v in values]
    y=selective_scan(*values); ref=sequential(*values)
    torch.testing.assert_close(y,ref,rtol=1e-10,atol=1e-10)
    probe=torch.randn_like(y)
    ga=torch.autograd.grad((y*probe).sum(),values,allow_unused=True)
    gb=torch.autograd.grad((ref*probe).sum(),values)
    for i,(a,b) in enumerate(zip(ga,gb)):
        if a is None:
            # At L=1, zero initial state makes A mathematically unused.
            assert length==1 and i==2 and torch.count_nonzero(b)==0
        else: torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-9)


def test_non_square_scan_inverse():
    x=torch.arange(15)
    for column in (False,True):
        order=scan_order(3,5,column,x.device)
        assert torch.equal(x[order][order.argsort()],x)


@pytest.mark.parametrize('mode',['conv','mamba','mamba_consistent'])
def test_zero_start_and_real_upstream_gradient(mode):
    torch.manual_seed(42)
    model=SpatialMixer(12,mode)
    x=torch.randn(2,15,12)
    row,col=model(x,3,5)
    assert torch.equal(x,row) and torch.equal(x,col)
    assert descriptor_consistency(row,col)==0
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    for _ in range(2):
        opt.zero_grad();row,col=model(x,3,5)
        (row.square().mean()+col.square().mean()).backward();opt.step()
    first=model.input.weight if mode=='conv' else model.ssm.in_proj.weight
    assert first.grad.abs().sum()>0 and torch.isfinite(first.grad).all()
    if mode!='conv':
        assert model.ssm.a_log.grad.abs().sum()>0
        assert model.ssm.select.weight.grad.abs().sum()>0
        assert not torch.equal(row,col)


def test_consistency_identity():
    x=torch.nn.functional.normalize(torch.randn(4,8),dim=-1)
    assert descriptor_consistency(x,x)==0
    assert descriptor_consistency(x,-x)>1.99


def test_mamba_arms_share_initialization_and_conv_capacity():
    torch.manual_seed(42);ordinary=SpatialMixer(384,'mamba')
    torch.manual_seed(42);consistent=SpatialMixer(384,'mamba_consistent')
    assert ordinary.state_dict().keys()==consistent.state_dict().keys()
    assert all(torch.equal(v,consistent.state_dict()[k]) for k,v in ordinary.state_dict().items())
    convolution=SpatialMixer(384,'conv')
    n=sum(p.numel() for p in ordinary.parameters())
    nc=sum(p.numel() for p in convolution.parameters())
    assert abs(n-nc)/n<.05


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Training-machine CUDA only')
def test_full_length_cuda_float32_scan():
    torch.manual_seed(9)
    u=torch.randn(2,400,4,device='cuda',requires_grad=True)
    dt=torch.rand_like(u)*.1
    a=-torch.arange(1,9,device='cuda').float().repeat(4,1)
    b=torch.randn(2,400,8,device='cuda')
    c=torch.randn_like(b);d=torch.ones(4,device='cuda')
    y=selective_scan(u,dt,a,b,c,d);r=sequential(u,dt,a,b,c,d)
    torch.testing.assert_close(y,r,rtol=2e-5,atol=2e-5)
    ga=torch.autograd.grad(y.square().mean(),u)[0]
    gb=torch.autograd.grad(r.square().mean(),u)[0]
    torch.testing.assert_close(ga,gb,rtol=2e-5,atol=2e-6)
