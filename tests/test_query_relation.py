import torch
from torch import nn
from src.models.query_relation import QueryRelationAttention,geometry,MODES


def test_geometry_translation_and_direction():
    a=torch.eye(4)[None]
    p=torch.tensor([[0.,0.],[1,0],[0,1],[1,1]])
    g=geometry(a,p)
    assert torch.allclose(g,geometry(a,p+2),atol=1e-6)
    assert torch.equal(g[0,0,1,:2],torch.tensor([1.,0.]))
    assert torch.equal(g[0,1,0,:2],torch.tensor([-1.,0.]))


def test_zero_start_capacity_and_two_step_gradients():
    counts=[]
    for mode in MODES:
        torch.manual_seed(5)
        original=nn.MultiheadAttention(16,4,batch_first=True)
        model=QueryRelationAttention(original,mode,4,grid=2)
        q,x=torch.randn(2,4,16),torch.randn(2,4,16)
        expected=original(q,x,x)[0]
        assert torch.allclose(model(q,x,x)[0],expected,atol=1e-6)
        active=[p for p in model.parameters() if p.requires_grad];counts.append(sum(p.numel() for p in active))
        opt=torch.optim.SGD(active,lr=.01)
        target=torch.randn_like(expected)
        for step in range(2):
            opt.zero_grad();(model(q,x,x)[0]*target).sum().backward()
            assert model.output.weight.grad.abs().sum()>0
            if step: assert model.project.weight.grad.abs().sum()>0 and model.edge[0].weight.grad.abs().sum()>0
            opt.step()
        assert all(p.grad is None for p in original.parameters())
        model.enabled=False
        assert torch.allclose(model(q,x,x)[0],expected,atol=1e-6)
    assert len(set(counts))==1


def test_wrong_geometry_changes_branch_after_nonzero_output():
    torch.manual_seed(6)
    model=QueryRelationAttention(nn.MultiheadAttention(16,4,batch_first=True),'aligned',4,grid=2)
    with torch.no_grad():model.output.weight.normal_();model.permutation.copy_(torch.tensor([3,2,1,0]))
    q,x=torch.randn(2,4,16),torch.randn(2,4,16)
    aligned=model(q,x,x)[0]
    model.mode='shuffled'
    assert not torch.allclose(aligned,model(q,x,x)[0],atol=1e-7)
