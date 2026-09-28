from types import SimpleNamespace
import torch
from torch import nn
from src.models.aggregators.boq import BoQ
from src.models.depth_query import DepthQueryVPR,SlotDepthRouter,MODES


def visual():
    backbone=nn.Module();backbone.dino=nn.Module();backbone.dino.blocks=nn.ModuleList([nn.Identity() for _ in range(12)])
    backbone.out_channels=12;backbone.crop_semantic_film=None;backbone.residual_clip_fusion=None
    return SimpleNamespace(backbone=backbone,aggregator=BoQ(in_channels=12,proj_channels=64,num_queries=4,num_layers=1,row_dim=2),semantic_region_gate=None,spatial_attn_head=None)


def test_all_modes_zero_start_and_dimension():
    torch.manual_seed(3);layers=[torch.randn(2,12,3,3) for _ in range(3)]
    for mode in MODES:
        v=visual();model=DepthQueryVPR(v,mode).eval()
        with torch.no_grad():
            assert torch.equal(model.aggregate(layers),v.aggregator(layers[-1])[0])
            assert model.aggregate(layers).shape==(2,128)


def test_router_is_per_slot_and_equal_control():
    r=SlotDepthRouter(4);slots=torch.randn(2,5,3,4)
    out,w=r(slots)
    assert w.shape==(2,5,3)
    assert torch.allclose(out,slots.mean(2),atol=1e-6)
    with torch.no_grad():r.score[-1].bias.copy_(torch.tensor([1.,0.,-1.]))
    assert not torch.allclose(r(slots)[0],out)
    assert torch.allclose(r(slots,equal=True)[0],out)


def test_active_depth_branch_receives_gradient_after_output_update():
    torch.manual_seed(1);model=DepthQueryVPR(visual(),'adaptive_depth')
    layers=[torch.randn(2,12,3,3) for _ in range(3)];target=torch.randn(2,128)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    for _ in range(2):
        opt.zero_grad();(model.aggregate(layers)*target).sum().backward();opt.step()
    for module in (model.align,model.residuals,model.routers):
        assert sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)>0


def test_last_repeat_ignores_mid_layers():
    torch.manual_seed(3);model=DepthQueryVPR(visual(),'last_repeat').eval()
    torch.nn.init.normal_(model.residuals[0].weight,std=.02)
    layers=[torch.randn(2,12,3,3) for _ in range(3)]
    with torch.no_grad():assert torch.equal(model.aggregate(layers),model.aggregate([layers[0]*3,layers[1]*5,layers[2]]))


def test_stateless_place_sampling_survives_resume():
    import random
    import numpy as np
    from scripts.train_depth_query import MatchedPlaces
    class ToyPlaces:
        places_ids=['a','b','c']
        def __getitem__(self,index):return (random.random(),float(np.random.rand()),float(torch.rand(())))
    dataset=ToyPlaces();full=MatchedPlaces(dataset,[0,1,2],0)
    expected=full[2]
    random.seed(987);np.random.seed(222);torch.manual_seed(111)
    assert MatchedPlaces(dataset,[2],0)[0]==expected
    assert MatchedPlaces(dataset,[2],1)[0]!=expected
