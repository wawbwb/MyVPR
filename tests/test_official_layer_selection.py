import numpy as np
import pytest

from src.official_layer_selection import LAYERS,COMBINATIONS,POLICY,fusion_indices,choose_layers,summarize


def candidates():
    scores=np.zeros((4,len(LAYERS),3),np.float32)
    scores[:,:,1]=1
    return scores,np.ones_like(scores,bool)


def test_predictions_use_scores_and_ties_keep_original_order():
    scores,valid=candidates()
    assert fusion_indices(scores,valid,(8,10,12)).tolist()==[1]*4
    scores[:]=0
    assert fusion_indices(scores,valid,(8,10,12)).tolist()==[0]*4


def test_invalid_queries_abstain_and_shape_rejected():
    scores,valid=candidates();valid[:]=False
    assert fusion_indices(scores,valid,(4,11,12)).tolist()==[0]*4
    with pytest.raises(ValueError):fusion_indices(scores[:,0],valid[:,0],(4,11,12))


def rows():
    return {'queries':100,'variants':{'8/10/12':dict(layers=[8,10,12],correct=90,correction_places=[1,2],net_vs_official=-5),
        '6/11/12':dict(layers=[6,11,12],correct=91,correction_places=[1,2,3],net_vs_official=-4)}}


def test_select_only_calibration_and_apply_all_gates():
    c=rows();s=choose_layers(c)
    assert s['gate'] and s['layers']==[6,11,12]
    c['variants']['6/11/12']['correct']=89
    assert not choose_layers(c)['gate']
    with pytest.raises(ValueError):choose_layers({'calibration':rows(),'evaluation':rows()})


def test_no_force_selection_when_novel_error_places_insufficient():
    c=rows();c['variants']['6/11/12']['correction_places']=[1,2]
    assert not choose_layers(c)['gate']


def test_summary_keeps_query_ids_and_place_groups_separate():
    scores,valid=candidates()
    summary=summarize(np.tile([0,1,2],(4,1)),np.ones(4,int),np.array([0,0,1,2]),scores,valid,np.array([8,9,10,11]))
    row=summary['variants']['8/10/12']
    assert row['corrections']==[8,9,10,11] and row['correction_places']==[0,1,2]


def test_fixed_budget_has_only_two_new_training_controls():
    assert POLICY['train_steps']==2048 and POLICY['rho']==.025 and len(COMBINATIONS)==10
    assert POLICY['token_lru_images']>=21 and POLICY['native_grid']==20


def test_feature_routing_keeps_native_layers_and_removes_hooks():
    import torch
    from torch import nn
    from types import SimpleNamespace
    from src.official_layer_selection import selected_features
    class Block(nn.Module):
        def __init__(self,index):super().__init__();self.index=index
        def forward(self,x):return x+self.index
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__();self.dino=nn.Module();self.dino.blocks=nn.ModuleList([Block(i+1) for i in range(12)])
        def forward(self,x):
            b,c,h,w=x.shape;t=x.flatten(2).transpose(1,2)
            t=torch.cat((torch.zeros(b,1,c),t),dim=1)
            for block in self.dino.blocks:t=block(t)
            return t[:,1:].transpose(1,2).reshape(b,c,h,w)
    backbone=Backbone();model=SimpleNamespace(visual=SimpleNamespace(backbone=backbone),fusion_layers=(6,11,12))
    x=torch.zeros(2,4,2,2);a,b,c=selected_features(model,x)
    assert torch.all(a==21) and torch.all(b==66) and torch.all(c==78)
    model.fusion_layers=(12,12,12)
    a,b,c=selected_features(model,x)
    assert torch.equal(a,b) and torch.equal(b,c)
    assert all(not block._forward_hooks for block in backbone.dino.blocks)
