"""Run only on the training machine: frozen layer identities and selection."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src.slgd_layer_protocol import (LayerCapture, quantized_tokens, fine_query_ids,
                                     summarize_layers, choose_layer)


class Add(nn.Module):
    def forward(self, tokens):
        return tokens+1


class Backbone(nn.Module):
    def __init__(self, alter=False):
        super().__init__()
        self.dino=nn.Module()
        self.dino.blocks=nn.ModuleList([Add() for _ in range(12)])
        self.alter=alter

    def forward(self, images):
        tokens=images.flatten(2).transpose(1,2)
        tokens=torch.cat((tokens[:,:1]*0,tokens),dim=1)
        for block in self.dino.blocks:
            tokens=block(tokens)
        result=tokens[:,1:].transpose(1,2).reshape_as(images)
        return result+1 if self.alter else result


def test_hook_uses_one_based_layers_and_reproduces_actual_legacy_forward():
    model=Backbone()
    cap=LayerCapture(model,[4,6,8,10,12])
    images=torch.zeros(2,4,20,20)
    features,last=cap(images)
    assert float(features[4].mean())==4 and float(features[12].mean())==12
    assert torch.equal(features[12],last) and cap.values=={}
    cap.close()
    assert not any(block._forward_hooks for block in model.dino.blocks)


def test_last_layer_must_not_gain_an_unrecorded_norm_or_transform():
    cap=LayerCapture(Backbone(alter=True),[12])
    with pytest.raises(ValueError,match='reproduce'):
        cap(torch.zeros(1,4,20,20))
    cap.close()


@pytest.mark.parametrize('layers',[[0],[13],[4,4],[]])
def test_invalid_layer_spec_rejected(layers):
    with pytest.raises(ValueError):
        LayerCapture(Backbone(),layers)


def test_quantization_happens_after_spatial_pool_and_unit_normalization():
    features=torch.randn(2,8,20,20)
    for side in (10,20):
        tokens=quantized_tokens(features,side)
        assert tokens.shape==(2,side*side,8) and tokens.dtype==np.float16
        assert np.max(np.abs(np.linalg.norm(tokens.astype(np.float32),axis=-1)-1))<.002
    with pytest.raises(ValueError):
        quantized_tokens(features,5)


def test_fine_subset_has_all_ru_hard_and_is_deterministic():
    groups=np.repeat(np.arange(8),3)
    hard=np.zeros(len(groups),bool)
    hard[[0,2,12,13]]=True
    ids=fine_query_ids(groups,hard,4,6)
    assert len(ids)==12 and {0,2,12,13}.issubset(ids)
    assert ids==fine_query_ids(groups,hard,4,6)
    assert sum(groups[i]<4 for i in ids)==6


def test_fine_subset_cannot_silently_discard_hard_queries():
    with pytest.raises(ValueError):
        fine_query_ids(np.arange(8),np.ones(8,bool),4,3)


def fixture():
    groups=np.arange(8)
    positive=groups.copy()
    candidates=np.stack((groups+100,groups),axis=1)
    scores={'L12_g10':np.tile([.8,.2],(8,1)),
            'L4_g10':np.tile([.2,.8],(8,1))}
    valid={name:np.ones((8,2),bool) for name in scores}
    return candidates,positive,groups,np.ones(8),scores,valid


def test_metrics_count_corrections_vs_ru_and_last_separately():
    candidates,pos,groups,gap,scores,valid=fixture()
    summary,outcome=summarize_layers(candidates,pos,groups,gap,scores,valid,4)
    stats=summary['calibration']['variants']['L4_g10']
    assert stats['corrections_vs_ru']==4 and stats['net_vs_last']==4
    assert stats['correction_places']==4 and stats['mean_positive_negative_margin']>0
    assert outcome['predictions']['L4_g10']==pos.tolist()


def test_no_positive_candidate_never_counts_as_correction():
    candidates,pos,groups,gap,scores,valid=fixture()
    candidates[:,1]+=1000
    summary,outcome=summarize_layers(candidates,pos,groups,gap,scores,valid,4)
    assert not any(outcome['reachable'])
    assert summary['evaluation']['variants']['L4_g10']['corrections_vs_ru']==0
    assert summary['evaluation']['variants']['L4_g10']['margin_queries']==0


def test_calibration_selection_is_unchanged_by_evaluation_reversal():
    candidates,pos,groups,gap,scores,valid=fixture()
    summary,_=summarize_layers(candidates,pos,groups,gap,scores,valid,4)
    selected=choose_layer(summary['calibration'],(4,12))
    assert selected['layer']==4
    scores['L4_g10'][4:]=scores['L12_g10'][4:]
    changed,_=summarize_layers(candidates,pos,groups,gap,scores,valid,4)
    assert choose_layer(changed['calibration'],(4,12))==selected


def test_selecting_a_layer_requires_hard_gain_and_no_all_query_loss():
    stats={'L12_g10':dict(hard_net_vs_last=0,net_vs_last=0),
           'L4_g10':dict(hard_net_vs_last=2,net_vs_last=-1)}
    assert choose_layer(dict(variants=stats),(4,12))['layer']==12
    stats['L4_g10']['net_vs_last']=0
    assert choose_layer(dict(variants=stats),(4,12))['layer']==4


def test_cache_shape_hash_unit_norm_identity_checks(tmp_path):
    from scripts.audit_slgd_layers import validate_layer_shard, NEW_LAYERS
    from scripts.audit_slgd_hard import save_npz
    path=tmp_path/'00000.npz'
    tokens=np.full((1,len(NEW_LAYERS),100,768),1/np.sqrt(768),np.float16)
    save_npz(path,dict(indices=np.array([0],np.int64),tokens=tokens))
    assert validate_layer_shard(path,[0]).shape==tokens.shape
    with pytest.raises(ValueError):
        validate_layer_shard(path,[1])
    save_npz(path,dict(indices=np.array([0],np.int64),tokens=tokens*0))
    with pytest.raises(ValueError):
        validate_layer_shard(path,[0])


def test_fine_rows_use_exact_old_query_reference_views():
    from scripts.audit_slgd_layers import FinePairs
    data=object.__new__(FinePairs)
    data.nq=2
    data.place_images=lambda place:torch.tensor([10*place+i for i in range(4)])
    assert int(data.row_image(3))==3 and int(data.row_image(7))==13
    assert int(data.row_image(8))==20 and int(data.row_image(9))==30


def test_launcher_does_not_train_or_modify_original_sources():
    root=Path(__file__).resolve().parents[1]
    content=(root/'scripts/run_slgd_layers.sh').read_bytes()
    assert b'\r' not in content
    assert b'CUDA_VISIBLE_DEVICES=1' in content and b'flock -n' in content
    assert b'train_slgd.py --' not in content and b'sed -i' not in content
