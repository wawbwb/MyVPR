"""CPU mechanics; executed only on the training machine."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

from src.models.official_boq import native_feature_layout, OfficialBoQ, verify_download, COMMIT, WEIGHT_URL, WEIGHT_SHA, SOURCE_SHA


@pytest.mark.parametrize('size',[20,23])
@pytest.mark.parametrize('batch',[1,3])
def test_native_layout_and_gradient(size,batch):
    x=torch.randn(batch,8,size,size,requires_grad=True)
    y=native_feature_layout(x)
    assert torch.equal(x,y)
    assert y.stride()[1:]==(1,size*8,8)
    # A singleton batch has no observable batch stride; PyTorch may choose
    # any equivalent leading stride when reshaping its post-CLS view.
    if batch>1:
        assert y.stride()[0]==(size*size+1)*8
    y.square().sum().backward()
    assert torch.equal(x.grad,2*x.detach())


def test_native_layout_rejects_non_grid():
    with pytest.raises(ValueError):native_feature_layout(torch.ones(2,4,8))


def test_official_forward_does_not_add_a_gate():
    class Readout(nn.Module):
        def forward(self,x):return x.sum(-1),[]
    model=OfficialBoQ(nn.Identity(),Readout())
    x=torch.randn(2,3,4)
    assert torch.equal(model(x),x.sum(-1))
    assert model.semantic_region_gate is None


def test_provenance_pin_is_not_local_ru():
    assert COMMIT=='1a4965ea7dfd9bd0dd846adf7a0e430f68101d12'
    assert WEIGHT_URL.endswith('/v1.0/dinov2_12288.pth')
    assert WEIGHT_SHA!='dba5770cd3351d5354decc2bd385a831107591cfa392a0f0d4e66e95bf0c316d'
    assert set(SOURCE_SHA)=={'backbones.py','boq.py','hubconf.py','LICENSE','README.md'}


def test_wrong_upstream_commit_rejected(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(commit='wrong',weight_url=WEIGHT_URL)))
    with pytest.raises(ValueError,match='provenance'):verify_download(tmp_path)


def test_different_download_digest_rejected(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(commit=COMMIT,weight_url=WEIGHT_URL,weight_sha256='wrong')))
    with pytest.raises(ValueError,match='SHA256'):verify_download(tmp_path)


def test_protocol_keeps_primary_official_size_and_all_seeds():
    from scripts.conv_official_boq import POLICY
    assert POLICY['train_size']==280 and POLICY['primary_eval_size']==322
    assert POLICY['seeds']==[42,43,44] and POLICY['steps']==2048
    assert POLICY['eval_sizes']==[280,322]
    assert POLICY['auxiliary_weight']==POLICY['rank_weight']==0
    assert 'Pitts30k-test' in POLICY['scope']


def test_offline_builder_loads_every_tensor_strictly(tmp_path):
    from src.models.official_boq import load_official
    class Backbone(nn.Module):
        def __init__(self):super().__init__();self.proj=nn.Linear(2,2)
    class Aggregator(nn.Module):
        def __init__(self,*args):super().__init__();self.proj=nn.Linear(2,2)
        def forward(self,x):return self.proj(x),[]
    class Backbones:DinoV2=Backbone
    class Aggregation:BoQ=Aggregator
    expected=OfficialBoQ(Backbone(),Aggregator())
    torch.save(expected.state_dict(),tmp_path/'dinov2_12288.pth')
    (tmp_path/'hubconf.py').touch()
    with patch('src.models.official_boq.verify_download'),patch('src.models.official_boq.load_source',side_effect=[Backbones,Aggregation]):
        loaded=load_official(tmp_path,tmp_path)
    assert set(loaded.state_dict())==set(expected.state_dict())
    assert all(torch.equal(p,expected.state_dict()[n]) for n,p in loaded.state_dict().items())
    assert all(not p.requires_grad for p in loaded.parameters())


def test_official_transfer_does_not_mutate_sealed_old_sources():
    root=Path(__file__).resolve().parents[1]
    source=(root/'scripts/conv_official_boq.py').read_text()
    assert 'original_local_baseline_not_used=True' in source
    assert "'logs/conv_official_boq_v1'" in source
    assert 'matches_official_one_decimal' in source
