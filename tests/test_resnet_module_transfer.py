"""CPU mechanics only; run on training machine, never claims retrieval gains."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.models.resnet_module_transfer import (
    ARMS, WEIGHT_URL, align_native_maps, make_neck, ResNetModuleTransfer,
    load_official_resnet, verify_resnet_download, gradient_groups)
from src.models.official_boq import OfficialBoQ, COMMIT, SOURCE_SHA
from src.resnet_transfer_protocol import POLICY, paired, assessment


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def active_state(neck):
    return {n:p.detach().clone() for n,p in neck.named_parameters() if p.requires_grad}


def maps(size=4):
    g = torch.Generator().manual_seed(123)
    return tuple(torch.randn(2,c,size,size,generator=g) for c in (1024,1024,1024))


def test_official_choice_and_sizes():
    assert WEIGHT_URL.endswith('/v1.0/resnet50_16384.pth')
    assert POLICY['primary_eval_size']==384 and POLICY['train_size']==320
    assert POLICY['eval_sizes']==[320,384] and POLICY['descriptor_dim']==16384
    assert POLICY['steps']==2048 and POLICY['smoke_steps']==8
    assert POLICY['rho']==.025 and POLICY['seed']==42


@pytest.mark.parametrize('size',[3,4,5])
def test_native_alignment_shapes_rms_and_last_identity(size):
    native = [torch.randn(2,c,size*scale,size*scale) for c,scale in ((256,4),(512,2),(1024,1))]
    aligned = align_native_maps(native)
    assert aligned[-1] is native[-1]
    assert all(x.shape==(2,1024,size,size) for x in aligned)
    for x,y in zip(native[:-1],aligned[:-1]):
        pooled = F.adaptive_avg_pool2d(x,(size,size))
        assert torch.allclose(pooled.square().mean(1), y.square().mean(1),atol=1e-6)
    assert not any(x.requires_grad for x in aligned)


def test_alignment_rejects_other_encoder():
    with pytest.raises(ValueError):
        align_native_maps([torch.ones(1,768,4,4)]*3)


@pytest.mark.parametrize('arm',ARMS)
def test_zero_start_reproduces_last_and_freezes_aux(arm):
    neck = make_neck(arm,42)
    x = maps()
    out, _, residual = neck(x)
    assert torch.equal(out,x[-1]) and torch.count_nonzero(residual)==0
    assert all(not p.requires_grad for p in neck.aux.parameters())
    assert all(isinstance(s.spatial,nn.Identity) for s in neck.stages)
    assert all(not hasattr(s,'ssm') for s in neck.stages)


@pytest.mark.parametrize('pair',[('conv_last','conv_cross'),('conditional_last','conditional_cross')])
def test_matched_capacity_initialization(pair):
    a,b = [active_state(make_neck(arm,42)) for arm in pair]
    assert a.keys()==b.keys()
    assert all(torch.equal(a[k],b[k]) for k in a)


@pytest.mark.parametrize('arm',ARMS)
def test_bound_and_nonzero_training_gradients(arm):
    neck = make_neck(arm,42)
    x = maps(3)
    target = torch.randn_like(x[-1])
    optimizer = torch.optim.Adam([p for p in neck.parameters() if p.requires_grad],lr=1e-3)
    class Container(nn.Module):
        def __init__(self):
            super().__init__()
            self.neck=neck
            self.arm=arm
    model=Container()
    seen={}
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        out,_,residual=neck(x)
        (out-target).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in neck.parameters() if p.requires_grad)
        for key,value in gradient_groups(model).items():
            seen[key]=seen.get(key,False) or value
        optimizer.step()
    assert all(seen.values()),seen
    assert torch.count_nonzero(residual)>0
    assert torch.all(residual.flatten(1).norm(dim=1) <= .025*x[-1].flatten(1).norm(dim=1)+1e-6)


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,8,1),nn.BatchNorm2d(8),nn.ReLU(),nn.Identity(),
            nn.Conv2d(8,256,1),nn.Conv2d(256,512,1),nn.Conv2d(512,1024,1))
    def forward(self,x):
        return self.net(x)


class FakeAggregator(nn.Module):
    def __init__(self,*args):
        super().__init__()
        self.fc=nn.Linear(1024,16)
    def forward(self,x):
        return F.normalize(self.fc(x.mean((2,3))),dim=1),[]


def visual():
    return OfficialBoQ(FakeBackbone(),FakeAggregator())


@pytest.mark.parametrize('arm',ARMS)
def test_wrapper_zero_start_bn_and_readout_gradient(arm):
    base=visual()
    model=ResNetModuleTransfer(base,arm).train()
    before={k:v.clone() for k,v in base.state_dict().items()}
    x=torch.randn(2,3,4,4)
    assert not base.training and not base.backbone.net[1].training
    direct=base(x)
    d=model(x)
    assert torch.allclose(d,direct,atol=1e-6,rtol=0)
    d[:,0].sum().backward()
    assert model.neck.up.weight.grad is not None
    assert not any(p.grad is not None for p in base.parameters())
    assert all(torch.equal(v,before[k]) for k,v in base.state_dict().items())
    assert all(not layer._forward_hooks for layer in base.backbone.net)


def test_offline_loader_strictly_loads_buffers_and_uses_no_pretraining(tmp_path):
    expected=visual()
    torch.save(expected.state_dict(),tmp_path/'resnet50_16384.pth')
    class Backbones:
        @staticmethod
        def ResNet(**kwargs):
            assert kwargs==dict(backbone_name='resnet50',pretrained=False,unfreeze_n_blocks=0,crop_last_block=True)
            return FakeBackbone()
    class Aggregation:
        BoQ=FakeAggregator
    with patch('src.models.resnet_module_transfer.verify_resnet_download'),patch(
            'src.models.resnet_module_transfer.load_source',side_effect=[Backbones,Aggregation]):
        loaded=load_official_resnet(tmp_path)
    assert all(torch.equal(v,expected.state_dict()[k]) for k,v in loaded.state_dict().items())
    assert all(not p.requires_grad for p in loaded.parameters())


def test_corrupt_digest_rejected(tmp_path):
    (tmp_path/'resnet50_16384.pth').write_bytes(b'bad')
    (tmp_path/'manifest.json').write_text(json.dumps(dict(commit=COMMIT,weight_url=WEIGHT_URL,
        source_sha256=SOURCE_SHA,weight_sha256='wrong')))
    with pytest.raises(ValueError,match='SHA256'):
        verify_resnet_download(tmp_path)


def test_paired_and_no_gain_does_not_mean_failure_to_execute():
    r=paired([False,True,False],[True,False,True])
    assert r['corrections']==[0,2] and r['regressions']==[1] and r['net']==1
    assert r['nominal_exact_two_sided_p']==1
    assert paired([True],[True])['nominal_exact_two_sided_p']==1
    with pytest.raises(ValueError):
        paired([1],[True])
    part={a:dict(vs_official=dict(net=0)) for a in ARMS}
    part.update(conv_cross_vs_conv_last=dict(net=0),conditional_cross_vs_conditional_last=dict(net=0))
    result=assessment({'msls-val_384':part,'pitts30k-val_384':part})
    assert not any(result['gain_checks'].values()) and not result['stable_gain_proven']


def test_no_implicit_training_or_old_run_mutation():
    root=Path(__file__).resolve().parents[1]
    source=(root/'scripts/resnet_module_transfer.py').read_text(encoding='utf-8')
    shell=(root/'scripts/run_resnet_module_transfer.sh').read_text(encoding='utf-8')
    assert "args.command=='train' and not args.allow_training" in source
    assert 'ACTION=${1:-preflight}' in shell
    assert "shared = (ROOT/'logs/query_relation_screen_v1.lock')" in source
    assert "self.visual.eval()" in (root/'src/models/resnet_module_transfer.py').read_text()
