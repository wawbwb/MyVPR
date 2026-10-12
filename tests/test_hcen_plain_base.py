"""Only execute on training machine; the sealed RU experiments stay unchanged."""
import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from src.hcen_plain_protocol import POLICY,PLAIN_SHA,PLAIN_CHECKPOINT,validate_checkpoint,paired_report,PlainHCENVPR,canonical_features
from src.models.hcen_vpr import HCENVPR

ROOT=Path(__file__).resolve().parents[1]


def config():
    return dict(distillation={'enabled':False},backbone={'class':'DinoV2','params':{'backbone_name':'dinov2_vitb14'}},aggregator={'class':'BoQ'})


def test_true_plain_identity_pinned():
    assert 'BoQ/version_8/checkpoints/epoch(29)' in PLAIN_CHECKPOINT
    assert 'semantic_region' not in PLAIN_CHECKPOINT
    assert PLAIN_SHA=='dba5770cd3351d5354decc2bd385a831107591cfa392a0f0d4e66e95bf0c316d'
    assert POLICY['screen_steps']==512 and POLICY['smoke_steps']==8
    assert POLICY['auxiliary_weight']==POLICY['rank_weight']==0
    validate_checkpoint(config(),{'backbone.weight':torch.ones(1),'aggregator.weight':torch.ones(1)})


@pytest.mark.parametrize('key',['semantic_region_gate.weight','spatial_attn_head.weight','backbone_fake.weight','_orig_mod.backbone.weight'])
def test_nonplain_or_uncanonical_state_rejected(key):
    with pytest.raises(ValueError):validate_checkpoint(config(),{'backbone.weight':torch.ones(1),'aggregator.weight':torch.ones(1),key:torch.ones(1)})


def test_nonplain_config_or_missing_module_rejected():
    c=config();c['distillation']['enabled']=True
    with pytest.raises(ValueError):validate_checkpoint(c,{'backbone.x':torch.ones(1),'aggregator.x':torch.ones(1)})
    for field in ['crop_semantic_film','residual_clip_fusion']:
        c=config();c['backbone']['params'][field]={'enabled':True}
        with pytest.raises(ValueError):validate_checkpoint(c,{'backbone.x':torch.ones(1),'aggregator.x':torch.ones(1)})
    for state in [{},{'backbone.x':torch.ones(1)},{'aggregator.x':torch.ones(1)}]:
        with pytest.raises(ValueError):validate_checkpoint(config(),state)


def test_paired_report_is_against_plain_not_ru():
    r=paired_report(dict(last_only=[True,False,True,True],hierarchy=[True,True,False,True]),
                    [True,False,False,True],[0,0,1,1],1)
    assert r['all']['variants']['last_only']['correct']==3
    assert r['all']['variants']['hierarchy']['corrections']==[1]
    assert r['all']['hierarchy_vs_last_only']==dict(corrections=[1],regressions=[2],net=0)
    assert r['calibration']['hierarchy_vs_last_only']['net']==1
    assert r['evaluation']['hierarchy_vs_last_only']['net']==-1
    r=paired_report(dict(last_only=[True],hierarchy=[True]),[True],[0],1)
    assert r['evaluation']['variants']['baseline']['r1'] is None
    json.dumps(r,allow_nan=False)
    with pytest.raises(ValueError):paired_report({'last_only':[True]},[True],[0],1)


class Visual(nn.Module):
    def __init__(self):
        super().__init__();self.semantic_region_gate=None;self.spatial_attn_head=None
        self.aggregator=nn.Sequential(nn.Flatten(),nn.Linear(8*4*4,16))


def test_model_zero_start_works_without_any_ru_gate_and_trains_neck_only():
    torch.manual_seed(42);visual=Visual();maps=[torch.randn(4,8,4,4) for _ in range(3)]
    direct=visual.aggregator(maps[-1]).detach();states=[]
    for mode in ['last_only','hierarchy']:
        torch.manual_seed(42);model=HCENVPR(visual,mode,channels=8,dim=8,inner=8,state=2)
        value=model.aggregate(*maps,auxiliary=True)
        assert torch.equal(value[0],direct)
        assert not value[-1].any() and visual.semantic_region_gate is None
        assert all(n.startswith('neck.') for n,p in model.named_parameters() if p.requires_grad)
        states.append(copy.deepcopy(model.neck.state_dict()))
    assert all(torch.equal(p,states[1][n]) for n,p in states[0].items())


def test_original_ru_runner_remains_exact_and_plain_loader_strict():
    old=ROOT/'scripts/train_hcen_aux_ablation.py'
    assert hashlib.sha256(old.read_bytes().replace(b'\r\n',b'\n')).hexdigest()=='c15851e4a50af08df85b5cf77c4245c576dbbeefa484ebe361ba38cfd2913316'
    new=(ROOT/'scripts/hcen_plain_base.py').read_text(encoding='utf-8')
    assert 'set(visual.state_dict())!=set(state)' in new
    assert 'if visual.semantic_region_gate is not None or visual.spatial_attn_head is not None:' in new
    assert 'visual.semantic_region_gate=None' not in new
    assert "result['baseline']!=read(baseline/path.name)['baseline']" in new
    assert "baseline_completed_sha256=reference_sha" in new
    calls=[n for n in ast.walk(ast.parse(new)) if isinstance(n,ast.Call)]
    assert not any(isinstance(n.func,ast.Name) and n.func.id=='ranking_regularizer' for n in calls)
    assert 'loss=primary+0.*auxiliary' in new


def test_native_plain_readout_layout_not_hook_or_channels_last():
    raw=torch.randn(2,8,4,4)
    channels_last=raw.contiguous(memory_format=torch.channels_last)
    canonical=canonical_features((channels_last,channels_last,channels_last))
    assert all(f.is_contiguous() and torch.equal(f,raw) for f in canonical)
    class Readout(nn.Module):
        def forward(self,x):
            assert x.is_contiguous()
            return x.mean((2,3))
    v=Visual();v.aggregator=Readout()
    model=PlainHCENVPR(v,'hierarchy',channels=8,dim=8,inner=8,state=2)
    assert torch.equal(model.readout(channels_last),raw.mean((2,3)))
    assert torch.equal(model.aggregate(*canonical)[0],raw.mean((2,3)))


def test_layout_fix_has_no_new_weights_and_preserves_gradient():
    v=Visual();torch.manual_seed(42);old=HCENVPR(v,'hierarchy',channels=8,dim=8,inner=8,state=2)
    torch.manual_seed(42);new=PlainHCENVPR(v,'hierarchy',channels=8,dim=8,inner=8,state=2)
    assert set(old.neck.state_dict())==set(new.neck.state_dict())
    assert all(torch.equal(p,new.neck.state_dict()[n]) for n,p in old.neck.state_dict().items())
    x=torch.randn(2,8,4,4).contiguous(memory_format=torch.channels_last).requires_grad_()
    new.readout(x).square().sum().backward()
    assert x.grad is not None and x.grad.abs().sum()>0


def test_launcher_lf_gpu1_gate_before_training():
    raw=(ROOT/'scripts/run_hcen_plain_base.sh').read_bytes();text=raw.decode()
    assert b'\r' not in raw
    assert 'CUDA_VISIBLE_DEVICES=1' in text and 'flock -n 9' in text
    assert 'logs/query_relation_screen_v1.lock' in text and 'SMOKE_PASS' in text
    assert text.index('scripts/hcen_plain_base.py baseline')<text.index('scripts/hcen_plain_base.py train')
    assert 'for MODE in last_only hierarchy' in text
    assert 'pip install' not in text and 'git pull' not in text
