"""Module inventory, shared initialization, real gradients and immutable reference."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.models.hcen_ablation import VARIANTS,COUNTS,build_model,common_parameters,module_gradient_status
from src.hcen_module_protocol import POLICY,summarize,aggregate_reports
from src.hcen_plain_protocol import PlainHCENVPR

ROOT=Path(__file__).resolve().parents[1]


class Visual(nn.Module):
    def __init__(self,channels=8):
        super().__init__();self.semantic_region_gate=None;self.spatial_attn_head=None
        self.aggregator=nn.Sequential(nn.Flatten(),nn.Linear(channels*4*4,16))


@pytest.mark.parametrize('variant',('full',*VARIANTS))
def test_real_parameter_counts_not_unused_fillers(variant):
    v=nn.Module();v.semantic_region_gate=None;v.spatial_attn_head=None;v.aggregator=nn.Identity()
    m=build_model(v,variant,42)
    assert sum(p.numel() for p in m.neck.parameters())==COUNTS[variant]
    assert all(n.startswith('neck.') for n,p in m.named_parameters() if p.requires_grad)
    names=list(m.state_dict())
    if variant=='linear':assert not any('.ssm.' in n or '.spatial.' in n or '.shallow.' in n or '.symmetric.' in n for n in names)
    if variant=='conv':assert not any('.ssm.' in n for n in names) and any('.conv_mixer.hidden.' in n for n in names)
    if variant=='no_sasf':assert not any('.spatial.' in n for n in names) and any('.ssm.select.' in n for n in names)


def test_full_factory_identical_to_original_and_all_shared_parameters_identical():
    torch.manual_seed(42);v=Visual()
    torch.manual_seed(43);original=PlainHCENVPR(v,'hierarchy',channels=8,dim=8,inner=8,state=2)
    full=build_model(v,'full',43,channels=8,dim=8,inner=8,state=2)
    assert all(torch.equal(p,full.neck.state_dict()[n]) for n,p in original.neck.state_dict().items())
    for variant in VARIANTS:
        m=build_model(v,variant,43,channels=8,dim=8,inner=8,state=2)
        assert all(torch.equal(p,common_parameters(m)[n]) for n,p in common_parameters(full).items())


def test_full_factory_reproduces_real_sealed_neck_initialization_if_available():
    # No CUDA or pretrained encoder needed: only neck trainable parameters enter this hash.
    from scripts.train_slgd import parameter_hash,read
    paths={42:ROOT/'logs/hcen_plain_base_v1',43:ROOT/'logs/hcen_plain_multiseed_v1/seed43',44:ROOT/'logs/hcen_plain_multiseed_v1/seed44'}
    if not all((p/'hierarchy_screen/initial.json').exists() for p in paths.values()):pytest.skip('Sealed reference not present')
    for s,p in paths.items():
        v=nn.Module();v.semantic_region_gate=None;v.aggregator=nn.Identity()
        assert parameter_hash(build_model(v,'full',s))==read(p/'hierarchy_screen/initial.json')['parameter_sha256']


@pytest.mark.parametrize('variant',VARIANTS)
def test_zero_start_bounds_and_actual_module_gradient(variant):
    torch.manual_seed(12);v=Visual();maps=[torch.randn(4,8,4,4) for _ in range(3)]
    m=build_model(v,variant,42,channels=8,dim=8,inner=8,state=2)
    out=m.aggregate(*maps,auxiliary=True)
    assert torch.equal(out[0],v.aggregator(maps[-1])) and not out[-1].any()
    optimizer=torch.optim.AdamW(m.neck.parameters(),lr=.001,weight_decay=0.)
    seen={'0':False,'1':False}
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True);out=m.aggregate(*maps,auxiliary=True)
        loss=out[0].square().mean()+sum(x.square().mean()*0 for x in out[1:3]);loss.backward()
        for k,yes in module_gradient_status(m).items():seen[k]|=yes
        assert all(p.grad is None for p in v.parameters())
        optimizer.step()
        ratio=out[-1].flatten(1).norm(dim=1)/maps[-1].flatten(1).norm(dim=1)
        assert (ratio<=.100001).all()
    assert all(seen.values())


def test_conv_and_full_fusion_have_equal_zero_mixer_start():
    v=Visual();full=build_model(v,'full',44,channels=8,dim=8,inner=8,state=2)
    conv=build_model(v,'conv',44,channels=8,dim=8,inner=8,state=2)
    low=torch.randn(2,8,4,4);high=torch.randn_like(low)
    for a,b in zip(full.neck.stages,conv.neck.stages):
        assert torch.equal(a(low,high),b(low,high))
        assert torch.equal(a.ssm.in_proj.weight,b.conv_mixer.in_proj.weight)
        assert torch.equal(a.ssm.out.weight,b.conv_mixer.out.weight)


def reports():
    result={}
    for s,net in zip((42,43,44),(1,2,-1)):
        values={v:[True,False,False] for v in ['full','last_only',*VARIANTS]}
        p=summarize(values,[True,False,False],[0,0,1],1)
        for part in p.values():
            for v in part['variants'].values():v['correct']=1
            part['comparisons']['full_vs_conv']['net']=net
        result[s]={'gsv_'+k:v for k,v in p.items()}
        result[s].update({'msls-val':copy.deepcopy(p['all']),'pitts30k-val':copy.deepcopy(p['all'])})
    return result


def test_all_controls_and_query_pairing():
    values={v:[True,False,False] for v in ['full','last_only',*VARIANTS]};values['full']=[True,True,False]
    p=summarize(values,[True,False,False],[0,0,1],1)
    assert p['all']['comparisons']['full_vs_conv']==dict(corrections=[1],regressions=[],net=1)
    assert len(p['all']['comparisons'])==4
    with pytest.raises(ValueError):summarize({'full':[True]},[True],[0],1)
    with pytest.raises(ValueError):summarize(values,[True],[0],1)
    assert summarize({v:[True] for v in values},[True],[0],1)['evaluation']['variants']['full']['r1'] is None


def test_aggregate_all_seeds_not_best_and_no_nan():
    out=aggregate_reports(reports());v=out['pitts30k-val']['comparisons']['full_vs_conv']
    assert v['net_by_seed']=={42:1,43:2,44:-1}
    assert v['mean_net']==pytest.approx(2/3) and not v['positive_every_seed']
    json.dumps(out,allow_nan=False)
    r=reports();r.pop(44)
    with pytest.raises(ValueError):aggregate_reports(r)


def test_old_reference_sources_unchanged():
    p=ROOT/'scripts/hcen_plain_multiseed.py'
    assert hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()=='f6801c13587cda6e8924cbc03190e1d29fe33a8beadc094de36d0b1cbcb4d7ea'
    assert POLICY['screen_steps']==512 and POLICY['trainable_parameters']['conv']==892356
    assert 'NOT equal parameter' in POLICY['fairness']


def test_launcher_waits_gpu1_without_killing_others_and_requires_all_smokes():
    raw=(ROOT/'scripts/run_hcen_module_ablation.sh').read_bytes();text=raw.decode()
    assert b'\r' not in raw and 'CUDA_VISIBLE_DEVICES=1' in text and 'flock -n 9' in text
    assert 'for SEED in 42 43 44; do train_stage "$SEED" smoke; done' in text
    assert 'for VARIANT in linear conv no_sasf' in text and 'SMOKE_PASS' in text
    assert text.index('hcen_module_ablation.py wait_gpu')<text.index('hcen_module_ablation.py train')
    helper=(ROOT/'scripts/hcen_module_ablation.py').read_text(encoding='utf-8')
    assert "if args.command=='wait_gpu':wait_gpu(args);return" in helper
    assert "used<=1500 and util<=5" in helper and 'stable>=3' in helper
    assert 'os.kill(' not in helper and 'nvidia-smi --gpu-reset' not in helper
