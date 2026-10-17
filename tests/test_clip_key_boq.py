"""CPU unit tests intended for the training server; not run on the local PC."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.models.clip_key_boq import ARMS, ClipKeyBoQ, KeyRouter, permute_patches


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(8,2,32,batch_first=True,dropout=0.)
        self.queries = nn.Parameter(torch.randn(1,3,8))
        self.self_attn = nn.MultiheadAttention(8,2,batch_first=True)
        self.norm_q = nn.LayerNorm(8)
        self.cross_attn = nn.MultiheadAttention(8,2,batch_first=True)
        self.norm_out = nn.LayerNorm(8)

    def forward(self,x):
        x = self.encoder(x)
        q = self.queries.repeat(len(x),1,1)
        q = self.norm_q(q + self.self_attn(q,q,q)[0])
        return x,self.norm_out(self.cross_attn(q,x,x)[0])


class Aggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_c = nn.Conv2d(8,8,3,padding=1)
        self.norm_input = nn.LayerNorm(8)
        self.boqs = nn.ModuleList([Block(),Block()])
        self.fc = nn.Linear(6,4)

    def forward(self,x):
        x = self.norm_input(self.proj_c(x).flatten(2).permute(0,2,1))
        outs = []
        for block in self.boqs:
            x,out = block(x)
            outs.append(out)
        out = self.fc(torch.cat(outs,1).permute(0,2,1)).flatten(1)
        return F.normalize(out,dim=-1),None


class Official(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(3,8,1)
        self.aggregator = Aggregator()

    def forward(self,x):
        return self.aggregator(self.backbone(x))[0]


def example():
    torch.manual_seed(42)
    model = ClipKeyBoQ(Official(),hidden=4)
    return model,torch.randn(2,3,4,5),torch.randn(2,512,3,3)


@pytest.mark.parametrize('arm',ARMS)
@pytest.mark.parametrize('training',[True,False])
def test_zero_start_matches_exact_original(arm,training):
    model,images,keys = example()
    model.train(training)
    with torch.no_grad():
        assert torch.equal(model(images,keys,arm),model.visual(images))


def test_gate_opens_at_zero_without_changing_original_weights():
    model,images,keys = example()
    original = copy.deepcopy(model.visual.state_dict())
    optimizer = torch.optim.Adam(model.routers.parameters(),lr=.01)
    for step in range(2):
        optimizer.zero_grad()
        out = model(images,keys)
        (out*torch.arange(out.shape[-1])).sum().backward()
        assert any(r.beta.grad.abs().sum()>0 for r in model.routers)
        if step:
            assert all(r.key.weight.grad.abs().sum()>0 and r.query.weight.grad.abs().sum()>0 for r in model.routers)
        optimizer.step()
    assert all(p.grad is None and not p.requires_grad for p in model.visual.parameters())
    assert all(torch.equal(value,original[n]) for n,value in model.visual.state_dict().items())
    assert not torch.equal(model(images,keys),model.visual(images))


def test_softmax_bias_is_bounded_and_gate_nonzero_derivative():
    router = KeyRouter(8,3,hidden=4,cap=2.)
    q,s = torch.randn(2,3,8),torch.randn(2,13,512)
    bias = router(q,s)
    assert torch.count_nonzero(bias)==0
    (bias*torch.randn_like(bias)).sum().backward()
    assert router.beta.grad.abs().sum()>0
    with torch.no_grad():
        router.beta.fill_(100.)
    assert router(q,s).abs().max() <= 2.


def test_constant_semantic_tokens_produce_no_spatial_bias():
    router = KeyRouter(8,3,hidden=4)
    router.beta.data.fill_(1.)
    keys = torch.ones(2,9,512)
    assert torch.count_nonzero(router(torch.randn(2,3,8),keys))==0


def test_permutation_is_image_keyed_vector_preserving_derangement():
    x = torch.arange(2*512*3*3).reshape(2,512,3,3).float()
    y = permute_patches(x,['a','b'])
    assert (x!=y).all()
    assert torch.equal(y[1],permute_patches(x[1:],['b'])[0])
    assert torch.equal(x.flatten(2).sort(-1).values,y.flatten(2).sort(-1).values)
    with pytest.raises(ValueError):
        permute_patches(x,['a'])


def test_guidance_changes_attention_without_touching_value_inputs():
    model,images,keys = example()
    for router in model.routers:
        router.beta.data.fill_(.5)
    seen = []
    handles = [block.cross_attn.register_forward_pre_hook(lambda m,a: seen.append(a[2].detach().clone()))
               for block in model.visual.aggregator.boqs]
    a = model(images,keys)
    first = [x.clone() for x in seen]
    seen.clear()
    b = model(images,permute_patches(keys,['a','b']))
    for handle in handles:
        handle.remove()
    assert not torch.allclose(a,b)
    assert all(torch.equal(x,y) for x,y in zip(first,seen))


def test_visual_control_does_not_depend_on_clip():
    model,images,keys = example()
    for router in model.routers:
        router.beta.data.fill_(.3)
    assert torch.equal(model(images,keys,'visual'),model(images,keys*0,'visual'))


def test_only_new_parameters_train_and_base_stays_eval():
    model,_,_ = example()
    model.train()
    assert all(n.startswith('routers.') for n,p in model.named_parameters() if p.requires_grad)
    assert model.routers.training and not model.visual.training
    assert not model.visual.aggregator.training


def test_three_arms_have_equal_initialization_capacity():
    states = []
    for _ in ARMS:
        model,_,_ = example()
        states.append({n:v.clone() for n,v in model.state_dict().items()})
    assert all(all(torch.equal(s[n],states[0][n]) for n in s) for s in states)


@pytest.mark.parametrize('arm',['aligned','shuffled'])
def test_missing_or_nonfinite_semantics_are_rejected(arm):
    model,images,keys = example()
    with pytest.raises(ValueError):
        model(images,None,arm)
    keys[0,0,0,0] = float('nan')
    with pytest.raises(ValueError):
        model(images,keys,arm)


def test_float16_cache_checks_unit_feature_norm_not_class_probabilities():
    from scripts.clip_key_study import validate_keys
    x = F.normalize(torch.randn(2,512,14,14),dim=1).numpy().astype(np.float16)
    validate_keys(x,2)
    with pytest.raises(ValueError):
        validate_keys(x*0,2)


def test_training_requires_explicit_authorization():
    script = Path(__file__).resolve().parents[1]/'scripts/clip_key_study.py'
    for command in ('smoke','train'):
        result = subprocess.run([sys.executable,str(script),command],capture_output=True,text=True)
        assert result.returncode != 0 and '--allow-training' in result.stderr


def test_config_seals_same_gsv_schedule_and_excludes_benchmark_tuning():
    from scripts.clip_key_study import configuration
    c = configuration(Path(__file__).resolve().parents[1]/'config/clip_key_boq.yaml')
    assert c['epochs']*c['train_places']//c['places_per_batch']==2560
    assert c['cuda_memory_fraction']<=.2 and c['microbatch']==2
    assert 'pitts30k-test' not in json.dumps(c)
    assert c['runs']=='logs/clip_key_boq_v1'


def test_owned_artifact_detects_corruption(tmp_path):
    from scripts.clip_key_study import finish,verified
    from src.semantic_region_protocol import owned_run,write
    output = tmp_path/'run'
    with owned_run(output,{'schema':'clip_key_boq_v1'}):
        write(output/'summary.json',{'x':1})
        finish(output)
    assert verified(output)['schema']=='clip_key_boq_v1'
    (output/'summary.json').write_text('{}')
    with pytest.raises(ValueError):
        verified(output)
