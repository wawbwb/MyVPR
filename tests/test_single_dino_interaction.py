"""CPU-only model/regression and fail-closed serial dependency tests."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.models.single_dino_interaction import ARMS, SingleDinoInteraction, SingleDinoVPR, local_permutation
from src.models.hetero_interleave import replay_backward
from src.single_dino_dependency import dependency_status, require_ready


def head(arm):
    torch.manual_seed(42)
    return SingleDinoInteraction(arm,channels=12,dim=8,rank=4)


@pytest.mark.parametrize('arm',ARMS)
def test_zero_start_and_finite_connected_new_parameters(arm):
    torch.manual_seed(9);x=torch.randn(3,12,2,3);m=head(arm)
    assert torch.equal(m(x)[0],x)
    opt=torch.optim.AdamW(m.parameters(),lr=.001)
    seen={n:False for n,p in m.named_parameters()}
    for _ in range(4):
        opt.zero_grad();out,aux,stats=m(x)
        ((out*torch.randn_like(out)).mean()+sum(z[:,0].mean() for z in aux)).backward()
        for n,p in m.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(),n
            seen[n]|=bool((p.grad!=0).any())
        opt.step()
        assert float(stats['relative_residual'].max())<=.025+1e-6
    assert all(seen.values()),[n for n,v in seen.items() if not v]


def test_three_same_capacity_identical_initialization_and_no_shared_weights():
    models=[head(a) for a in ARMS];state=models[0].state_dict()
    assert len({sum(p.numel() for p in m.parameters()) for m in models})==1
    for m in models[1:]:assert all(torch.equal(v,m.state_dict()[n]) for n,v in state.items())
    assert not set(map(id,models[0].parameters()))&set(map(id,models[1].parameters()))


def test_three_routings_differ_after_nonzero_output():
    x=torch.randn(2,12,3,4);models=[head(a) for a in ARMS]
    with torch.no_grad():
        models[0].output.weight.normal_(std=.2)
        for m in models[1:]:m.load_state_dict(models[0].state_dict())
    outputs=[m(x)[0] for m in models]
    assert not torch.allclose(outputs[0],outputs[1],atol=1e-7,rtol=1e-7)
    assert not torch.allclose(outputs[1],outputs[2],atol=1e-7,rtol=1e-7)


def test_permutation_deterministic_without_mutating_global_rng():
    state=torch.random.get_rng_state().clone()
    a=local_permutation(3,4,'cpu');b=local_permutation(3,4,'cpu')
    assert torch.equal(a,b) and torch.equal(a.sort().values,torch.arange(12))
    assert torch.equal(torch.random.get_rng_state(),state)


class TinyNeck(nn.Module):
    def __init__(self):super().__init__();self.weight=nn.Parameter(torch.ones(12))
    def forward(self,layers,auxiliary=False):
        z=layers[-1]*self.weight[None,:,None,None];return z,(),torch.zeros_like(z)


class TinyAnchor(nn.Module):
    def __init__(self):super().__init__();self.neck=TinyNeck();self.calls=0
    def features(self,x):self.calls+=1;return x,x,x
    def readout(self,x):return F.normalize(x.flatten(1),dim=1)


def test_actual_single_encoder_wrapper_one_encode_and_frozen_anchor():
    anchor=TinyAnchor();m=SingleDinoVPR(anchor,'aligned',channels=12,dim=8,rank=4)
    x=torch.randn(2,12,3,4);features=m.encode(x)
    assert anchor.calls==1 and not hasattr(m,'resnet')
    assert features[2].data_ptr()==features[5].data_ptr()
    z,g,l,_=m.describe(features)
    (z[:,0].mean()+g[:,0].mean()+l[:,0].mean()).backward()
    assert all(not p.requires_grad and p.grad is None for p in anchor.parameters())
    m.train();assert not anchor.training and m.fusion.training


def test_single_operator_gradient_replay_matches_direct():
    torch.manual_seed(4);x=torch.randn(4,12,2,3)
    direct=SingleDinoVPR(TinyAnchor(),'aligned',channels=12,dim=8,rank=4)
    with torch.no_grad():direct.fusion.output.weight.normal_(std=.01)
    replay=copy.deepcopy(direct);features=direct.encode(x)
    outputs=direct.describe(features)[:3];weights=[torch.randn_like(y) for y in outputs]
    sum((a*b).sum() for a,b in zip(outputs,weights)).backward()
    with torch.no_grad():leaves=[y.detach().requires_grad_() for y in replay.describe(features)[:3]]
    sum((a*b).sum() for a,b in zip(leaves,weights)).backward()
    replay_backward(replay,features,[y.grad for y in leaves],2)
    for (n,p),(k,q) in zip(direct.fusion.named_parameters(),replay.fusion.named_parameters()):
        assert n==k;torch.testing.assert_close(p.grad,q.grad,rtol=5e-4,atol=2e-5,msg=n)


def test_private_runner_binding_never_changes_live_dual_module():
    from scripts import hetero_interleave_study as original
    from scripts import single_dino_interaction as single
    schema,arms,builder=original.SCHEMA,original.ARMS,original.build
    bound=single.bound_engine()
    assert bound is not original and bound.SCHEMA=='single_dino_interaction_v1'
    assert bound.ARMS==ARMS and bound.build is single.build
    assert original.SCHEMA==schema and original.ARMS==arms and original.build is builder


def test_single_build_never_calls_resnet_loader(monkeypatch):
    from scripts import single_dino_interaction as single
    import src.models.resnet_module_transfer as legacy
    def forbidden(*a,**k):raise AssertionError('ResNet must never be loaded')
    monkeypatch.setattr(legacy,'load_official_resnet',forbidden)
    monkeypatch.setattr(single,'make_anchor',lambda c:TinyAnchor())
    c=dict(seed=42,device='cpu',dim=8,rank=4,rho=.025)
    models=single.build(c)
    assert set(models)==set(ARMS) and all(not hasattr(m,'resnet') for m in models.values())


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf8')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(directory):
    dump(directory/'completed.json',dict(complete=True,schema='hetero_interleave_v1',
        files={p.name:digest(p) for p in directory.iterdir() if p.is_file() and p.name!='completed.json'}))


def fake_upstream(root,phase='complete'):
    run=root/'logs/dual';launcher=root/'.cache/dual_launcher'
    dump(run/'train/contract.json',dict(stage='train'))
    (run/'train/last.pt').write_bytes(b'existing-final-checkpoint')
    dump(run/'train/summary.json',dict(steps=1024,checks=dict(complete=True)))
    seal(run/'train')
    dump(run/'eval/contract.json',dict(stage='eval',checkpoint_sha256=digest(run/'train/last.pt')))
    dump(run/'eval/summary.json',dict(checks=dict(frozen=True),verdict='NO_MATCHED_HETERO_INTERACTION_GAIN'))
    seal(run/'eval')
    dump(launcher/'progress.json',dict(phase=phase,completed_stages=['cpu_tests','plan','smoke','train','eval']))
    return dict(wait_for_run='logs/dual',wait_for_launcher='.cache/dual_launcher',
        wait_for_train_contract_sha256=digest(run/'train/contract.json'))


@pytest.mark.parametrize('phase',['train','eval','smoke'])
def test_not_ready_while_upstream_still_running_even_if_seal_exists(tmp_path,phase):
    c=fake_upstream(tmp_path,phase)
    assert dependency_status(tmp_path,c,False)['state']=='waiting'
    with pytest.raises(RuntimeError):require_ready(tmp_path,c)


def test_ready_after_train_and_eval_without_requiring_positive_gain(tmp_path):
    c=fake_upstream(tmp_path)
    d=dependency_status(tmp_path,c,False)
    assert d['state']=='ready' and 'eval_completed_sha256' in d


def test_upstream_failure_cannot_start_new_training(tmp_path):
    c=fake_upstream(tmp_path,'failed')
    assert dependency_status(tmp_path,c,False)['state']=='blocked'


@pytest.mark.parametrize('file',['train/summary.json','eval/summary.json','train/contract.json','train/last.pt'])
def test_tampering_or_wrong_run_blocks_dependency(tmp_path,file):
    c=fake_upstream(tmp_path)
    target=tmp_path/'logs/dual'/file;target.write_bytes(target.read_bytes()+b' ')
    assert dependency_status(tmp_path,c,False)['state']=='blocked'


def test_missing_evaluation_blocks_even_after_training(tmp_path):
    c=fake_upstream(tmp_path);(tmp_path/'logs/dual/eval/completed.json').unlink()
    assert dependency_status(tmp_path,c,False)['state']=='blocked'


def test_direct_cli_guard_precedes_gpu_and_output_creation(monkeypatch):
    from scripts import single_dino_interaction as s
    def wait(*a,**k):raise RuntimeError('UPSTREAM_NOT_COMPLETE')
    def forbidden(*a,**k):raise AssertionError('No GPU engine construction permitted')
    monkeypatch.setattr(s,'require_ready',wait);monkeypatch.setattr(s,'bound_engine',forbidden)
    monkeypatch.setattr('sys.argv',['single_dino_interaction.py','smoke','--allow-training'])
    with pytest.raises(RuntimeError,match='UPSTREAM_NOT_COMPLETE'):s.main()


def test_waiting_queue_imports_no_torch_or_cuda():
    import ast
    root=Path(__file__).resolve().parents[1]
    for file in ('scripts/queue_single_dino_interaction.py','src/single_dino_dependency.py'):
        tree=ast.parse((root/file).read_text(encoding='utf8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):assert all(not alias.name.startswith('torch') for alias in node.names)
            if isinstance(node,ast.ImportFrom):assert not (node.module or '').startswith('torch')


def test_single_evaluator_uses_new_arm_names():
    import inspect
    from scripts.single_dino_interaction import evaluate
    text=inspect.getsource(evaluate)
    assert "reports['aligned']" in text and "reports['permuted']" in text
    assert "reports['hetero_interleaved']" not in text


def test_single_completion_does_not_claim_a_frozen_cnn(tmp_path):
    from scripts.single_dino_interaction import finish,bound_engine
    dump(tmp_path/'contract.json',dict(stage='smoke'))
    dump(tmp_path/'summary.json',dict(checks=dict(frozen_anchor_and_cnn_unchanged=True,complete=True)))
    finish(tmp_path)
    report=json.loads((tmp_path/'summary.json').read_text())
    assert report['encoder_count']==1 and report['checks']['frozen_anchor_unchanged'] is True
    assert 'frozen_anchor_and_cnn_unchanged' not in report['checks']
    assert bound_engine().verified(tmp_path)==dict(stage='smoke')
