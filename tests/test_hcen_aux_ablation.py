"""Run on the training machine; preserve the sealed 128-step implementation."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.hcen_aux_ablation import ARMS, POLICY, training_schedule, objective, gradient_diagnostic, paired_report
from src.models.hcen_vpr import HCENVPR

ROOT = Path(__file__).resolve().parents[1]


def triplets(count=8192):
    a = np.arange(count)*4
    return np.stack((a, a+1, ((np.arange(count)+101)%count)*4), axis=1)


def test_uniform_budget_and_smoke_prefix():
    full = training_schedule(triplets(), 'screen')
    smoke = training_schedule(triplets(), 'smoke')
    assert len(full) == 512 and len(smoke) == 8 and smoke == full[:8]
    assert [b[:8] for b in full] == np.arange(0, 8192, 2).reshape(512, 8).tolist()
    assert all(8 <= len(b) <= 16 and len(set(b)) == len(b) for b in full)


@pytest.mark.parametrize('rows', [triplets(100), np.zeros((8192, 2), int), triplets().astype(float),
                                 np.zeros((8192, 3), int), -triplets()])
def test_invalid_schedules_rejected(rows):
    with pytest.raises(ValueError): training_schedule(rows, 'screen')


def test_bad_positive_and_negative_rejected():
    for column in (1, 2):
        rows = triplets()
        rows[0, column] = 20 if column == 1 else 0
        with pytest.raises(ValueError): training_schedule(rows, 'screen')
    with pytest.raises(ValueError): training_schedule(triplets(), 'unknown')


@pytest.mark.parametrize('arm', ARMS)
def test_exact_objective_weights_and_gradients(arm):
    a = torch.tensor(2., requires_grad=True); b = torch.tensor(3., requires_grad=True)
    loss = objective(a, b, arm); loss.backward()
    weight = ARMS[arm]['auxiliary_weight']
    assert float(loss) == pytest.approx(2+weight*3)
    assert float(a.grad) == 1 and float(b.grad) == pytest.approx(weight)


def test_zero_weight_does_not_update_auxiliary_head():
    head = nn.Linear(3, 2, bias=False); initial = head.weight.detach().clone()
    primary = torch.tensor(1., requires_grad=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0.)
    objective(primary, head(torch.ones(1, 3)).square().mean(), 'last_noaux').backward()
    optimizer.step()
    assert torch.equal(initial, head.weight)
    assert int(optimizer.state[head.weight]['step']) == 1


def test_gradient_diagnostic_uses_weighted_loss_and_excludes_auxiliary_head():
    up = nn.Parameter(torch.tensor([2.])); trunk = nn.Parameter(torch.tensor([3.])); head = nn.Parameter(torch.tensor([4.]))
    named = [('neck.up.weight', up), ('neck.trunk.weight', trunk), ('neck.aux.0.weight', head)]
    primary = (up*trunk).sum(); auxiliary = (trunk*head).sum()
    off = gradient_diagnostic(primary, 0*auxiliary, named)
    low = gradient_diagnostic(primary, .001*auxiliary, named)
    assert off['primary_shared_norm'] == pytest.approx(13**.5)
    assert off['primary_up_norm'] == 3
    assert off['weighted_auxiliary_shared_norm'] == 0
    assert low['weighted_auxiliary_shared_norm'] == pytest.approx(.004)
    assert low['weighted_auxiliary_up_norm'] == 0
    assert up.grad is None and trunk.grad is None and head.grad is None


def test_four_paired_contrasts_and_interaction():
    outcomes = dict(last_noaux=[True, False, False, True], hierarchy_noaux=[True, True, False, True],
                    last_lowaux=[True, False, True, True], hierarchy_lowaux=[True, True, True, False])
    report = paired_report(outcomes, [True, False, False, True], [0, 0, 1, 1], [False, True, True, False], 1)
    all_ = report['all']; pairs = all_['comparisons']
    assert pairs['hierarchy_noaux_vs_last_noaux'] == dict(corrections=1, regressions=0, net=1)
    assert pairs['hierarchy_lowaux_vs_last_lowaux'] == dict(corrections=1, regressions=1, net=0)
    assert all_['depth_auxiliary_interaction_count'] == -1
    assert all_['variants']['hierarchy_lowaux']['correction_ids'] == [1, 2]
    assert all_['variants']['hierarchy_lowaux']['regression_ids'] == [3]
    assert report['evaluation']['queries'] == 2


def test_empty_partition_is_json_safe_and_mismatch_rejected():
    outcomes = {a: [True, False] for a in ARMS}
    report = paired_report(outcomes, [True, False], [0, 0], [False, True], 1)
    assert report['evaluation']['variants']['ru']['r1'] is None
    json.dumps(report, allow_nan=False)
    with pytest.raises(ValueError): paired_report({'last_noaux': [True]}, [True], [0], [False], 1)
    with pytest.raises(ValueError): paired_report(outcomes, [True], [0], [False], 1)


class Visual(nn.Module):
    def __init__(self):
        super().__init__(); self.aggregator = nn.Sequential(nn.Flatten(), nn.Linear(8*4*4, 16))
        self.semantic_region_gate = None


def test_four_arms_have_identical_initialization_and_descriptor():
    torch.manual_seed(12); visual = Visual(); features = [torch.randn(4, 8, 4, 4) for _ in range(3)]
    states = []; outputs = []
    for arm, spec in ARMS.items():
        torch.manual_seed(42)
        model = HCENVPR(visual, spec['mode'], channels=8, dim=8, inner=8, state=2)
        states.append({n: p.detach().clone() for n, p in model.neck.named_parameters()})
        outputs.append(model.aggregate(*features)[0].detach())
    assert all(all(torch.equal(p, s[n]) for n, p in states[0].items()) for s in states[1:])
    assert all(torch.equal(outputs[0], o) for o in outputs[1:])
    assert set(s['auxiliary_weight'] for s in ARMS.values()) == {0., .001}
    assert POLICY['rank_weight'] == 0 and POLICY['feature_relative_cap'] == .1


def test_original_trainer_unchanged_and_new_stage_does_not_optimize_ranking():
    source = (ROOT/'scripts/train_hcen_vpr.py').read_text(encoding='utf-8')
    tree = ast.parse(source); function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    assert hashlib.sha256(ast.dump(function, include_attributes=False).encode()).hexdigest() == '1e629e782005381e7f503827394c967459ac9dea7ac66f0e3faf31493b0080f0'
    new = (ROOT/'scripts/train_hcen_aux_ablation.py').read_text(encoding='utf-8')
    calls = [n for n in ast.walk(ast.parse(new)) if isinstance(n, ast.Call)]
    assert not any(isinstance(n.func, ast.Name) and n.func.id == 'ranking_regularizer' for n in calls)
    assert 'previous_provenance(args.previous)' in new
    assert 'HCENVPR(visual, ARMS[args.arm]' in new
    assert 'previous/' not in new  # No trained pilot checkpoint load.


def test_launcher_lf_gpu1_lock_all_arms():
    path = ROOT/'scripts/run_hcen_aux_ablation.sh'; raw = path.read_bytes(); source = raw.decode()
    assert b'\r' not in raw
    assert 'CUDA_VISIBLE_DEVICES=1' in source and 'flock -n 9' in source
    assert 'logs/query_relation_screen_v1.lock' in source
    assert 'for ARM in last_noaux hierarchy_noaux last_lowaux hierarchy_lowaux' in source
    assert 'compare_smoke' in source and 'SMOKE_PASS' in source
    assert 'git pull' not in source and 'pip install' not in source
