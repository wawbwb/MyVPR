"""CPU mechanics and protocol tests; not evidence of GPU training or benefit."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.models.conv_residual_ablation import (
    ARMS, RHO, PARAMETERS, build_model, residual_from_raw, image_diagnostics,
    module_gradient_status,
)
from src.models.hcen_followup import build_model as historical_model
from src.conv_residual_protocol import (
    POLICY, CONTRASTS, arm_contract, training_schedule,
    diagnostic_summary, summarize, aggregate_reports,
)

ROOT = Path(__file__).resolve().parents[1]


class Visual(nn.Module):
    def __init__(self, channels=8):
        super().__init__()
        self.semantic_region_gate = None
        self.aggregator = nn.Sequential(nn.Flatten(), nn.Linear(channels * 4 * 4, 16))


OPTIONS = dict(channels=8, dim=8, inner=8, state=2)


@pytest.mark.parametrize('seed', (42, 43, 44))
def test_all_arms_exact_same_parameters_and_names_as_old_conv(seed):
    visual = Visual()
    old = historical_model(visual, 'conv', seed, **OPTIONS)
    expected = old.neck.state_dict()
    for arm in ARMS:
        model = build_model(visual, arm, seed, **OPTIONS)
        assert model.mode == 'hierarchy'
        assert set(model.neck.state_dict()) == set(expected)
        assert all(torch.equal(value, expected[name]) for name, value in model.neck.state_dict().items())
        assert not any('.ssm.' in name or '.spatial.' in name for name in model.state_dict())
        assert all(name.startswith('neck.') for name, p in model.named_parameters() if p.requires_grad)


@pytest.mark.parametrize('arm', ARMS)
def test_real_width_parameter_count(arm):
    visual = nn.Module(); visual.semantic_region_gate = None; visual.aggregator = nn.Identity()
    model = build_model(visual, arm, 42)
    assert sum(p.numel() for p in model.neck.parameters()) == PARAMETERS
    assert sum(p.numel() for p in model.neck.aux.parameters()) == 65536


@pytest.mark.parametrize('seed', (42, 43, 44))
def test_rms_function_matches_old_nonzero_projection(seed):
    visual = Visual(); old = historical_model(visual, 'conv', seed, **OPTIONS)
    model = build_model(visual, 'rms_tanh', seed, **OPTIONS)
    features = [torch.randn(3, 8, 4, 4) for _ in range(3)]
    with torch.no_grad():
        old.neck.up.weight.normal_(0, .05)
        model.neck.up.load_state_dict(old.neck.up.state_dict())
    actual = model.neck(features, auxiliary=True); expected = old.neck(features, auxiliary=True)
    assert expected[-1].any()
    assert torch.equal(actual[0], expected[0]) and torch.equal(actual[-1], expected[-1])
    assert all(torch.equal(a, b) for a, b in zip(actual[1], expected[1]))
    assert 'last_raw' not in model.state_dict()


def test_formulas_limits_and_rms_stopgradient():
    last = torch.full((2, 8, 4, 4), .01, requires_grad=True)
    raw = torch.full_like(last, 2., requires_grad=True)
    linear = residual_from_raw(last, raw, 'linear_residual')
    fixed = residual_from_raw(last, raw, 'fixed_tanh')
    rms = residual_from_raw(last, raw, 'rms_tanh')
    assert torch.equal(linear, .1 * raw)
    assert torch.equal(fixed, .1 * raw.tanh())
    assert torch.equal(rms, .1 * last.square().mean(1, keepdim=True).sqrt().detach() * raw.tanh())
    last_norm = last.flatten(1).norm(dim=1)
    assert (linear.flatten(1).norm(dim=1) / last_norm > .1).all()
    assert (fixed.flatten(1).norm(dim=1) / last_norm > .1).all()
    assert (rms.flatten(1).norm(dim=1) / last_norm <= .100001).all()
    rms.sum().backward()
    assert last.grad is None and raw.grad is not None
    for arm in ARMS:
        assert not residual_from_raw(last, torch.zeros_like(raw), arm).any()
    with pytest.raises(ValueError): residual_from_raw(last, raw, 'unknown')
    with pytest.raises(ValueError): residual_from_raw(last, raw[:, :, :3], 'rms_tanh')
    with pytest.raises(ValueError): residual_from_raw(last, raw, 'fixed_tanh', rho=float('nan'))


@pytest.mark.parametrize('arm', ARMS)
def test_zero_start_and_two_actual_fusion_gradients(arm):
    visual = Visual(); model = build_model(visual, arm, 42, **OPTIONS)
    features = [torch.randn(4, 8, 4, 4) for _ in range(3)]
    out = model.aggregate(*features, auxiliary=True)
    assert torch.equal(out[0], visual.aggregator(features[-1]))
    assert not out[-1].any()
    auxiliary = {name: p.detach().clone() for name, p in model.neck.aux.named_parameters()}
    optimizer = torch.optim.AdamW(model.neck.parameters(), lr=.001, weight_decay=0.)
    seen = {'0': False, '1': False}
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        out = model.aggregate(*features, auxiliary=True)
        loss = out[0][:, 0].square().mean() + sum(x.square().mean() * 0. for x in out[1:3])
        loss.backward()
        for key, yes in module_gradient_status(model).items():
            seen[key] |= yes
        assert all(p.grad is None for p in visual.parameters())
        optimizer.step()
    assert all(seen.values())
    assert all(torch.equal(p, auxiliary[name]) for name, p in model.neck.aux.named_parameters())


def descriptors(batch):
    return F.normalize(torch.ones(batch, 16), dim=1)


def test_diagnostics_do_not_impose_rms_limit_on_other_arms():
    last = torch.full((2, 8, 4, 4), .01); raw = torch.full_like(last, 2.)
    d = descriptors(2)
    for arm in ARMS:
        residual = residual_from_raw(last, raw, arm)
        rows = image_diagnostics(last, raw, residual, d, d, arm)
        assert len(rows) == 2
        if arm == 'rms_tanh':
            assert rows[0]['image_relative_l2'] <= .100001
        else:
            assert rows[0]['image_relative_l2'] > .1
        if arm == 'linear_residual':
            assert rows[0]['element_bound_violation_fraction'] is None
        else:
            assert rows[0]['element_bound_violation_fraction'] == 0
        assert rows[0]['valid_reference_patch_count'] == 16
        if arm == 'linear_residual':
            assert rows[0]['applied_tanh_saturation_fraction'] is None
        else:
            assert rows[0]['applied_tanh_saturation_fraction'] == 0
        json.dumps(diagnostic_summary(rows), allow_nan=False)
    with pytest.raises(ValueError, match='bound'):
        image_diagnostics(last, raw, torch.ones_like(last), d, d, 'rms_tanh')
    with pytest.raises(ValueError, match='bound'):
        image_diagnostics(last, raw, torch.ones_like(last), d, d, 'fixed_tanh')


def test_zero_energy_missing_ratios_are_explicit_not_infinite():
    last = torch.zeros(2, 8, 4, 4); raw = torch.ones_like(last); d = descriptors(2)
    for arm in ARMS:
        residual = residual_from_raw(last, raw, arm)
        rows = image_diagnostics(last, raw, residual, d, d, arm)
        assert rows[0]['image_relative_l2'] is None
        assert rows[0]['patch_relative_l2_p95'] is None
        assert rows[0]['zero_reference_patch_count'] == 16
        assert rows[0]['zero_reference_nonzero_residual_patch_count'] == (0 if arm == 'rms_tanh' else 16)
        summary = diagnostic_summary(rows)
        assert summary['metrics']['image_relative_l2']['n'] == 0
        json.dumps(summary, allow_nan=False)
    with pytest.raises(ValueError, match='Nonfinite'):
        image_diagnostics(last, raw * float('nan'), last, d, d, 'rms_tanh')


def test_no_checkpoint_recompute_statistic_side_effects():
    visual = Visual(); model = build_model(visual, 'rms_tanh', 42, **OPTIONS)
    features = [torch.randn(2, 8, 4, 4) for _ in range(3)]
    model.aggregate(*features)
    assert model.neck.last_raw is not None
    model.aggregate(*features)
    assert not hasattr(model.neck, 'diagnostic_rows') and not hasattr(model.neck, 'statistics')


def test_same_schedule_all_seeds_and_fixed_budget():
    anchor = np.arange(8192) * 4
    triplets = np.stack((anchor, anchor + 1, np.roll(anchor, 1)), 1)
    schedule = training_schedule(triplets, 'screen')
    assert len(schedule) == 2048
    assert schedule[:512] == schedule[512:1024] == schedule[1024:1536] == schedule[1536:]
    assert training_schedule(triplets, 'smoke') == schedule[:8]
    first = dict(arm=ARMS[0], residual_policy=POLICY, schedule=schedule, code={'all': 'sealed'}, seed=42)
    second = {**first, 'arm': ARMS[1]}
    assert arm_contract(first) == arm_contract(second)
    second = copy.deepcopy(second); second['schedule'][0] = [999]
    assert arm_contract(first) != arm_contract(second)
    assert POLICY['zero_initialization'].endswith('NOT a zero-init ablation')


def test_paired_metrics_all_seeds_not_best_case():
    values = dict(linear_residual=[True, False, False], fixed_tanh=[True, False, True], rms_tanh=[True, True, False])
    parts = summarize(values, [True, False, False], [0, 0, 1], 1)
    assert parts['all']['comparisons']['rms_tanh_vs_fixed_tanh'] == dict(corrections=[1], regressions=[2], net=0)
    reports = {s: {'gsv_' + role: copy.deepcopy(p) for role, p in parts.items()} for s in (42, 43, 44)}
    for s in reports:
        reports[s].update({'msls-val': copy.deepcopy(parts['all']), 'pitts30k-val': copy.deepcopy(parts['all'])})
    result = aggregate_reports(reports)
    assert result['msls-val']['variants']['rms_tanh']['mean_r1'] == pytest.approx(2/3)
    assert result['msls-val']['comparisons']['rms_tanh_vs_linear_residual']['positive_every_seed']
    reports.pop(44)
    with pytest.raises(ValueError): aggregate_reports(reports)
    with pytest.raises(ValueError): summarize({ARMS[0]: [True]}, [True], [0], 1)
    with pytest.raises(ValueError): diagnostic_summary([])
    empty = summarize({arm: [True] for arm in ARMS}, [True], [0], 1)
    assert empty['evaluation']['variants']['rms_tanh']['r1'] is None
    json.dumps(empty, allow_nan=False)


def test_atomic_resume_and_new_safe_explicit_launcher():
    script = (ROOT / 'scripts/conv_residual_ablation.py').read_text(encoding='utf-8')
    raw = (ROOT / 'scripts/run_conv_residual_ablation.sh').read_bytes(); shell = raw.decode()
    assert b'\r' not in raw
    assert 'CUDA_VISIBLE_DEVICES=1' in shell and 'flock -n 9' in shell
    assert 'for ARM in linear_residual fixed_tanh rms_tanh' in shell
    assert 'for SEED in 42 43 44; do train_stage "$SEED" smoke; done' in shell
    assert shell.index('conv_residual_ablation.py wait_gpu') < shell.index('conv_residual_ablation.py train')
    assert 'SMOKE_PASS' in shell and 'from scripts.train_slgd import verify,read' in shell
    assert 'state[\'steps\'] % 16 == 0' in script
    assert "save(args.output / 'last.pt'" in script and "restore(args.output / 'last.pt'" in script
    assert 'probe_cpu_rng' in script and 'probe_cuda_rng' in script
    assert 'gpu-reset' not in script and 'os.kill(' not in script
    assert CONTRASTS == (('rms_tanh', 'fixed_tanh'), ('fixed_tanh', 'linear_residual'),
                         ('rms_tanh', 'linear_residual'))
    assert 'historical_rms_predictions_match' in script and 'diagnostics_files' in script
