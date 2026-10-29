"""Run on the training machine only; toy CPU tensors require no model download."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from src.models.conditional_complement import (
    ARMS, ConditionalComplement, AnchoredConditionalNeck, conditional_gradient_status)
from src.conditional_complement_protocol import (
    POLICY, named_hash, matched_contract, compare_outcomes, candidate_verdict)
from src.models.conv_residual_ablation import residual_from_raw


class ToyStage(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mix = nn.Conv2d(2 * dim, dim, 1)

    def forward(self, low, high):
        return .5 * (low + high) + .1 * self.mix(torch.cat((low, high), 1))


class ToyAnchor(nn.Module):
    def __init__(self, channels=8, dim=4):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(channels) for _ in range(3)])
        self.projections = nn.ModuleList([nn.Linear(channels, dim, bias=False) for _ in range(3)])
        self.stages = nn.ModuleList([ToyStage(dim), ToyStage(dim)])
        self.aux = nn.ModuleList([nn.Linear(dim, 2, bias=False) for _ in range(2)])
        self.up = nn.Conv2d(dim, channels, 1, bias=False)
        self.output_scale = .25

    def forward(self, last):
        b, _, h, w = last.shape
        aligned = [p(n(last.flatten(2).transpose(1, 2))).transpose(1, 2).reshape(b, -1, h, w)
                   for p, n in zip(self.projections, self.norms)]
        raw = self.up(self.stages[1](self.stages[0](aligned[0], aligned[1]), aligned[2]))
        residual = .25 * residual_from_raw(last, raw, 'rms_tanh')
        return last + residual, residual


def features(height=3, width=4):
    torch.manual_seed(123)
    return tuple(torch.randn(2, 8, height, width) for _ in range(3))


def neck(arm):
    torch.manual_seed(42)
    return AnchoredConditionalNeck(ToyAnchor(), arm, channels=8, dim=4, rank=2)


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('grid', [(3, 4), (4, 5)])
def test_zero_addition_exactly_reproduces_nonzero_trained_anchor(arm, grid):
    layers = features(*grid)
    torch.manual_seed(42)
    original = ToyAnchor()
    expected, expected_residual = original(layers[-1])
    current = AnchoredConditionalNeck(deepcopy(original), arm, channels=8, dim=4, rank=2)
    output, _, residual = current(layers)
    assert torch.equal(output, expected)
    assert torch.equal(residual, expected_residual)
    assert bool(residual.any())  # Not a trivial all-zero anchor test.
    assert not any(p.requires_grad for p in current.aux.parameters())


def test_bc_have_identical_parameter_names_initialization_and_capacity():
    b, c = neck('b_last_condition'), neck('c_cross_condition')
    assert named_hash(dict(b.named_parameters())) == named_hash(dict(c.named_parameters()))
    assert {n: p.shape for n, p in b.named_parameters()} == {n: p.shape for n, p in c.named_parameters()}
    assert sum(p.numel() for p in b.parameters()) == sum(p.numel() for p in c.parameters())
    assert neck('a_continue').complement is None


def activate_complement(value):
    with torch.no_grad():
        for branch in value.complement.branches:
            branch.up.weight.fill_(.2)


@pytest.mark.parametrize('arm,changes', [('a_continue', False), ('b_last_condition', False),
                                         ('c_cross_condition', True)])
def test_only_c_uses_true_intermediate_layers(arm, changes):
    value = neck(arm)
    if value.complement is not None:
        activate_complement(value)
    layers = features()
    altered = (layers[0] + torch.sin(layers[0]) * 2,
               layers[1].flip(-1), layers[2])
    assert (not torch.equal(value(layers)[0], value(altered)[0])) is changes


def test_dynamic_gates_respond_to_image_content():
    module = neck('c_cross_condition').complement
    layers = features()
    _, first = module(layers)
    _, second = module((layers[0].square(), layers[1], layers[2]))
    assert not torch.equal(first['branch0_channel_gate_mean'], second['branch0_channel_gate_mean'])
    assert not torch.equal(first['branch0_spatial_gate_mean'], second['branch0_spatial_gate_mean'])
    assert torch.all(first['branch0_spatial_gate_std'] > 0)


@pytest.mark.parametrize('arm', ARMS)
def test_relative_cap_and_zero_energy_reference(arm):
    value = neck(arm)
    if value.complement is not None:
        activate_complement(value)
    layers = features()
    _, _, delta = value(layers)
    assert torch.all(delta.flatten(1).norm(dim=1) <= .025 * layers[-1].flatten(1).norm(dim=1) + 1e-6)
    zeros = (*layers[:2], torch.zeros_like(layers[-1]))
    out, _, delta = value(zeros)
    assert torch.count_nonzero(out) == 0 and torch.count_nonzero(delta) == 0


@pytest.mark.parametrize('arm', ARMS[1:])
def test_zero_up_does_not_permanently_block_conditional_gradients(arm):
    value = neck(arm)
    parameters = [p for p in value.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=.01, weight_decay=0.)
    layers = features()
    seen = {k: False for k in conditional_gradient_status(value)}
    target = torch.cos(layers[-1])
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss = (value(layers)[0] - target).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        for key, yes in conditional_gradient_status(value).items():
            seen[key] |= yes
        optimizer.step()
    assert all(seen.values()), seen
    assert not any(p.grad is not None for p in value.aux.parameters())


def test_state_roundtrip_includes_both_anchor_and_conditioning(tmp_path):
    value = neck('c_cross_condition')
    activate_complement(value)
    path = tmp_path / 'state.pt'
    torch.save(value.state_dict(), path)
    restored = neck('c_cross_condition')
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert torch.equal(value(features())[0], restored(features())[0])


@pytest.mark.parametrize('dims', [(0, 4, 2), (8, 4, 5), (8, -1, 2)])
def test_invalid_bottleneck_sizes_rejected(dims):
    with pytest.raises(ValueError):
        ConditionalComplement(*dims)


def test_invalid_features_arm_and_anchor_cap_rejected():
    with pytest.raises(ValueError):
        neck('unknown')
    value = neck('c_cross_condition')
    with pytest.raises(ValueError):
        value(features()[:2])
    with pytest.raises(ValueError):
        value((features()[0], features()[1][..., :2], features()[2]))
    with pytest.raises(ValueError):
        value.complement(tuple(torch.randn(2, 7, 3, 4) for _ in range(3)))
    original = ToyAnchor()
    original.output_scale = 1.
    with pytest.raises(ValueError):
        AnchoredConditionalNeck(original, ARMS[0])


def outcome_report(a, b, c, official, warm):
    return compare_outcomes(dict(zip(ARMS, (a, b, c))), official, warm)


def test_protocol_requires_all_three_boolean_paired_arms():
    with pytest.raises(ValueError):
        compare_outcomes({}, [True], [True])
    with pytest.raises(ValueError):
        outcome_report([1], [True], [True], [True], [True])
    with pytest.raises(ValueError):
        outcome_report([True], [True], [True, False], [True], [True])
    with pytest.raises(ValueError):
        outcome_report([], [], [], [], [])


def test_pairing_distinguishes_warm_continuation_capacity_and_true_depth():
    r = outcome_report([True, False, False], [True, True, False], [True, True, True],
                       [True, False, False], [False, False, False])
    assert r['b_vs_a']['net'] == 1 and r['c_vs_a']['net'] == 2 and r['c_vs_b']['net'] == 1
    assert r['c_cross_condition']['vs_warm_start']['net'] == 3
    assert r['c_vs_b']['corrections'] == [2]


def test_candidate_gate_is_two_dataset_primary_and_never_stable_gain():
    r = outcome_report([True, False], [True, False], [True, True],
                       [True, False], [True, False])
    reports = {f'{ds}_{size}': deepcopy(r) for size in (280, 322)
               for ds in ('msls-val', 'pitts30k-val')}
    result = candidate_verdict(reports)
    assert result['primary_cross_depth_candidate'] and not result['stable_gain_proven']
    reports['msls-val_322']['c_vs_b']['net'] = 0
    assert not candidate_verdict(reports)['primary_cross_depth_candidate']
    with pytest.raises(ValueError):
        candidate_verdict({'msls-val_322': r})


def test_fixed_contract_never_drops_other_differences():
    first = dict(arm=ARMS[0], policy=POLICY, schedule=[1, 2])
    second = dict(first, arm=ARMS[1])
    assert matched_contract(first) == matched_contract(second)
    second['schedule'] = [2, 1]
    assert matched_contract(first) != matched_contract(second)
    assert POLICY['steps'] == 2048 and POLICY['smoke_steps'] == 8
    assert POLICY['primary_eval_size'] == 322 and POLICY['eval_sizes'] == [280, 322]


def test_runner_imports_and_checkpointed_forward_collects_each_image_once(monkeypatch):
    from types import SimpleNamespace
    from torch.nn import functional as F
    from scripts.conditional_complement import training_forward
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *args, **kwargs: self)
    value = neck('c_cross_condition')
    activate_complement(value)
    def aggregate(a, b, c, auxiliary=False):
        fused, _, delta = value((a, b, c), auxiliary)
        flat = fused.mean((2, 3)).repeat(1, 1536)
        return F.normalize(flat, dim=1), delta
    model = SimpleNamespace(neck=value, features=lambda x: (x * .5, x * .8, x),
                            aggregate=aggregate)
    images = torch.randn(7, 8, 3, 4)
    descriptor, delta, stats = training_forward(model, images)
    assert descriptor.shape == (7, 12288) and delta.shape == (7,)
    assert stats['addition_rms'] > 0
    assert value.last_raw is None and value.last_conditional_stats is None
    # This exercises backward recomputation after the transient stats are reset.
    descriptor[:, 0].sum().backward()
    assert value.complement.branches[0].spatial_gate.weight.grad is not None


def test_atomic_checkpoint_restores_adam_and_exact_next_update(tmp_path, monkeypatch):
    from scripts.train_slgd import save, restore, parameter_hash
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda _: None)
    def setup():
        model = nn.Module()
        model.neck = neck('c_cross_condition')
        original = [p for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith('neck.complement.')]
        added = [p for n, p in model.named_parameters() if n.startswith('neck.complement.')]
        optimizer = torch.optim.AdamW([
            dict(params=original, lr=1e-5), dict(params=added, lr=1e-4)], weight_decay=0.)
        return model, optimizer
    def update(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        values = features()
        (model.neck(values)[0] - values[-1].sin()).square().mean().backward()
        optimizer.step()
    model, optimizer = setup()
    update(model, optimizer)
    payload = dict(steps=1, contract_sha256='test-contract')
    path = tmp_path / 'last.pt'
    save(path, model, optimizer, payload)
    update(model, optimizer)
    expected = parameter_hash(model)
    restored, other_optimizer = setup()
    state = restore(path, restored, other_optimizer, 'test-contract')
    assert state == payload
    update(restored, other_optimizer)
    assert parameter_hash(restored) == expected
    assert all(int(s['step']) == 2 for s in other_optimizer.state.values())
    with pytest.raises(ValueError):
        restore(path, restored, other_optimizer, 'wrong-contract')
