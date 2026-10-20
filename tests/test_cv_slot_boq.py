"""Small CPU unit tests; no GSV data, GPU jobs, or training loop is started."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cv_slot_protocol import correspondence
from src.models.cv_slot_boq import (ARMS, CVSlotBoQ, ChannelAdapter, transported_js,
    transport_matrix, correspondence_loss, slot_gram_loss, replay_backward)

spec = importlib.util.spec_from_file_location('_cv_slot_tiny_official', ROOT/'tests/test_clip_key_boq.py')
tiny = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tiny)


def example():
    torch.manual_seed(42)
    m = CVSlotBoQ(tiny.Official(), hidden=4, grid=2)
    return m, torch.randn(4, 8, 4, 4)


def test_zero_start_and_baseline_match_official():
    m, features = example()
    m.train()
    with torch.no_grad():
        reference = m.visual.aggregator(features)[0]
        z, slots = m.describe(features)
        z0, s0 = m.describe(features, baseline=True)
    assert torch.allclose(z, reference, atol=1e-7)
    assert torch.equal(z, z0) and torch.equal(slots, s0)
    assert slots.shape == (4, 2, 3, 4)
    assert torch.allclose(slots.sum(-1), torch.ones(4, 2, 3), atol=1e-6)


def test_only_new_weights_train_and_visual_stays_eval():
    m, _ = example()
    m.train()
    assert not m.visual.training
    assert all(n.startswith(('adapter.', 'routers.')) for n, p in m.named_parameters() if p.requires_grad)
    assert all(not p.requires_grad for p in m.visual.parameters())


def test_inference_has_no_clip_or_paired_inputs():
    m, _ = example()
    with torch.no_grad():
        z = m(torch.randn(2, 3, 4, 4))
    assert z.shape == (2, 32)
    assert torch.allclose(z.norm(dim=-1), torch.ones(2), atol=1e-6)
    assert not any('clip' in n or 'teacher' in n for n in m.state_dict())
    with pytest.raises(TypeError):
        m(torch.randn(2, 3, 4, 4), torch.ones(2))


def test_zero_gates_receive_gradient_without_visual_updates():
    m, features = example()
    old = copy.deepcopy(m.visual.state_dict())
    z, slots = m.describe(features)
    loss = (z*torch.randn_like(z)).sum()+(slots*torch.randn_like(slots)).sum()
    loss.backward()
    assert m.adapter.up.weight.grad.abs().sum() > 0
    assert all(r.beta.grad.abs().sum() > 0 for r in m.routers)
    assert all(p.grad is None for p in m.visual.parameters())
    assert all(torch.equal(v, old[n]) for n, v in m.visual.state_dict().items())


@pytest.mark.parametrize('nonzero', [False, True])
def test_microbatch_replay_matches_direct_chain_rule(nonzero):
    m, features = example()
    if nonzero:
        with torch.no_grad():
            m.adapter.up.weight.normal_(std=.01)
            for r in m.routers: r.beta.fill_(.1)
    other = copy.deepcopy(m)
    z, slots = m.describe(features)
    gz, gs = torch.randn_like(z), torch.randn_like(slots)
    torch.autograd.backward((z, slots), (gz, gs))
    replay_backward(other, features, gz, gs, microbatch=2)
    a = dict(m.named_parameters())
    for n, p in other.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and a[n].grad is not None
            assert torch.allclose(p.grad, a[n].grad, atol=5e-6, rtol=5e-4), n


def test_channel_adapter_does_not_mix_neighbor_pixels():
    torch.manual_seed(42)
    a = ChannelAdapter(8, hidden=4)
    with torch.no_grad(): a.up.weight.normal_()
    x = torch.randn(1, 8, 4, 4)
    y = x.clone()
    y[:, :, 0, 0] += 1
    change = (a(x)-a(y)).abs().sum(1)
    assert change[0, 0, 0] > 0
    change[0, 0, 0] = 0
    assert torch.count_nonzero(change) == 0


def test_transport_support_and_reliability_are_dino_only():
    torch.manual_seed(42)
    d, c = torch.randn(4, 8), torch.randn(4, 6)
    g = correspondence(d, d.roll(1, 0), c, c.roll(1, 0), topk=4)
    matrices = {arm: transport_matrix(g, arm, .25) for arm in ('dino', 'clip', 'shuffled')}
    for matrix in matrices.values():
        assert not matrix.requires_grad
        assert torch.allclose(matrix.sum(-1), matrices['dino'].sum(-1), atol=1e-6)
        assert torch.equal(matrix > 0, matrices['dino'] > 0)


def test_perfect_geometric_transport_has_zero_loss():
    a = torch.tensor([[[.1, .2, .3, .4], [.4, .3, .2, .1]]], requires_grad=True)
    matrix = torch.eye(4).roll(1, dims=1)
    b = (a.detach() @ matrix).requires_grad_()
    value = transported_js(a, b, matrix)
    assert abs(float(value)) < 1e-7
    value.backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()


def test_transport_loss_trains_both_views_and_empty_abstains():
    a = torch.tensor([[[.1, .2, .3, .4]]], requires_grad=True)
    b = torch.tensor([[[.4, .3, .2, .1]]], requires_grad=True)
    transported_js(a, b, torch.eye(4)).backward()
    assert a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0
    a.grad.zero_()
    b.grad.zero_()
    zero = transported_js(a, b, torch.zeros(4, 4))
    zero.backward()
    assert float(zero) == 0 and a.grad.abs().sum() == b.grad.abs().sum() == 0


def test_slot_gram_detaches_reference_and_penalizes_collapse():
    reference = torch.eye(4)[None, None].requires_grad_()
    slots = torch.ones_like(reference).div(4).requires_grad_()
    loss = slot_gram_loss(slots, reference)
    assert float(loss) > .5
    loss.backward()
    assert reference.grad is None and slots.grad is not None
    assert float(slot_gram_loss(reference, reference)) == 0


def test_no_correspondence_is_connected_zero():
    slots = torch.rand(4, 2, 3, 4, requires_grad=True)
    value = correspondence_loss(slots, [], 'no_correspondence')
    value.backward()
    assert float(value) == 0 and slots.grad.abs().sum() == 0


def test_student_checkpoint_does_not_contain_official_weights():
    m, _ = example()
    state = copy.deepcopy(m.student_state())
    assert set(state) == {'adapter', 'routers'}
    with torch.no_grad(): m.adapter.up.bias.fill_(.1)
    m.load_student(state)
    assert torch.equal(m.adapter.up.bias, state['adapter']['up.bias'])
    with pytest.raises(ValueError): m.load_student(dict(state, visual={}))


def test_four_arm_initial_states_are_equal_but_not_shared():
    torch.manual_seed(42)
    official = tiny.Official()
    models = []
    for _ in ARMS:
        torch.manual_seed(42)
        models.append(CVSlotBoQ(official, hidden=4, grid=2))
    assert all(m.visual is official for m in models)
    for m in models[1:]:
        assert all(torch.equal(p, q) for p, q in zip(m.adapter.parameters(), models[0].adapter.parameters()))
        assert all(torch.equal(p, q) for p, q in zip(m.routers.parameters(), models[0].routers.parameters()))
        assert not {id(p) for p in m.parameters() if p.requires_grad} & {
            id(p) for p in models[0].parameters() if p.requires_grad}


def test_training_needs_explicit_authorization_before_loading_data():
    from scripts.cv_slot_study import train
    from types import SimpleNamespace
    with pytest.raises(ValueError, match='allow-training'):
        train({}, SimpleNamespace(allow_training=False))


def test_failed_teacher_blocks_training_before_cuda_or_output(tmp_path, monkeypatch):
    from scripts import cv_slot_study as study
    from src.semantic_region_protocol import write
    c = dict(teacher_audit=str(tmp_path), plan=str(tmp_path))
    write(tmp_path/'summary.json', dict(verdict='STOP_TEACHER_NOT_SUPPORTED',
        partitions=dict(holdout=dict(checks=dict(enough=True, clip_gain=False)))))
    monkeypatch.setattr(study.audit, 'verify', lambda _: dict(code=study.key.source_hashes(study.audit.SOURCES)))
    with pytest.raises(ValueError, match='Teacher gate NOT PASS'):
        study.require_teacher(c)


def test_training_cli_refuses_without_allow_flag():
    result = subprocess.run([sys.executable, str(ROOT/'scripts/cv_slot_study.py'), 'train'],
                            capture_output=True, text=True)
    assert result.returncode != 0 and 'Need --allow-training' in result.stderr


def test_study_configuration_and_image_keyed_teacher_graphs():
    from scripts.cv_slot_study import configuration, make_graphs
    c = configuration(ROOT/'config/cv_slot_boq.yaml')
    c.update(grid=2, topk=4)
    torch.manual_seed(42)
    features = torch.randn(4, 8, 4, 4)
    keys = torch.randn(4, 6, 2, 2)
    ids = torch.tensor([20, 21, 22, 23])
    graphs = make_graphs(features, keys, ids, c)
    assert [(a, b) for a, b, _, _ in graphs] == [(0, 1), (2, 3)]
    again = make_graphs(features, keys, ids, c)
    assert torch.equal(graphs[0][2]['shuffled_weights'], again[0][2]['shuffled_weights'])
    with pytest.raises(ValueError): make_graphs(features[:3], keys[:3], ids[:3], c)
