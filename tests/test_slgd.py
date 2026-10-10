"""CPU unit tests; run on the training machine, not the development PC."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.models.slgd import (LocalProjector, SLGDStudent, compression_loss,
                             local_objective, local_tokens, mine_pairs, mutual_score)
from scripts.train_slgd import POLICY, teacher_checks, parameter_hash, write, verify, seal


class FakeBackbone(nn.Module):
    crop_semantic_film = None
    residual_clip_fusion = None

    def forward(self, images):
        return images


class FakeAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 6)

    def forward(self, features):
        return F.normalize(self.linear(features.mean((2, 3))), dim=-1), None


def student():
    visual = nn.Module()
    visual.backbone = FakeBackbone()
    visual.aggregator = FakeAggregator()
    visual.semantic_region_gate = None
    return SLGDStudent(visual, LocalProjector(8, 128), slots=4, beta=.2)


def test_mutual_identity_and_gradients():
    left = torch.eye(4)[None].requires_grad_()
    score, count = mutual_score(left, left.detach())
    assert count.tolist() == [4]
    torch.testing.assert_close(score, torch.ones(1))
    score.sum().backward()
    assert left.grad.abs().sum() > 0


def test_mutual_matching_is_token_permutation_invariant():
    left = torch.eye(4)[None]
    torch.testing.assert_close(mutual_score(left, left[:, [2, 0, 3, 1]])[0], torch.ones(1))


def test_mutual_rejects_invalid_shapes():
    with pytest.raises(ValueError):
        mutual_score(torch.ones(2, 4), torch.ones(2, 4))
    with pytest.raises(ValueError):
        mutual_score(torch.ones(2, 0, 4), torch.ones(2, 3, 4))


def test_mining_never_uses_self_or_positive_as_negative():
    descriptors = F.normalize(torch.randn(8, 12), dim=-1)
    labels = torch.arange(8)//2
    pos, neg = mine_pairs(descriptors, labels)
    assert (pos != torch.arange(8)).all()
    assert (labels[pos] == labels).all()
    assert (labels[neg] != labels).all()
    with pytest.raises(ValueError):
        mine_pairs(descriptors, torch.arange(8))


def test_invalid_tied_matches_abstain_without_nan():
    tokens = torch.zeros(4, 10, 8, requires_grad=True)
    loss, margin, valid = local_objective(tokens, torch.tensor([1, 0, 3, 2]), torch.tensor([2, 3, 0, 1]))
    assert not valid.any()
    assert float(loss) == 0 and torch.isfinite(margin).all()
    loss.backward()
    assert tokens.grad is not None


def test_compression_detaches_teacher_and_updates_student():
    d = F.normalize(torch.randn(4, 12), dim=-1).detach().requires_grad_()
    margin = torch.full((4,), .1, requires_grad=True)
    loss = compression_loss(d, torch.tensor([1, 0, 3, 2]), torch.tensor([2, 3, 0, 1]), margin, torch.ones(4, dtype=torch.bool))
    assert float(loss) >= -1e-6
    loss.backward()
    assert margin.grad is None
    assert d.grad.abs().sum() > 0


def test_compression_zero_coverage_is_differentiable_zero():
    d = torch.randn(4, 12, requires_grad=True)
    loss = compression_loss(d, torch.tensor([1, 0, 3, 2]), torch.tensor([2, 3, 0, 1]), torch.ones(4), torch.zeros(4, dtype=torch.bool))
    assert float(loss) == 0
    loss.backward()
    assert torch.equal(d.grad, torch.zeros_like(d))


def test_fixed_vector_norm_and_weighted_dot_product():
    torch.manual_seed(42)
    model = student().eval()
    descriptor, tokens = model.aggregate(torch.randn(2, 8, 4, 4))
    assert descriptor.shape == (2, 6+4*128)
    assert tokens.shape == (2, 100, 128)
    torch.testing.assert_close(descriptor.norm(dim=1), torch.ones(2), atol=1e-6, rtol=1e-6)
    g = descriptor[:, :6]/(.8**.5)
    local = descriptor[:, 6:]/(.2**.5)
    torch.testing.assert_close(descriptor@descriptor.T, .8*(g@g.T)+.2*(local@local.T))


def test_slot_descriptor_does_not_use_pair_matching(monkeypatch):
    import src.models.slgd as module
    def forbidden(*args, **kwargs):
        raise AssertionError('Matching must not run at inference')
    monkeypatch.setattr(module, 'mutual_score', forbidden)
    descriptor = student().eval()(torch.randn(2, 8, 4, 4))
    assert torch.isfinite(descriptor).all()


def test_student_all_branches_receive_retrieval_gradients():
    model = student().train()
    descriptor, _ = model.aggregate(torch.randn(3, 8, 4, 4))
    (descriptor*torch.randn_like(descriptor)).sum().backward()
    for group in (model.visual.aggregator, model.local, model.slot_score):
        assert sum(float(p.grad.abs().sum()) for p in group.parameters() if p.grad is not None) > 0
    assert model.dustbin.grad is not None
    assert not model.visual.backbone.training


def test_equal_seed_gives_equal_initial_student():
    torch.manual_seed(42)
    a = student()
    torch.manual_seed(42)
    b = student()
    assert parameter_hash(a) == parameter_hash(b)
    x = torch.randn(2, 8, 4, 4)
    torch.testing.assert_close(a(x), b(x), atol=0, rtol=0)


@pytest.mark.parametrize('beta', [0., 1., -1.])
def test_invalid_beta_rejected(beta):
    m = student()
    with pytest.raises(ValueError):
        SLGDStudent(m.visual, LocalProjector(8, 128), slots=4, beta=beta)


def test_ot_requires_more_tokens_than_slots():
    with pytest.raises(ValueError):
        student().aggregate(torch.randn(2, 8, 2, 2))


def test_teacher_gate_requires_advantage_over_raw_not_only_learning():
    report = dict(valid_fraction=1., pair_accuracy=.8, raw_pair_accuracy=.82, mean_margin=.1)
    assert not teacher_checks(report)['raw_gain']
    report['raw_pair_accuracy'] = .7
    assert all(teacher_checks(report).values())


def test_completed_outputs_detect_tampering(tmp_path):
    write(tmp_path/'contract.json', dict(policy=POLICY))
    seal(tmp_path)
    assert verify(tmp_path)['policy'] == POLICY
    write(tmp_path/'contract.json', dict(policy='changed'))
    with pytest.raises(ValueError):
        verify(tmp_path)
