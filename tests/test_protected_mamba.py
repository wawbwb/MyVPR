"""Run only on training machine, including frozen-BoQ input-gradient checks."""
import ast
from pathlib import Path
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from src.models.protected_mamba import bounded_tangent, preservation_loss, ProtectedMambaVPR


def test_tangent_bound_and_zero_start():
    torch.manual_seed(3)
    ref = F.normalize(torch.randn(20, 16), dim=-1)
    actual, delta = bounded_tangent(ref, ref)
    torch.testing.assert_close(actual, ref)
    assert torch.count_nonzero(delta) == 0
    proposed = torch.randn_like(ref).requires_grad_()
    d, delta = bounded_tangent(ref, proposed)
    assert delta.norm(dim=-1).max() <= .1
    torch.testing.assert_close((delta*ref).sum(-1), torch.zeros(20), atol=2e-7, rtol=0)
    assert (d*ref).sum(-1).min() >= (1+.1**2)**-.5-2e-7
    d.square().sum(dim=0)[0].backward()
    assert torch.isfinite(proposed.grad).all() and proposed.grad.abs().sum() > 0


def test_preservation_zero_and_teacher_detached():
    ref = F.normalize(torch.randn(8, 12), dim=-1).detach().requires_grad_()
    x = ref.detach().clone().requires_grad_()
    kl, drift = preservation_loss(x, ref)
    assert abs(float(kl)) < 1e-7 and drift == 0
    x = F.normalize(x+.1*torch.randn_like(x), dim=-1)
    kl, drift = preservation_loss(x, ref)
    assert kl > 0 and drift > 0
    (kl+.1*drift).backward()
    assert ref.grad is None


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(12, 12)

    def forward(self, x):
        y = torch.tanh(self.linear(x))
        return y, y[:, :3]+y.mean(1, keepdim=True), None


class ToyBase(nn.Module):
    def __init__(self, visual, mode):
        super().__init__()
        self.aggregator = nn.Module()
        self.aggregator.proj_c = nn.Conv2d(12, 12, 1)
        self.aggregator.norm_input = nn.LayerNorm(12)
        self.aggregator.boqs = nn.ModuleList([ToyBlock()])
        self.aggregator.fc = nn.Linear(3, 2)


@pytest.mark.parametrize('mode', ['conv_preserved', 'mamba_plain', 'mamba_preserved'])
def test_frozen_ru_and_live_mixer_input_gradient(monkeypatch, mode):
    import src.models.depth_query as depth
    monkeypatch.setattr(depth, 'DepthQueryVPR', ToyBase)
    torch.manual_seed(42)
    model = ProtectedMambaVPR(None, mode)
    assert all(not p.requires_grad for p in model.base.parameters())
    model.train()
    assert not model.base.training and not model.base.aggregator.training
    saved = {k: v.clone() for k, v in model.base.state_dict().items()}
    x = torch.randn(4, 12, 3, 5)
    d, ref, delta = model.aggregate(x)
    torch.testing.assert_close(d, ref, atol=2e-7, rtol=1e-6)
    opt = torch.optim.AdamW(model.mixer.parameters(), lr=.001, weight_decay=0.)
    for _ in range(3):
        opt.zero_grad()
        d, ref, delta = model.aggregate(x)
        loss = (d-torch.ones_like(d)*.3).square().sum()
        loss.backward(); opt.step()
    first = model.mixer.input.weight if mode.startswith('conv') else model.mixer.ssm.select.weight
    assert first.grad.abs().sum() > 0 and torch.isfinite(first.grad).all()
    assert all(p.grad is None for p in model.base.parameters())
    assert all(torch.equal(v, saved[k]) for k, v in model.base.state_dict().items())
    d, ref, _ = model.aggregate(x)
    assert torch.equal(model.aggregate(x, bypass=True)[0], ref)
    assert (d-ref).norm() > 0


def test_mamba_objective_arms_identical_initialization(monkeypatch):
    import src.models.depth_query as depth
    monkeypatch.setattr(depth, 'DepthQueryVPR', ToyBase)
    torch.manual_seed(42); a = ProtectedMambaVPR(None, 'mamba_plain')
    torch.manual_seed(42); b = ProtectedMambaVPR(None, 'mamba_preserved')
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def test_trainer_protocol_static():
    source = Path('scripts/train_protected_mamba.py').read_text()
    tree = ast.parse(source)
    assert 'lr=1e-4' in source and 'full_epochs=3' in source
    assert 'Only mixer parameters may train' in source
    assert 'best_epoch' not in source
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'fixed_step'
                           for child in node.body for n in ast.walk(child)) or 'state' not in ast.unparse(node.test)
