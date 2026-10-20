"""Training-machine CPU tests, no weights/download/GPU/private PDF required."""
import copy
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.models.hetero_interleave import (ARMS, HeterogeneousInteraction,
    LocalGlobalRefinement, interleave, uninterleave, proxy_channels, replay_backward)


def maps(batch=2):
    torch.manual_seed(17)
    return tuple(torch.randn(batch, 12, 2, 3) for _ in range(3)), tuple(
        torch.randn(batch, c, 2, 3) for c in (3, 5, 7))


def head(arm):
    torch.manual_seed(42)
    return HeterogeneousInteraction(arm, channels=12, cnn_channels=(3, 5, 7), dim=8, rank=4)


def test_same_position_interleaving_exact_roundtrip():
    a = torch.arange(12.).reshape(1, 3, 4)
    b = a+100
    t = interleave(a, b)
    assert t.shape == (1, 6, 4)
    torch.testing.assert_close(t[:, :2], torch.stack((a[:, 0], b[:, 0]), 1))
    aa, bb = uninterleave(t)
    torch.testing.assert_close(aa, a); torch.testing.assert_close(bb, b)


def test_interleaving_rejects_incompatible_streams():
    with pytest.raises(ValueError): interleave(torch.zeros(2, 3, 4), torch.zeros(2, 2, 4))
    with pytest.raises(ValueError): uninterleave(torch.zeros(2, 3, 4))


@pytest.mark.parametrize('arm', ARMS)
def test_zero_output_preserves_frozen_anchor(arm):
    a, b = maps()
    out, aux, stats = head(arm)(a, b)
    assert torch.equal(out, a[-1])
    assert torch.equal(stats['relative_residual'], torch.zeros(2))
    for z in aux: torch.testing.assert_close(z.norm(dim=1), torch.ones(2))


def test_all_three_arms_equal_parameters_and_identical_initial_values():
    models = [head(arm) for arm in ARMS]
    state = models[0].state_dict()
    assert len({sum(p.numel() for p in m.parameters()) for m in models}) == 1
    for m in models[1:]:
        assert state.keys() == m.state_dict().keys()
        assert all(torch.equal(v, m.state_dict()[k]) for k, v in state.items())
    assert not set(map(id, models[0].parameters())) & set(map(id, models[1].parameters()))


@pytest.mark.parametrize('arm', ARMS)
def test_all_new_parameters_receive_finite_gradients_after_updates(arm):
    a, b = maps()
    m = head(arm)
    opt = torch.optim.AdamW(m.parameters(), lr=.001)
    seen = {k: False for k, p in m.named_parameters()}
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        out, aux, _ = m(a, b)
        loss = (out*torch.linspace(-1, 1, out.numel()).reshape_as(out)).mean()+sum(z.square().mean()+z[:, 0].mean() for z in aux)
        loss.backward()
        for k, p in m.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), k
            seen[k] |= bool((p.grad != 0).any())
        opt.step()
    assert all(seen.values()), [k for k,v in seen.items() if not v]


def test_proxy_ignores_actual_cnn_while_heterogeneous_model_uses_it():
    a, b = maps()
    for arm in ('dino_proxy', 'hetero_interleaved'):
        m = head(arm)
        with torch.no_grad(): m.output.weight.normal_(std=.2)
        out = m(a, b)[0]
        changed = m(a, tuple(x.flip(-1)+3 for x in b))[0]
        if arm == 'dino_proxy': assert torch.equal(out, changed)
        else: assert not torch.allclose(out, changed)


def test_separate_and_interleaved_scan_differ_with_same_weights():
    a,b = maps(); x=head('separate_scan'); y=head('hetero_interleaved')
    with torch.no_grad():
        x.output.weight.normal_(std=.2)
        y.load_state_dict(x.state_dict())
    assert not torch.allclose(x(a,b)[0], y(a,b)[0], atol=1e-7, rtol=1e-7)


def test_refinement_matches_private_manuscript_equation_15():
    m=LocalGlobalRefinement(8,4)
    x=torch.randn(2,8,3,4)
    n=m.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
    gate=m.gate(n.mean((2,3)))[:,:,None,None]
    expected=x+m.local(n)*(gate*n)
    actual,w=m(x)
    torch.testing.assert_close(actual,expected);torch.testing.assert_close(w,gate)


def test_cap_holds_even_at_saturation_and_zero_anchor():
    a,b=maps();m=head('hetero_interleaved')
    with torch.no_grad(): m.output.weight.fill_(100)
    out,_,stats=m(a,b)
    assert float(stats['relative_residual'].max())<=.025+1e-6
    assert torch.isfinite(out).all()
    a=(*a[:2],torch.zeros_like(a[-1]))
    assert torch.equal(m(a,b)[0],a[-1])


def test_proxy_channel_geometry_and_gradients():
    x=torch.randn(2,12,3,4,requires_grad=True)
    z=proxy_channels(x,17)
    assert z.shape==(2,17,3,4)
    z.sum().backward();assert torch.isfinite(x.grad).all()


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.fusion=head('hetero_interleaved')

    def describe(self, features):
        x,aux,stats=self.fusion(features[:3],features[3:])
        return F.normalize(x.flatten(1),dim=1),*aux,stats


def test_gradient_cache_replay_matches_direct_full_batch():
    a,b=maps(4);features=(*a,*b)
    direct=Tiny();replay=copy.deepcopy(direct)
    # Cover real trained-output gradients, not only the zero-start projection.
    with torch.no_grad():
        direct.fusion.output.weight.normal_(std=.01)
        replay.load_state_dict(direct.state_dict())
    outputs=direct.describe(features)[:3]
    weights=[torch.randn_like(x) for x in outputs]
    sum((x*w).sum() for x,w in zip(outputs,weights)).backward()
    with torch.no_grad():
        leaves=[z.detach().requires_grad_() for z in replay.describe(features)[:3]]
    sum((x*w).sum() for x,w in zip(leaves,weights)).backward()
    replay_backward(replay,features,[z.grad for z in leaves],2)
    for (n,p),(k,q) in zip(direct.named_parameters(),replay.named_parameters()):
        assert n==k
        torch.testing.assert_close(p.grad,q.grad,rtol=5e-4,atol=2e-5,msg=n)


def test_adam_checkpoint_next_step_reproduces(tmp_path):
    m=head('hetero_interleaved');a,b=maps();opt=torch.optim.AdamW(m.parameters(),lr=.001)
    def step(model,optimizer):
        optimizer.zero_grad();out,aux,_=model(a,b)
        (out.square().mean()+sum(z[:,0].mean() for z in aux)).backward();optimizer.step()
    step(m,opt)
    target=tmp_path/'last.pt';torch.save(dict(model=m.state_dict(),opt=opt.state_dict()),target)
    saved=torch.load(target,weights_only=True)
    restored=head('hetero_interleaved');restored.load_state_dict(saved['model'])
    opt2=torch.optim.AdamW(restored.parameters(),lr=.001);opt2.load_state_dict(saved['opt'])
    step(m,opt);step(restored,opt2)
    for p,q in zip(m.parameters(),restored.parameters()): assert torch.equal(p,q)


def test_pilot_config_is_locked_and_protects_archive():
    from scripts.hetero_interleave_study import configuration
    c=configuration()
    assert c['pilot_steps']==1024 and c['student_size']==322 and c['cuda_memory_fraction']==.2
    assert c['anchor_run']=='logs/conditional_complement_v1/train_c_cross_condition'
    source=(Path(__file__).resolve().parents[1]/'scripts/hetero_interleave_study.py').read_text(encoding='utf8')
    assert 'Explicit --allow-training required' in source
    assert 'private_reference_sha256' in source
