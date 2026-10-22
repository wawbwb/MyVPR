import pytest
import torch
from torch.nn import functional as F
from src.models.relation_metric import RelationMetric,calibrated,relation_basis,whitening,orthogonal_check


def setup():
    torch.manual_seed(42)
    u=torch.linalg.qr(torch.randn(24,8),mode='reduced')[0]
    z=F.normalize(torch.randn(16,24),dim=1)
    return u,z


@pytest.mark.parametrize('static',[False,True])
def test_zero_exact_and_all_parameters_train(static):
    u,z=setup();m=RelationMetric(u,static,hidden=6);o=torch.optim.AdamW(m.parameters(),lr=.01)
    assert torch.equal(m(z),z)
    gradients={k:False for k,p in m.named_parameters()}
    for _ in range(4):
        x=m(z);loss=(x[0]*x[2]).sum()-(x[0]*x[1]).sum()
        o.zero_grad();loss.backward()
        for k,p in m.named_parameters():gradients[k]|=p.grad is not None and bool((p.grad!=0).any())
        o.step()
    assert all(gradients.values())
    assert not torch.equal(m(z),z)
    assert m.basis.grad is None


@pytest.mark.parametrize('static',[False,True])
def test_bound_unit_and_batch_independence(static):
    u,z=setup();m=RelationMetric(u,static,hidden=6)
    with torch.no_grad():
        for p in m.parameters():p.uniform_(-20,20)
    out,delta,a=m(z,True)
    assert delta.norm(dim=1).max()<=.050001
    assert torch.allclose(out.norm(dim=1),torch.ones(len(z)),atol=1e-6,rtol=0)
    torch.testing.assert_close(out,torch.cat([m(t[None]) for t in z]),atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(delta-delta@u@u.T,torch.zeros_like(delta),atol=1e-6,rtol=0)


def test_pair_both_end_gradients():
    u,z=setup();m=RelationMetric(u,hidden=6);x=m(z);x.retain_grad()
    loss=F.softplus(((x[:8]*x[8:]).sum(1)-(x[:8]*x[:8].roll(1,0)).sum(1)+.05)/.05).mean()
    loss.backward();assert x.grad[:8].norm()>0 and x.grad[8:].norm()>0


def test_covariance_and_whitening_formula():
    torch.manual_seed(1);z=torch.randn(20,12);p=torch.eye(12);neg=(torch.arange(20)+4)%20
    u,w,b,e=relation_basis(z,p,neg,rank=5)
    expected=[]
    for v in z.reshape(-1,4,12):
        for i in range(4):
            for j in range(i):expected.append((v[i]-v[j]).double())
    diff=torch.stack(expected)
    torch.testing.assert_close(w,diff.T@diff/len(diff),atol=1e-6,rtol=1e-6)
    orthogonal_check(u);wu,wa=whitening(p,w,rank=5);orthogonal_check(wu)
    assert wa.abs().max()<=.05
    normalized=F.normalize(z,dim=1);out,delta=calibrated(normalized,wu,wa)
    assert delta.norm(dim=1).max()<=.050001
    assert torch.isfinite(out).all()


def test_reject_same_place_and_nonorthogonal():
    with pytest.raises(ValueError,match='Same-place'):relation_basis(torch.randn(8,4),torch.eye(4),torch.arange(8),rank=2)
    with pytest.raises(ValueError,match='Nonorthogonal'):RelationMetric(torch.ones(24,8))


def test_saved_int32_pair_indices_match_int64():
    torch.manual_seed(7);z=torch.randn(20,12);p=torch.eye(12)
    negative=(torch.arange(20)+4)%20
    a=relation_basis(z,p,negative,rank=5)
    b=relation_basis(z,p,negative.to(torch.int32),rank=5)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
    with pytest.raises(ValueError,match='Bad negative'):
        relation_basis(z,p,negative.float(),rank=5)


def test_whitening_zero_variation_finite():
    u,a=whitening(torch.eye(8),torch.zeros(8,8,dtype=torch.float64),rank=4)
    assert torch.isfinite(a).all();assert torch.allclose(a,torch.zeros_like(a),atol=1e-7)


def test_verdict_requires_anchor_and_controls():
    from scripts.relation_metric import verdict
    names=('innovation1','random_basis','pca_basis','relation_static','relation_conditional','bounded_whitening')
    r={tag:{'correct':dict.fromkeys(names,n)} for tag,n in [('pitts30k-val',7276),('msls-val',695),('gsv-adaptation-holdout',3049)]}
    assert not verdict(r)['anchor_gain']
    r['pitts30k-val']['correct']['relation_conditional']=7277
    assert verdict(r)['matched_mechanism_signal']
    r['pitts30k-val']['correct']['bounded_whitening']=7277
    assert verdict(r)['anchor_gain'] and not verdict(r)['matched_mechanism_signal']
    r['msls-val']['correct']['relation_conditional']=694
    assert not verdict(r)['anchor_gain']


def test_optimizer_checkpoint_resume(tmp_path):
    u,z=setup();m=RelationMetric(u,hidden=6);o=torch.optim.AdamW(m.parameters(),lr=.001)
    def step(model,opt):
        out=model(z);loss=(out[0]*out[1]).sum();opt.zero_grad();loss.backward();opt.step()
    step(m,o);path=tmp_path/'x.pt';torch.save({'model':m.state_dict(),'optimizer':o.state_dict()},path)
    copy=RelationMetric(u,hidden=6);opt=torch.optim.AdamW(copy.parameters(),lr=.001)
    saved=torch.load(path,weights_only=True);copy.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
    step(m,o);step(copy,opt)
    for a,b in zip(m.parameters(),copy.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
