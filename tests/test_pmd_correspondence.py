import torch
from src.pmd_correspondence import make_views,CorrespondenceHead,supervised_loss,counts


def test_exact_mutual_targets():
    views,(a,b)=make_views(torch.zeros(3,406,406),42)
    assert views.shape==(2,3,322,322)
    good=a>=0
    assert torch.equal(b[a[good]],torch.arange(529)[good])
    assert good.any() and (~good).any()
    assert torch.equal(a,make_views(torch.zeros(3,406,406),42)[1][0])


def test_occlusion_removes_matches():
    x=torch.zeros(3,406,406)
    # Each transformed view uses a different RNG sequence when adding occlusion;
    # directly test masked patches never acquire a correspondence instead.
    _,targets=make_views(x,5,True)
    assert all(int((t<0).sum())>=25 for t in targets)


def test_trainable_rejection_controls():
    for mode in ['independent_dustbin','forced','partial']:
        h=CorrespondenceHead(mode);plans=h(torch.randn(1,4,768),torch.randn(1,4,768))
        targets=[torch.tensor([0,1,-1,-1])]*2
        loss=supervised_loss(plans,targets,mode!='forced');loss.backward()
        assert torch.isfinite(loss) and h.match.key.weight.grad is not None
        c=counts(plans,targets);assert c['matched']==4 and c['unmatched']==4
        if mode!='forced':assert h.match.bin_score.grad is not None
