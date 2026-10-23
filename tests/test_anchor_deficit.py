import torch
from src.models.anchor_deficit import Complement,fuse,required_margin,balanced_loss


def test_descriptor_score_and_bypass():
    z=torch.nn.functional.normalize(torch.randn(8,12),dim=1)
    h=torch.nn.functional.normalize(torch.randn(8,4),dim=1)
    d=fuse(z,h)
    torch.testing.assert_close(d@d.T,.9*(z@z.T)+.1*(h@h.T))
    torch.testing.assert_close(d.norm(dim=1),torch.ones(8))
    assert torch.equal(fuse(z,h,0)[:,:12],z)


def test_required_margin_feasibility():
    g=torch.tensor([.3,.005,-.02,-.4])
    t,clipped=required_margin(g)
    torch.testing.assert_close(t,torch.tensor([0.,.055,.28,1.8]))
    assert clipped.tolist()==[False,False,False,True]
    assert torch.all(.9*g[:3]+.1*t[:3]>=.01-1e-7)


def test_hard_balance_not_diluted_by_easy_count():
    x=torch.tensor([0.,0.],requires_grad=True);y=torch.tensor([1.,0.]);mask=torch.tensor([True,False])
    a=balanced_loss(x,y,mask)
    b=balanced_loss(torch.cat((x[:1],x[1:].repeat(100))),torch.cat((y[:1],y[1:].repeat(100))),torch.tensor([True]+[False]*100))
    torch.testing.assert_close(a,b)
    a.backward();assert x.grad[0]!=0


def test_batch_independence_gradients_and_resume(tmp_path):
    torch.manual_seed(42);m=Complement(12,8,4);x=torch.randn(9,12);opt=torch.optim.AdamW(m.parameters(),lr=.001)
    torch.testing.assert_close(m(x),torch.cat([m(v[None]) for v in x]),atol=1e-6,rtol=1e-6)
    def step(model,optimizer):
        h=model(x);g=(h[:3]*h[3:6]).sum(1)-(h[:3]*h[6:]).sum(1)
        loss=balanced_loss(g,torch.tensor([.2,0.,.1]),torch.tensor([True,False,True]))
        optimizer.zero_grad();loss.backward();optimizer.step()
    step(m,opt);assert all(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters())
    path=tmp_path/'state.pt';torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict()),path)
    copy=Complement(12,8,4);other=torch.optim.AdamW(copy.parameters(),lr=.001)
    p=torch.load(path,weights_only=True);copy.load_state_dict(p['model']);other.load_state_dict(p['optimizer'])
    step(m,opt);step(copy,other)
    for a,b in zip(m.parameters(),copy.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
