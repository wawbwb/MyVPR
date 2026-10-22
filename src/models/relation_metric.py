"""Train-only relation basis; bounded per-image calibration after frozen BoQ."""
import torch
from torch import nn
from torch.nn import functional as F

ARMS=('random_basis','pca_basis','relation_static','relation_conditional')


def orthogonal_check(u):
    if u.ndim!=2 or not torch.isfinite(u).all():raise ValueError('Invalid basis')
    if not torch.allclose(u.T@u,torch.eye(u.shape[1],device=u.device,dtype=u.dtype),atol=2e-4,rtol=0):
        raise ValueError('Nonorthogonal basis')


def calibrated(z,u,a):
    delta=((z@u)*a)@u.T
    # FP32 unit inputs need not have a bit-exact unit norm. Subtract the
    # unchanged normalization to preserve the archived zero-update descriptor.
    out=z+(F.normalize(z+delta,dim=-1)-F.normalize(z,dim=-1))
    return out,delta


class RelationMetric(nn.Module):
    def __init__(self,basis,static=False,hidden=32,cap=.05):
        super().__init__();orthogonal_check(basis)
        if not 0<cap<1:raise ValueError('Invalid cap')
        self.register_buffer('basis',basis.detach().clone());self.cap=cap
        r=basis.shape[1]
        if static:self.coefficients=nn.Parameter(torch.zeros(r));self.gate=None
        else:
            self.register_parameter('coefficients',None)
            self.gate=nn.Sequential(nn.Linear(r,hidden),nn.GELU(),nn.Linear(hidden,r))
            nn.init.zeros_(self.gate[-1].weight);nn.init.zeros_(self.gate[-1].bias)

    def forward(self,z,details=False):
        a=self.cap*torch.tanh(self.coefficients if self.gate is None else self.gate(z@self.basis))
        out,delta=calibrated(z,self.basis,a)
        return (out,delta,a) if details else out


def canonical_columns(u):
    idx=u.abs().argmax(0);sign=u[idx,torch.arange(u.shape[1],device=u.device)].sign()
    return u*sign[None,:]


def relation_basis(projected,pca,negative_ids,rank=64,eps=1e-8):
    if len(projected)%4 or negative_ids.shape!=(len(projected),):raise ValueError('Expected place-major four views')
    if negative_ids.dtype not in (torch.int32,torch.int64) or (negative_ids<0).any() or (negative_ids>=len(projected)).any():
        raise ValueError('Bad negative index')
    negative_ids=negative_ids.long()
    own=torch.arange(len(projected),device=projected.device)//4
    if (own==own[negative_ids]).any():raise ValueError('Same-place negative')
    views=projected.reshape(-1,4,projected.shape[1]).double()
    w=torch.zeros((projected.shape[1],projected.shape[1]),device=projected.device,dtype=torch.float64)
    for i in range(4):
        for j in range(i):
            d=views[:,i]-views[:,j];w+=d.T@d/(len(views)*6)
    d=(projected-projected[negative_ids]).double();b=d.T@d/len(d)
    contrast=b/(b.trace()+eps)-w/(w.trace()+eps)
    val,v=torch.linalg.eigh(contrast);ids=torch.argsort(val.abs(),descending=True,stable=True)[:rank]
    u=canonical_columns(pca@v[:,ids].to(pca.dtype));orthogonal_check(u)
    return u,w,b,val[ids]


def whitening(pca,within,rank=64,cap=.05):
    w=within[:rank,:rank];c=.9*w+.1*w.trace()/rank*torch.eye(rank,device=w.device,dtype=w.dtype)
    values,q=torch.linalg.eigh(c);values=values.clamp_min(1e-8)
    logscale=.5*(values.log().mean()-values.log())
    u=canonical_columns(pca[:,:rank]@q.to(pca.dtype));orthogonal_check(u)
    return u,(cap*logscale.tanh()).to(pca.dtype)
