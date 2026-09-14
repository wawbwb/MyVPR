"""Frozen pair evidence -> residual scores, with optional density correction."""
import torch
from torch import nn
import torch.nn.functional as F

MODES=['independent','set','density','consistency','competition']


class CandidateSet(nn.Module):
    def __init__(self,mode,input_dim=1536,width=128,db_dim=512):
        super().__init__()
        if mode not in MODES:raise ValueError(mode)
        self.mode=mode;self.width=width
        self.encode=nn.Sequential(nn.LayerNorm(input_dim+db_dim),nn.Linear(input_dim+db_dim,width),nn.GELU())
        self.qkv=nn.Linear(width,3*width)
        self.mix=nn.Linear(width,width)
        self.norm=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width))
        self.out=nn.Linear(width,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    def forward(self,evidence,base,db_vectors):
        # All arms receive the same evidence and reference global descriptors.
        combined=torch.cat([evidence,F.normalize(db_vectors,dim=-1)*db_vectors.shape[-1]**.5],dim=-1)
        h=self.encode(combined);q,k,v=self.qkv(h).chunk(3,-1)
        logits=q@k.transpose(-1,-2)/self.width**.5
        n=h.shape[1]
        if self.mode=='independent':
            logits=logits.masked_fill(~torch.eye(n,dtype=torch.bool,device=h.device)[None],float('-inf'))
        elif self.mode in ['density','competition']:
            # Visual similarity only: no labels/GPS. Soft density is not a true place grouping.
            z=F.normalize(db_vectors,dim=-1)
            density=torch.exp(((z@z.transpose(-1,-2))-1)/.02).sum(-1)
            logits=logits-density.clamp_min(1).log()[:,None,:]
        h=self.norm(h+self.mix(logits.softmax(-1)@v));h=h+self.ff(h)
        return base+4*torch.tanh(self.out(h).squeeze(-1))


def list_loss(scores,labels):
    """Multi-positive set likelihood; unreachable queries are not relabeled."""
    valid=labels.any(-1)&(~labels).any(-1)
    if not valid.any():return scores.sum()*0,0
    s=scores[valid];y=labels[valid]
    return (s.logsumexp(-1)-s.masked_fill(~y,float('-inf')).logsumexp(-1)).mean(),int(valid.sum())


def duplicate_consistency(original,augmented):
    # Compare original unique entries, excluding appended duplicates from normalization.
    logp=F.log_softmax(original,dim=-1)
    logq=F.log_softmax(augmented[:,:original.shape[1]],dim=-1)
    return .5*((logp.exp()*(logp-logq)).sum(-1)+(logq.exp()*(logq-logp)).sum(-1)).mean()
