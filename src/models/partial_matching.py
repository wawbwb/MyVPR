"""Pair-conditioned patch transport inside a frozen Pair-VPR decoder.

The dustbin/Sinkhorn construction is established matching machinery, not a
novelty claim. Scores remain the original classifier applied to updated tokens.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class PartialMatching(nn.Module):
    MODES=('ordinary','forced','partial')

    def __init__(self,dim=768,width=128,mode='partial',iterations=20):
        super().__init__()
        if mode not in self.MODES:raise ValueError('Unknown matching mode')
        self.mode=mode;self.iterations=iterations
        self.norm=nn.LayerNorm(dim)
        self.key=nn.Linear(dim,width,bias=False)
        self.value=nn.Linear(dim,width,bias=False)
        self.output=nn.Linear(width,dim,bias=False)
        self.bin_score=nn.Parameter(torch.tensor(0.))
        nn.init.zeros_(self.output.weight)

    def assignment(self,score):
        b,n,m=score.shape
        if self.mode=='ordinary':return score.softmax(-1),score.transpose(1,2).softmax(-1)
        if self.mode=='partial':
            dust=self.bin_score.expand(b,n,1)
            z=torch.cat([torch.cat([score,dust],2),self.bin_score.expand(b,1,m+1)],1)
            mu=torch.cat([score.new_ones(n),score.new_tensor([float(m)])])/(n+m)
            nu=torch.cat([score.new_ones(m),score.new_tensor([float(n)])])/(n+m)
        else:
            z=score;mu=score.new_full((n,),1/n);nu=score.new_full((m,),1/m)
        u=torch.zeros_like(z[:,:,0]);v=torch.zeros_like(z[:,0,:])
        for _ in range(self.iterations):
            u=mu.log()[None]-torch.logsumexp(z+v[:,None,:],2)
            v=nu.log()[None]-torch.logsumexp(z+u[:,:,None],1)
        plan=torch.exp(z+u[:,:,None]+v[:,None,:])[:,:n,:m]
        # Do NOT renormalize matched mass to one: rejected mass must attenuate messages.
        return plan/mu[:n][None,:,None],plan.transpose(1,2)/nu[:m][None,:,None]

    def forward(self,x,y):
        # x has a CLS token; y is patch-only. Never transport the CLS token.
        a=self.norm(x[:,1:]);b=self.norm(y)
        logits=self.key(a)@self.key(b).transpose(1,2)/math.sqrt(self.key.out_features)
        p,q=self.assignment(logits.float())
        dx=self.output((p@self.value(b).float()).to(x.dtype))
        dy=self.output((q@self.value(a).float()).to(y.dtype))
        outx=torch.cat([x[:,:1],x[:,1:]+dx],1)
        return outx,y+dy,dict(matched_mass_x=p.sum(-1).mean(),matched_mass_y=q.sum(-1).mean())


class PMDPair(nn.Module):
    """Insert after block11, before block12; prefix frozen and detached.

    One final official decoder block propagates updated patch evidence to CLS.
    No hooks/monkeypatches, no changes to original repository files.
    """
    def __init__(self,base,mode='partial',width=128):
        super().__init__()
        if len(base.dec_blocks)!=12 or base.decoder_clstoken is None:raise ValueError('Expected stage2 Pair-VPR12')
        self.base=base.eval()
        for param in base.parameters():param.requires_grad_(False)
        self.match=PartialMatching(base.decoder_clstoken.shape[-1],width,mode)

    def prefix(self,a,b):
        with torch.no_grad():
            x=self.base.decoder_embed(a);y=self.base.decoder_embed(b)
            x=torch.cat([self.base.decoder_clstoken.expand(len(x),-1,-1),x],1)+self.base.dec_pos_embed_cls
            y=y+self.base.dec_pos_embed
            for block in self.base.dec_blocks[:-1]:x,y=block(x,y)
        return x.detach(),y.detach()

    def finish(self,state,bypass=False):
        x,y=state;stats={}
        if not bypass:x,y,stats=self.match(x,y)
        x,_=self.base.dec_blocks[-1](x,y)
        return self.base.classvprmodule(self.base.dec_norm(x)[:,0]).flatten(),stats

    def forward(self,a,b):return self.finish(self.prefix(a,b))[0]
