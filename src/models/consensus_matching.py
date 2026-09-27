"""NCNet/CHM-inspired factorized correlation consensus, not a novelty claim."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.models.partial_matching import PMDPair


class CorrelationConsensus(nn.Module):
    """Alternate convolution over source/target grid axes of the 4D volume."""
    def __init__(self,channels=8,pointwise=False):
        super().__init__();k=1 if pointwise else 3
        self.source=nn.Conv2d(1,channels,k,padding=k//2)
        self.target=nn.Conv2d(channels,channels,k,padding=k//2)
        self.readout=nn.Conv2d(channels,1,1)

    def forward(self,scores):
        batch,n,m=scores.shape;h=math.isqrt(n);w=math.isqrt(m)
        if h*h!=n or w*w!=m:raise ValueError('Expected square patch grids')
        # Every target location gets a source-grid convolution, then vice versa.
        x=scores.transpose(1,2).reshape(batch*m,1,h,h)
        x=F.gelu(self.source(x));c=x.shape[1]
        x=x.reshape(batch,m,c,n).permute(0,3,2,1).reshape(batch*n,c,w,w)
        x=self.readout(F.gelu(self.target(x)))
        return x.reshape(batch,n,m)


class ConsensusMatching(nn.Module):
    def __init__(self,dim=768,width=128,mode='spatial'):
        super().__init__()
        if mode not in ['spatial','pointwise','shuffled']:raise ValueError('Unknown mode')
        self.mode=mode;self.norm=nn.LayerNorm(dim);self.key=nn.Linear(dim,width,bias=False)
        self.value=nn.Linear(dim,width,bias=False);self.output=nn.Linear(width,dim,bias=False)
        self.consensus=CorrelationConsensus(pointwise=mode=='pointwise')
        nn.init.zeros_(self.output.weight)

    def contextual_bias(self,scores):
        if self.mode!='shuffled':return self.consensus(scores)
        # Fixed coordinate-scrambling control; restore indexing before messaging.
        g=torch.Generator().manual_seed(42);n,m=scores.shape[1:]
        p=torch.randperm(n,generator=g).to(scores.device);q=torch.randperm(m,generator=g).to(scores.device)
        shuffled=scores[:,p][:,:,q]
        return self.consensus(shuffled)[:,p.argsort()][:,:,q.argsort()]

    def forward(self,x,y):
        a=self.norm(x[:,1:]);b=self.norm(y)
        scores=self.key(a)@self.key(b).transpose(1,2)/math.sqrt(self.key.out_features)
        # Symmetrize axis-factorized filters; no dustbin or matched-mass rejection.
        bias=(self.contextual_bias(scores)+self.contextual_bias(scores.transpose(1,2)).transpose(1,2))/2
        logits=scores+bias
        p=logits.softmax(-1);q=logits.transpose(1,2).softmax(-1)
        dx=self.output(p@self.value(b));dy=self.output(q@self.value(a))
        return torch.cat([x[:,:1],x[:,1:]+dx],1),y+dy,dict(context_abs_mean=bias.abs().mean())


class ConsensusPair(PMDPair):
    def __init__(self,base,mode='spatial',width=128):
        super().__init__(base)
        self.match=ConsensusMatching(base.decoder_clstoken.shape[-1],width,mode)
