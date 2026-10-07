"""Research SC-Mamba-BoQ: shared spatial scans and descriptor consistency.

Independent real-diagonal Mamba-1 recurrence implementation, following
Gu & Dao arXiv:2312.00752. No official fused CUDA kernels or pretrained Mamba.
The associative reference scan is O(L log L), NOT the linear-work CUDA kernel.
Four-direction scanning is prior art (VMamba), not our novelty claim.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

MODES = ('boq', 'conv', 'mamba', 'mamba_consistent')


def selective_scan(u, delta, a, b, c, skip):
    """u/delta B,L,D; a D,S; b/c B,L,S. h[t]=exp(dt*A)*h[t-1]+dt*B*u."""
    transition = torch.exp(delta[..., None] * a[None, None])
    state = delta[..., None] * b[:, :, None, :] * u[..., None]
    offset = 1
    # Compose affine transitions (a2,b2)o(a1,b1)=(a2*a1,b2+a2*b1).
    while offset < u.shape[1]:
        state = torch.cat((state[:, :offset], state[:, offset:] +
                           transition[:, offset:] * state[:, :-offset]), dim=1)
        transition = torch.cat((transition[:, :offset], transition[:, offset:] *
                                transition[:, :-offset]), dim=1)
        offset *= 2
    return (state * c[:, :, None, :]).sum(-1) + u * skip


def scan_order(height, width, column, device):
    grid = torch.arange(height*width, device=device).reshape(height, width)
    return grid.T.flatten() if column else grid.flatten()


class SelectiveMixer(nn.Module):
    def __init__(self, dim, inner=64, state=8, rank=4):
        super().__init__()
        self.state, self.rank = state, rank
        self.in_proj = nn.Linear(dim, 2*inner, bias=False)
        self.conv = nn.Conv1d(inner, inner, 4, padding=3, groups=inner)
        self.select = nn.Linear(inner, rank+2*state, bias=False)
        self.dt = nn.Linear(rank, inner)
        nn.init.uniform_(self.dt.weight, -rank**-.5, rank**-.5)
        dt = torch.exp(torch.rand(inner)*(math.log(.1)-math.log(.001))+math.log(.001))
        with torch.no_grad(): self.dt.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.a_log = nn.Parameter(torch.arange(1,state+1).float().log().repeat(inner,1))
        self.skip = nn.Parameter(torch.ones(inner))
        self.out = nn.Linear(inner, dim, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, tokens):
        x,z = self.in_proj(tokens).chunk(2,-1)
        x = F.silu(self.conv(x.transpose(1,2))[..., :tokens.shape[1]].transpose(1,2))
        dt,b,c = torch.split(self.select(x), (self.rank,self.state,self.state), dim=-1)
        y = selective_scan(x, F.softplus(self.dt(dt)), -self.a_log.exp(), b,c,self.skip)
        return self.out(y*F.silu(z))


class SpatialMixer(nn.Module):
    def __init__(self, dim, mode):
        super().__init__()
        self.mode = mode
        self.norm = nn.LayerNorm(dim)
        if mode == 'conv':
            self.input = nn.Linear(dim,128,bias=False)
            self.depthwise = nn.Conv2d(64,64,3,padding=1,groups=64)
            self.hidden = nn.Linear(64,64)
            self.output = nn.Linear(64,dim,bias=False)
            nn.init.zeros_(self.output.weight)
        else:
            self.ssm = SelectiveMixer(dim)

    def forward(self, tokens, height, width):
        x = self.norm(tokens)
        if self.mode == 'conv':
            u,z = self.input(x).chunk(2,-1)
            u = self.depthwise(u.transpose(1,2).reshape(len(x),64,height,width)).flatten(2).transpose(1,2)
            y = tokens + self.output(F.silu(self.hidden(F.silu(u)))*F.silu(z))
            return y,y
        row = scan_order(height,width,False,x.device)
        col = scan_order(height,width,True,x.device)
        orderings = (row,row.flip(0),col,col.flip(0))
        packed = torch.cat([x[:,order] for order in orderings],0)
        scanned = self.ssm(packed).chunk(4,0)
        restored = [y[:,order.argsort()] for y,order in zip(scanned,orderings)]
        return tokens + (restored[0]+restored[1])*.5, tokens + (restored[2]+restored[3])*.5


def descriptor_consistency(row, col):
    # Equals 1-cosine for unit vectors, without tiny negative roundoff values.
    return .5*(row-col).square().sum(-1).mean()


class ScanMambaVPR(nn.Module):
    def __init__(self, visual, mode):
        super().__init__()
        from src.models.depth_query import DepthQueryVPR
        if mode not in MODES: raise ValueError('Unknown mode')
        self.mode = mode
        self.base = DepthQueryVPR(visual,'baseline').requires_grad_(False)
        self.base.aggregator.requires_grad_(True)
        dim = self.base.aggregator.proj_c.out_channels
        self.mixer = None if mode=='boq' else SpatialMixer(dim,mode)

    def features(self, images):
        return self.base.features(images)[-1]

    def aggregate(self, features, bypass=False):
        agg = self.base.aggregator
        tokens = agg.norm_input(agg.proj_c(features).flatten(2).transpose(1,2))
        def readout(x):
            outs=[]
            for block in agg.boqs:
                x,y,_ = block(x); outs.append(y)
            return F.normalize(agg.fc(torch.cat(outs,1).transpose(1,2)).flatten(1),dim=-1)
        if self.mixer is None or bypass:
            d=readout(tokens);return d,d,d
        row,col = self.mixer(tokens,*features.shape[-2:])
        dr = readout(row)
        if self.mode=='conv': return dr,dr,dr
        dc = readout(col)
        return F.normalize(dr+dc,dim=-1),dr,dc

    def forward(self,images): return self.aggregate(self.features(images))[0]
