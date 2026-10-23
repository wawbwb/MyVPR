"""Independent complementary embedding with explicit anchor-margin targets."""
import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS=('constant_target','deficit_target','shuffled_target')


class Complement(nn.Module):
    def __init__(self,inputs=256,hidden=256,outputs=128):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(inputs),nn.Linear(inputs,hidden),nn.GELU(),nn.Linear(hidden,outputs))

    def forward(self,x):return F.normalize(self.net(x),dim=-1)


def fuse(anchor,side,weight=.1):
    if not 0<=weight<1:raise ValueError('Invalid score weight')
    return torch.cat((math.sqrt(1-weight)*anchor,math.sqrt(weight)*side),dim=-1)


def required_margin(anchor_margin,weight=.1,goal=.01,cap=1.8):
    if not 0<weight<1 or not 0<cap<=2:raise ValueError('Invalid margin budget')
    raw=(goal-(1-weight)*anchor_margin)/weight
    return raw.clamp(0,cap),raw>cap


def balanced_loss(side_gap,target,hard):
    error=F.smooth_l1_loss(side_gap,target,reduction='none',beta=.1)
    pieces=[error[mask].mean() for mask in (hard,~hard) if mask.any()]
    if not pieces:raise ValueError('Empty training batch')
    return torch.stack(pieces).mean()
