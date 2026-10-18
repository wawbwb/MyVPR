"""Within-place CLIP residual adversary; CLIP/predictor are TRAINING ONLY.

The inference student reuses the isolated visual-only BoQ router. Neither the
official network nor innovation-1 sources are modified. Residuals are proxies,
not verified weather/dynamic-object ground truth.
"""
import hashlib
import math

import numpy as np
import torch
from torch import nn

from src.models.clip_key_boq import ClipKeyBoQ

ARMS = ('visual', 'aligned', 'shuffled')


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,strength):
        ctx.strength = float(strength)
        return value.view_as(value)

    @staticmethod
    def backward(ctx,gradient):
        return -ctx.strength*gradient,None


def reverse(value,strength=1.):
    return Reverse.apply(value,strength)


def within_place(value,labels):
    if value.ndim != 2 or labels.ndim != 1 or len(labels) != len(value):
        raise ValueError('Need feature rows and aligned place labels')
    _,inverse,counts = torch.unique(labels,return_inverse=True,return_counts=True)
    if bool((counts < 2).any()):
        raise ValueError('Within-place residual requires at least two views per place')
    sums = torch.zeros(len(counts),value.shape[1],device=value.device,dtype=value.dtype)
    sums = sums.index_add(0,inverse,value)
    means = sums/counts.to(value.dtype)[:,None]
    return value-means[inverse]


def center_four(value):
    value = np.asarray(value,dtype=np.float32)
    if value.ndim != 2 or len(value)%4:
        raise ValueError('Need four contiguous views per place')
    x = value.reshape(-1,4,value.shape[-1])
    return (x-x.mean(1,keepdims=True)).reshape(value.shape)


def projection(dim=12288,outputs=256,seed=42):
    generator = torch.Generator().manual_seed(seed+7103)
    value = torch.randn(dim,outputs,generator=generator)
    return torch.nn.functional.normalize(value,dim=0)*math.sqrt(dim)


def donor_indices(place_names,seed=42):
    """Within each place, permute complete view vectors with no fixed points.

    Spatially shuffling a map does NOT change its pooled scene vector, so it is
    not a valid corruption for this new experiment.
    """
    indices=[]
    for p,name in enumerate(place_names):
        key=int.from_bytes(hashlib.sha256(f'wpcrd:{seed}:{name}'.encode()).digest()[:8],'little')
        order=np.random.default_rng(key).permutation(4)
        mapping=np.empty(4,np.int64)
        mapping[order]=np.roll(order,1)
        indices.extend((p*4+mapping).tolist())
    return np.asarray(indices,np.int64)


def whitening(scene,train_rows,components=32):
    if train_rows%4 or train_rows<components+1 or not 0<components<=scene.shape[1]:
        raise ValueError('Invalid training-only whitening population')
    residual=center_four(scene)
    train=residual[:train_rows].astype(np.float64)
    covariance=train.T@train/max(1,len(train)-1)
    eigenvalues,eigenvectors=np.linalg.eigh(covariance)
    order=np.argsort(eigenvalues)[::-1][:components]
    selected=eigenvalues[order]
    if not np.isfinite(selected).all() or selected[-1]<=1e-10:
        raise ValueError('CLIP within-place residuals are degenerate')
    matrix=(eigenvectors[:,order]/np.sqrt(selected)[None]).astype(np.float32)
    # Bound rare target outliers, then restore each place's zero residual mean.
    targets=center_four(np.clip(residual@matrix,-5.,5.))
    return matrix,targets,dict(explained_variance=float(selected.sum()/eigenvalues.clip(0).sum()),
                             eigenvalues=selected.tolist(),target_train_variance=float(targets[:train_rows].var()))


class Regressor(nn.Module):
    def __init__(self,inputs=256,hidden=128,targets=32):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(inputs,hidden),nn.GELU(),nn.Linear(hidden,targets))

    def forward(self,x):
        return 5.*torch.tanh(self.layers(x.clamp(-8.,8.))/5.)


class ResidualPredictor(nn.Module):
    def __init__(self,scale,descriptor_dim=12288,hidden=128,targets=32,seed=42):
        super().__init__()
        scale=torch.as_tensor(scale,dtype=torch.float32)
        if scale.ndim!=1 or not torch.isfinite(scale).all() or bool((scale<=0).any()):
            raise ValueError('Need positive training-only baseline feature scales')
        self.register_buffer('projection',projection(descriptor_dim,len(scale),seed))
        self.register_buffer('scale',scale)
        self.head=Regressor(len(scale),hidden,targets)

    def forward(self,descriptors,labels,adversarial=False):
        x=within_place(descriptors@self.projection,labels)/self.scale
        return self.head(reverse(x) if adversarial else x)


class InvariantBoQ(ClipKeyBoQ):
    """Inference has exactly one input, RGB; CLIP is NOT an inference dependency."""
    def forward(self,images):
        return super().forward(images,None,'visual')

    def describe(self,features):
        return super().aggregate(features,None,'visual')
