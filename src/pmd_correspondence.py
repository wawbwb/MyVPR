"""Exact patch-grid crop/flip/occlusion labels, not place-label correspondences."""
import torch
from torch import nn
from src.models.partial_matching import PartialMatching


def make_views(image,seed,occlude=True,patch=14,grid=23):
    """Input is a normalized 29x29-patch image. Crops are NOT resized again."""
    if image.shape!=(3,29*patch,29*patch):raise ValueError('Expected 406x406 RGB tensor')
    rng=torch.Generator().manual_seed(seed);views=[];ids=[]
    coordinates=torch.arange(29*29).reshape(29,29)
    for _ in range(2):
        top,left=torch.randint(0,7,(2,),generator=rng).tolist()
        x=image[:,top*patch:(top+grid)*patch,left*patch:(left+grid)*patch].clone()
        tags=coordinates[top:top+grid,left:left+grid].clone()
        if bool(torch.randint(0,2,(),generator=rng)):
            x=x.flip(-1);tags=tags.flip(-1)
        if occlude:
            y,z=torch.randint(0,grid-5,(2,),generator=rng).tolist()
            x[:,y*patch:(y+5)*patch,z*patch:(z+5)*patch]=0
            tags[y:y+5,z:z+5]=-1
        views.append(x);ids.append(tags.flatten())
    targets=[]
    for a,b in [(ids[0],ids[1]),(ids[1],ids[0])]:
        equal=(a[:,None]==b[None,:]) & (a[:,None]>=0)
        targets.append(torch.where(equal.any(1),equal.long().argmax(1),-1))
    return torch.stack(views),targets


class CorrespondenceHead(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        self.match=PartialMatching(mode='partial' if mode=='partial' else 'forced')
        for p in list(self.match.value.parameters())+list(self.match.output.parameters()):p.requires_grad_(False)
        if mode=='forced':self.match.bin_score.requires_grad_(False)

    def forward(self,a,b):
        m=self.match;a=m.key(m.norm(a));b=m.key(m.norm(b))
        score=(a@b.transpose(1,2))/(a.shape[-1]**.5)
        if self.mode=='independent_dustbin':
            def one(s):return torch.cat([s,m.bin_score.expand(*s.shape[:-1],1)],-1).softmax(-1)[...,:-1]
            return one(score),one(score.transpose(1,2))
        return m.assignment(score)


def supervised_loss(plans,targets,rejection=True):
    terms=[]
    for p,t in zip(plans,targets):
        p=p[0];t=t.to(p.device);valid=t>=0
        if valid.any():terms.append(-p[valid,t[valid]].clamp_min(1e-8).log().mean())
        if rejection and (~valid).any():terms.append(-(1-p[~valid].sum(-1)).clamp_min(1e-8).log().mean())
    return torch.stack(terms).mean()


def counts(plans,targets):
    out=dict(matched=0,matched_correct=0,unmatched=0,rejected_unmatched=0,false_rejected_matched=0)
    for p,t in zip(plans,targets):
        p=p[0];t=t.to(p.device);matched=t>=0;reject=(1-p.sum(-1))>p.max(-1).values
        out['matched']+=int(matched.sum());out['unmatched']+=int((~matched).sum())
        out['matched_correct']+=int(((p.argmax(-1)==t)&matched&~reject).sum())
        out['rejected_unmatched']+=int((reject&~matched).sum())
        out['false_rejected_matched']+=int((reject&matched).sum())
    return out
