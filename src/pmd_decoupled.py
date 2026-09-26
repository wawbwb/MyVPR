"""Known-transform experiment: detach rejection from correspondence learning."""
import torch
from torch import nn
from src.pmd_correspondence import CorrespondenceHead
from scripts.audit_pmd_rejection import explicit_assignment


class DecoupledHead(CorrespondenceHead):
    def __init__(self,mode):
        if mode not in ['forced','partial','decoupled']:raise ValueError('Unknown mode')
        super().__init__('forced' if mode=='decoupled' else mode)
        self.experiment_mode=mode
        if mode=='decoupled':self.reject=nn.Sequential(nn.Linear(386,64),nn.ReLU(),nn.Linear(64,1))

    def forward(self,a,b):
        assignments=explicit_assignment(self,a,b,200)
        if self.experiment_mode=='partial':
            # Normalize explicit real+dustbin probabilities, not 1-row_sum.
            return [(p/(p.sum(-1)+d).clamp_min(1e-12)[...,None],d/(p.sum(-1)+d).clamp_min(1e-12)) for p,d in assignments]
        plans=[p/p.sum(-1,keepdim=True).clamp_min(1e-12) for p,d in assignments]
        if self.experiment_mode=='forced':return [(p,p.new_zeros(p.shape[:-1])) for p in plans]
        # Rejection cannot change norm/key/transport; shared head in both directions.
        with torch.no_grad():
            x=self.match.key(self.match.norm(a));y=self.match.key(self.match.norm(b))
            features=[]
            for p,u,v in [(plans[0],x,y),(plans[1],y,x)]:
                message=p@v
                entropy=-(p*p.clamp_min(1e-12).log()).sum(-1,keepdim=True)
                features.append(torch.cat([u,message,(u-message).abs(),p.max(-1,keepdim=True).values,entropy],-1))
        return [(p,self.reject(f.detach()).squeeze(-1).sigmoid()) for p,f in zip(plans,features)]


def supervised_loss(assignments,targets,rejection=True):
    terms=[]
    # Partial plan carries matched probability; decoupled/forced plan sums to1.
    for (p,d),t in zip(assignments,targets):
        p=p[0];d=d[0];t=t.to(p.device);valid=t>=0
        if valid.any():terms.append(-p[valid,t[valid]].clamp_min(1e-8).log().mean())
        if rejection:
            # Explicit binary supervision, both classes equally weighted.
            if valid.any():terms.append(-(1-d[valid]).clamp_min(1e-8).log().mean())
            if (~valid).any():terms.append(-d[~valid].clamp_min(1e-8).log().mean())
    return torch.stack(terms).sum()/2


def counts(assignments,targets):
    out=dict(matched=0,matched_correct=0,location_correct=0,unmatched=0,rejected_unmatched=0,false_rejected_matched=0)
    for (p,d),t in zip(assignments,targets):
        p=p[0];d=d[0];t=t.to(p.device);valid=t>=0;correct=p.argmax(-1)==t;reject=d>.5
        out['matched']+=int(valid.sum());out['unmatched']+=int((~valid).sum())
        out['location_correct']+=int((correct&valid).sum())
        out['matched_correct']+=int((correct&valid&~reject).sum())
        out['rejected_unmatched']+=int((reject&~valid).sum())
        out['false_rejected_matched']+=int((reject&valid).sum())
    return out
