"""Training-only fixed gallery supervision; no inference classifier or extra encoder."""
import numpy as np
import torch
from torch.nn import functional as F
from src.models.joint_local_refine import JointLocalVPR,image_seeds

ARMS=('continue_ms','refine_ms','continue_proto','refine_proto')


class GalleryVPR(JointLocalVPR):
    def __init__(self,anchor,arm):
        if arm not in ARMS:raise ValueError('Unknown arm')
        super().__init__(anchor,'joint_refine' if arm.startswith('refine') else 'neck_continue')


def distance(a,b):
    """Radians -> pairwise great-circle metres, with ordinary broadcasting."""
    h=torch.sin((a[...,0]-b[...,0])/2).square()
    h+=torch.cos(a[...,0])*torch.cos(b[...,0])*torch.sin((a[...,1]-b[...,1])/2).square()
    return 6371000*2*torch.asin(h.clamp(0,1).sqrt())


def positive_without_self(bank,sums,indices):
    x=sums[indices//4]-bank[indices]
    norm=np.linalg.norm(x,axis=1,keepdims=True)
    if (norm<1e-6).any():raise ValueError('Degenerate positive prototype')
    return x/norm


class FixedGallery:
    def __init__(self,bank,row_ids,coordinates,device):
        if bank.ndim!=2 or len(bank)%4 or len(row_ids)!=len(bank) or len(set(row_ids))!=len(bank):raise ValueError('Invalid bank')
        self.bank=bank;self.sums=bank.reshape(-1,4,bank.shape[1]).sum(1)
        self.proxies=F.normalize(torch.tensor(self.sums,device=device),dim=1)
        self.row_map={int(r):i for i,r in enumerate(row_ids)}
        coords=torch.deg2rad(torch.tensor(coordinates,dtype=torch.float64,device=device)).reshape(-1,4,2)
        self.centers=coords.mean(1)
        self.radius=distance(coords,self.centers[:,None,:]).amax(1)

    def loss(self,z,row_ids,labels,coordinates):
        ids=np.array([self.row_map[int(i)] for i in row_ids],dtype=np.int64)
        target=torch.tensor(ids//4,device=z.device)
        if not torch.equal(target,labels):raise ValueError('Place labels disagree with sealed bank')
        pos=torch.tensor(positive_without_self(self.bank,self.sums,ids),device=z.device)
        # Replace own-class score by the prototype of the OTHER three views.
        scores=z@self.proxies.T
        scores=scores.scatter(1,target[:,None],(z*pos).sum(1,keepdim=True))
        xy=torch.deg2rad(coordinates.double().to(z.device))
        safe=distance(xy[:,None,:],self.centers[None,:,:])-self.radius[None,:]>=25.
        safe.scatter_(1,target[:,None],True)
        if not (safe.sum(1)>=2).all():raise ValueError('No legal negative prototypes')
        logits=(20*scores).masked_fill(~safe,-torch.inf)
        loss=F.cross_entropy(logits,target)
        stats=dict(gallery_classes=self.proxies.shape[0],safe_negative_mean=float((safe.sum(1)-1).double().mean()),
            training_prototype_top1=int((logits.detach().argmax(1)==target).sum()),
            positive_excludes_self=True)
        return loss,stats
