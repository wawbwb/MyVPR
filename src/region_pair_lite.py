"""Fixed, untrained region correspondence screen. No labels used in scoring."""
import numpy as np


def normalize(x):
    return x / np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)


def pool(tokens, masks):
    m=masks.reshape(len(masks),-1).astype(np.float32)
    if not len(m) or not m.any(1).all():
        raise ValueError('Empty membership')
    y,x=np.mgrid[:20,:20]
    xy=np.stack([x.ravel(),y.ravel()],1)/19
    return normalize(m@tokens/m.sum(1,keepdims=True)), m@xy/m.sum(1,keepdims=True)


def match_regions(q,d,qxy,dxy):
    if min(len(q),len(d))<3: return 0.,0
    s=normalize(q.astype(np.float32))@normalize(d.astype(np.float32)).T
    right=s.argmax(1); left=s.argmax(0)
    ids=np.flatnonzero((left[right]==np.arange(len(q))) & (s[np.arange(len(q)),right]>=.6))
    if len(ids)<3: return 0.,0
    displacement=dxy[right[ids]]-qxy[ids]
    groups=np.linalg.norm(displacement[:,None]-displacement[None,:],axis=2)<=.2
    # Deterministic translation-consensus heuristic, not geometric ground truth.
    best=int(groups.sum(1).argmax()); kept=ids[groups[best]]
    values=np.sort(s[kept,right[kept]])[::-1][:4]
    return float(values.sum()/4),len(kept)


def conservative_order(ru_scores, region_scores, supports):
    order=np.arange(len(ru_scores))
    best=int(np.argmax(region_scores))
    if (best and ru_scores[0]-ru_scores[best]<=.02 and supports[best]>=3
            and region_scores[best]-region_scores[0]>=.05):
        order=np.concatenate(([best],order[order!=best]))
    return order
