"""Explicit landmark CLIP prior; auxiliary BoQ keeps selected tokens only."""
import hashlib
import numpy as np
from src.models.clip_token_drop import DYNAMIC,STATIC,SETTINGS,qq_attention_only
LANDMARKS=['building','wall','bridge','traffic sign']


def selection(fraction,identity,mode):
    f=np.asarray(fraction)
    if f.shape!=(20,20) or not np.isfinite(f).all() or (f<0).any() or (f>1).any():
        raise ValueError('Invalid static coverage')
    if mode not in ['full','random','static']: raise ValueError('Invalid branch')
    keep=f>=.75
    available=int(keep.sum())>=16
    if not available or mode=='full': return np.zeros((20,20),bool),available
    if mode=='random':
        seed=int(hashlib.sha256(('static-control:'+identity).encode()).hexdigest()[:16],16)
        offset=int(np.random.default_rng(seed).integers(1,400));dy,dx=divmod(offset,20)
        keep=np.roll(keep,(dy,dx),(0,1))
    return ~keep,available


def coordinates(path):
    from pathlib import PurePosixPath
    parts=PurePosixPath(path).parts
    city=parts[1]; name=parts[-1]
    if not name.startswith(city+'_'): raise ValueError('Unexpected GSV filename')
    fields=name[len(city)+1:].split('_')
    lat,lon=float(fields[4]),float(fields[5])
    if not (-90<=lat<=90 and -180<=lon<=180): raise ValueError('Invalid coordinates')
    return lat,lon


def make_batches(centroids,coords):
    z=np.asarray(centroids); coords=np.radians(coords)
    delta=coords[:,None]-coords[None,:]
    hav=np.sin(delta[:,:,0]/2)**2+np.cos(coords[:,0,None])*np.cos(coords[None,:,0])*np.sin(delta[:,:,1]/2)**2
    distance=6371000*2*np.arcsin(np.sqrt(np.clip(hav,0,1)))
    similarity=z@z.T
    result=[]
    for epoch in range(2):
        anchors=np.random.default_rng(42+epoch).permutation(len(z))[:64]
        batches=[]
        for anchor in anchors:
            order=np.argsort(-similarity[anchor],kind='stable')
            chosen=[int(anchor)]
            for j in order:
                if all(distance[j,k]>=100 for k in chosen): chosen.append(int(j))
                if len(chosen)==16: break
            if len(chosen)!=16: raise ValueError('Not enough geographically separated negative places')
            batches.append(chosen)
        result.append(batches)
    return result


class StaticTeacher:
    def __init__(self, device):
        from src.models.clip_teacher import CLIPTeacherEncoder
        from src.models.cc_lsa import module_state_sha256
        self.encoder=CLIPTeacherEncoder(dynamic_categories=DYNAMIC+STATIC).to(device).eval()
        self.device=device
        self.identity={'visual_sha256':module_state_sha256(self.encoder.visual),
                       'text_sha256':hashlib.sha256(self.encoder.dynamic_text_feats.cpu().numpy().tobytes()).hexdigest()}

    def fractions(self, images):
        import torch
        import torch.nn.functional as F
        with torch.inference_mode():
            t=self.encoder; v=t.visual
            # Inputs and cached masks share the same 280x280 geometry.
            x=images.to(self.device)*t.imagenet_std+t.imagenet_mean
            x=F.interpolate(x,size=(560,560),mode='bicubic',align_corners=False,antialias=True)
            x=(x-t.clip_mean)/t.clip_std
            x=v.conv1(x); gh,gw=x.shape[-2:]
            if (gh,gw)!=(35,35): raise ValueError('Requires ViT-B/16 at 560')
            x=x.flatten(2).transpose(1,2)
            cls=v.class_embedding[None,None].expand(x.shape[0],1,-1)
            x=torch.cat([cls,x],1)
            pos=v.positional_embedding
            # Match official ClearCLIP positional interpolation convention.
            old=int((pos.shape[0]-1)**.5)
            pp=pos[1:].reshape(1,old,old,-1).permute(0,3,1,2)
            pp=F.interpolate(pp,scale_factor=((gh+.1)/old,(gw+.1)/old),mode='bicubic',align_corners=False)
            pos=torch.cat([pos[:1],pp.flatten(2).transpose(1,2)[0]],0)
            x=v.ln_pre(x+pos[None])
            tr=v.transformer; batch_first=tr.batch_first
            state=x if batch_first else x.transpose(0,1)
            for block in tr.resblocks[:-1]: state=block(state)
            x=state if batch_first else state.transpose(0,1)
            x=qq_attention_only(tr.resblocks[-1],x)
            x=v.ln_post(x)[:,1:]@v.proj
            x=F.normalize(x.float(),dim=-1)
            cosine=x@t.dynamic_text_feats.float().T
            # Interpolate class scores before argmax/margin, then area-pool.
            maps=cosine.transpose(1,2).reshape(-1,len(DYNAMIC)+len(STATIC),gh,gw)
            maps=F.interpolate(maps,size=(280,280),mode='bilinear',align_corners=False)
            all_names=DYNAMIC+STATIC
            chosen=[all_names.index(c) for c in LANDMARKS]
            other=[i for i in range(len(all_names)) if i not in chosen]
            margin=maps[:,chosen].amax(1)-maps[:,other].amax(1)
            binary=(margin>SETTINGS['margin']).float()
            fraction=F.avg_pool2d(binary[:,None],14,14)[:,0]
            if not torch.isfinite(fraction).all(): raise ValueError('Nonfinite teacher output')
            return fraction.cpu().numpy().astype(np.float16)
