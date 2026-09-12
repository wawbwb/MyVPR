"""ClearCLIP-style dense teacher and pre-encoder token exclusion for plain RU-BoQ.

Independent implementation of the last-layer QQ attention-only construction in
ClearCLIP (Lan et al., ECCV 2024). This is a VPR adaptation, not an official
ClearCLIP segmentation benchmark reproduction (fixed full-image 560 resize).
"""
import hashlib
import numpy as np

DYNAMIC = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person']
STATIC = ['building', 'road', 'sidewalk', 'wall', 'fence', 'tree', 'vegetation',
          'sky', 'bridge', 'traffic sign', 'traffic light', 'pole', 'terrain']
SETTINGS = dict(schema='clearclip_token_drop_v1', model='ViT-B-16', pretrained='openai',
                teacher_size=560, image_size=280, grid=20, margin=.02,
                fraction=.75, min_keep=32, seed=42)


def drop_mask(fraction, identity, mode, min_keep=32):
    f=np.asarray(fraction)
    if f.shape!=(20,20) or not np.isfinite(f).all() or (f<0).any() or (f>1).any():
        raise ValueError('Expected finite 20x20 dynamic pixel fractions in [0,1]')
    if mode not in ('none','aligned','shuffled') or not 1<=min_keep<=400:
        raise ValueError('Invalid mode/minimum token count')
    mask=f>=SETTINGS['fraction']
    fallback=int((~mask).sum())<min_keep
    if fallback or mode=='none': mask=np.zeros_like(mask)
    if mode=='shuffled':
        seed=int(hashlib.sha256(('42:'+identity).encode()).hexdigest()[:16],16)
        rng=np.random.default_rng(seed)
        mask=np.stack([rng.permutation(row) for row in mask])
    return mask, fallback


def qq_attention_only(block, x):
    """B,N,C input; intentionally no final residual or FFN."""
    import torch.nn.functional as F
    attn=block.attn
    q, _, value=F.linear(block.ln_1(x),attn.in_proj_weight,attn.in_proj_bias).chunk(3,-1)
    b,n,c=q.shape; h=attn.num_heads; width=c//h
    q=q.reshape(b,n,h,width).transpose(1,2)
    value=value.reshape(b,n,h,width).transpose(1,2)
    weights=((q@q.transpose(-1,-2))*(width**-.5)).softmax(-1)
    output=(weights@value).transpose(1,2).reshape(b,n,c)
    return attn.out_proj(output)


class DenseCLIPTeacher:
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
            margin=maps[:,:len(DYNAMIC)].amax(1)-maps[:,len(DYNAMIC):].amax(1)
            binary=(margin>SETTINGS['margin']).float()
            fraction=F.avg_pool2d(binary[:,None],14,14)[:,0]
            if not torch.isfinite(fraction).all(): raise ValueError('Nonfinite teacher output')
            return fraction.cpu().numpy().astype(np.float16)


def compact_tokens(tokens, drop):
    """Physically gather kept tokens; mask only the remaining batch padding."""
    import torch
    if drop.dtype!=torch.bool or drop.shape!=tokens.shape[:2] or drop.all(1).any():
        raise ValueError('Invalid exclusion mask or no retained tokens')
    lengths=(~drop).sum(1); width=int(lengths.max())
    # Stable order preserves the original flattened spatial order.
    indices=torch.argsort(drop.to(torch.int32),dim=1,stable=True)[:,:width]
    compact=tokens.gather(1,indices[:,:,None].expand(-1,-1,tokens.shape[-1]))
    padding=torch.arange(width,device=tokens.device)[None]>=lengths[:,None]
    return compact.masked_fill(padding[:,:,None],0),padding


def encode_compact(aggregator, tokens, padding):
    """Every encoder/cross-attention layer excludes padded keys and values."""
    import torch
    outs=[]
    for block in aggregator.boqs:
        tokens=block.encoder(tokens,src_key_padding_mask=padding)
        tokens=tokens.masked_fill(padding[:,:,None],0)
        q=block.queries.expand(tokens.shape[0],-1,-1)
        q=block.norm_q(q+block.self_attn(q,q,q)[0])
        out=block.cross_attn(q,tokens,tokens,key_padding_mask=padding,need_weights=False)[0]
        outs.append(block.norm_out(out))
    out=aggregator.fc(torch.cat(outs,dim=1).transpose(1,2)).flatten(1)
    return torch.nn.functional.normalize(out,dim=-1)


def aggregate(aggregator, features, drop):
    """Frozen RU features enter the unchanged 3x3 projection, then token gather.

    DINO and the 3x3 projection already mix neighboring information; this does
    not claim complete removal of dynamic information from retained tokens.
    """
    if aggregator.semantic_num_classes is not None:
        raise ValueError('Requires plain RU-BoQ, not semantic-conditioned BoQ')
    if drop.shape!=(features.shape[0],features.shape[2],features.shape[3]):
        raise ValueError('Feature/mask grid mismatch')
    if not drop.any(): return aggregator(features)[0]
    tokens=aggregator.norm_input(aggregator.proj_c(features).flatten(2).transpose(1,2))
    tokens,padding=compact_tokens(tokens,drop.flatten(1))
    return encode_compact(aggregator,tokens,padding)
