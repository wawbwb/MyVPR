"""Controlled compositing and paired challenge summaries; no model imports."""
import hashlib
import numpy as np
from PIL import Image

SEEDS=(11,29,47)
LOADS=(.05,.15)
SCHEMES=('none','clip','clip_shift','known','known_shift')


def rng(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big'))


def shifted(mask,key):
    mask=np.asarray(mask,dtype=bool)
    if not mask.any(): return mask.copy()
    yy,xx=np.where(mask); h,w=mask.shape
    dy0,dy1=-int(yy.min()),h-int(yy.max())
    dx0,dx1=-int(xx.min()),w-int(xx.max())
    choices=[(dy,dx) for dy in range(dy0,dy1) for dx in range(dx0,dx1) if dy or dx]
    if not choices: raise ValueError('Mask cannot translate without clipping')
    dy,dx=choices[int(rng(key).integers(len(choices)))]
    out=np.zeros_like(mask); out[yy+dy,xx+dx]=True
    return out


def shift_scores(score,key):
    """Toroidal translation preserves soft values/shape on a periodic grid; record wrapping."""
    h,w=score.shape
    k=int(rng(key).integers(1,h*w)); dy,dx=divmod(k,w)
    return np.roll(score,(dy,dx),(0,1))


def scaled_object(rgb,mask,area):
    scale=(area*280*280/max(int(mask.sum()),1))**.5
    h,w=mask.shape
    nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
    if nw>224 or nh>196: raise ValueError('Donor too sparse/tall for matched target area')
    resized=np.array(Image.fromarray(rgb).resize((nw,nh),Image.Resampling.BILINEAR))
    m=np.array(Image.fromarray(mask.astype('uint8')*255).resize((nw,nh),Image.Resampling.NEAREST))>0
    if abs(float(m.sum())/(280*280)-area)>.005: raise ValueError('Resized coverage differs')
    return resized,m


def composite(background,rgb,mask,area,key):
    patch,m=scaled_object(rgb,mask,area)
    h,w=m.shape
    generator=rng('placement:'+key)
    # Shared bottom/centre for both size levels of the same donor and seed.
    x=int(generator.integers(112,169))-w//2; bottom=int(generator.integers(210,281)); y=bottom-h
    support=np.zeros((280,280),bool); support[y:y+h,x:x+w]=m
    object_image=background.copy(); neutral=background.copy()
    object_image[y:y+h,x:x+w][m]=patch[m]
    # Same silhouette/location/area; remove recognizable internal object appearance.
    neutral[y:y+h,x:x+w][m]=np.round(patch[m].mean(0)).astype('uint8')
    return object_image,neutral,support


def summary(rows):
    by={(r['case'],r['scheme']):r for r in rows}
    if len(by)!=len(rows): raise ValueError('Duplicate results')
    cases={r['case'] for r in rows}
    if set(by)!={(c,s) for c in cases for s in SCHEMES}: raise ValueError('Incomplete schemes')
    clean={r['query']:r for r in rows if r['content']=='clean' and r['scheme']=='none'}
    result={}
    groups=sorted({(r['content'],r['seed'],r['load']) for r in rows})
    for content,seed,load in groups:
        base=[r for r in rows if r['scheme']=='none' and (r['content'],r['seed'],r['load'])==(content,seed,load)]
        if {r['query'] for r in base}!=set(clean): raise ValueError('Query set mismatch')
        damaged={r['case'] for r in base if clean[r['query']]['top1_correct'] and not r['top1_correct']}
        group={'queries':len(base),'clean_correct':sum(clean[r['query']]['top1_correct'] for r in base),
               'damaged_from_clean':len(damaged),'schemes':{}}
        for scheme in SCHEMES:
            current=[by[r['case'],scheme] for r in base]
            group['schemes'][scheme]={
                'correct':sum(r['top1_correct'] for r in current),
                'rescued_damaged':[r['query'] for r in current if r['case'] in damaged and r['top1_correct']],
                'new_regressions':[r['query'] for r,b in zip(current,base) if b['top1_correct'] and not r['top1_correct']],
                'corrected_vs_unmasked':[r['query'] for r,b in zip(current,base) if not b['top1_correct'] and r['top1_correct']],
                'mean_margin_delta':float(np.mean([r['positive_negative_margin']-b['positive_negative_margin'] for r,b in zip(current,base)]))}
        result[f'{content}|seed={seed}|area={load}']=group
    return {'groups':result,'warning':'Paired synthetic challenge; seeds reuse queries, not independent samples. Known mask is paste support, not recovered background.'}
