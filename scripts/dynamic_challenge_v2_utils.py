"""Low-overlap controls and eligible-pair summaries, without torch."""
import numpy as np
from scripts.dynamic_challenge_utils import rng,SCHEMES


def low_overlap(mask,key,periodic=False,limit=.1):
    f=np.asarray(mask,dtype=np.float32)
    if f.ndim!=2 or not np.isfinite(f).all() or (f<0).any() or (f>1).any():
        raise ValueError('Invalid mask')
    mass=float(f.sum()); h,w=f.shape
    if mass==0: return f.copy(),{'valid':True,'overlap':0.,'reason':'empty','shift':[0,0]}
    if periodic:
        choices=[(y,x) for y in range(h) for x in range(w) if y or x]
    else:
        yy,xx=np.where(f>0)
        # Pixel translations on a 14-pixel lattice; no clipping or wrap.
        choices=[(y,x) for y in range(-int(yy.min()),h-int(yy.max()))
                 for x in range(-int(xx.min()),w-int(xx.max())) if (y or x) and y%14==x%14==0]
    order=rng(key).permutation(len(choices)); minimum=1.
    for i in order:
        dy,dx=choices[int(i)]
        if periodic: shifted=np.roll(f,(dy,dx),(0,1))
        else:
            shifted=np.zeros_like(f); shifted[yy+dy,xx+dx]=f[yy,xx]
        overlap=float(np.minimum(f,shifted).sum()/mass); minimum=min(minimum,overlap)
        if overlap<=limit:
            return shifted,{'valid':True,'overlap':overlap,'reason':'matched','shift':[dy,dx]}
    return np.zeros_like(f),{'valid':False,'overlap':None,'min_overlap':minimum,
                             'reason':'no_translation_below_limit','shift':None}


def summary(rows):
    by={(r['case'],r['scheme']):r for r in rows}
    cases={r['case'] for r in rows}
    if len(rows)!=len(by) or set(by)!={(c,s) for c in cases for s in SCHEMES}:
        raise ValueError('Missing or duplicate schemes')
    clean={r['query']:r for r in rows if r['content']=='clean' and r['scheme']=='none'}
    groups={}
    for content,seed,load in sorted({(r['content'],r['seed'],r['load']) for r in rows}):
        base=[r for r in rows if r['scheme']=='none' and (r['content'],r['seed'],r['load'])==(content,seed,load)]
        if {r['query'] for r in base}!=set(clean): raise ValueError('Different query sets')
        damaged={r['case'] for r in base if clean[r['query']]['top1_correct'] and not r['top1_correct']}
        g={'queries':len(base),'damaged_from_clean':len(damaged),'schemes':{},'paired_controls':{}}
        for scheme in SCHEMES:
            paired=[(b,by[b['case'],scheme]) for b in base if by[b['case'],scheme]['eligible']]
            g['schemes'][scheme]={'eligible':len(paired),'excluded':len(base)-len(paired),
                'correct':sum(r['top1_correct'] for _,r in paired),
                'eligible_damaged':sum(b['case'] in damaged for b,_ in paired),
                'rescued_damaged':[r['query'] for b,r in paired if b['case'] in damaged and r['top1_correct']],
                'regressed':[r['query'] for b,r in paired if b['top1_correct'] and not r['top1_correct']]}
        for aligned,control in [('known','known_shift'),('clip','clip_shift')]:
            eligible=[b for b in base if by[b['case'],control]['eligible']]
            g['paired_controls'][aligned]={'queries':len(eligible),
                'aligned_only_correct':[b['query'] for b in eligible if by[b['case'],aligned]['top1_correct'] and not by[b['case'],control]['top1_correct']],
                'control_only_correct':[b['query'] for b in eligible if not by[b['case'],aligned]['top1_correct'] and by[b['case'],control]['top1_correct']]}
        groups[f'{content}|seed={seed}|area={load}']=g
    return {'groups':groups,'warning':'Compare controls only on common eligible cases. Missing controls are not failures or identity interventions. Repeated queries are not independent.'}
