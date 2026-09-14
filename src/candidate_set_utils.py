"""CPU utilities for a non-semantic, city-disjoint adaptation screen."""
import hashlib
import numpy as np


def stable_key(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def choose_views(records):
    """Distinct panoramas; earliest is query, two latest are references."""
    unique={}
    for r in sorted(records,key=lambda r:(r['date'],r['path'])):
        unique.setdefault(r['panoid'],r)
    ordered=list(unique.values())
    if len(ordered)<3:return None
    return ordered[0],ordered[-2:]


def make_plan(groups,train_queries=1024,eval_queries=512,train_places=4096):
    cities={}
    for label,rows in groups.items():
        chosen=choose_views(rows)
        if chosen is not None:
            cities.setdefault(rows[0]['city'],[]).append((label,chosen))
    eligible=sorted([c for c,v in cities.items() if len(v)>=max(eval_queries,1024)],key=stable_key)
    if len(eligible)<2:raise ValueError('Need two cities with >=1024 eligible places (3 distinct panoramas/place)')
    dev,test=eligible[:2]
    training=[x for c,v in cities.items() if c not in [dev,test] for x in v]
    training=sorted(training,key=lambda x:stable_key(x[0]))[:train_places]
    if len(training)<max(train_queries,1024):raise ValueError('Insufficient training places outside held-out cities')
    result={}
    for split,pool,count in [('train',training,train_queries),('dev',cities[dev],eval_queries),('test',cities[test],eval_queries)]:
        pool=sorted(pool,key=lambda x:stable_key(x[0]));database=[];queries=[]
        for label,(q,refs) in pool:
            database.extend([dict(r,label=label) for r in refs])
        reference_panos={(r['city'],r['panoid']) for r in database}
        for label,(q,refs) in pool:
            if (q['city'],q['panoid']) not in reference_panos:queries.append(dict(q,label=label))
            if len(queries)==count:break
        if len(queries)!=count:raise ValueError('Too few queries after preventing cross-place panorama leakage')
        result[split]={'database':database,'queries':queries}
    result['split_info']={'dev_city':dev,'test_city':test,'train_places':len(training),
        'warning':'GSV city-disjoint for this adaptation only; official pretrained model may have seen all cities. Place-ID GT, not official benchmark recall.'}
    return result


def summary(scores,labels,base):
    scores=np.asarray(scores);labels=np.asarray(labels,bool);base=np.asarray(base)
    if scores.shape!=labels.shape or scores.shape!=base.shape or not np.isfinite(scores).all():raise ValueError('Invalid score table')
    order=np.argsort(-scores,axis=1,kind='stable');old=np.argsort(-base,axis=1,kind='stable')
    hit=np.take_along_axis(labels,order,1);oldhit=np.take_along_axis(labels,old,1)[:,0]
    return {'queries':len(scores),'correct':int(hit[:,0].sum()),'recall':{str(k):float(hit[:,:k].any(1).mean()) for k in [1,5,10,20]},
        'reachable':int(labels.any(1).sum()),'corrected':np.flatnonzero(hit[:,0]&~oldhit).tolist(),
        'regressed':np.flatnonzero(~hit[:,0]&oldhit).tolist(),'top1_candidate_positions':order[:,0].tolist()}
