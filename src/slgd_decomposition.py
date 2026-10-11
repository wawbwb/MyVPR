"""Fixed-checkpoint branch diagnostics, not fusion-weight/model selection."""
import numpy as np


HEADS=('initial','retrieval','last12','mid10')
BETA=.2
GLOBAL_DIM=12288
LOCAL_DIM=2048
VARIANTS=('ru_global','initial_hybrid')+tuple(
    mode+'_'+branch for mode in HEADS[1:]
    for branch in ('global','hybrid','ru_global_trained_local'))


def descriptor_variant(descriptors,name):
    """Keep the original sqrt weights; no alpha sweep or score calibration."""
    if name=='ru_global':
        return descriptors['initial'][...,:GLOBAL_DIM]/np.float32(np.sqrt(1-BETA))
    if name=='initial_hybrid':return descriptors['initial']
    for mode in HEADS[1:]:
        if name==mode+'_global':
            return descriptors[mode][...,:GLOBAL_DIM]/np.float32(np.sqrt(1-BETA))
        if name==mode+'_hybrid':return descriptors[mode]
        if name==mode+'_ru_global_trained_local':
            return np.concatenate((descriptors['initial'][...,:GLOBAL_DIM],
                                   descriptors[mode][...,GLOBAL_DIM:]),axis=-1)
    raise ValueError('Unknown fixed diagnostic variant: '+name)


def validate_shard(arrays,indices):
    if set(arrays)!=set(HEADS)|{'indices','coordinates'}:
        raise ValueError('Diagnostic shard fields differ')
    if arrays['indices'].dtype!=np.int64 or not np.array_equal(arrays['indices'],indices):
        raise ValueError('Diagnostic shard row identities differ')
    c=arrays['coordinates']
    if c.shape!=(len(indices),2) or c.dtype!=np.float64 or not np.isfinite(c).all():
        raise ValueError('Invalid diagnostic coordinates')
    for head in HEADS:
        d=arrays[head]
        if d.shape!=(len(indices),GLOBAL_DIM+LOCAL_DIM) or d.dtype!=np.float32 or not np.isfinite(d).all():
            raise ValueError('Invalid diagnostic descriptor: '+head)
        for block,target in ((d[...,:GLOBAL_DIM],1-BETA),(d[...,GLOBAL_DIM:],BETA)):
            if np.max(np.abs(np.square(block).sum(-1)-target))>1e-4:
                raise ValueError('Weighted branch normalization differs: '+head)


def paired_summary(outcomes,groups,hard,cutoff):
    groups=np.asarray(groups);hard=np.asarray(hard,bool)
    result={}
    comparisons=[('initial_hybrid','ru_global')]
    for mode in HEADS[1:]:
        comparisons.extend([(mode+'_global','ru_global'),(mode+'_hybrid','initial_hybrid'),
            (mode+'_hybrid',mode+'_global'),(mode+'_ru_global_trained_local','ru_global'),
            (mode+'_hybrid',mode+'_ru_global_trained_local')])
    for role,mask in [('calibration',groups<cutoff),('evaluation',groups>=cutoff),
                      ('all',np.ones(len(groups),bool))]:
        ru=np.asarray(outcomes['ru_global'],bool)
        variants={}
        for name,values in outcomes.items():
            ok=np.asarray(values,bool)
            if ok.shape!=groups.shape or hard.shape!=groups.shape:
                raise ValueError('Unmatched diagnostic outcomes')
            variants[name]=dict(correct=int(ok[mask].sum()),r1=float(ok[mask].mean()),
                hard_correct=int(ok[mask & hard].sum()),
                corrections_vs_ru=int((mask & ~ru & ok).sum()),
                regressions_vs_ru=int((mask & ru & ~ok).sum()))
        pairs={}
        for candidate,reference in comparisons:
            c=np.asarray(outcomes[candidate],bool);r=np.asarray(outcomes[reference],bool)
            pairs[candidate+'_vs_'+reference]=dict(corrections=int((mask & ~r & c).sum()),
                regressions=int((mask & r & ~c).sum()),
                net=int(c[mask].sum())-int(r[mask].sum()),
                hard_net=int(c[mask & hard].sum())-int(r[mask & hard].sum()))
        result[role]=dict(queries=int(mask.sum()),hard_queries=int((mask & hard).sum()),
                          variants=variants,comparisons=pairs)
    return result
