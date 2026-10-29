"""Calibration-only layer proposals, followed by matched frozen-BoQ controls."""
import itertools

import numpy as np

LAYERS=(4,6,8,10,11,12)
REFERENCE=(8,10,12)
COMBINATIONS=tuple((*p,12) for p in itertools.combinations(LAYERS[:-1],2))
POLICY=dict(revision=1,layers=list(LAYERS),reference_layers=list(REFERENCE),seed=42,
    native_grid=20,candidate_k=20,min_matches=3,token_lru_images=64,
    train_steps=2048,rho=.025,lr=1e-4,weight_decay=0.,clip=1.,microbatch=4,
    min_correction_places=3,min_new_correction_places=1,
    screening='Official full-gallery top20 with original GSV GPS filter; native400 raw patches, unit FP16 tokens re-normalized for FP32 MNN scoring',
    combination='Equal mean of within-query candidate percentile ranks for two auxiliary layers plus L12; screening proxy, NOT the trained fusion network',
    selection='Calibration only: at least3 corrected official-error places, at least1 new correction place vs81012, total correct >=81012; maximize correction places, net vs official, new places, then smaller layer tuple',
    training='If calibration gate passes, two NEW fresh runs only: selected triple and L12/L12/L12; reuse sealed E3 rho0.025 as81012 control; identical initial parameters/schedule/loss/budget',
    eval_sizes=[280,322],primary_eval_size=322,
    scope='Exploratory official-layer selection on exposed GSV partitions, then exposed MSLS/Pitts validation; not proof that local-MNN proxy predicts trainable-fusion gain or independent test')


def fusion_indices(scores,valid,layers):
    """No GT enters predictions; exact ties retain official candidate order."""
    scores,valid=np.asarray(scores),np.asarray(valid,bool)
    if scores.ndim!=3 or scores.shape!=valid.shape or scores.shape[1]!=len(LAYERS) or not np.isfinite(scores).all():
        raise ValueError('Expected query/layer/candidate scores')
    if len(layers)!=3 or layers[-1]!=12 or len(set(layers))!=3 or any(x not in LAYERS for x in layers):
        raise ValueError('Expected two distinct auxiliary layers plus12')
    indexes=[LAYERS.index(x) for x in layers]
    chosen=scores[:,indexes];ok=valid[:,indexes]
    masked=np.where(ok,chosen,-np.inf)
    percentile=(masked[:,:,:,None]>masked[:,:,None,:]).sum(-1)/max(1,scores.shape[-1]-1)
    combined=percentile.mean(1);joint=ok.all(1)
    rank=np.where(joint,combined,-np.inf).argmax(1)
    rank[~joint.any(1)]=0
    return rank


def summarize(candidates,positive,groups,scores,valid,query_ids):
    """Return a partition only; choose_layers must never receive evaluation."""
    candidates,positive,groups,query_ids=map(np.asarray,(candidates,positive,groups,query_ids))
    if candidates.ndim!=2 or len(candidates)==0 or any(x.shape!=(len(candidates),) for x in (positive,groups,query_ids)):
        raise ValueError('Unmatched query identities')
    baseline=candidates[:,0]==positive;truth=candidates==positive[:,None]
    result=dict(queries=len(candidates),official_correct=int(baseline.sum()),
        official_error_places=len(set(groups[~baseline].tolist())),reachable_errors=int((~baseline & truth.any(1)).sum()),
        variants={})
    for layers in COMBINATIONS:
        prediction=fusion_indices(scores,valid,layers)
        correct=truth[np.arange(len(truth)),prediction]
        fixes=~baseline & correct;regressions=baseline & ~correct
        key='/'.join(map(str,layers))
        result['variants'][key]=dict(layers=list(layers),correct=int(correct.sum()),
            corrections=query_ids[fixes].tolist(),regressions=query_ids[regressions].tolist(),
            correction_places=sorted(set(groups[fixes].tolist())),
            net_vs_official=int(correct.sum()-baseline.sum()))
    result['single_layers']={}
    for i,layer in enumerate(LAYERS):
        selected=np.where(valid[:,i],scores[:,i],-np.inf).argmax(1)
        selected[~valid[:,i].any(1)]=0
        correct=truth[np.arange(len(truth)),selected]
        pm=truth & valid[:,i];nm=~truth & valid[:,i]
        mask=pm.any(1)&nm.any(1)
        margins=(np.where(pm,scores[:,i],-np.inf).max(1)[mask]
                 -np.where(nm,scores[:,i],-np.inf).max(1)[mask])
        result['single_layers'][str(layer)]=dict(correct=int(correct.sum()),
            corrections=query_ids[~baseline & correct].tolist(),
            regressions=query_ids[baseline & ~correct].tolist(),
            mean_positive_negative_margin=float(margins.mean()) if len(margins) else None,
            valid_candidate_fraction=float(valid[:,i].mean()))
    return result


def choose_layers(calibration):
    if 'evaluation' in calibration or 'calibration' in calibration:
        raise ValueError('Pass calibration partition ONLY')
    reference=calibration['variants']['8/10/12'];reference_places=set(reference['correction_places'])
    eligible=[]
    for row in calibration['variants'].values():
        layers=tuple(row['layers']);places=set(row['correction_places'])
        new=len(places-reference_places)
        if layers!=REFERENCE and len(places)>=POLICY['min_correction_places'] and new>=POLICY['min_new_correction_places'] and row['correct']>=reference['correct']:
            eligible.append((row,new))
    if not eligible:
        return dict(gate=False,layers=list(REFERENCE),reason='NO_CALIBRATION_SUPPORTED_ALTERNATIVE; no new training')
    row,new=max(eligible,key=lambda item:(len(item[0]['correction_places']),item[0]['net_vs_official'],item[1],tuple(-x for x in item[0]['layers'])))
    return dict(gate=True,layers=row['layers'],new_correction_places_vs_reference=new,
        reason='CALIBRATION_LOCKED_PROXY_PROPOSAL; evaluation cannot change selection')


def selected_features(self,images):
    from src.slgd_layer_protocol import LayerCapture
    layers=tuple(self.fusion_layers)
    if len(layers)!=3 or layers[-1]!=12 or any(x not in LAYERS for x in layers):
        raise ValueError('Invalid fixed fusion layers')
    capture=LayerCapture(self.visual.backbone,tuple(sorted(set(layers))))
    try:
        features,_=capture(images)
        return tuple(features[x] for x in layers)
    finally:capture.close()
