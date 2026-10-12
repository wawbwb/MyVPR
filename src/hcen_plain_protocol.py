"""Pinned true plain-BoQ base, not a disabled gate on RU-trained weights."""
import numpy as np
from src.models.hcen_vpr import HCENVPR

PLAIN_SHA = 'dba5770cd3351d5354decc2bd385a831107591cfa392a0f0d4e66e95bf0c316d'
PLAIN_CHECKPOINT = 'logs/dinov2_vitb14/BoQ/version_8/checkpoints/epoch(29)_step(46890)_R1[0.9122]_R5[0.9568].ckpt'
MODES = ('last_only', 'hierarchy')
POLICY = dict(revision=1, seed=42, smoke_steps=8, screen_steps=512,
    backbone='True locally trained plain DINOv2-ViT-B/14 + BoQ, no RU module',
    layers=[8,10,12],dim=128,inner=32,state=4,feature_relative_cap=.1,
    lr=1e-4,weight_decay=0.,clip=1.,precision='fp32',microbatch=4,
    auxiliary_weight=0.,rank_weight=0.,historical_msls_correct=675,
    feature_layout='Native NCHW contiguous before plain-BoQ readout; match legacy backbone output, not hook memory strides',
    data='Reuse fixed original GSV images/views/schedule and historical RU-mined training indices, NOT RU descriptors',
    selection='Pinned historical version8 epoch29; fixed last512; no new benchmark model selection',
    scope='Exploratory plain-base transfer on previously exposed queries. Different base training histories; not a pure RU-gate causal ablation or independent final validation')


class PlainHCENVPR(HCENVPR):
    """No new weights: preserve the original plain backbone's readout layout."""
    def readout(self,features):
        return super().readout(features.contiguous())


def canonical_features(features):
    return tuple(f.contiguous() for f in features)


def validate_checkpoint(config, state):
    if config.get('distillation',{}).get('enabled',False):
        raise ValueError('Plain checkpoint must not enable distillation')
    if config.get('backbone',{}).get('class') != 'DinoV2' or config['backbone'].get('params',{}).get('backbone_name') != 'dinov2_vitb14':
        raise ValueError('Expected original DINOv2 ViT-B/14')
    if config.get('aggregator',{}).get('class') != 'BoQ':
        raise ValueError('Expected original BoQ')
    for name in ('crop_semantic_film','residual_clip_fusion'):
        if config['backbone']['params'].get(name,{}).get('enabled',False):
            raise ValueError('Nonplain backbone branch')
    if not state or any(not k.startswith(('backbone.','aggregator.')) for k in state):
        raise ValueError('Unexpected checkpoint tensor: not a plain backbone/BoQ')
    if not any(k.startswith('backbone.') for k in state) or not any(k.startswith('aggregator.') for k in state):
        raise ValueError('Incomplete plain checkpoint')


def paired_report(outcomes, baseline, groups, cutoff):
    if set(outcomes) != set(MODES):raise ValueError('Both matched neck controls required')
    baseline=np.asarray(baseline,bool);groups=np.asarray(groups)
    values={m:np.asarray(v,bool) for m,v in outcomes.items()}
    if baseline.ndim!=1 or groups.shape!=baseline.shape or any(v.shape!=baseline.shape for v in values.values()):
        raise ValueError('Unmatched paired queries')
    report={}
    for role,mask in [('calibration',groups<cutoff),('evaluation',groups>=cutoff),('all',np.ones(len(groups),bool))]:
        variants={}
        for mode,ok in dict(baseline=baseline,**values).items():
            variants[mode]=dict(correct=int(ok[mask].sum()),r1=float(ok[mask].mean()) if mask.any() else None,
                corrections=np.flatnonzero(mask&~baseline&ok).tolist(),regressions=np.flatnonzero(mask&baseline&~ok).tolist())
        c,r=values['hierarchy'],values['last_only']
        fixes=np.flatnonzero(mask&~r&c).tolist();damage=np.flatnonzero(mask&r&~c).tolist()
        report[role]=dict(queries=int(mask.sum()),variants=variants,
            hierarchy_vs_last_only=dict(corrections=fixes,regressions=damage,net=len(fixes)-len(damage)))
    return report
