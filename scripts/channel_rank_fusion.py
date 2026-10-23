"""CPU-only fixed rank fusion of archived independent retrieval branches."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np


def fuse(left,right,k=60):
    """Equal reciprocal ranks. Absent items contribute zero; label-free ties."""
    left,right=np.asarray(left),np.asarray(right)
    if left.ndim!=2 or left.shape!=right.shape or left.shape[1]!=20 or k!=60:
        raise ValueError('Expected paired Qx20 rankings and fixed k60')
    if not all(np.issubdtype(x.dtype,np.integer) for x in (left,right)) or (left<0).any() or (right<0).any():
        raise ValueError('Expected nonnegative integer reference IDs')
    result=[]
    for a,b in zip(left,right):
        if len(set(a.tolist()))!=20 or len(set(b.tolist()))!=20:raise ValueError('Duplicate reference')
        score={};priority={int(v):i for i,v in enumerate(a)}
        for row in (a,b):
            for rank,value in enumerate(row,1):score[int(value)]=score.get(int(value),0.)+1/(k+rank)
        result.append(sorted(score,key=lambda v:(-score[v],priority.get(v,20),v))[:20])
    return np.asarray(result,dtype=np.int64)


def oracle(left,right,gt):
    """Label-dependent opportunity counts only, never used in actual fusion."""
    a,b=np.asarray(left),np.asarray(right)
    hit_a=np.array([row[0] in g for row,g in zip(a,gt)])
    hit_b=np.array([row[0] in g for row,g in zip(b,gt)])
    deep_a=np.array([np.isin(row,g).any() for row,g in zip(a,gt)])
    deep_b=np.array([np.isin(row,g).any() for row,g in zip(b,gt)])
    return dict(top1_oracle_correct=int((hit_a|hit_b).sum()),
        branch_only_correct_ids=np.flatnonzero(~hit_a&hit_b).tolist(),
        baseline_only_correct_ids=np.flatnonzero(hit_a&~hit_b).tolist(),
        union_top20_reachable=int((deep_a|deep_b).sum()),
        added_top20_reachable_ids=np.flatnonzero(~deep_a&deep_b).tolist())


def main():
    from scripts import original_channel_transfer as transfer
    from scripts import spatial_adapter_repeat as original
    from scripts import eval_clip_key_study as metrics
    from src.semantic_region_protocol import read,write,sha,owned_run
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'doc/channel_rank_fusion_v1')
    a=p.parse_args();output=a.output.resolve()
    current=ROOT/'logs/original_channel_transfer_v1/eval'
    prior=ROOT/'logs/spatial_adapter_repeat_v1/eval_seed42'
    transfer.verified(current);original.verified(prior)
    c=transfer.configuration();sets=metrics.populations(c,transfer.old.key.open_plan(c))
    files=('scripts/channel_rank_fusion.py','tests/test_channel_rank_fusion.py','doc/CHANNEL_RANK_FUSION_V1.md')
    inputs=[p for folder in (current,prior) for p in folder.glob('*.json')]
    contract=dict(schema='channel_rank_fusion_v1',k=60,weights=[1,1],seed_source=42,
        source_hashes={f:sha(ROOT/f) for f in files},
        inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        scope='CPU archived top20, two independent branches, no tuning/training; repeated development sets',
        oracle_is_not_deployable=True)
    with owned_run(output,contract,False):
        all_results={}
        for d in sets:
            tag=d['tag'];now=read(current/(tag+'.json'));old=read(prior/(tag+'.json'))
            expected=dict(references=d['references'],queries=d['queries'],order=metrics.order_digest(d['names']),gt=metrics.gt_digest(d['ground_truth']))
            if now['dataset']!=expected:raise ValueError('Current population changed')
            for x,y in [('references','references'),('queries','queries'),('image_order_sha256','order'),('ground_truth_sha256','gt')]:
                if old['dataset'][x]!=expected[y]:raise ValueError('Original population mismatch')
            variants=dict(innovation1=now['variants']['innovation1'],official=old['variants']['official'],original_channel=old['variants']['channel'])
            for r in variants.values():
                if metrics.retrieval_report(np.array(r['predictions']),d['ground_truth'],d['references'])!=r:
                    raise ValueError('Archived outcomes do not match GT')
            baseline=np.array(variants['innovation1']['predictions'])
            for branch in ('official','original_channel'):
                pred=np.array(variants[branch]['predictions'])
                variants['fusion_'+branch]=metrics.retrieval_report(fuse(baseline,pred),d['ground_truth'],d['references'])
            result=dict(correct={n:r['correct'] for n,r in variants.items()},
                vs_innovation1={n:metrics.paired_report(variants['innovation1'],variants[n]) for n in ('fusion_official','fusion_original_channel')},
                channel_vs_official_fusion=metrics.paired_report(variants['fusion_official'],variants['fusion_original_channel']),
                opportunity={n:oracle(baseline,np.array(variants[n]['predictions']),d['ground_truth']) for n in ('official','original_channel')})
            write(output/(tag+'.json'),dict(dataset=expected,variants=variants,comparison=result))
            all_results[tag]=result
            print(tag,result['correct'],flush=True)
        gain=all_results['pitts30k-val']['vs_innovation1']['fusion_original_channel']['net']>0 and all(
            all_results[t]['vs_innovation1']['fusion_original_channel']['net']>=0 for t in ('msls-val','gsv-adaptation-holdout'))
        specific=gain and all_results['pitts30k-val']['channel_vs_official_fusion']['net']>0 and all(
            all_results[t]['channel_vs_official_fusion']['net']>=0 for t in ('msls-val','gsv-adaptation-holdout'))
        checks=dict(inputs_unchanged=all(sha(ROOT/n)==h for n,h in contract['inputs'].items()))
        write(output/'summary.json',dict(results=all_results,checks=checks,anchor_gain=gain,channel_specific_signal=specific,
            training_started=False,stable_gain_proven=False,blind_test_used=False,
            verdict='EXPLORATORY_CHANNEL_FUSION_SIGNAL' if specific else 'NO_CHANNEL_SPECIFIC_FUSION_SIGNAL'))
        if not all(checks.values()):raise ValueError('Inputs mutated')
        write(output/'completed.json',dict(complete=True,files={p.name:sha(p) for p in output.glob('*.json') if p.name!='completed.json'}))


if __name__=='__main__':main()
