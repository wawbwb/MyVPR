"""Replication changes initialization, not the sealed loss/update loop or data."""
import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from src.hcen_aux_ablation import ARMS, POLICY as BASE_POLICY
from src.hcen_replication import SEEDS, effective_policy, comparable_contract, aggregate_reports
from src.models.hcen_vpr import HCENVPR

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('seed', SEEDS)
def test_only_policy_seed_changes(seed):
    policy = effective_policy(seed)
    assert policy['seed'] == seed
    assert {k:v for k,v in policy.items() if k != 'seed'} == {k:v for k,v in BASE_POLICY.items() if k != 'seed'}
    assert policy['screen_steps'] == 512 and policy['rank_weight'] == 0
    with pytest.raises(ValueError): effective_policy(45)


def contract(seed):
    return dict(arm='hierarchy_lowaux', mode='hierarchy', auxiliary_weight=.001,
        stage='screen', policy=effective_policy(seed), ru_sha256='ru', cache_completed_sha256='cache',
        source_completed_sha256='source', previous={'paper':'same'}, metadata={'city':'same'},
        benchmark_metadata={'msls':'same'}, train_places=['same'], schedule=[[0,1]],
        original_code={'model':'same'}, versions={'torch':'same'}, code={'runner':str(seed)},
        initialization_seed=seed)


def test_matched_contracts_reject_scientific_changes_not_seed():
    base = contract(42); new = contract(43)
    assert comparable_contract(base) == comparable_contract(new)
    for key, value in [('schedule',[[2,3]]), ('train_places',['other']), ('ru_sha256','other'),
                       ('auxiliary_weight',.1), ('benchmark_metadata',{'msls':'other'})]:
        altered=copy.deepcopy(new); altered[key]=value
        assert comparable_contract(base) != comparable_contract(altered)
    bad=copy.deepcopy(new);bad['initialization_seed']=44
    with pytest.raises(ValueError): comparable_contract(bad)
    assert new['policy']['seed'] == 43  # No mutation during canonicalization.


def reports():
    result={}
    for s, gain in zip(SEEDS,(1,2,-1)):
        part=dict(queries=100, variants={'ru':{'correct':90}, **{a:{'correct':90+gain} for a in ARMS}},
                  comparisons={'hierarchy_noaux_vs_last_noaux':{'net':gain}})
        result[s]={d:copy.deepcopy(part) for d in ('gsv_calibration','gsv_evaluation','gsv_all','msls-val','pitts30k-val')}
    return result


def test_aggregate_preserves_all_seeds_and_sample_std():
    out=aggregate_reports(reports());v=out['pitts30k-val']['variants']['hierarchy_noaux']
    assert v['delta_correct_by_seed'] == {42:1,43:2,44:-1}
    assert v['mean_delta_correct'] == pytest.approx(2/3)
    assert v['sample_std_delta_correct'] == pytest.approx((7/3)**.5)
    assert v['positive_seeds'] == 2 and v['negative_seeds'] == 1
    c=out['pitts30k-val']['comparisons']['hierarchy_noaux_vs_last_noaux']
    assert not c['positive_every_seed'] and not c['negative_every_seed']
    json.dumps(out,allow_nan=False)


def test_missing_seed_and_population_or_baseline_mismatch_rejected():
    r=reports();r.pop(44)
    with pytest.raises(ValueError): aggregate_reports(r)
    for key,value in [('queries',101),('queries',0)]:
        r=reports();r[43]['gsv_all'][key]=value
        with pytest.raises(ValueError): aggregate_reports(r)
    r=reports();r[44]['msls-val']['variants']['ru']['correct']=89
    with pytest.raises(ValueError): aggregate_reports(r)


class Visual(nn.Module):
    def __init__(self):
        super().__init__();self.semantic_region_gate=None
        self.aggregator=nn.Sequential(nn.Flatten(),nn.Linear(8*4*4,16))


def test_different_necks_same_ru_zero_start_and_within_seed_matching():
    torch.manual_seed(42);visual=Visual();features=[torch.randn(2,8,4,4) for _ in range(3)]
    hashes={};descriptors=[]
    for s in SEEDS:
        hashes[s]=set()
        for arm in ARMS:
            torch.manual_seed(s);model=HCENVPR(visual,ARMS[arm]['mode'],channels=8,dim=8,inner=8,state=2)
            h=hashlib.sha256()
            for p in model.neck.parameters():h.update(p.detach().numpy().tobytes())
            hashes[s].add(h.hexdigest());descriptors.append(model.aggregate(*features)[0].detach())
    assert all(len(v)==1 for v in hashes.values())
    assert len(set.union(*hashes.values()))==3
    assert all(torch.equal(descriptors[0],d) for d in descriptors[1:])


def train_loop(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    return next(n for n in train.body if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple))


def test_original_runner_hash_and_exact_update_loop():
    original=ROOT/'scripts/train_hcen_aux_ablation.py'
    assert hashlib.sha256(original.read_bytes().replace(b'\r\n',b'\n')).hexdigest()=='c15851e4a50af08df85b5cf77c4245c576dbbeefa484ebe361ba38cfd2913316'
    assert ast.dump(train_loop(original),include_attributes=False)==ast.dump(train_loop(ROOT/'scripts/hcen_multiseed.py'),include_attributes=False)
    text=(ROOT/'scripts/hcen_multiseed.py').read_text(encoding='utf-8')
    assert 'seed(42); visual = load_visual(args)' in text
    assert 'seed(args.seed); model = HCENVPR' in text
    assert 'PlaceBatches(data, cache[\'train_indices\'], schedule[state[\'steps\']:])' in text


def test_launcher_preregisters_both_seeds_all_arms_and_smoke_gate():
    raw=(ROOT/'scripts/run_hcen_multiseed.sh').read_bytes();text=raw.decode()
    assert b'\r' not in raw and 'CUDA_VISIBLE_DEVICES=1' in text
    assert 'flock -n 9' in text and 'logs/query_relation_screen_v1.lock' in text
    assert 'for SEED in 43 44; do train_stage "$SEED" smoke; done' in text
    assert 'for SEED in 43 44; do train_stage "$SEED" screen; done' in text
    assert 'for ARM in last_noaux hierarchy_noaux last_lowaux hierarchy_lowaux' in text
    assert 'scripts/hcen_multiseed.py aggregate' in text
    assert 'SMOKE_PASS' in text and 'git pull' not in text and 'pip install' not in text
