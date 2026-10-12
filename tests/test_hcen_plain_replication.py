"""Plain-base replication fairness and immutable seed42 loop regressions."""
import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from src.hcen_plain_protocol import POLICY as BASE_POLICY,PlainHCENVPR
from src.hcen_plain_replication import SEEDS,effective_policy,comparable_contract,aggregate_reports

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('s',SEEDS)
def test_same_policy_except_initialization_seed(s):
    p=effective_policy(s)
    assert p['seed']==s
    assert {k:v for k,v in p.items() if k!='seed'}=={k:v for k,v in BASE_POLICY.items() if k!='seed'}
    assert p['screen_steps']==512 and p['auxiliary_weight']==p['rank_weight']==0
    assert BASE_POLICY['seed']==42
    with pytest.raises(ValueError):effective_policy(45)


def contract(s):
    return dict(mode='hierarchy',stage='screen',policy=effective_policy(s),initialization_seed=s,
        checkpoint_sha256='plain',source_completed_sha256='source',cache_completed_sha256='cache',
        metadata={'city':'same'},train_places=['same'],original_code={'model':'same'},benchmark_metadata={'msls':'same'},
        versions={'torch':'same'},schedule=[[0,1]],baseline_completed_sha256='baseline',code={'runner':str(s)})


def test_semantic_contract_matches_seed42_and_rejects_data_weight_or_base_changes():
    a=contract(42);b=contract(43)
    assert comparable_contract(a)==comparable_contract(b)
    for field,value in [('checkpoint_sha256','ru'),('schedule',[[2,3]]),('baseline_completed_sha256','other'),('metadata',{'city':'other'})]:
        c=copy.deepcopy(b);c[field]=value
        assert comparable_contract(a)!=comparable_contract(c)
    c=copy.deepcopy(b);c['policy']['feature_relative_cap']=.2
    assert comparable_contract(a)!=comparable_contract(c)
    b['initialization_seed']=44
    with pytest.raises(ValueError):comparable_contract(b)
    assert a['policy']['seed']==42


def reports():
    result={}
    for s,net in zip(SEEDS,(1,3,-1)):
        part=dict(queries=100,variants=dict(baseline={'correct':90},last_only={'correct':90},hierarchy={'correct':90+net}),
                  hierarchy_vs_last_only={'net':net})
        result[s]={d:copy.deepcopy(part) for d in ('gsv_calibration','gsv_evaluation','gsv_all','msls-val','pitts30k-val')}
    return result


def test_three_seed_mean_std_no_best_seed_selection():
    out=aggregate_reports(reports());p=out['pitts30k-val'];v=p['variants']['hierarchy']
    assert v['correct_by_seed']=={42:91,43:93,44:89}
    assert v['mean_delta_correct']==1 and v['sample_std_delta_correct']==2
    assert not v['positive_every_seed'] and not v['negative_every_seed']
    assert p['hierarchy_vs_last_only']['net_by_seed']=={42:1,43:3,44:-1}
    assert p['hierarchy_vs_last_only']['mean_delta_pp']==1
    json.dumps(out,allow_nan=False)


def test_reject_missing_seed_or_changed_query_baseline():
    r=reports();r.pop(44)
    with pytest.raises(ValueError):aggregate_reports(r)
    r=reports();r[44]['gsv_all']['queries']=99
    with pytest.raises(ValueError):aggregate_reports(r)
    r=reports();r[43]['msls-val']['variants']['baseline']['correct']=89
    with pytest.raises(ValueError):aggregate_reports(r)


class Visual(nn.Module):
    def __init__(self):
        super().__init__();self.semantic_region_gate=None;self.spatial_attn_head=None
        self.aggregator=nn.Sequential(nn.Flatten(),nn.Linear(8*4*4,16))


def test_different_necks_same_plain_zero_start():
    torch.manual_seed(42);visual=Visual();maps=[torch.randn(2,8,4,4) for _ in range(3)]
    hashes={};outputs=[]
    for s in SEEDS:
        hashes[s]=set()
        for mode in ['last_only','hierarchy']:
            torch.manual_seed(s);model=PlainHCENVPR(visual,mode,channels=8,dim=8,inner=8,state=2)
            h=hashlib.sha256()
            for p in model.neck.parameters():h.update(p.detach().numpy().tobytes())
            hashes[s].add(h.hexdigest());outputs.append(model.aggregate(*maps)[0].detach())
    assert all(len(v)==1 for v in hashes.values()) and len(set.union(*hashes.values()))==3
    assert all(torch.equal(outputs[0],x) for x in outputs[1:])


def update_loop(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    return next(n for n in f.body if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple))


def test_old_plain_runner_immutable_and_updates_exactly_identical():
    original=ROOT/'scripts/hcen_plain_base.py';new=ROOT/'scripts/hcen_plain_multiseed.py'
    assert hashlib.sha256(original.read_bytes().replace(b'\r\n',b'\n')).hexdigest()=='ee277187730ee16dde319a7b4278dac09de016a30353fba3849eea7bfe7a654d'
    assert ast.dump(update_loop(original),include_attributes=False)==ast.dump(update_loop(new),include_attributes=False)
    text=new.read_text(encoding='utf-8')
    assert 'seed(args.seed);model=PlainHCENVPR' in text
    assert 'context(args,args.mode,args.stage)' in text
    assert 'load_plain(args)' in text and 'forward_plain_batch' in text
    assert 'reference_identity(args)' in text and 'if result[\'baseline\']!=original[\'baseline\']' in text


def test_launcher_both_smokes_before_four_512_runs():
    raw=(ROOT/'scripts/run_hcen_plain_multiseed.sh').read_bytes();text=raw.decode()
    assert b'\r' not in raw and 'CUDA_VISIBLE_DEVICES=1' in text
    assert 'logs/query_relation_screen_v1.lock' in text and 'flock -n 9' in text
    assert 'for SEED in 43 44; do train_stage "$SEED" smoke; done' in text
    assert 'for SEED in 43 44; do train_stage "$SEED" screen; done' in text
    assert 'for MODE in last_only hierarchy' in text and 'SMOKE_PASS' in text
    assert 'scripts/hcen_plain_multiseed.py aggregate' in text
    assert 'git pull' not in text and 'pip install' not in text
