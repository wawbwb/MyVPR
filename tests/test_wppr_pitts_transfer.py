from scripts.wppr_pitts_transfer import paired
import numpy as np
import pytest
from scripts.wppr_pitts_transfer import reconcile_candidates,compatible_legacy,LEGACY_CODE,TIE_POLICY
from scripts.wppr_pitts_transfer import TIE_CODE,ACCURACY_POLICY,validate_saved


def test_paired_counts_and_original_ids():
    rows=[dict(query=7,group='a',selected_correct=True,full20_correct=False),
          dict(query=11,group='b',selected_correct=False,full20_correct=True),
          dict(query=15,group='c',selected_correct=True,full20_correct=True)]
    result=paired(rows,'full20_correct')
    assert result['corrections']==[7]
    assert result['regressions']==[11]
    assert result['selected_correct']==result['baseline_correct']==2
    assert result['delta_r1_pp']==0


def tie_inputs():
    expected=np.arange(44);actual=expected.copy();actual[28:30]=[29,28]
    cached=np.linspace(1,0,50);cached[29]=cached[28]-1e-8
    scores=cached.copy();scores[29]=scores[28]+1e-8
    return expected,actual,scores,cached,np.zeros(512),np.full(512,2e-7)


def test_near_tie_preserves_original_order():
    args=tie_inputs();ids,audit=reconcile_candidates(*args)
    assert np.array_equal(ids,args[0])
    assert audit['ranks_1based']==[29,30]


@pytest.mark.parametrize('failure',['set','descriptor','gap','cached_order','nan'])
def test_unsafe_differences_rejected(failure):
    args=list(tie_inputs())
    if failure=='set':args[1][29]=49
    if failure=='descriptor':args[4][0]=1e-3
    if failure=='gap':args[2][29]+=1e-3
    if failure=='cached_order':args[3][29]+=1e-3
    if failure=='nan':args[2][0]=np.nan
    with pytest.raises(ValueError):reconcile_candidates(*args)


def test_migration_only_known_code_and_identical_sources():
    new=dict(head='abc',code={'scripts/wppr_pitts_transfer.py':'new','src/wppr_runtime.py':'same'},tie_policy=TIE_POLICY)
    old=dict(head='abc',code={'scripts/wppr_pitts_transfer.py':LEGACY_CODE,'src/wppr_runtime.py':'same'})
    assert compatible_legacy(old,new)
    old['head']='other'
    assert not compatible_legacy(old,new)


def test_tie_contract_migration_retains_runtime_and_sources():
    new=dict(head='abc',accuracy_policy=ACCURACY_POLICY,code={'scripts/wppr_pitts_transfer.py':'new','src/wppr_runtime.py':'same'})
    old=dict(head='abc',tie_policy=TIE_POLICY,code={'scripts/wppr_pitts_transfer.py':TIE_CODE,'src/wppr_runtime.py':'same'})
    assert compatible_legacy(old,new)
    old['code']['src/wppr_runtime.py']='changed'
    assert not compatible_legacy(old,new)


def test_reused_shards_match_frozen_source():
    prediction=-np.arange(44,dtype=float);keep=np.arange(12)
    ref=dict(scores=prediction,labels=np.zeros(44,dtype=bool))
    row=dict(keep=keep,prediction=prediction,teacher=prediction.copy(),labels=ref['labels'].copy(),scores=prediction[:12].copy())
    validate_saved(row,ref)
    row['teacher'][20]+=1
    with pytest.raises(ValueError):validate_saved(row,ref)
