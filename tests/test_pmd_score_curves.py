import pytest
from scripts.audit_pmd_score_curves import curve_report


def test_perfect_ranking():
    r,c=curve_report([False,False,True,True],[.1,.2,.8,.9])
    assert r['auroc']==1 and r['average_precision']==1
    assert all(v['rejection_recall']==1 and v['actual_false_rejection']==0 for v in r['recall_at_false_rejection_caps'].values())


def test_ties_not_split_to_invent_operating_point():
    r,c=curve_report([False,True],[.5,.5])
    assert r['auroc']==.5
    assert all(v['rejection_recall']==0 for v in r['recall_at_false_rejection_caps'].values())
    with pytest.raises(ValueError):curve_report([True,True],[.1,.2])
