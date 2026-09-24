import numpy as np
import pytest
from scripts.eval_pmd_expanded import remaining_ids,paired_counts,validate_row


def test_excluded_queries_never_reused():
    assert remaining_ids(5,[1,3])==[0,2,4]
    with pytest.raises(ValueError):remaining_ids(5,[1,1])
    with pytest.raises(ValueError):remaining_ids(5,[5])


def test_paired_identity_not_just_net():
    result=paired_counts([True,False],[False,True],[3,8])
    assert result['net']==0 and result['corrections']==[8] and result['regressions']==[3]


def test_resume_row_validation():
    source=dict(candidates=np.arange(20),labels=np.arange(20)==2)
    row=dict(source,query=np.array(4),scores=np.zeros((4,20)))
    validate_row(row,4,source)
    with pytest.raises(ValueError):validate_row(row,5,source)
    row['scores'][0,0]=np.nan
    with pytest.raises(ValueError):validate_row(row,4,source)
