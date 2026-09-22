from scripts.wppr_pitts_transfer import paired


def test_paired_counts_and_original_ids():
    rows=[dict(query=7,group='a',selected_correct=True,full20_correct=False),
          dict(query=11,group='b',selected_correct=False,full20_correct=True),
          dict(query=15,group='c',selected_correct=True,full20_correct=True)]
    result=paired(rows,'full20_correct')
    assert result['corrections']==[7]
    assert result['regressions']==[11]
    assert result['selected_correct']==result['baseline_correct']==2
    assert result['delta_r1_pp']==0
