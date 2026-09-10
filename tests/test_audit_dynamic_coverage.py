import numpy as np
import pytest

from scripts.audit_dynamic_coverage import GROUPS, VARIANTS, group, paired, sample_queries, summarize, safe_image


@pytest.mark.parametrize('value,index', [(0, 0), (.001, 1), (.0499, 1), (.05, 2), (.15, 3), (.30, 4), (1, 4)])
def test_boundaries(value, index):
    assert group(value) == GROUPS[index]


@pytest.mark.parametrize('value', [-.1, 1.01, float('nan'), float('inf')])
def test_invalid_coverage(value):
    with pytest.raises(ValueError): group(value)


def test_paired_uses_global_query_indices():
    result = paired([0, 1, 0, 1], [0, 0, 1, 1], [1, 2])
    assert result['left_only_query_ids'] == [1]
    assert result['right_only_query_ids'] == [2]
    assert result['net'] == 0
    assert paired([], [], [])['delta_r1_pp'] is None


def test_all_queries_partitioned_and_empty_groups_are_null():
    hits = {v: [1, 0, 1, 0] for v in VARIANTS}
    hits['aligned'] = [1, 1, 0, 0]
    result = summarize([0, .01, .2, .8], hits)
    assert sum(result[g]['n'] for g in GROUPS) == result['all']['n'] == 4
    assert result['all']['vs_zero_bias']['aligned']['net'] == 0
    assert result[GROUPS[2]]['r1']['aligned'] is None


def test_sampling_is_identity_based_and_not_outcome_based():
    paths = ['cph/query/images/a.jpg', 'cph/query/images/b.jpg', 'sf/query/images/c.jpg']
    selected = sample_queries(paths, [.2, .2, .2], 2)[GROUPS[3]]
    perm = [2, 0, 1]
    shuffled = [paths[i] for i in perm]
    again = sample_queries(shuffled, [.2, .2, .2], 2)[GROUPS[3]]
    assert [paths[i] for i in selected] == [shuffled[i] for i in again]
    assert all(not ids for ids in sample_queries(paths, [.2]*3, 0).values())


def test_path_escape_refused(tmp_path):
    with pytest.raises(ValueError): safe_image(tmp_path, '../outside.jpg')
