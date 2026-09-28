import numpy as np
from scripts.audit_competitive_mechanism import groups, summarize


def test_groups_are_exhaustive_and_empty_safe():
    g = groups([False, True, True, False], [True, False, True, False])
    assert [np.flatnonzero(g[k]).tolist() for k in ('corrections', 'regressions', 'stable_correct', 'stable_error')] == [[0], [1], [2], [3]]
    m = np.ones((4, 4, 2, 3))
    m[:, 2, :, 0] -= .1
    report = summarize(m, np.zeros((4, 4)), g)
    assert abs(report['corrections']['competitive']['blocks'][0]['attention_overlap_cosine']['delta_vs_ru']['mean']+.1) < 1e-10
    empty = summarize(m, np.zeros((4, 4)), {'empty': np.zeros(4, dtype=bool)})
    assert empty['empty']['ru']['blocks'][0]['slot_effective_rank']['value']['mean'] is None
