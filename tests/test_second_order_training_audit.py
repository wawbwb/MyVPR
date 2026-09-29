import numpy as np
import pytest
import torch
from scripts.audit_second_order_training import margins, summarize


def test_self_exclusion_and_other_place_negatives():
    f = torch.eye(2).repeat_interleave(4, dim=0)
    rows = margins(f, np.zeros(8), chunk=3)
    assert all(r['correct'] and r['margin'] == 1 for r in rows)
    assert all(r['positive'] != r['query'] and r['positive']//4 == r['query']//4 for r in rows)
    assert all(r['negative']//4 != r['query']//4 for r in rows)


def test_fixed_pairs_and_batch_exposure():
    torch.manual_seed(2)
    f = torch.nn.functional.normalize(torch.randn(16, 7), dim=1)
    batches = np.repeat([0, 0, 1, 1], 4)
    base = margins(f, batches)
    g = torch.nn.functional.normalize(f+torch.randn_like(f)*.1, dim=1)
    pairs = [[r['positive'], r['negative']] for r in base]
    rows = margins(g, batches, pairs)
    for i, row in enumerate(rows):
        assert row['fixed_pair_margin'] == pytest.approx(float(g[i]@g[pairs[i][0]]-g[i]@g[pairs[i][1]]), abs=1e-6)
        assert row['batch_margin'] >= row['margin']-1e-6
        assert row['hardest_negative_in_batch'] == (batches[i] == batches[row['negative']])


def test_groups_frozen_to_ru_and_empty_groups():
    base = [dict(query=i, correct=c, margin=m, batch_margin=m, hardest_negative_in_batch=True) for i, (c, m) in enumerate([(False, -.1), (True, .01), (True, .5)])]
    current = [dict(r, correct=True, margin=r['margin']+.2, fixed_pair_margin=r['margin']+.1) for r in base]
    report = summarize(base, current)
    assert report['errors']['corrections'] == 1
    assert report['low_margin_correct']['queries'] == 1
    assert report['ordinary_correct']['queries'] == 1
    assert report['all']['fixed_pair_delta_mean'] == pytest.approx(.1)
    assert summarize([], [])['errors']['queries'] == 0
    with pytest.raises(ValueError): summarize(base, current[::-1])


def test_invalid_batch_and_features_rejected():
    with pytest.raises(ValueError): margins(torch.ones(8, 2), np.repeat([0, 1], 4))
    with pytest.raises(ValueError): margins(torch.full((8, 2), float('nan')), np.zeros(8))
