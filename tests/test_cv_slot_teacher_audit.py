"""CPU-only tests, intended to run on the training machine."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch

from src.cv_slot_protocol import (ARMS, balanced_places, correspondence, crop_targets,
    geometry_metrics, graph_score, paired_interval, decide)


def features():
    g = torch.Generator().manual_seed(7)
    return [torch.randn(n, d, generator=g) for n, d in ((16, 12), (16, 12), (16, 8), (16, 8))]


def test_deterministic_candidates_and_weight_multisets():
    x = features()
    graph = correspondence(*x, topk=4)
    changed = correspondence(*x[:2], -x[2], x[3].flip(0), topk=4)
    assert torch.equal(graph['destinations'], changed['destinations'])
    assert torch.equal(graph['support'], changed['support'])
    assert torch.equal(graph['distributions']['dino'], changed['distributions']['dino'])
    for row in range(16):
        support = graph['support'][row]
        assert torch.equal(graph['clip_weights'][row, support].sort().values,
                           graph['shuffled_weights'][row, support].sort().values)
    assert torch.equal(graph['shuffled_weights'], correspondence(*x, topk=4)['shuffled_weights'])


def test_different_permutation_seeds():
    a, b = correspondence(*features(), topk=8, seed=1), correspondence(*features(), topk=8, seed=2)
    assert torch.equal(a['destinations'], b['destinations'])
    assert not torch.equal(a['shuffled_weights'], b['shuffled_weights'])


def test_probability_support_normalization_and_detach():
    x = [v.requires_grad_() for v in features()]
    graph = correspondence(*x, topk=4)
    valid = graph['support'].any(-1)
    for p in graph['distributions'].values():
        assert not p.requires_grad
        assert torch.isfinite(p).all()
        assert torch.all(p[~graph['support']] == 0)
        assert torch.allclose(p.sum(-1), valid.float(), atol=1e-6)


def test_constant_clip_is_exactly_dino():
    x = features()
    x[2:] = [torch.ones_like(v) for v in x[2:]]
    graph = correspondence(*x, topk=4)
    for p in graph['distributions'].values():
        assert torch.allclose(p, graph['distributions']['dino'], atol=1e-7)


def test_one_candidate_cannot_be_reweighted():
    g = correspondence(*features(), topk=1)
    assert torch.equal(g['distributions']['clip'], g['distributions']['dino'])
    assert torch.equal(g['distributions']['shuffled'], g['distributions']['dino'])


@pytest.mark.parametrize('bad', ['nan', 'temperature', 'shape', 'topk'])
def test_invalid_inputs_fail(bad):
    x, options = features(), dict(topk=4)
    if bad == 'nan': x[0][0, 0] = float('nan')
    elif bad == 'shape': x[2] = x[2][:-1]
    elif bad == 'temperature': options['clip_temperature'] = 0
    else: options['topk'] = 17
    with pytest.raises(ValueError):
        correspondence(*x, **options)


def test_crop_targets_are_inverse_and_have_known_overlap():
    valid, target = crop_targets()
    rvalid, rtarget = crop_targets(reverse=True)
    assert valid.sum() == rvalid.sum() == 144
    assert torch.equal(rtarget[target[valid]], torch.arange(196)[valid])
    assert torch.all(rvalid[target[valid]])


def test_geometry_metric_counts_empty_rows_as_wrong():
    destinations = torch.arange(4)[:, None]
    graph = dict(destinations=destinations, support=torch.tensor([[1], [1], [0], [1]], dtype=torch.bool),
        distributions={arm: torch.tensor([[1.], [1.], [0.], [1.]]) for arm in ARMS},
        similarities=torch.ones(4, 1))
    result = geometry_metrics(graph, torch.ones(4, dtype=torch.bool), torch.arange(4))
    assert result['dino']['accuracy'] == result['dino']['gt_probability'] == .75
    assert result['candidate_gt_coverage'] == .75
    assert graph_score(graph, 'clip') == 1.


def test_city_balanced_sampling_is_stable_and_split_disjoint():
    places = [dict(city=c, split=s, place=f'{s}:{c}:{i}')
        for s in ('train', 'holdout') for c in ('A', 'B', 'C') for i in range(10)]
    plan = dict(places=places)
    a = balanced_places(plan, 'train', 8, 42)
    b = balanced_places(plan, 'holdout', 8, 42)
    assert a == balanced_places(plan, 'train', 8, 42)
    assert len(set(a)) == 8 and not set(a)&set(b)
    counts = [sum(places[i]['city'] == c for i in a) for c in ('A', 'B', 'C')]
    assert max(counts)-min(counts) <= 1


def test_paired_interval_is_place_level_and_reproducible():
    value = paired_interval([1., 1., 1.], [0., 0., 0.])
    assert value == dict(n=3, mean=1., ci95=[1., 1.])
    assert paired_interval([0., 0.], [1., 1.])['ci95'] == [-1., -1.]
    with pytest.raises(ValueError): paired_interval([float('nan'), 1], [0, 0])


def example_rows():
    geometry = [{arm: dict(gt_probability=({'dino': .5, 'clip': .6, 'shuffled': .4})[arm],
                           accuracy=.9) for arm in ARMS} for _ in range(128)]
    real = [{arm: dict(pairwise_win=({'dino': .8, 'clip': .8, 'shuffled': .7})[arm])
             for arm in ARMS} for _ in range(128)]
    thresholds = dict(min_geometry_places=128, min_real_places=120,
        min_probability_gain=.001, accuracy_tolerance=.002, real_pair_tolerance=0.)
    return geometry, real, thresholds


def test_gate_demands_clip_beat_both_controls_and_preserve_real_pairs():
    geometry, real, t = example_rows()
    assert decide(geometry, real, t)['verdict'] == 'PASS'
    for row in geometry: row['shuffled']['gt_probability'] = .6
    assert decide(geometry, real, t)['verdict'] == 'STOP_TEACHER_NOT_SUPPORTED'
    for row in geometry: row['shuffled']['gt_probability'] = .4
    for row in real: row['clip']['pairwise_win'] = .79
    assert not decide(geometry, real, t)['checks']['real_pair_noninferior']


def test_no_real_pairs_does_not_pass_or_emit_nan():
    geometry, _, t = example_rows()
    value = decide(geometry, [], t)
    assert value['verdict'] == 'STOP_TEACHER_NOT_SUPPORTED'
    assert value['real_pairwise_win']['clip'] is None


def test_configuration_locked_before_extraction():
    from scripts.cv_slot_teacher_audit import configuration
    c = configuration(ROOT/'config/cv_slot_teacher_audit.yaml')
    assert c['sample_per_split'] == 128 and c['cuda_memory_fraction'] <= .2
    assert c['thresholds']['min_probability_gain'] == .001
