"""Run on the training machine; no GPU/weights/downloads required."""
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from src.region_vlad import (grid_masks, neighbour_union, patch_membership,
                             region_vlad, image_vote, equal_budget_union)


@pytest.mark.parametrize('count', [1, 3, 7, 16])
def test_grid_exact_partition(count):
    masks = grid_masks(count, 20)
    assert masks.shape == (count, 20, 20)
    assert (masks.sum(0) == 1).all()
    assert masks.reshape(count, -1).any(1).all()


def test_any_pixel_membership():
    masks = np.zeros((1, 280, 280), bool)
    masks[0, 13, 14] = True
    patches = patch_membership(masks)
    assert patches.sum() == 1 and patches[0, 1]


def test_neighbours_keep_seed_and_union_without_double_counting():
    masks = grid_masks(16, 20)
    merged, fallback = neighbour_union(masks, 1)
    assert not fallback
    assert np.all((merged | masks) == merged)
    assert np.all(merged.sum((1, 2)) >= masks.sum((1, 2)))
    original, _ = neighbour_union(masks, 0)
    np.testing.assert_array_equal(original, masks)


def test_degenerate_centroids_safe_identity():
    masks = np.ones((4, 20, 20), bool)
    merged, fallback = neighbour_union(masks)
    assert fallback
    np.testing.assert_array_equal(merged, masks)


def test_vlad_matches_explicit_cluster_loop():
    torch.manual_seed(1)
    x = F.normalize(torch.randn(12, 8), dim=1)
    centers = torch.randn(3, 8)
    masks = torch.rand(4, 12) > .3
    actual = region_vlad(x, centers, masks)
    assignment = (x @ F.normalize(centers, dim=1).T).argmax(1)
    expected = []
    for mask in masks:
        pieces = []
        for k in range(3):
            residual = (x[mask & (assignment == k)]-centers[k]).sum(0)
            pieces.append(F.normalize(residual, dim=0))
        expected.append(F.normalize(torch.cat(pieces), dim=0))
    torch.testing.assert_close(actual, torch.stack(expected))


def test_vlad_rejects_empty_region():
    with pytest.raises(ValueError):
        region_vlad(torch.randn(4, 2), torch.randn(2, 2), torch.zeros(1, 4, dtype=torch.bool))


def test_vote_aggregates_region_hits_not_image_id_order():
    ids = np.array([[0, 1, 2]])
    sims = np.array([[.9, .8, .7]])
    owners = np.array([5, 2, 2])
    result = image_vote(ids, sims, owners, 0, 1, 4)
    np.testing.assert_array_equal(result, [2, 5, -1, -1])


def test_equal_budget_no_gt_unique_twenty():
    result = equal_budget_union(np.arange(20), np.arange(8, 28))
    assert len(result) == len(set(result)) == 20
    assert np.array_equal(result[:10], np.arange(10))


def test_shift_preserves_area_and_token_budget():
    m = grid_masks(7)
    shifted = np.roll(m, (140, 140), (1, 2))
    np.testing.assert_array_equal(m.sum((1, 2)), shifted.sum((1, 2)))
    np.testing.assert_array_equal(patch_membership(m).sum(1), patch_membership(shifted).sum(1))


def test_empty_regions_explicit_all_four_abstain():
    from scripts.region_vlad_screen import empty_regions, MODES
    masks, stats = empty_regions(3)
    for mode in MODES:
        assert masks[mode].shape == (0, 400)
        assert stats[mode]['count'] == 0
    assert stats['sam']['raw_count'] == 3
    np.testing.assert_array_equal(equal_budget_union(np.arange(20), np.full(20, -1)), np.arange(20))


def test_migration_only_accepts_known_script_and_unchanged_settings():
    from scripts.region_vlad_screen import compatible_contract, LEGACY_SCRIPT_SHA
    old = {'implementation': {'scripts/region_vlad_screen.py': LEGACY_SCRIPT_SHA, 'other': 'same'}, 'weight': 'a'}
    new = {'implementation': {'scripts/region_vlad_screen.py': 'new', 'other': 'same'}, 'weight': 'a',
           'empty_region_policy': 'all_four_zero_regions_keep_ru_query_abstains_v1'}
    assert compatible_contract(old, new)
    assert compatible_contract(new, new)
    assert not compatible_contract(old, {**new, 'weight': 'changed'})
    assert not compatible_contract({**old, 'implementation': {'scripts/region_vlad_screen.py': 'unknown'}}, new)


def test_abstaining_query_still_counted_as_miss():
    from scripts.region_vlad_screen import measure
    result = measure(np.full((1, 20), -1), [np.array([0])], np.array([True]))
    assert result['r1'] == 0 and result['regressions'] == [0]
