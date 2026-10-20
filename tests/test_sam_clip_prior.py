"""CPU tests for training machine; no weights/network/GPU required."""
import numpy as np
import pytest
import torch

from src.region_vlad import grid_masks, region_vlad
from src.sam_clip_prior import (ARMS, CLASSES, fragment_risk, pixel_risk,
    interventions, weighted_vlad, pair_score, fixed_ranking)


def example_cosines():
    s = np.zeros((3, 2, len(CLASSES)), np.float32)
    s[0, :, 0] = .8  # agreed car
    s[1, :, 6] = .8  # agreed building
    s[2, 0, 0] = .8
    s[2, 1, 6] = .8  # crop/background-sensitive: abstain
    return s


def test_consensus_requires_both_views_and_static_competitors():
    r = fragment_risk(example_cosines(), 100)
    assert r[0] > .99 and np.array_equal(r[1:], [0, 0])


def test_uncertain_is_not_half_dynamic():
    r = fragment_risk(np.zeros((2, 2, len(CLASSES))), 100)
    np.testing.assert_array_equal(r, [0, 0])


def test_ego_vehicle_is_not_external_car():
    s = example_cosines()[:1]
    s[:, :, 16] = .9
    assert fragment_risk(s, 100)[0] == 0


@pytest.mark.parametrize('scale', [0, -1, float('nan')])
def test_invalid_scale_rejected(scale):
    with pytest.raises(ValueError):
        fragment_risk(example_cosines(), scale)


def test_overlap_dynamic_priority_and_uncovered_pixels():
    masks = np.zeros((2, 4, 4), bool)
    masks[0, :3] = True  # building
    masks[1, 1, 1] = True  # car overlapping building
    r = pixel_risk(masks, [0, .8])
    assert r[1, 1] == pytest.approx(.8)
    assert r[3, 3] == 0 and r[0, 0] == 0


def test_building_centred_supersegment_cannot_reintroduce_car():
    masks = grid_masks(4, 4)
    # n=4 nondegenerate, centroid Delaunay includes neighbours.
    arms, stats = interventions(masks, [1, 0, 0, 0], 'image', grid=4)
    sam, _ = arms['sam']
    pre, rel = arms['pre']
    center, center_rel = arms['center_only']
    assert len(pre) == 4 and stats['region_count'] == 4
    assert np.all(pre[:, masks[0].ravel()] <= .5+1e-6)
    # At least one static-centred supersegment contains the car.
    static = np.flatnonzero(sam[1:, masks[0].ravel()].sum(1) > 0)+1
    assert len(static) > 0
    assert (rel[static] < 1).all()
    np.testing.assert_array_equal(center, sam)
    np.testing.assert_array_equal(center_rel[static], np.ones(len(static)))


def test_car_centred_supersegment_keeps_building_pixels():
    masks = grid_masks(4, 4)
    arms, _ = interventions(masks, [1, 0, 0, 0], 'image', grid=4)
    sam, _ = arms['sam']
    pre, _ = arms['pre']
    building = (~masks[0]).ravel()
    assert sam[0, building].sum() > 0
    np.testing.assert_array_equal(pre[0, building], sam[0, building])


def test_zero_risk_and_zero_strength_reproduce_baseline():
    masks = grid_masks(7, 20)
    for risks, strength in [(np.zeros(7), .5), (np.ones(7), 0)]:
        arms, _ = interventions(masks, risks, 'abc', strength, grid=20)
        for arm in ARMS:
            np.testing.assert_array_equal(arms[arm][0], arms['sam'][0])
            np.testing.assert_array_equal(arms[arm][1], np.ones(7))


def test_uniform_control_matches_image_pixel_mass_not_region_mask_count():
    masks = grid_masks(7, 20)
    arms, stats = interventions(masks, np.linspace(0, 1, 7), 'abc', grid=20)
    np.testing.assert_allclose(arms['uniform_mass'][1], 1-.5*stats['pixel_risk_mean'], atol=1e-6)


def test_shuffle_deterministic_does_not_alter_topology():
    masks = grid_masks(7, 20)
    a, s = interventions(masks, np.linspace(0, 1, 7), 'abc', grid=20)
    b, t = interventions(masks, np.linspace(0, 1, 7), 'abc', grid=20)
    assert s == t
    for arm in ARMS:
        np.testing.assert_array_equal(a[arm][0] > 0, a['sam'][0] > 0)
        np.testing.assert_array_equal(a[arm][0], b[arm][0])


def test_fractional_token_coverage_not_any_pixel():
    m = np.zeros((1, 4, 4), bool)
    m[0, 0, 0] = True
    arms, _ = interventions(m, [1], 'tiny', order=0, grid=2)
    assert arms['sam'][0][0, 0] == .25
    assert arms['pre'][0][0, 0] == .125


def test_boolean_weighted_vlad_matches_existing_implementation():
    torch.manual_seed(3)
    x, centers = torch.randn(12, 8), torch.randn(3, 8)
    masks = torch.ones(2, 12, dtype=torch.bool)
    masks[1, :4] = False
    torch.testing.assert_close(weighted_vlad(x, centers, masks), region_vlad(x, centers, masks))


def test_uniform_scalar_cancels_vlad_but_not_matching_reliability():
    torch.manual_seed(4)
    x, centers, mask = torch.randn(12, 8), torch.randn(3, 8), torch.ones(1, 12)
    a, b = weighted_vlad(x, centers, mask), weighted_vlad(x, centers, mask*.5)
    torch.testing.assert_close(a, b)
    assert pair_score(a.numpy(), a.numpy(), [1], [1]) == pytest.approx(1)
    assert pair_score(b.numpy(), b.numpy(), [.5], [.5]) == pytest.approx(.5)


def test_nonuniform_vlad_changes_aggregation():
    torch.manual_seed(7)
    x, centers = torch.randn(12, 8), torch.randn(3, 8)
    m = torch.ones(1, 12)
    n = m.clone()
    n[:, :6] = .5
    assert not torch.allclose(weighted_vlad(x, centers, m), weighted_vlad(x, centers, n))


def test_empty_regions_abstain_and_stable_global_ranking():
    assert pair_score(np.empty((0, 8)), np.ones((1, 8)), [], [1]) == 0
    np.testing.assert_array_equal(fixed_ranking([.9, .8], [0, 0]), [0, 1])
    np.testing.assert_array_equal(fixed_ranking([.8, .8], [.5, .5]), [0, 1])


def test_pipeline_has_no_training_or_old_code_mutations():
    from pathlib import Path
    import ast
    root = Path(__file__).resolve().parents[1]
    source = (root/'scripts/sam_clip_prior.py').read_text(encoding='utf8')
    ast.parse(source)
    assert 'optimizer' not in source and '.backward(' not in source
    assert 'load_inference_model_from_ckpt' not in source


def test_masked_crops_keep_geometry_and_are_square():
    from scripts.sam_clip_prior import crop_views
    rgb = np.full((20, 20, 3), 200, np.uint8)
    m = np.zeros((20, 20), bool)
    m[5:10, 5:13] = True
    box, masked = crop_views(rgb, m, .05)
    assert box.size == masked.size and box.width == box.height
    assert not np.array_equal(np.asarray(box), np.asarray(masked))


def test_shard_validates_scores_masks_and_unit_descriptors(tmp_path):
    from scripts.sam_clip_prior import validate_shard
    scores = example_cosines()
    risks = fragment_risk(scores, 100)
    arrays = {'image_sha': 'image', 'cosines': scores, 'risks': risks, 'scale': 100.,
        'masks_packbits': np.zeros((3, 9800), np.uint8)}
    for arm in ARMS:
        d = np.zeros((3, 8), np.float16)
        d[:, 0] = 1
        arrays[arm], arrays[arm+'_reliability'] = d, np.ones(3, np.float32)
    target = tmp_path/'shard.npz'
    np.savez_compressed(target, **arrays)
    validate_shard(target, 'image', 8)
    with pytest.raises(ValueError, match='Image changed'):
        validate_shard(target, 'different', 8)
    arrays['risks'] = np.zeros(3, np.float32)
    np.savez_compressed(target, **arrays)
    with pytest.raises(ValueError, match='Risk reconstruction'):
        validate_shard(target, 'image', 8)


def test_empty_shard_is_valid_abstention(tmp_path):
    from scripts.sam_clip_prior import validate_shard
    arrays = {'image_sha': 'image', 'cosines': np.empty((0, 2, len(CLASSES)), np.float32),
        'risks': np.empty(0, np.float32), 'scale': 100.,
        'masks_packbits': np.empty((0, 9800), np.uint8)}
    for arm in ARMS:
        arrays[arm] = np.empty((0, 8), np.float16)
        arrays[arm+'_reliability'] = np.empty(0, np.float32)
    target = tmp_path/'empty.npz'
    np.savez_compressed(target, **arrays)
    validate_shard(target, 'image', 8)


def test_atomic_shard_metadata_does_not_collide_with_writer_path(tmp_path):
    from src.semantic_region_protocol import save_npz
    target = tmp_path/'atomic.npz'
    save_npz(target, image_path='cph/database/images/example.jpg', risks=np.empty(0))
    with np.load(target, allow_pickle=False) as z:
        assert str(z['image_path']) == 'cph/database/images/example.jpg'
    from pathlib import Path
    import ast
    tree = ast.parse((Path(__file__).resolve().parents[1]/'scripts/sam_clip_prior.py').read_text(encoding='utf8'))
    # Guard against reintroducing a path key in the pipeline arrays literal.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'arrays' for t in node.targets):
            if isinstance(node.value, ast.Dict):
                keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
                assert 'image_path' in keys and 'path' not in keys
