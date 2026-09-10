import copy

import numpy as np
import pytest

from scripts.region_pair_lite import (
    LEGACY_SCRIPT_SHA, compatible_legacy_contract, inspect_shard, read_shard, save_shard,
)


def arrays(n=3):
    result = {}
    for mode in ('sam', 'grid', 'shifted'):
        result[mode] = np.ones((n, 768), dtype=np.float16)
        result[mode+'_xy'] = np.full((n, 2), .5, dtype=np.float32)
    return result


@pytest.mark.parametrize('n', [0, 3, 16])
def test_valid_shards_and_abstention(tmp_path, n):
    path = tmp_path/'region.npz'
    save_shard(path, arrays(n))
    assert inspect_shard(path) is None
    assert read_shard(path)['sam'].shape == (n, 768)


def test_missing_empty_and_truncated(tmp_path):
    path = tmp_path/'region.npz'
    assert inspect_shard(path) == 'missing'
    path.touch()
    assert inspect_shard(path) is not None
    save_shard(path, arrays())
    path.write_bytes(path.read_bytes()[:64])
    assert inspect_shard(path) is not None


@pytest.mark.parametrize('kind', ['keys', 'shape', 'dtype', 'nan', 'centroid', 'count', 'scalar'])
def test_bad_schema(tmp_path, kind):
    data = arrays()
    if kind == 'keys': del data['grid']
    if kind == 'shape': data['grid'] = data['grid'][:1]
    if kind == 'dtype': data['grid'] = data['grid'].astype(np.float32)
    if kind == 'nan': data['grid'][0, 0] = np.nan
    if kind == 'centroid': data['sam_xy'][0, 0] = 2
    if kind == 'count': data = arrays(2)
    if kind == 'scalar': data['sam'] = np.array(1)
    path = tmp_path/'region.npz'
    np.savez_compressed(path, **data)
    assert inspect_shard(path) is not None


def test_invalid_new_shard_does_not_replace_good_one(tmp_path):
    path = tmp_path/'region.npz'
    save_shard(path, arrays())
    original = path.read_bytes()
    with pytest.raises(ValueError): save_shard(path, arrays(1))
    assert path.read_bytes() == original


def test_contract_migration_only_allows_known_script_change():
    old = {'code': {'scripts/region_pair_lite.py': LEGACY_SCRIPT_SHA,
                    'src/region_pair_lite.py': 'scorer'}, 'ru': 'checkpoint'}
    current = copy.deepcopy(old)
    current['code']['scripts/region_pair_lite.py'] = 'new-script'
    assert compatible_legacy_contract(old, current)
    assert old['code']['scripts/region_pair_lite.py'] == LEGACY_SCRIPT_SHA
    for key in ('ru', 'scorer', 'unknown_script'):
        changed = copy.deepcopy(old)
        if key == 'ru': changed['ru'] = 'other'
        if key == 'scorer': changed['code']['src/region_pair_lite.py'] = 'other'
        if key == 'unknown_script': changed['code']['scripts/region_pair_lite.py'] = 'other'
        assert not compatible_legacy_contract(changed, current)
