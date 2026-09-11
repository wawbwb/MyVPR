import json
import sys

import numpy as np
import unittest
import tempfile
from PIL import Image
from pathlib import Path

from scripts.audit_dynamic_mask_regions import main, sha, spatial_stats


def inputs(tmp_path):
    root, run_dir = tmp_path / 'data', tmp_path / 'run'
    root.mkdir()
    run_dir.mkdir()
    paths = ['db.jpg', 'q.jpg', 'zero.jpg']
    for p in paths:
        Image.new('RGB', (40, 30), 'gray').save(root / p)
    np.save(root / 'msls_val_dbImages.npy', paths[:1])
    np.save(root / 'q.npy', paths[1:])
    np.save(root / 'gt.npy', np.array([[0], [0]], dtype=np.int64))
    masks = np.zeros((3, 20, 20), dtype=np.float32)
    masks[1, 15:] = 1
    cache = tmp_path / 'cache.npz'
    np.savez(cache, image_paths=paths, masks=masks, num_references=1)
    split = {'name': 'test', 'protocol': 'synthetic', 'num_queries': 2,
             'manifests': {role: {'path': name, 'sha256': sha(root / name)}
                           for role, name in [('queries', 'q.npy'), ('ground_truth', 'gt.npy')]}}
    run = {'schema_version': 3, 'method': 'frozen_dynamic_category_negative_attention_prior',
           'mask_cache': {'sha256': sha(cache), 'num_references': 1},
           'datasets': [split, dict(split, name='overlapping')]}
    (run_dir / 'run.json').write_text(json.dumps(run), encoding='utf-8')
    output = tmp_path / 'out'
    args = ['--run', str(run_dir), '--mask-cache', str(cache), '--dataset-root', str(root),
            '--output', str(output)]
    return args, output, root, cache, run_dir


class AuditTests(unittest.TestCase):
    def test_band_denominators_and_zero_mask(self):
        mask = np.zeros((20, 20))
        self.assertIsNone(spatial_stats(mask)['bottom_share_of_mask'])
        mask[15:] = 1
        result = spatial_stats(mask)
        self.assertEqual(result['coverage'], .25)
        self.assertEqual(result['bottom_share_of_mask'], 1)
        self.assertEqual(result['above_bottom_area_fraction_of_image'], 0)
        mask[:] = .4
        result = spatial_stats(mask)
        self.assertAlmostEqual(result['bottom_share_of_mask'], .25)
        self.assertAlmostEqual(sum(result['band_area_fraction_of_image'].values()), result['coverage'])
        self.assertAlmostEqual(result['above_bottom_area_fraction_of_image'], .3)

    def test_invalid_masks(self):
        for value in (-1, 2, float('nan')):
            with self.assertRaisesRegex(ValueError, 'Invalid'):
                spatial_stats(np.full((20, 20), value))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.inputs = inputs(Path(self.temp.name))

    def test_cpu_report_without_outcomes_and_overlap_deduplication(self):
        args, output, root, cache, run_dir = self.inputs
        before = sha(cache)
        main(args)
        assert sha(cache) == before
        result = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
        assert result['performance_audit_complete'] is False
        assert result['unique_query_union']['all']['n'] == 2
        assert result['datasets']['test']['strata']['[15%,30%)']['bottom_share_ge_half_count'] == 1
        assert len(list((output / 'images').glob('*.jpg'))) == 3
        reviews = json.loads((output / 'review_samples.json').read_text(encoding='utf-8'))
        assert reviews['reviews'][0]['annotation']['visible_static_overlap'] == 'unknown'
        assert all(im['annotation']['ego_region_selected'] == 'unknown'
                   for r in reviews['reviews'] for im in r['images'])
        assert '<script id="data"' in (output / 'index.html').read_text(encoding='utf-8')
        assert 'torch' not in sys.modules
        with self.assertRaisesRegex(ValueError, 'never overwritten'):
            main(args)


    def test_changed_cache_rejected_before_output(self):
        args, output, root, cache, run_dir = self.inputs
        with cache.open('ab') as f:
            f.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            main(args)
        assert not output.exists()


    def test_changed_manifest_rejected_before_output(self):
        args, output, root, cache, run_dir = self.inputs
        np.save(root / 'q.npy', ['zero.jpg', 'q.jpg'])
        with self.assertRaisesRegex(ValueError, 'Manifest SHA mismatch'):
            main(args)
        assert not output.exists()


    def test_missing_preview_rejected_before_output(self):
        args, output, root, cache, run_dir = self.inputs
        (root / 'q.jpg').unlink()
        with self.assertRaisesRegex(ValueError, 'Image missing'):
            main(args)
        assert not output.exists()

if __name__ == "__main__":
    unittest.main()
