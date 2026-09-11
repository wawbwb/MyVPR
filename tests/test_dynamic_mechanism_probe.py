import copy
import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.dynamic_mechanism_probe import CASES, polygon_mask, pool, rank, validate_annotations, prepare


class ProbeTests(unittest.TestCase):
    def test_prepare_creates_fixed_pending_cases(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('q.jpg', 'ref.jpg'):
                Image.new('RGB', (40,30)).save(root/name)
            split = {'query_paths': ['q.jpg'] * 740, 'fixed_gt_indices': [0] * 740}
            args = SimpleNamespace(output=root/'out', dataset_root=root, run=root, mask_cache=root/'unused')
            with patch('scripts.dynamic_mechanism_probe.load_inputs', return_value=(['ref.jpg'], None, 1, {'msls-val': split}, {})):
                prepare(args)
                with self.assertRaises(ValueError):
                    prepare(args)
            template = json.loads((args.output/'template.json').read_text())
            self.assertEqual([c['query_index'] for c in template['cases']], list(CASES))
            self.assertEqual(len(list((args.output/'images').glob('*.jpg'))), 16)
            self.assertTrue(all(c['decision']=='pending' for c in template['cases']))
            with self.assertRaisesRegex(ValueError, 'include/skip'):
                validate_annotations(template, template)

    def test_shift_preserves_rows_and_area(self):
        mask = polygon_mask([[[.1,.2],[.3,.2],[.2,.6]]])
        shifted = np.roll(mask, 140, axis=1)
        np.testing.assert_array_equal(mask.sum(1), shifted.sum(1))
        self.assertAlmostEqual(float(pool(mask).mean()), float(mask.mean()), places=6)
        np.testing.assert_allclose(pool(shifted), np.roll(pool(mask), 10, axis=1))

    def test_invalid_polygons(self):
        for poly in ([[[0,0],[1,1]]], [[[0,0],[2,1],[0,1]]], [[[0,0],[float('nan'),1],[0,1]]]):
            with self.assertRaises(ValueError):
                polygon_mask(poly)

    def test_rank_full_reference_ids_and_margin(self):
        result = rank(np.array([.2,.8,.7,.9]), [1,2])
        self.assertEqual(result['top1_reference_index'], 3)
        self.assertEqual(result['best_gt_rank'], 2)
        self.assertAlmostEqual(result['positive_negative_margin'], -.1)
        self.assertEqual(result['top1_correct'], 0)

    def example(self):
        return {'probe_id': 'fixed', 'cases': [{'query_index': 4, 'query_path': 'q.jpg',
                    'decision': 'include', 'static_overlap': 'yes', 'notes': '',
                    'external': {'status': 'reviewed', 'polygons': [[[.1,.1],[.2,.1],[.2,.2]]]},
                    'ego': {'status': 'absent', 'polygons': []}}]}

    def test_manual_gate_and_overlap(self):
        template = self.example()
        self.assertEqual(len(validate_annotations(template, copy.deepcopy(template))), 1)
        for key, value in [('decision','pending'), ('static_overlap','unknown')]:
            data = copy.deepcopy(template)
            data['cases'][0][key] = value
            with self.assertRaises(ValueError):
                validate_annotations(template, data)
        data = copy.deepcopy(template)
        data['cases'][0]['ego'] = copy.deepcopy(data['cases'][0]['external'])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_annotations(template, data)
        data = copy.deepcopy(template)
        data['cases'][0]['external']['status'] = 'absent'
        with self.assertRaises(ValueError):
            validate_annotations(template, data)


if __name__ == '__main__':
    unittest.main()
