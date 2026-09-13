import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from pairvpr_cls_utils import remap_fraction, shuffled, selected_pairs, summarize


class GeometryTests(unittest.TestCase):
    def test_coverage_conservation(self):
        f = np.zeros((20,20)); f[:5,:10] = 1
        out = remap_fraction(f)
        self.assertEqual(out.shape, (23,23))
        self.assertAlmostEqual(float(out.mean()), float(f.mean()), places=7)
        np.testing.assert_allclose(remap_fraction(np.ones((20,20))), 1)
        self.assertEqual(float(remap_fraction(np.zeros((20,20))).sum()), 0)

    def test_shuffle(self):
        f = np.arange(529).reshape(23,23)/529
        a = shuffled(f,'image',11)
        np.testing.assert_array_equal(np.sort(a.ravel()), np.sort(f.ravel()))
        np.testing.assert_array_equal(a, shuffled(f,'image',11))
        self.assertFalse(np.array_equal(a, shuffled(f,'image',29)))

    def test_selection_and_summary(self):
        cases, excluded = selected_pairs([[2,1,0],[0,2,1],[0,1,2]], [[1],[0],[4]])
        self.assertEqual(cases[0]['positive'], 1)
        self.assertEqual(cases[0]['negative'], 2)
        self.assertEqual(cases[1]['group'], 'success')
        self.assertEqual(excluded[0]['reason'], 'no_positive')
        records = [dict(cases[0], margins={'original':-1., 'aligned':1., 'shuffle_11':-.5})]
        s = summarize(records)['groups']['reachable_error']
        self.assertEqual(s['variants']['aligned']['negative_to_positive'], 1)
        self.assertEqual(s['aligned_above_all_shuffles'], 1)


if __name__ == '__main__':
    unittest.main()
