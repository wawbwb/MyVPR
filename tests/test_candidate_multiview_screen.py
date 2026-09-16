import unittest
import numpy as np
from scripts.candidate_multiview_screen import crop_boxes, candidate_sets, outcome, summarize


class MultiViewTests(unittest.TestCase):
    def test_boxes(self):
        self.assertEqual(crop_boxes(400, 200), [(0, 0, 300, 200), (50, 0, 350, 200), (100, 0, 400, 200)])

    def test_box_bounds(self):
        for w in (5, 101, 1024):
            for left, top, right, bottom in crop_boxes(w, 37):
                self.assertTrue(0 <= left < right <= w)
                self.assertEqual((top, bottom), (0, 37))

    def test_union_budget_and_preservation(self):
        full = -np.arange(100, dtype=float)
        crops = np.stack([np.roll(full, n) for n in (20, 40, 60)])
        result = candidate_sets(full, crops)
        self.assertEqual(len(result['union']), 80)
        self.assertTrue(set(result['full20']).issubset(result['union']))

    def test_no_refill(self):
        full = -np.arange(100, dtype=float)
        result = candidate_sets(full, np.tile(full, (3, 1)))
        self.assertEqual(len(result['union']), 20)
        self.assertEqual(len(result['full80']), 80)

    def test_ties_use_database_order(self):
        result = candidate_sets(np.zeros(100), np.zeros((3, 100)))
        np.testing.assert_array_equal(result['full80'], np.arange(80))

    def test_invalid(self):
        with self.assertRaises(ValueError): candidate_sets(np.zeros(10), np.zeros((3, 10)))
        with self.assertRaises(ValueError): candidate_sets(np.zeros(100), np.zeros((2, 100)))
        with self.assertRaises(ValueError): candidate_sets(np.full(100, np.nan), np.zeros((3, 100)))

    def test_complementarity_not_r1(self):
        sets = dict(full20=np.arange(20), full80=np.arange(80), union=np.array(list(range(20))+[90]))
        row = outcome(8, sets, [90], False)
        report = summarize([row])
        self.assertEqual(report['crop_unique_gain_ids'], [8])
        self.assertEqual(report['newly_reachable_ids'], [8])
        self.assertFalse(report['actual_pair_scoring'])


if __name__ == '__main__': unittest.main()
