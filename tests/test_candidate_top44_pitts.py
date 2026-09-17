import unittest
import numpy as np
from scripts.eval_candidate_top44_pitts import ranking, remap_report


class PittsTop44Tests(unittest.TestCase):
    def test_ranking(self):
        np.testing.assert_array_equal(ranking(-np.arange(100.), np.arange(20)), np.arange(44))

    def test_stable_tie(self):
        np.testing.assert_array_equal(ranking(np.zeros(100), np.arange(20)), np.arange(44))

    def test_mismatch(self):
        with self.assertRaises(ValueError): ranking(-np.arange(100.), np.arange(20)[::-1])

    def test_invalid_scores(self):
        for s in (np.zeros(20), np.full(100, np.nan), np.zeros((2, 100))):
            with self.assertRaises(ValueError): ranking(s, np.arange(20))

    def test_original_query_ids(self):
        r = dict(corrections=[0], regressions=[1], newly_reachable_ids=[0], realized_newly_reachable_corrections=[0])
        out, rows = remap_report(r, [dict(query_index=0), dict(query_index=1)], [501, 9])
        self.assertEqual(out['corrections'], [501]); self.assertEqual(out['regressions'], [9])
        self.assertEqual(rows[1]['subset_index'], 1); self.assertEqual(rows[1]['query_index'], 9)
        self.assertEqual(r['corrections'], [0])

    def test_bad_mapping(self):
        with self.assertRaises(ValueError): remap_report({}, [dict(query_index=0)]*2, [9, 9])


if __name__ == '__main__': unittest.main()
