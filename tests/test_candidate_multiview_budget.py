import unittest
import numpy as np
from scripts.audit_candidate_multiview_budget import budget_sets, query_record, report


def saved():
    return dict(full20=np.arange(20), full80=np.arange(80),
        crop_top20=np.tile(np.arange(20, 40), (3, 1)), union=np.arange(40))


class BudgetTests(unittest.TestCase):
    def test_fixed_and_matched(self):
        s = budget_sets(saved(), 100)
        self.assertEqual(len(s['full44']), 44)
        self.assertEqual(len(s['fullM']), len(s['union']))
        np.testing.assert_array_equal(s['fullM'], np.arange(40))

    def test_no_refill(self):
        x = saved(); x['crop_top20'] = np.tile(np.arange(20), (3, 1)); x['union'] = np.arange(20)
        self.assertEqual(len(budget_sets(x, 100)['fullM']), 20)

    def test_bad_prefix(self):
        x = saved(); x['full80'] = x['full80'][::-1]
        with self.assertRaises(ValueError): budget_sets(x, 100)

    def test_bad_union(self):
        x = saved(); x['union'] = np.arange(41)
        with self.assertRaises(ValueError): budget_sets(x, 100)

    def test_invalid_ids(self):
        for value in (-1, 100):
            x = saved(); x['crop_top20'][0, 0] = value
            with self.assertRaises(ValueError): budget_sets(x, 100)

    def test_duplicate(self):
        x = saved(); x['full80'][1] = 0
        with self.assertRaises(ValueError): budget_sets(x, 100)

    def test_comparison(self):
        x = saved(); x['crop_top20'] = np.tile(np.arange(80, 100), (3, 1))
        x['union'] = np.concatenate([np.arange(20), np.arange(80, 100)])
        s = budget_sets(x, 100)
        rows = [query_record(3, s, [90], False), query_record(8, s, [30], False)]
        r = report(rows)
        self.assertEqual(r['comparisons']['union_vs_fullM']['union_only_ids'], [3])
        self.assertEqual(r['comparisons']['union_vs_fullM']['full_only_ids'], [8])
        self.assertEqual(r['comparisons']['union_vs_fullM']['net_reachable'], 0)
        self.assertEqual(r['variants']['fullM']['additional_pairs'], r['variants']['union']['additional_pairs'])

    def test_empty(self):
        self.assertIsNone(report([])['variants']['fullM']['coverage'])


if __name__ == '__main__': unittest.main()
