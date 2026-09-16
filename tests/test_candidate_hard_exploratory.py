import unittest
import numpy as np
from scripts.candidate_hard_exploratory import group_report, groups_for


class ExploratoryTests(unittest.TestCase):
    def values(self):
        return dict(query_indices=np.array([97, 200, 513]),
                    labels=np.array([[True, False], [True, False], [False, False]]),
                    base=np.array([[0., 1.], [1., 0.], [0., 1.]]),
                    scores=np.array([[2., 1.], [0., 1.], [1., 0.]]))

    def test_global_query_ids_not_subset_rows(self):
        r = group_report(self.values(), {'all': [97, 200, 513]})['all']
        self.assertEqual(r['corrected'], [97])
        self.assertEqual(r['regressed'], [200])
        self.assertEqual(r['net'], 0)

    def test_empty_group(self):
        self.assertEqual(group_report(self.values(), {'empty': []})['empty']['queries'], 0)

    def test_unknown_id_rejected(self):
        with self.assertRaises(ValueError): group_report(self.values(), {'bad': [999]})

    def test_duplicate_ids_rejected(self):
        v = self.values(); v['query_indices'] = np.array([97, 97, 513])
        with self.assertRaises(ValueError): group_report(v, {'all': [97, 513]})

    def test_unreachable_never_corrected(self):
        r = group_report(self.values(), {'unreachable': [513]})['unreachable']
        self.assertEqual(r['correct'], 0); self.assertEqual(r['corrected'], [])

    def test_groups_range_boundary(self):
        rows = [dict(query_index=0, category='reachable_error', residual_range_only_fixable=True,
                     residual_range_blocked=False, positive_minus_negative_margin=-2),
                dict(query_index=1, category='reachable_correct', residual_range_only_fixable=False,
                     residual_range_blocked=False, positive_minus_negative_margin=8)]
        r = groups_for(rows)
        self.assertEqual(r['range_fixable_errors'], [0])
        self.assertEqual(r['correct_margin_below_8'], [])


if __name__ == '__main__': unittest.main()
