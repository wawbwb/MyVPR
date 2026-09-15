import unittest
import numpy as np
from scripts.audit_candidate_training import inspect_query, aggregate


class TrainingAuditTests(unittest.TestCase):
    def test_categories(self):
        for labels, expected in [([0, 0], 'unreachable'), ([1, 1], 'all_positive'),
                                 ([1, 0], 'reachable_correct'), ([0, 1], 'reachable_error')]:
            self.assertEqual(inspect_query([2, 1], labels)['category'], expected)

    def test_stable_multi_positive_loss(self):
        r = inspect_query([1000, 1000, 1000], [1, 1, 0])
        self.assertAlmostEqual(r['frozen_list_loss'], np.log(1.5))
        self.assertAlmostEqual(r['negative_softmax_mass'], 1/3)

    def test_range_and_tie(self):
        for gap, expected in [(7.99, True), (8, False), (9, False), (0, True)]:
            self.assertEqual(inspect_query([gap, 0], [0, 1])['residual_range_only_fixable'], expected)
        self.assertEqual(inspect_query([0, 0], [0, 1])['best_positive_rank'], 2)

    def test_excluded_loss(self):
        self.assertIsNone(inspect_query([1, 2], [0, 0])['frozen_list_loss'])
        self.assertIsNone(inspect_query([1, 2], [1, 1])['frozen_list_loss'])

    def test_concentration(self):
        rows = []
        for i in range(100):
            r = inspect_query([0, 0], [0, 1])
            r.update(query_index=i, city='city', label=str(i)); rows.append(r)
        result = aggregate(rows)
        self.assertEqual(result['reachable_errors'], 100)
        self.assertAlmostEqual(result['loss_effective_query_count'], 100)
        self.assertAlmostEqual(result['loss_concentration_valid_denominator']['0.05']['loss_fraction'], .05)
        self.assertEqual(result['loss_concentration_valid_denominator']['0.01']['query_ids'], [0])

    def test_empty_valid(self):
        r = inspect_query([1, 2], [0, 0]); r.update(query_index=0, city='x', label='x')
        self.assertIsNone(aggregate([r])['loss_effective_query_count'])


if __name__ == '__main__': unittest.main()
