import unittest
import numpy as np
from scripts.diagnose_candidate_hard_last import diagnose_rows, group_stats, compare, correlation


class LastDiagnosticTests(unittest.TestCase):
    def test_changing_best_negative(self):
        v = dict(base=np.array([[0., 2., 1.]]), scores=np.array([[1., 0., 3.]]),
                 labels=np.array([[True, False, False]]), query_indices=np.array([42]))
        rows, _ = diagnose_rows(v)
        self.assertEqual(rows[0]['margin_change'], 0)
        self.assertEqual(rows[0]['fixed_pair_delta_advantage'], 3)
        self.assertFalse(rows[0]['corrected'])

    def test_unreachable_and_empty_group(self):
        v = dict(base=np.zeros((1, 2)), scores=np.ones((1, 2)),
                 labels=np.zeros((1, 2), bool), query_indices=np.array([7]))
        rows, _ = diagnose_rows(v)
        self.assertIsNone(rows[0]['margin_change'])
        self.assertEqual(group_stats(rows, {'empty': []})['empty']['margin_queries'], 0)

    def test_bound_and_saturation(self):
        v = dict(base=np.zeros((1, 2)), scores=np.array([[4., -4.]]),
                 labels=np.array([[True, False]]), query_indices=np.array([7]))
        rows, _ = diagnose_rows(v)
        self.assertEqual(rows[0]['saturation_fraction'], 1)
        v['scores'][0, 0] = 5
        with self.assertRaises(ValueError): diagnose_rows(v)

    def test_common_offsets_removed(self):
        a = np.array([[1., 2., 3.], [3., 2., 1.]])
        b = a+np.array([[10.], [-7.]])
        r = compare(a, b)
        self.assertAlmostEqual(r['centered_delta_correlation'], 1)
        self.assertAlmostEqual(r['centered_mae'], 0)

    def test_constant_correlation_is_null(self):
        self.assertIsNone(correlation(np.ones(4), np.ones(4)))

    def test_multi_positive_margin(self):
        v = dict(base=np.array([[0., 2., 3.]]), scores=np.array([[4., 2., 3.]]),
                 labels=np.array([[True, True, False]]), query_indices=np.array([70]))
        rows, _ = diagnose_rows(v)
        self.assertEqual(rows[0]['margin_before'], -1)
        self.assertEqual(rows[0]['margin_after'], 1)
        self.assertTrue(rows[0]['corrected'])


if __name__ == '__main__': unittest.main()
