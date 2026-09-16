import unittest
import numpy as np
from scripts.diagnose_candidate_reference_oracle import score_graph, summarize


class ReferenceOracleTests(unittest.TestCase):
    def test_empty_graph_preserves_ranking(self):
        base = np.array([2., -1., 0.])
        scores, kept = score_graph(base, np.arange(3), [])
        self.assertEqual(kept, [])
        np.testing.assert_array_equal(np.argsort(-scores), np.argsort(-base))

    def test_formula(self):
        scores, _ = score_graph([0., 0., 0.], np.arange(3), [(0, 1, .8)])
        np.testing.assert_allclose(scores, [.5, .5, 1/3])

    def test_wrong_same_place_is_retained(self):
        _, kept = score_graph([1., 2., 3.], np.arange(3), [(0, 1, .8), (1, 2, .7)], ['wrong', 'wrong', 'correct'])
        self.assertEqual(kept, [(0, 1)])

    def test_max_not_degree_vote(self):
        one, _ = score_graph([0., 0., 0.], np.arange(3), [(0, 1, .8)])
        two, _ = score_graph([0., 0., 0.], np.arange(3), [(0, 1, .8), (0, 2, .7)])
        self.assertEqual(one[0], two[0])

    def test_permutation_equivariance(self):
        ids = np.array([4, 2, 8]); base = np.array([.5, 1., -1.]); edges = [(4, 2, .9)]
        original, _ = score_graph(base, ids, edges)
        order = np.array([2, 0, 1])
        shuffled, _ = score_graph(base[order], ids[order], edges)
        np.testing.assert_allclose(shuffled, original[order])

    def test_shift_invariant_and_finite(self):
        a, _ = score_graph([0., 1.], np.arange(2), [(0, 1, .9)])
        b, _ = score_graph([10000., 10001.], np.arange(2), [(0, 1, .9)])
        np.testing.assert_allclose(a, b)

    def test_invalid_inputs(self):
        for base, ids, edges in [([np.nan, 0.], [0, 1], []), ([0., 1.], [1, 1], []),
            ([0., 1.], [0, 1], [(0, 2, 1.)]), ([0., 1.], [0, 1], [(0, 0, 1.)]),
            ([0., 1.], [0, 1], [(0, 1, 1.), (1, 0, 1.)])]:
            with self.assertRaises(ValueError): score_graph(base, ids, edges)

    def test_empty_summary(self):
        self.assertIsNone(summarize([])['frozen']['r1'])

    def test_paired_outcomes(self):
        rows = []
        for qi, old, new in [(3, False, True), (7, True, False)]:
            row = dict(query_index=qi, frozen_correct=old, frozen_id=0)
            for mode in ('visual_graph', 'label_filtered_graph'):
                row[mode+'_correct'] = new; row[mode+'_id'] = 1
            rows.append(row)
        result = summarize(rows)['label_filtered_graph']
        self.assertEqual(result['corrections'], [3]); self.assertEqual(result['regressions'], [7])
        self.assertEqual(result['net'], 0)


if __name__ == '__main__': unittest.main()
