import unittest
import numpy as np
from scripts.audit_candidate_reference_graph import graph_edges, edge_diagnostics, compact_plan, coverage


class ReferenceGraphTests(unittest.TestCase):
    def test_two_references_single_undirected_edge(self):
        selected, pre = graph_edges(np.array([7, 2]), np.array([[1., 0.], [1., 0.]]))
        self.assertEqual(selected, [(2, 7, 1.)]); self.assertEqual(pre, selected)

    def test_permutation_and_tie_invariance(self):
        ids = np.array([9, 2, 7, 4]); z = np.eye(4)
        expected = graph_edges(ids, z)
        p = np.array([2, 0, 3, 1])
        self.assertEqual(graph_edges(ids[p], z[p]), expected)
        for i, j, _ in expected[0]: self.assertLess(i, j)

    def test_cap_and_no_duplicate_edges(self):
        ids = np.arange(6); z = np.eye(6)
        chosen, all_edges = graph_edges(ids, z, neighbors=5, cap=3)
        self.assertEqual(len(chosen), 3); self.assertEqual(len(all_edges), 15)
        self.assertEqual(len(set((a, b) for a, b, _ in all_edges)), 15)

    def test_reject_duplicate_ids(self):
        with self.assertRaises(ValueError): graph_edges(np.array([1, 1]), np.eye(2))

    def test_reject_nan(self):
        with self.assertRaises(ValueError): graph_edges(np.array([1, 2]), np.array([[1., 0.], [np.nan, 0.]]))

    def test_labels_used_only_for_diagnostics(self):
        edges, _ = graph_edges(np.array([0, 1]), np.eye(2))
        good = edge_diagnostics(edges, ['a', 'a'], 'a')
        wrong = edge_diagnostics(edges, ['b', 'b'], 'a')
        self.assertEqual(good['positive_positive'], 1)
        self.assertEqual(wrong['wrong_same_place'], 1)
        self.assertEqual(good['same_place'], wrong['same_place'])

    def test_packed_pairs_reused_across_queries(self):
        p = compact_plan([[(1, 2, .5)], [(1, 2, .5), (2, 3, .2)], []])
        np.testing.assert_array_equal(p['pairs'], [[1, 2], [2, 3]])
        np.testing.assert_array_equal(p['edge_indices'], [0, 0, 1])
        np.testing.assert_array_equal(p['offsets'], [0, 1, 3, 3])

    def test_empty_plan_and_coverage(self):
        p = compact_plan([[], []]); self.assertEqual(p['pairs'].shape, (0, 2))
        self.assertIsNone(coverage([])['same_place_fraction'])
        self.assertIsNone(coverage([])['positive_edge_coverage_of_two_positive_queries'])

    def test_wrong_consensus_not_counted_as_positive(self):
        d = edge_diagnostics([(0, 1, .9)], ['wrong', 'wrong'], 'query')
        row = dict(positive_candidates=0, selected=d, pre_cap=d, selected_edge_count=1)
        c = coverage([row])
        self.assertEqual(c['queries_with_positive_edge'], 0)
        self.assertEqual(c['queries_with_wrong_same_place_edge'], 1)


if __name__ == '__main__': unittest.main()
