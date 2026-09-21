import copy
import unittest
import numpy as np
from src.top44_confirmation import panorama_group, partition, cluster_interval, exact_mcnemar, paired_statistics
from scripts.pitts_top44_confirmation import validate_new


class ConfirmationTests(unittest.TestCase):
    def test_panorama(self):
        self.assertEqual(panorama_group('queries_real/008/008043_pitch1_yaw10.jpg'),
                         panorama_group('queries_real/008/008043_pitch2_yaw1.jpg'))
        with self.assertRaises(ValueError): panorama_group('unknown.jpg')

    def test_partition(self):
        idx = dict(query_indices=[1, 0], all_queries=2, database=['d'], index_sha256={'i': 'x'},
                   queries=['q/2_pitch1_yaw1.jpg', 'q/1_pitch1_yaw1.jpg'], gt=[[0], [0]])
        old = dict(idx, query_indices=[1], queries=idx['queries'][:1], gt=[[0]])
        rows = partition(idx, old)
        self.assertEqual([r['query_index'] for r in rows], [0, 1])
        self.assertEqual([r['prior_query'] for r in rows], [False, True])
        bad = dict(old, queries=[])
        with self.assertRaises(ValueError): partition(idx, bad)

    def test_exact(self):
        self.assertEqual(exact_mcnemar(0, 0), 1)
        self.assertAlmostEqual(exact_mcnemar(2, 0), .5)

    def test_cluster(self):
        self.assertIsNone(cluster_interval([1, 0], ['a', 'a']))
        self.assertEqual(cluster_interval([0, 0], ['a', 'b']), [0., 0.])
        self.assertEqual(cluster_interval([1, 1], ['a', 'b']), [100., 100.])
        self.assertEqual(cluster_interval([1, -1, 0], ['a', 'b', 'b']),
                         cluster_interval([1, -1, 0], ['a', 'b', 'b']))

    def test_paired_ids(self):
        rows = [dict(query_index=90, group='a', old_correct=False, new_correct=True),
                dict(query_index=2, group='b', old_correct=True, new_correct=False)]
        result = paired_statistics(rows)
        self.assertEqual(result['corrections'], [90])
        self.assertEqual(result['regressions'], [2])
        self.assertEqual(result['interpretation'], 'INCONCLUSIVE')
        with self.assertRaises(ValueError): paired_statistics(rows+rows)

    def test_shard(self):
        record = dict(query_index=500, image_sha256='x', gt=[4])
        row = dict(candidates=np.arange(44), scores=np.zeros(44), global_scores=-np.arange(44.),
                   labels=np.arange(44) == 4, query_index=np.asarray(500), image_sha256=np.asarray('x'),
                   seconds=np.zeros(3))
        validate_new(row, record, 100)
        for key, bad in [('candidates', np.zeros(44, int)), ('scores', np.full(44, np.nan)),
                         ('query_index', np.asarray(0)), ('labels', np.zeros(44, bool)),
                         ('seconds', np.array([-1., 0., 0.]))]:
            changed = copy.deepcopy(row); changed[key] = bad
            with self.assertRaises(ValueError): validate_new(changed, record, 100)
