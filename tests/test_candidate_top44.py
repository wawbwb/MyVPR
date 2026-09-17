import unittest
import numpy as np
from scripts.eval_candidate_top44 import validate_result, paired_record, summarize


def fixture(positives=(0,)):
    scores = -np.arange(44, dtype=np.float32)
    row = dict(candidates=np.arange(44), scores=scores, labels=np.isin(np.arange(44), positives),
        reproduced_score=np.asarray(0.))
    return row, dict(base=scores[:20].copy())


class Top44Tests(unittest.TestCase):
    def test_validate(self):
        row, old = fixture(); validate_result(row, np.arange(44), old, [0])

    def test_modified_old_scores(self):
        row, old = fixture(); row['scores'][1] = 3
        with self.assertRaises(ValueError): validate_result(row, np.arange(44), old, [0])

    def test_bad_reproduction(self):
        row, old = fixture(); row['reproduced_score'] = np.asarray(1.)
        with self.assertRaises(ValueError): validate_result(row, np.arange(44), old, [0])

    def test_bad_gt(self):
        row, old = fixture(); row['labels'][1] = True
        with self.assertRaises(ValueError): validate_result(row, np.arange(44), old, [0])

    def test_nonfinite(self):
        row, old = fixture(); row['scores'][30] = np.nan
        with self.assertRaises(ValueError): validate_result(row, np.arange(44), old, [0])

    def test_new_correction(self):
        row, _ = fixture((25,)); row['scores'][25] = 1
        r = paired_record(9, row)
        self.assertTrue(r['correction_from_added_candidate']); self.assertTrue(r['newly_reachable'])
        self.assertEqual(r['query_index'], 9)

    def test_regression(self):
        row, _ = fixture(); row['scores'][25] = 1
        self.assertTrue(paired_record(0, row)['regression_from_added_candidate'])

    def test_stable_tie(self):
        row, _ = fixture(); row['scores'][25] = 0
        self.assertEqual(paired_record(0, row)['new_id'], 0)

    def test_paired_summary(self):
        good, _ = fixture((25,)); good['scores'][25] = 1
        bad, _ = fixture(); bad['scores'][25] = 1
        out, _ = summarize([good, bad])
        self.assertEqual(out['corrections'], [0]); self.assertEqual(out['regressions'], [1])
        self.assertEqual(out['net'], 0); self.assertEqual(out['top20']['correct'], 1)
        self.assertEqual(out['realized_newly_reachable_corrections'], [0])


if __name__ == '__main__': unittest.main()
