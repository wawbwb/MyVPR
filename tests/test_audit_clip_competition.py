import unittest
import numpy as np
from scripts.audit_clip_competition import region_diagnostic
from scripts.clip_dynamic_localization import maps, FINE_BOXES
from scripts.clip_dynamic_screen import BOXES, DYNAMIC, EGO, STATIC


class CompetitionTest(unittest.TestCase):
    def test_overlap_weights_reproduce_region_score(self):
        rng = np.random.default_rng(17)
        target = np.zeros((280, 280), bool)
        target[13:179, 41:227] = True
        for boxes in (BOXES, FINE_BOXES):
            c = rng.normal(0, .1, (len(boxes), len(DYNAMIC)+len(EGO)+len(STATIC)))
            r = region_diagnostic(c, boxes, target)
            expected = np.repeat(np.repeat(maps(c, boxes)['external_positive'], 14, 0), 14, 1)[target].mean()
            self.assertAlmostEqual(r['positive_score_inside'], float(expected), places=6)
            self.assertAlmostEqual(sum(v['region_weight'] for v in r['crops']), 1)
            self.assertAlmostEqual(sum(r['weighted_group_wins'].values()), 1)

    def test_competitor_and_ties(self):
        c = np.zeros((16, len(DYNAMIC)+len(EGO)+len(STATIC)))
        t = np.ones((280, 280), bool)
        r = region_diagnostic(c, BOXES, t)
        self.assertAlmostEqual(r['weighted_group_wins']['group_tie'], 1)
        self.assertEqual(r['positive_score_inside'], 0)
        c[:, len(DYNAMIC)] = .2
        r = region_diagnostic(c, BOXES, t)
        self.assertAlmostEqual(r['weighted_group_wins']['ego'], 1)
        self.assertAlmostEqual(r['mean_margin'], -.2)

    def test_invalid_data(self):
        c = np.zeros((16, 17))
        with self.assertRaises(ValueError):
            region_diagnostic(c, BOXES, np.zeros((280, 280), bool))
        c[0, 0] = np.nan
        with self.assertRaises(ValueError):
            region_diagnostic(c, BOXES, np.ones((280, 280), bool))


if __name__ == '__main__':
    unittest.main()
