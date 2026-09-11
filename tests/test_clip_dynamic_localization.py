import unittest
import numpy as np

from scripts.clip_dynamic_localization import maps,statistics,FINE_BOXES
from scripts import clip_dynamic_screen as base


class LocalizationTests(unittest.TestCase):
    def scores(self, boxes):
        return np.zeros((len(boxes),len(base.DYNAMIC)+len(base.EGO)+len(base.STATIC)))

    def test_coarse_sigmoid_reproduces_previous_map(self):
        c=np.random.default_rng(42).uniform(-.2,.5,(16,17))
        old,ego=base.heatmap(c); new=maps(c,base.BOXES)
        np.testing.assert_array_equal(old,new['external_sigmoid'])
        np.testing.assert_array_equal(ego,new['ego_sigmoid'])

    def test_ties_and_static_winners_get_zero_positive_evidence(self):
        c=self.scores(FINE_BOXES)
        r=maps(c,FINE_BOXES)
        np.testing.assert_array_equal(r['external_positive'],np.zeros((20,20)))
        np.testing.assert_array_equal(r['external_sigmoid'],np.full((20,20),.5))
        c[:,-1]=.2
        r=maps(c,FINE_BOXES)
        self.assertEqual(float(r['external_positive'].max()),0)
        self.assertEqual(float(r['ego_positive'].max()),0)

    def test_local_evidence_does_not_spread_beyond_window(self):
        c=self.scores(FINE_BOXES); c[0,0]=.1
        m=maps(c,FINE_BOXES)['external_positive']
        self.assertGreater(float(m[:4,:4].min()),0)
        self.assertEqual(float(m[4:].sum()+m[:,4:].sum()),0)

    def test_ego_competes_with_external(self):
        c=self.scores(FINE_BOXES); c[:,0]=.1; c[:,len(base.DYNAMIC)]=.3
        r=maps(c,FINE_BOXES)
        self.assertEqual(float(r['external_positive'].max()),0)
        self.assertGreater(float(r['ego_positive'].min()),.9)

    def test_constant_zero_is_not_a_spatial_success(self):
        s=statistics(np.zeros((2,20,20)))
        self.assertEqual(s['all_zero_queries'],[0,1])
        self.assertIsNone(s['horizontal_variance_fraction'])

    def test_invalid_input_and_uncovered_grid(self):
        with self.assertRaises(ValueError): maps(np.zeros((2,17)),FINE_BOXES)
        with self.assertRaises(ValueError): maps(np.full((81,17),np.nan),FINE_BOXES)
        with self.assertRaises(ValueError): maps(np.zeros((1,17)),[(0,0,4,4)])


if __name__=='__main__': unittest.main()
