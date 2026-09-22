import unittest
import numpy as np
from scripts.diagnose_pair_budget import features,auc,diagnose

class DiagnosticTests(unittest.TestCase):
    def test_prefix_only(self):
        with self.assertRaises(ValueError):features(np.zeros((2,44)))
        f=features(np.zeros((2,20)))
        self.assertTrue(all(np.isfinite(v).all() for v in f.values()))
    def test_auc(self):
        self.assertEqual(auc([1,2],[False,True]),1)
        self.assertEqual(auc([1,1],[False,True]),.5)
        self.assertIsNone(auc([1],[True]))
    def test_high_confidence_missed_fix(self):
        s=np.zeros((2,44));s[:,0]=10;s[:,25]=20
        y=np.zeros((2,44),bool);y[0,25]=True;y[1,0]=True
        r,c=diagnose(s,y,[91,38],1)
        self.assertEqual(r['groups']['missed_corrections']['query_ids'],[91])
        self.assertEqual(r['groups']['regressions']['query_ids'],[38])
        self.assertEqual(c[0]['new_winner_rank'],26)
