import unittest
import numpy as np
from scripts.wppr_extension import remaining_indices,metrics

class ExtensionTests(unittest.TestCase):
    def test_complement(self):
        s={'calibration':{'indices':list(range(128))},'evaluation':{'indices':list(range(128,384))}}
        self.assertEqual(remaining_indices(s),list(range(384,2048)))
        s['evaluation']['indices'][0]=0
        with self.assertRaises(ValueError):remaining_indices(s)
    def test_lost_winner_preserves_accuracy(self):
        t=np.zeros((1,44));t[0,43]=9;t[0,0]=8
        pred=-np.arange(44.)[None,:];labels=np.zeros((1,44),bool);labels[0,[0,43]]=True
        r,c,_=metrics(pred,t,labels,[91])
        self.assertEqual(r['winner_retained'],0)
        self.assertEqual(r['selected12_full_correct'],1)
        self.assertEqual(r['regressions_vs44'],[])
        self.assertEqual(c[0]['query_index'],91)
