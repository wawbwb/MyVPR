import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.dynamic_challenge_v2_utils import low_overlap,summary
from scripts.dynamic_challenge_utils import SCHEMES


class Controls(unittest.TestCase):
    def test_known_budget_and_overlap(self):
        m=np.zeros((280,280),bool); m[160:230,90:160]=True
        control,status=low_overlap(m,'fixed')
        self.assertTrue(status['valid']); self.assertLessEqual(status['overlap'],.1)
        self.assertEqual(control.sum(),m.sum())
        a,b=np.argwhere(m),np.argwhere(control)
        np.testing.assert_array_equal(a-a.min(0),b-b.min(0))
        np.testing.assert_array_equal(control,low_overlap(m,'fixed')[0])

    def test_soft_controls_and_impossible_case(self):
        m=np.zeros((20,20),np.float32); m[10:15,10:15]=.8
        control,status=low_overlap(m,'fixed',periodic=True)
        self.assertTrue(status['valid']); self.assertLessEqual(status['overlap'],.1)
        np.testing.assert_array_equal(np.sort(control.ravel()),np.sort(m.ravel()))
        _,status=low_overlap(np.ones((20,20)),'fixed',periodic=True)
        self.assertFalse(status['valid']); self.assertEqual(status['min_overlap'],1.)

    def test_unavailable_is_not_failure(self):
        rows=[]
        for content,seed,load in [('clean',0,0.),('object',11,.15)]:
            for q in [0,1]:
                for scheme in SCHEMES:
                    eligible=not(content=='object' and q==1 and scheme=='clip_shift')
                    row=dict(case=f'{content}{q}',query=q,content=content,seed=seed,load=load,scheme=scheme,eligible=eligible)
                    if eligible: row['top1_correct']=int(content=='clean' or scheme=='clip')
                    rows.append(row)
        g=summary(rows)['groups']['object|seed=11|area=0.15']
        self.assertEqual(g['schemes']['clip_shift']['eligible'],1)
        self.assertEqual(g['schemes']['clip_shift']['excluded'],1)
        self.assertEqual(g['paired_controls']['clip']['queries'],1)
        self.assertEqual(g['paired_controls']['clip']['aligned_only_correct'],[0])
        with self.assertRaises(ValueError): summary(rows[:-1])


if __name__=='__main__': unittest.main()
