import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dynamic_challenge_utils import composite,shifted,shift_scores,summary,SCHEMES


class ChallengeTests(unittest.TestCase):
    def test_matched_support_and_no_background_change(self):
        background=np.full((280,280,3),30,np.uint8)
        donor=np.zeros((80,100,3),np.uint8); donor[:]=[120,180,220]; donor[10:30]=255
        mask=np.ones((80,100),bool); mask[:10,:20]=False
        obj,flat,support=composite(background,donor,mask,.15,'fixed')
        np.testing.assert_array_equal(obj[~support],background[~support])
        np.testing.assert_array_equal(flat[~support],background[~support])
        self.assertLess(abs(support.mean()-.15),.005)
        self.assertFalse(np.array_equal(obj,flat))
        np.testing.assert_array_equal(obj,composite(background,donor,mask,.15,'fixed')[0])

    def test_translation_controls_preserve_budget(self):
        m=np.zeros((280,280),bool); m[40:100,90:130]=True
        s=shifted(m,'fixed'); self.assertEqual(m.sum(),s.sum())
        aa,bb=np.argwhere(m),np.argwhere(s)
        np.testing.assert_array_equal(aa-aa.min(0),bb-bb.min(0))
        f=np.arange(400).reshape(20,20)/400
        np.testing.assert_array_equal(np.sort(f.ravel()),np.sort(shift_scores(f,'fixed').ravel()))

    def test_rescue_counts_use_clean_damage(self):
        rows=[]
        for content,seed,load in [('clean',0,0.),('object',11,.15)]:
            for q in [0,1]:
                for scheme in SCHEMES:
                    correct=(q==0) if content=='clean' else (scheme=='known')
                    rows.append(dict(case=f'{content}{q}',query=q,content=content,seed=seed,load=load,
                                     scheme=scheme,top1_correct=int(correct),positive_negative_margin=float(correct)))
        result=summary(rows)['groups']['object|seed=11|area=0.15']
        self.assertEqual(result['damaged_from_clean'],1)
        self.assertEqual(result['schemes']['known']['rescued_damaged'],[0])
        self.assertEqual(result['schemes']['known']['corrected_vs_unmasked'],[0,1])
        with self.assertRaises(ValueError): summary(rows[:-1])


if __name__=='__main__': unittest.main()
