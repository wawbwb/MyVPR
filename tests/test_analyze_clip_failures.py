import unittest
import numpy as np
from scripts.analyze_clip_failures import anchor_changes


class AnchorTest(unittest.TestCase):
    def test_fixed_anchors_when_winner_changes(self):
        s=np.array([[.7,.8,.1],[.6,.5,.9],[.8,.7,.1]])
        r=anchor_changes(s,0,1)
        self.assertAlmostEqual(r[1]['positive_delta'],-.1)
        self.assertAlmostEqual(r[1]['negative_delta'],-.3)
        self.assertAlmostEqual(r[1]['fixed_margin_delta'],.2)
        self.assertEqual(s[1].argmax(),2)

    def test_invalid_anchors(self):
        with self.assertRaises(ValueError): anchor_changes(np.zeros((3,4)),1,1)
        with self.assertRaises(ValueError): anchor_changes(np.full((3,4),np.nan),1,2)


if __name__=='__main__': unittest.main()
