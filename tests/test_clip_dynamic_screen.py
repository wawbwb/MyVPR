import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.clip_dynamic_screen import DYNAMIC,EGO,STATIC,heatmap,controls,outcome,shard_read,localization_diagnostics


class ClipDynamicTests(unittest.TestCase):
    def scores(self):
        return np.zeros((16,len(DYNAMIC)+len(EGO)+len(STATIC)))

    def test_ties_are_half_not_claimed_zero_dynamic(self):
        mask,ego=heatmap(self.scores())
        np.testing.assert_array_equal(mask,np.full((20,20),.5))
        np.testing.assert_array_equal(ego,mask)

    def test_competing_categories_and_full_spatial_support(self):
        scores=self.scores(); scores[:,0]=.3
        mask,ego=heatmap(scores)
        self.assertGreater(mask.min(),.99)
        self.assertLess(ego.max(),.01)
        scores[:,len(DYNAMIC)]=.6
        mask,ego=heatmap(scores)
        self.assertLess(mask.max(),.01)
        self.assertGreater(ego.min(),.99)

    def test_controls_preserve_vertical_area_and_are_identity_deterministic(self):
        mask=np.arange(400,dtype=np.float32).reshape(20,20)/400
        c=controls(mask,'query.jpg')
        np.testing.assert_array_equal(c['shuffle'],controls(mask,'query.jpg')['shuffle'])
        self.assertFalse(np.array_equal(c['shuffle'],mask))
        for i in range(20): np.testing.assert_array_equal(np.sort(c['shuffle'][i]),np.sort(mask[i]))
        for value in c.values(): np.testing.assert_allclose(value.mean(1),mask.mean(1),atol=1e-7)

    def test_rank_uses_all_gt_and_stable_ties(self):
        result=outcome(np.array([.1,.8,.8,.9]),np.array([1,2]))
        self.assertEqual(result['best_gt_reference_index'],1)
        self.assertEqual(result['best_gt_rank'],2)
        self.assertEqual(result['top1_correct'],0)
        self.assertAlmostEqual(result['positive_negative_margin'],-.1)

    def test_shard_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'shard.npz'; scores=self.scores(); mask,ego=heatmap(scores)
            np.savez(file,cosines=scores,mask=mask,ego_mask=ego,image_sha='fixed')
            np.testing.assert_array_equal(shard_read(file,'fixed')[0],mask)
            with self.assertRaises(ValueError): shard_read(file,'changed')
            np.savez(file,cosines=scores,mask=np.zeros((20,20)),ego_mask=ego,image_sha='fixed')
            with self.assertRaises(ValueError): shard_read(file,'fixed')

    def test_annotation_diagnostic_does_not_modify_heatmap(self):
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'annotations.json'
            self.assertFalse(localization_diagnostics(None,None,None,file)['available'])
            file.write_text(json.dumps({'cases':[{'decision':'include','query_index':0,'query_path':'q.jpg',
                'external':{'polygons':[[[0,0],[.2,0],[.2,.2],[0,.2]]]},'ego':{'polygons':[]}}]}))
            masks=np.full((1,20,20),.5,dtype=np.float32); original=masks.copy()
            result=localization_diagnostics(masks,masks,['q.jpg'],file)
            self.assertEqual(result['regions'][0]['inside_minus_outside'],0)
            np.testing.assert_array_equal(masks,original)


if __name__=='__main__': unittest.main()
