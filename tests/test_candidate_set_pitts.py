import unittest
import numpy as np
from scripts.candidate_set_pitts import validate_row,validate_outcomes
from src.candidate_set_utils import summary


class CacheTests(unittest.TestCase):
    def row(self):
        z=np.zeros((20,512),np.float32);z[:,0]=1
        return dict(evidence=np.zeros((20,1536),np.float32),base=np.zeros(20,np.float32),
            db_vectors=z,candidates=np.arange(20),global_scores=np.arange(20,0,-1,dtype=np.float32),
            labels=np.isin(np.arange(20),[2,7]))

    def test_multiple_positives_and_unreachable(self):
        validate_row(self.row(),[2,7,25],30)
        row=self.row();row['labels'][:]=False;validate_row(row,[25],30)

    def test_wrong_gt_or_candidate_order_rejected(self):
        with self.assertRaisesRegex(ValueError,'GT'):validate_row(self.row(),[2],30)
        row=self.row();row['candidates'][0]=row['candidates'][1]
        with self.assertRaisesRegex(ValueError,'indices'):validate_row(row,[2,7],30)
        row=self.row();row['global_scores'][1]=100
        with self.assertRaisesRegex(ValueError,'order'):validate_row(row,[2,7],30)

    def test_corrupt_feature_rejected(self):
        row=self.row();row['evidence'][0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'evidence'):validate_row(row,[2,7],30)
        row=self.row();row['db_vectors']*=2
        with self.assertRaisesRegex(ValueError,'Unnormalized'):validate_row(row,[2,7],30)

    def test_admission_reference_and_mapping(self):
        # Admission pair corrections are relative to global, not to pair itself.
        data=dict(base=np.tile([2.,1.],(12,1)),global_scores=np.tile([2.,1.],(12,1)),
            labels=np.tile([False,True],(12,1)),candidates=np.tile([0,1],(12,1)))
        data['base'][0]=[1.,2.]
        qids=np.arange(100,112);outcomes={**data,'query_indices':qids.copy()}
        decision={'passed':True,'pair':summary(data['base'],data['labels'],data['global_scores'])}
        validate_outcomes(data,outcomes,qids,decision)
        with self.assertRaisesRegex(ValueError,'mapping'):validate_outcomes(data,outcomes,qids[::-1],decision)
        wrong={**decision,'pair':summary(data['base'],data['labels'],data['base'])}
        with self.assertRaisesRegex(ValueError,'changed'):validate_outcomes(data,outcomes,qids,wrong)
