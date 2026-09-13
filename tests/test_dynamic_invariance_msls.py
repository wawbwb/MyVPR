import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.eval_dynamic_invariance_msls import summarize


class SummaryTests(unittest.TestCase):
    def fixture(self):
        hits={'frozen':[1,0,1],'plain':[1,1,0],'random':[0,1,1],'semantic':[1,1,1]}
        return [dict(query_index=q,variant=m,top1_correct=h,best_gt_rank=1 if h else 6)
                for m,values in hits.items() for q,h in enumerate(values)]

    def test_paired_outcomes(self):
        s=summarize(self.fixture(),3)['models']
        self.assertEqual(s['plain']['comparisons']['frozen'],{'corrected':[1],'regressed':[2],'net':0})
        self.assertEqual(s['semantic']['comparisons']['random']['corrected'],[0])
        self.assertEqual(s['frozen']['recall']['R@5'],2/3)

    def test_reject_incomplete_and_duplicate(self):
        rows=self.fixture()
        with self.assertRaises(ValueError): summarize(rows[:-1],3)
        with self.assertRaises(ValueError): summarize(rows+[rows[0]],3)


if __name__=='__main__': unittest.main()
