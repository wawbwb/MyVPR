import unittest
from scripts.eval_clip_token_cross import CELLS,name,summarize


class CrossTest(unittest.TestCase):
    def rows(self):
        return [{'query_index':q,'variant':v,'top1_correct':int(q==0),'best_gt_rank':1 if q==0 else 2}
                for q in range(2) for v in ['frozen']+[name(t,i) for t,i in CELLS]]

    def test_row_baseline_not_other_training_arm(self):
        rows=self.rows()
        for r in rows:
            if r['variant']=='aligned__none' and r['query_index']==0:r.update(top1_correct=0,best_gt_rank=2)
        s=summarize(rows,2)
        self.assertEqual(s['within_checkpoint']['aligned__aligned']['vs_same_checkpoint_all_keep']['corrected'],[0])
        self.assertEqual(s['within_checkpoint']['none__aligned']['vs_same_checkpoint_all_keep']['corrected'],[])
        self.assertEqual(len(s['matrix']),9)

    def test_incomplete_or_duplicate_rejected(self):
        rows=self.rows()
        for invalid in [rows[:-1],rows+[rows[0]]]:
            with self.assertRaises(ValueError):summarize(invalid,2)


if __name__=='__main__':unittest.main()
