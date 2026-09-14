import tempfile
from pathlib import Path
import unittest
import numpy as np
from scripts.pitts_pair_admission import load_index


class IndexTests(unittest.TestCase):
    def fixture(self,root):
        db=['000/a.jpg','001/b.jpg'];q=['queries_real/a.jpg','queries_real/b.jpg']
        for name in db+q:
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'placeholder')
        np.save(root/'pitts30k_val_dbImages.npy',np.array(db))
        np.save(root/'pitts30k_val_qImages.npy',np.array(q))
        gt=np.empty(2,dtype=object);gt[0]=np.array([0,1]);gt[1]=np.array([1])
        np.save(root/'pitts30k_val_gt_25m.npy',gt)

    def test_fixed_sample_full_database_multigt(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root);a=load_index(root,1)
            self.assertEqual(len(a['database']),2);self.assertEqual(len(a['queries']),1)
            self.assertEqual(a,load_index(root,1))
            full=load_index(root,1024)
            for i,g in zip(full['query_indices'],full['gt']):self.assertEqual(g,[0,1] if i==0 else [1])

    def test_bad_gt_and_missing_image_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            np.save(root/'pitts30k_val_gt_25m.npy',np.array([[0],[9]]))
            with self.assertRaises(ValueError):load_index(root)
            self.fixture(root);(root/'000/a.jpg').unlink()
            with self.assertRaises(FileNotFoundError):load_index(root)


if __name__=='__main__':unittest.main()
