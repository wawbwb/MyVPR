import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dynamic_invariance_utils import masks,views,make_split


class Controls(unittest.TestCase):
    def test_translation_preserves_shape_no_wrap(self):
        f=np.zeros((20,20)); f[5:9,7:10]=1; f[9,9]=1
        a,b,reason=masks(f,'test')
        self.assertEqual(reason,'active'); self.assertEqual(a.sum(),b.sum())
        aa=np.argwhere(a); bb=np.argwhere(b)
        np.testing.assert_array_equal(aa-aa.min(0),bb-bb.min(0))
        self.assertFalse(np.array_equal(a,b))
        np.testing.assert_array_equal(b,masks(f,'test')[1])

    def test_unmovable_and_large_skip_both(self):
        f=np.zeros((20,20)); f[0,0]=1; f[-1,-1]=1
        a,b,reason=masks(f,'test')
        self.assertEqual(reason,'unshiftable'); self.assertFalse(a.any() or b.any())
        self.assertFalse(masks(np.ones((20,20)),'test')[0].any())

    def test_perturbation_support_and_determinism(self):
        x=np.full((3,280,280),.5,np.float32)
        m=np.zeros((20,20),bool); m[2:4,2:4]=True
        a,b=views(x,m,'test',0); outside=~np.repeat(np.repeat(m,14,0),14,1)
        np.testing.assert_array_equal(a[:,outside],x[:,outside])
        self.assertFalse(np.array_equal(a,b))
        np.testing.assert_array_equal(a,views(x,m,'test',0)[0])
        np.testing.assert_array_equal(x,views(x,np.zeros_like(m),'test',0)[0])

    def test_city_and_place_separation(self):
        records=[[f'Images/{city}/{label}_{view}.jpg',label] for city,offset in [('A',0),('B',300)]
                 for label in range(offset,offset+256) for view in range(4)]
        split=make_split(records)
        train={records[i][0].split('/')[1] for _,ids in split['train'] for i in ids}
        dev={records[i][0].split('/')[1] for _,ids in split['dev'] for i in ids}
        self.assertFalse(train&dev)
        self.assertEqual(split,make_split(records))


if __name__=='__main__': unittest.main()
