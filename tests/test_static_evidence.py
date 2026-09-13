import unittest
import numpy as np
from src.models.static_evidence import selection,coordinates,make_batches


class StaticEvidenceTests(unittest.TestCase):
    def test_selection_and_fallback(self):
        f=np.zeros((20,20));f[:4,:4]=.75
        mask,ok=selection(f,'a','static')
        self.assertTrue(ok);self.assertEqual((~mask).sum(),16)
        self.assertFalse(selection(f,'a','full')[0].any())
        f[0,0]=.74
        mask,ok=selection(f,'a','static')
        self.assertFalse(ok);self.assertFalse(mask.any())
        with self.assertRaises(ValueError):selection(np.full((20,20),np.nan),'a','static')

    def test_matched_random(self):
        f=np.zeros((20,20));f[4:12,2:8]=1
        mask,_=selection(f,'example','static');random,_=selection(f,'example','random')
        self.assertEqual(mask.sum(),random.sum())
        self.assertFalse(np.array_equal(mask,random))
        np.testing.assert_array_equal(random,selection(f,'example','random')[0])

    def test_coordinates(self):
        self.assertEqual(coordinates('Images/San_Francisco/San_Francisco_123_2020_01_90_37.5_-122.5_pano.jpg'),(37.5,-122.5))

    def test_shared_hard_batches(self):
        coords=np.array([[0,i*.002] for i in range(256)])
        # Place 1 is a false-negative candidate too close to place 0.
        coords[1]=[0,.00001]
        z=np.random.default_rng(7).normal(size=(256,8));z/=np.linalg.norm(z,axis=1,keepdims=True)
        batches=make_batches(z,coords)
        self.assertEqual(batches,make_batches(z,coords))
        self.assertEqual([len(e) for e in batches],[64,64])
        for epoch in batches:
            for batch in epoch:
                self.assertEqual(len(set(batch)),16)
                self.assertFalse(0 in batch and 1 in batch)
        with self.assertRaises(ValueError):make_batches(z,np.zeros((256,2)))


if __name__=='__main__':unittest.main()
