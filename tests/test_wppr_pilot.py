import unittest
import numpy as np
from scripts.wppr_pilot import retention,capture_cls

class PilotTests(unittest.TestCase):
    def test_retention(self):
        x=np.arange(44.)[None,:]
        h,k,w=retention(x,x)
        self.assertTrue(h[0]);self.assertEqual(w[0],43)
        self.assertFalse(retention(-x,x)[0][0])
    def test_invalid(self):
        with self.assertRaises(ValueError):retention(np.zeros((2,20)),np.zeros((2,20)))
    def test_capture(self):
        import torch
        x=torch.randn(1,5,768)
        y=capture_cls((x,x));self.assertEqual(tuple(y.shape),(1,768))
        self.assertFalse(y.requires_grad)
        x[:,1:]=999
        torch.testing.assert_close(y,capture_cls((x,x)))
