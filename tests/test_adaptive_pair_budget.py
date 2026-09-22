import unittest
import numpy as np
from src.adaptive_pair_budget import prefix_margin,decide,outcomes,fixed_random

class BudgetTests(unittest.TestCase):
    def test_no_future_input(self):
        with self.assertRaises(ValueError):decide(np.zeros((2,44)),1)
        x=np.arange(88.).reshape(2,44)
        a=decide(x[:,:20],1)
        x[:,20:]=-99999
        np.testing.assert_array_equal(a,decide(x[:,:20],1))
    def test_margin_and_tie(self):
        np.testing.assert_array_equal(prefix_margin(np.tile(np.arange(20.),(2,1))),[1,1])
        np.testing.assert_array_equal(decide(np.zeros((2,20)),0),[44,44])
    def test_budget_blocks_future_winner(self):
        x=np.arange(44.)[None,:];y=np.zeros((1,44),bool);y[0,43]=True
        self.assertFalse(outcomes(x,y,np.array([20]))[0][0])
        self.assertTrue(outcomes(x,y,np.array([44]))[0][0])
    def test_random_fixed(self):
        np.testing.assert_array_equal(fixed_random([3,7],.5),fixed_random([3,7],.5))
        np.testing.assert_array_equal(fixed_random([3,7],0),[20,20])
    def test_invalid(self):
        with self.assertRaises(ValueError):outcomes(np.zeros((1,44)),np.zeros((1,44),bool),np.array([45]))
