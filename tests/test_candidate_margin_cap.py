import unittest
from unittest.mock import patch
import torch
from scripts import candidate_margin_cap as cap
from scripts import candidate_margin_preserve as original


class CappedMarginTests(unittest.TestCase):
    def penalty(self, before, after):
        s = torch.tensor([[float(after), 0.]], requires_grad=True)
        b = torch.tensor([[float(before), 0.]], requires_grad=True)
        y = torch.tensor([[True, False]])
        return cap.objective(s, b, y, 1), s, b

    def test_easy_margin_can_shrink(self):
        (total, retrieval, penalty, _), _, _ = self.penalty(25, 24)
        self.assertEqual(penalty.item(), 0)
        torch.testing.assert_close(total, retrieval)

    def test_cap_breach_penalized(self):
        (_, _, penalty, _), s, b = self.penalty(25, .5)
        self.assertAlmostEqual(penalty.item(), .5)
        penalty.backward()
        self.assertLess(s.grad[0, 0].item(), 0)
        self.assertIsNone(b.grad)

    def test_near_correct_original_margin_kept(self):
        (_, _, penalty, _), _, _ = self.penalty(.3, .1)
        self.assertAlmostEqual(penalty.item(), .2, places=6)

    def test_wrong_frozen_query_not_preserved(self):
        (_, _, penalty, n), _, _ = self.penalty(-1, -3)
        self.assertEqual(penalty.item(), 0)
        self.assertEqual(n.item(), 0)

    def test_mean_over_all_queries(self):
        _, _, penalty, _ = cap.objective(torch.tensor([[.5, 0.], [-3., 0.]]),
            torch.tensor([[25., 0.], [-1., 0.]]), torch.tensor([[True, False], [True, False]]), 1)
        self.assertAlmostEqual(penalty.item(), .25)

    def test_original_arm_gradient_unchanged(self):
        s = torch.tensor([[.1, .5, -.5]], requires_grad=True)
        b = torch.tensor([[3., 0., 2.]])
        y = torch.tensor([[True, False, True]])
        loss0 = original.objective(s, b, y, 0)[0]
        loss1 = cap.objective(s, b, y, 0)[0]
        torch.testing.assert_close(loss0, loss1, rtol=0, atol=0)
        g0 = torch.autograd.grad(loss0, s, retain_graph=True)[0]
        g1 = torch.autograd.grad(loss1, s)[0]
        torch.testing.assert_close(g0, g1, rtol=0, atol=0)

    def test_multi_positive_best_can_change(self):
        _, _, p, _ = cap.objective(torch.tensor([[0., 1.5, 0.]]),
            torch.tensor([[25., 2., 0.]]), torch.tensor([[True, True, False]]), 1)
        self.assertEqual(p.item(), 0)

    def test_zero_start(self):
        (_, _, p, _), _, _ = self.penalty(.3, .3)
        self.assertEqual(p.item(), 0)

    def test_unreachable_rejected(self):
        with self.assertRaises(ValueError):
            cap.objective(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool), 1)

    def test_runner_scoped_configuration_restored(self):
        saved = original.objective, original.POLICY, original.codes
        def inspect():
            self.assertIs(original.objective, cap.objective)
            self.assertEqual(original.POLICY['preservation_cap'], 1.)
            self.assertIn('scripts/candidate_margin_cap.py', original.codes())
            import sys
            self.assertIn('logs/candidate_margin_cap_v1', sys.argv)
        with patch.object(original, 'main', side_effect=inspect):
            cap.main(['--acknowledge-exploratory'])
        self.assertEqual((original.objective, original.POLICY, original.codes), saved)

    def test_restore_on_failure(self):
        saved = original.objective, original.POLICY, original.codes
        with patch.object(original, 'main', side_effect=ValueError('stop')):
            with self.assertRaises(ValueError): cap.main(['--acknowledge-exploratory'])
        self.assertEqual((original.objective, original.POLICY, original.codes), saved)


if __name__ == '__main__': unittest.main()
