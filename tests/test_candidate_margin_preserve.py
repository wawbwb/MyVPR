import unittest
import torch
from scripts.candidate_margin_preserve import objective
from src.models.candidate_set import CandidateSet


class MarginPreserveTests(unittest.TestCase):
    def test_correct_margin_drop_penalized(self):
        base = torch.tensor([[3., 0.]])
        scores = torch.tensor([[2., 0.]], requires_grad=True)
        y = torch.tensor([[True, False]])
        total, retrieval, penalty, n = objective(scores, base, y, 1)
        self.assertAlmostEqual(penalty.item(), 1.)
        self.assertEqual(n.item(), 1)
        torch.testing.assert_close(total, retrieval+penalty)
        penalty.backward()
        self.assertLess(scores.grad[0, 0].item(), 0)
        self.assertGreater(scores.grad[0, 1].item(), 0)

    def test_correct_improvement_not_penalized(self):
        total, retrieval, penalty, _ = objective(torch.tensor([[4., 0.]]), torch.tensor([[3., 0.]]),
                                                torch.tensor([[True, False]]), 1)
        self.assertEqual(penalty.item(), 0.)
        torch.testing.assert_close(total, retrieval)

    def test_wrong_query_has_no_preservation(self):
        _, _, penalty, n = objective(torch.tensor([[-5., 0.]]), torch.tensor([[-1., 0.]]),
                                     torch.tensor([[True, False]]), 1)
        self.assertEqual(penalty.item(), 0.)
        self.assertEqual(n.item(), 0)

    def test_mean_over_all_queries(self):
        _, _, penalty, _ = objective(torch.tensor([[2., 0.], [-5., 0.]]),
            torch.tensor([[3., 0.], [-1., 0.]]), torch.tensor([[True, False], [True, False]]), 1)
        self.assertAlmostEqual(penalty.item(), .5)

    def test_multi_positive_and_common_offset(self):
        b = torch.tensor([[3., 2., 0.]])
        y = torch.tensor([[True, True, False]])
        _, _, p1, _ = objective(torch.tensor([[1., 3., 0.]]), b, y, 1)
        _, _, p2, _ = objective(torch.tensor([[11., 13., 10.]]), b, y, 1)
        self.assertEqual(p1.item(), 0.)
        self.assertEqual(p2.item(), 0.)

    def test_reference_detached(self):
        b = torch.tensor([[3., 0.]], requires_grad=True)
        s = torch.tensor([[2., 0.]], requires_grad=True)
        total, _, _, _ = objective(s, b, torch.tensor([[True, False]]), 1)
        total.backward(); self.assertIsNone(b.grad)

    def test_zero_weight_matches_original_gradient(self):
        y = torch.tensor([[True, False, True]])
        s = torch.tensor([[1., 3., 2.]], requires_grad=True)
        total, retrieval, _, _ = objective(s, torch.tensor([[4., 0., 1.]]), y, 0)
        a = torch.autograd.grad(total, s, retain_graph=True)[0]
        b = torch.autograd.grad(retrieval, s)[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_unreachable_rejected(self):
        with self.assertRaises(ValueError):
            objective(torch.ones(1, 2), torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.bool), 1)

    def test_zero_start_penalty_and_update(self):
        torch.manual_seed(42)
        model = CandidateSet('independent', input_dim=6, db_dim=3, width=8)
        x = torch.randn(2, 4, 6); z = torch.randn(2, 4, 3)
        b = torch.tensor([[2., 0., 0., 0.], [0., 2., 0., 0.]])
        y = torch.tensor([[True, False, False, False], [True, False, False, False]])
        s = model(x, b, z); torch.testing.assert_close(s, b, rtol=0, atol=0)
        loss, _, penalty, _ = objective(s, b, y, 1)
        self.assertEqual(penalty.item(), 0.)
        loss.backward(); self.assertGreater(model.out.weight.grad.abs().sum().item(), 0)


if __name__ == '__main__': unittest.main()
