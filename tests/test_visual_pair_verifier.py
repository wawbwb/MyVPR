import torch
from src.visual_pair_verifier import VisualPairVerifier, pair_edges, anchored_score


def test_zero_start_exact_baseline_and_gradient():
    model = VisualPairVerifier()
    edges = torch.randn(3, 64, 8)
    edges[..., 7] = 1
    ru = torch.tensor([0.2, 0.5, 0.8])
    residual = model(edges, ru)
    assert torch.equal(anchored_score(ru, residual, 0.1), ru)
    residual.sum().backward()
    assert model.head[-1].weight.grad.abs().sum() > 0


def test_empty_edges_finite():
    model = VisualPairVerifier()
    assert torch.isfinite(model(torch.zeros(2, 64, 8), torch.ones(2))).all()


def test_identity_edges_and_broadcast():
    tokens = torch.eye(400)[None]
    edges = pair_edges(tokens, tokens.expand(2, -1, -1))
    assert edges.shape == (2, 64, 8)
    assert torch.all(edges[..., 7] == 1)
    assert torch.allclose(edges[..., 0], torch.ones(2, 64))
    assert torch.all(edges[..., 5:7] == 0)


def test_edge_order_invariance():
    model = VisualPairVerifier()
    torch.nn.init.normal_(model.head[-1].weight)
    edges = torch.randn(2, 64, 8)
    edges[..., 7] = 1
    ru = torch.ones(2)
    assert torch.allclose(model(edges, ru), model(edges[:, torch.randperm(64)], ru), atol=1e-6)
