import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.cc_lsa_features import extract_ru_descriptor_and_local, mutual_nearest_edges_batch
from src.cc_lsa_gate_a import mutual_nearest_edges


def test_batched_mnn_matches_scalar_definition_and_ties():
    query = np.eye(3, dtype=np.float32)
    candidates = np.stack([query, query[[2, 0, 1]]])
    batched = mutual_nearest_edges_batch(query, candidates, device=torch.device("cpu"))
    for candidate, edges in zip(candidates, batched):
        scalar = mutual_nearest_edges(query, candidate)
        for actual, expected in zip(edges, scalar):
            np.testing.assert_allclose(actual, expected)
    tied = np.ones((1, 3, 3), dtype=np.float32)
    left, right, _ = mutual_nearest_edges_batch(query, tied, device=torch.device("cpu"))[0]
    assert left.tolist() == [0]
    assert right.tolist() == [0]


def test_mnn_rejects_corrupt_zero_norm_cache():
    with pytest.raises(RuntimeError, match="zero norm"):
        mutual_nearest_edges_batch(np.zeros((3, 2)), np.ones((2, 3, 2)), device=torch.device("cpu"))


class CountingBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        return images


class SpatialGate(nn.Module):
    def forward(self, value):
        gate = torch.linspace(0.8, 1.2, value.shape[-1]).view(1, 1, 1, -1)
        return value * gate, None, gate


class MockRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CountingBackbone()
        self.semantic_region_gate = SpatialGate()
        self.aggregator = nn.Flatten()


def test_ru_feature_capture_is_before_gate_and_uses_one_forward():
    model = MockRU()
    images = torch.rand(2, 3, 20, 20)
    descriptor, local = extract_ru_descriptor_and_local(model, images)
    assert model.backbone.calls == 1
    expected_local = F.normalize(F.adaptive_avg_pool2d(images, (14, 14)).flatten(2).transpose(1, 2), dim=-1)
    expected_descriptor = F.normalize(model.semantic_region_gate(images)[0].flatten(1), dim=-1)
    torch.testing.assert_close(local, expected_local)
    torch.testing.assert_close(descriptor, expected_descriptor)
