import torch
from torch import nn

from scripts.eval_condition_robustness import InferenceModel
from src.models.semantic_region_gate import SemanticRegionGate


def test_condition_inference_applies_semantic_gate_to_tuple_local_features() -> None:
    class TupleBackbone(nn.Module):
        def forward(self, images):
            return images, images.mean(dim=(2, 3))

    class CaptureAggregator(nn.Module):
        def __init__(self):
            super().__init__()
            self.local = None
            self.cls = None

        def forward(self, output):
            self.local, self.cls = output
            return self.local.flatten(1)

    gate = SemanticRegionGate(2, alpha=0.2)
    with torch.no_grad():
        gate.proj.weight.zero_()
        gate.proj.bias.fill_(10.0)
    aggregator = CaptureAggregator()
    model = InferenceModel(
        TupleBackbone(), aggregator, semantic_region_gate=gate
    )
    images = torch.ones(2, 2, 2, 2)

    model(images)

    torch.testing.assert_close(
        aggregator.local, images * (1.0 + 0.2 * torch.tanh(torch.tensor(10.0)))
    )
    torch.testing.assert_close(aggregator.cls, images.mean(dim=(2, 3)))
