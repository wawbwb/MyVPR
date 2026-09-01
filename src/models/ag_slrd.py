"""Semantic-layout teacher used by the AG-SLRD Phase-0 screen.

The module deliberately contains no RGB encoder and no connection to the
student retrieval model.  It consumes a hard, coarse semantic label grid and
produces a unit-normalised descriptor.  Absolute layout is retained by
coordinate channels and spatial-pyramid pooling; label-specific features are
combined with learned, per-image class weights rather than handwritten class
priors.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


AG_SLRD_TEACHER_CHECKPOINT_SCHEMA = "openvpr_ag_slrd_semantic_teacher"
AG_SLRD_TEACHER_CHECKPOINT_VERSION = 1


def file_sha256(path: str | Path) -> str:
    """Return the SHA256 of *path* without reading the entire file at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 1 or result != value:
        raise ValueError(f"{name} must be a positive integer")
    return result


class _ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        groups = min(16, out_channels)
        while out_channels % groups:
            groups -= 1
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )


class SemanticLayoutEncoder(nn.Module):
    """Encode a coarse semantic layout into a VPR descriptor.

    Args:
        num_classes: Number of coarse semantic classes in the immutable cache.
        embed_dim: Width of the learned categorical embedding.
        channels: Three convolutional stage widths.
        descriptor_dim: Output descriptor dimension.
        ignore_index: Optional cache padding value.  Normal cache cells must be
            in ``[0, num_classes)``; ignored cells do not contribute to class
            pooling.

    Input is an integer tensor of shape ``(B,H,W)``.  Passing floating-point
    pseudo-labels is rejected: class IDs must remain categorical.  The default
    canonical input is 70x70, but tests and diagnostic extraction may use any
    spatial size at least 8x8.
    """

    def __init__(
        self,
        num_classes: int = 12,
        embed_dim: int = 32,
        channels: Sequence[int] = (64, 128, 256),
        descriptor_dim: int = 512,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.num_classes = _positive_int(num_classes, name="num_classes")
        self.embed_dim = _positive_int(embed_dim, name="embed_dim")
        self.descriptor_dim = _positive_int(
            descriptor_dim, name="descriptor_dim"
        )
        if len(tuple(channels)) != 3:
            raise ValueError("channels must contain exactly three stage widths")
        self.channels = tuple(
            _positive_int(value, name=f"channels[{index}]")
            for index, value in enumerate(channels)
        )
        if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
            raise TypeError("ignore_index must be an integer")
        if 0 <= ignore_index < self.num_classes:
            raise ValueError("ignore_index must lie outside the valid class IDs")
        self.ignore_index = int(ignore_index)

        # One extra entry is reserved for ignored cells.  It is learned because
        # an unknown patch still has a spatial footprint, but it is excluded
        # from every label-specific pool below.
        self.class_embedding = nn.Embedding(
            self.num_classes + 1, self.embed_dim
        )
        stages: list[nn.Module] = []
        in_channels = self.embed_dim + 2
        for out_channels in self.channels:
            stages.append(_ConvBlock(in_channels, out_channels))
            in_channels = out_channels
        self.encoder = nn.Sequential(*stages)

        feature_dim = self.channels[-1]
        class_dim = max(32, feature_dim // 2)
        self.class_dim = class_dim
        self.global_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim * 4 * 4, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )
        self.class_projection = nn.Sequential(
            nn.Linear(feature_dim, class_dim),
            nn.LayerNorm(class_dim),
            nn.GELU(),
        )
        self.class_weight_head = nn.Sequential(
            nn.Linear(class_dim + 1, class_dim),
            nn.GELU(),
            nn.Linear(class_dim, 1),
        )
        self.class_weight_bias = nn.Parameter(torch.zeros(self.num_classes))
        self.descriptor_head = nn.Sequential(
            nn.Linear(feature_dim + class_dim, self.descriptor_dim),
            nn.LayerNorm(self.descriptor_dim),
        )

    def export_config(self) -> dict[str, Any]:
        """Return the complete constructor configuration for checkpoints."""

        return {
            "num_classes": self.num_classes,
            "embed_dim": self.embed_dim,
            "channels": list(self.channels),
            "descriptor_dim": self.descriptor_dim,
            "ignore_index": self.ignore_index,
        }

    def _validate_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(labels):
            raise TypeError("semantic labels must be a torch.Tensor")
        if labels.ndim != 3:
            raise ValueError(
                "semantic labels must have shape (B,H,W), got "
                f"{tuple(labels.shape)}"
            )
        if labels.shape[0] < 1 or min(labels.shape[-2:]) < 8:
            raise ValueError(
                "semantic labels require a non-empty batch and an at least "
                "8x8 spatial grid"
            )
        if labels.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("semantic labels must use an integer dtype")
        labels = labels.to(dtype=torch.long)
        valid = (labels >= 0) & (labels < self.num_classes)
        ignored = labels == self.ignore_index
        if bool((~valid & ~ignored).any()):
            bad = labels[~valid & ~ignored][0].item()
            raise ValueError(
                f"semantic label {bad} is neither a valid class nor ignore_index"
            )
        if bool((~valid.flatten(1).any(dim=1)).any()):
            raise ValueError("every image must contain at least one valid class cell")
        return labels

    @staticmethod
    def _coordinate_channels(
        batch: int,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((yy, xx), dim=0).unsqueeze(0).expand(
            batch, -1, -1, -1
        )

    def forward(
        self, labels: torch.Tensor, *, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        labels = self._validate_labels(labels)
        valid = (labels >= 0) & (labels < self.num_classes)
        safe_labels = torch.where(
            valid, labels, torch.full_like(labels, self.num_classes)
        )
        embedded = self.class_embedding(safe_labels).permute(0, 3, 1, 2)
        coordinates = self._coordinate_channels(
            embedded.shape[0],
            embedded.shape[2],
            embedded.shape[3],
            device=embedded.device,
            dtype=embedded.dtype,
        )
        features = self.encoder(torch.cat((embedded, coordinates), dim=1))

        # A fixed 4x4 pyramid retains coarse absolute geometry.  A pure global
        # average would make many road/sky/building layouts indistinguishable.
        global_feature = self.global_projection(
            F.adaptive_avg_pool2d(features, output_size=(4, 4))
        )

        one_hot = F.one_hot(
            safe_labels.clamp_max(self.num_classes - 1),
            num_classes=self.num_classes,
        ).permute(0, 3, 1, 2)
        one_hot = one_hot.to(dtype=features.dtype)
        one_hot = one_hot * valid.unsqueeze(1).to(dtype=features.dtype)
        class_masks = F.adaptive_avg_pool2d(
            one_hot, output_size=features.shape[-2:]
        )
        mass = class_masks.sum(dim=(-2, -1))
        pooled = torch.einsum("bchw,bkhw->bkc", features, class_masks)
        pooled = pooled / mass.clamp_min(1e-6).unsqueeze(-1)
        class_features = self.class_projection(pooled)
        presence = mass > 0
        area_fraction = mass / float(features.shape[-2] * features.shape[-1])
        weight_input = torch.cat(
            (class_features, area_fraction.unsqueeze(-1)), dim=-1
        )
        weight_logits = self.class_weight_head(weight_input).squeeze(-1)
        weight_logits = weight_logits + self.class_weight_bias.unsqueeze(0)
        weight_logits = weight_logits.masked_fill(~presence, -torch.inf)
        class_weights = torch.softmax(weight_logits.float(), dim=-1).to(
            dtype=class_features.dtype
        )
        class_feature = torch.sum(
            class_weights.unsqueeze(-1) * class_features, dim=1
        )

        descriptor = self.descriptor_head(
            torch.cat((global_feature, class_feature), dim=-1)
        )
        descriptor = F.normalize(descriptor.float(), p=2, dim=-1).to(
            dtype=global_feature.dtype
        )
        if not return_aux:
            return descriptor
        return descriptor, {
            "class_weights": class_weights,
            "class_presence": presence,
            "class_area_fraction": area_fraction,
            "feature_map": features,
        }


def build_teacher_checkpoint(
    model: SemanticLayoutEncoder,
    *,
    mode: str,
    epoch: int,
    global_step: int,
    cache_provenance: Mapping[str, Any],
    data_config: Mapping[str, Any],
    trainer_config: Mapping[str, Any],
    optimizer_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the fail-closed, self-describing teacher checkpoint payload."""

    mode = str(mode).lower()
    if mode not in {"aligned", "shuffled"}:
        raise ValueError("teacher mode must be aligned or shuffled")
    if isinstance(epoch, bool) or int(epoch) < 0:
        raise ValueError("epoch must be a non-negative integer")
    if isinstance(global_step, bool) or int(global_step) < 0:
        raise ValueError("global_step must be a non-negative integer")
    result: dict[str, Any] = {
        "schema": AG_SLRD_TEACHER_CHECKPOINT_SCHEMA,
        "version": AG_SLRD_TEACHER_CHECKPOINT_VERSION,
        "mode": mode,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_config": model.export_config(),
        "state_dict": model.state_dict(),
        "cache": dict(cache_provenance),
        "data": dict(data_config),
        "trainer": dict(trainer_config),
    }
    if optimizer_state is not None:
        result["optimizer_state"] = dict(optimizer_state)
    return result


def load_semantic_layout_teacher(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[SemanticLayoutEncoder, dict[str, Any]]:
    """Strictly reconstruct a semantic-layout teacher checkpoint."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"semantic-layout checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("semantic-layout checkpoint must be a mapping")
    if checkpoint.get("schema") != AG_SLRD_TEACHER_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported semantic-layout checkpoint schema")
    if checkpoint.get("version") != AG_SLRD_TEACHER_CHECKPOINT_VERSION:
        raise ValueError("unsupported semantic-layout checkpoint version")
    model_config = checkpoint.get("model_config")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(model_config, Mapping) or not isinstance(
        state_dict, Mapping
    ):
        raise ValueError(
            "semantic-layout checkpoint requires model_config and state_dict"
        )
    model = SemanticLayoutEncoder(**dict(model_config))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, dict(checkpoint)


__all__ = [
    "AG_SLRD_TEACHER_CHECKPOINT_SCHEMA",
    "AG_SLRD_TEACHER_CHECKPOINT_VERSION",
    "SemanticLayoutEncoder",
    "build_teacher_checkpoint",
    "file_sha256",
    "load_semantic_layout_teacher",
]
