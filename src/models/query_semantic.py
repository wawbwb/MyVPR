"""Training-only semantic targets for query-conditioned BoQ.

The segmentation teacher is represented by a compact offline cache.  The
student semantic head and all query-conditioning parameters live in BoQ and are
therefore the only semantic components needed at inference.
"""

from __future__ import annotations

import math
import hashlib
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import torch
from torch.nn import functional as F


QUERY_SEMANTIC_MODES = frozenset(
    {"architecture_only", "aligned", "shuffled", "random"}
)


def _canonical_checkpoint_state(state: Mapping) -> OrderedDict:
    prefix = "_orig_mod."
    return OrderedDict(
        (
            str(key)[len(prefix) :] if str(key).startswith(prefix) else str(key),
            value,
        )
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_query_semantic_cache_hashes(
    cache_dir: str | Path,
    manifest: Mapping,
) -> dict[str, str]:
    """Verify every immutable array recorded by a completed cache manifest."""

    cache_dir = Path(cache_dir).expanduser().resolve()
    declared = manifest.get("array_sha256")
    required = (
        "labels.npy",
        "confidence.npy",
        "shuffled_indices.npy",
    )
    if not isinstance(declared, Mapping) or set(declared) != set(required):
        raise ValueError(
            "query semantic cache manifest must contain SHA256 values for "
            f"exactly {required}"
        )

    actual_hashes: dict[str, str] = {}
    for filename in required:
        expected = declared.get(filename)
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError(
                f"query semantic cache has an invalid SHA256 for {filename}"
            )
        path = cache_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"query semantic cache array not found: {path}"
            )
        actual = _file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"query semantic cache SHA256 mismatch for {filename}: "
                f"expected {expected}, found {actual}"
            )
        actual_hashes[filename] = actual
    return actual_hashes


def _validate_ru_checkpoint_config(checkpoint: Mapping) -> dict[str, object]:
    config = checkpoint.get("hyper_parameters")
    if not isinstance(config, Mapping):
        raise RuntimeError(
            "RU warm start requires checkpoint hyper_parameters for provenance"
        )
    distillation = config.get("distillation")
    if not isinstance(distillation, Mapping):
        raise RuntimeError("RU checkpoint has no distillation configuration")
    semantic_region = distillation.get("semantic_region")
    if not isinstance(semantic_region, Mapping):
        raise RuntimeError("RU checkpoint has no semantic_region configuration")
    if not bool(semantic_region.get("enabled", False)):
        raise RuntimeError("warm-start checkpoint has no trained semantic gate")
    if str(semantic_region.get("mode", "")).lower() != (
        "repeatability_uniqueness_only"
    ):
        raise RuntimeError(
            "warm-start checkpoint is not repeatability+uniqueness (RU)"
        )
    try:
        region_weight = float(semantic_region.get("lambda_target", 0.0))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("RU checkpoint has an invalid semantic-region weight") from exc
    if not math.isfinite(region_weight) or region_weight != 0.02:
        raise RuntimeError(
            "RU checkpoint must use the matched semantic-region lambda=0.02"
        )
    query_semantic = distillation.get("query_semantic") or {}
    if not isinstance(query_semantic, Mapping) or bool(
        query_semantic.get("enabled", False)
    ):
        raise RuntimeError(
            "warm-start checkpoint already contains a query-semantic experiment"
        )

    backbone = config.get("backbone") or {}
    aggregator = config.get("aggregator") or {}
    datamodule = config.get("datamodule") or {}
    if not all(
        isinstance(section, Mapping)
        for section in (backbone, aggregator, datamodule)
    ):
        raise RuntimeError("RU checkpoint has malformed model/data configuration")
    if backbone.get("class") != "DinoV2" or aggregator.get("class") != "BoQ":
        raise RuntimeError("RU checkpoint must contain the DINOv2-BoQ model")
    if list(datamodule.get("train_image_size", ())) != [280, 280]:
        raise RuntimeError("RU checkpoint must have been trained at 280x280")
    if str(datamodule.get("augmentation_mode", "")).lower() != "photometric":
        raise RuntimeError("RU checkpoint must use photometric-only augmentation")
    if int(datamodule.get("batch_size", -1)) != 40 or int(
        datamodule.get("img_per_place", -1)
    ) != 4:
        raise RuntimeError("RU checkpoint must use the matched P=40, K=4 setup")
    if datamodule.get("train_set_name") != "gsv-cities" or datamodule.get(
        "cities"
    ) not in ("all", None):
        raise RuntimeError("RU checkpoint must use full GSV-Cities")
    if int(config.get("seed", -1)) != 42:
        raise RuntimeError("RU checkpoint must use the matched seed 42")
    if float(semantic_region.get("alpha", float("nan"))) != 0.2:
        raise RuntimeError("RU checkpoint must use semantic gate alpha=0.2")
    trainer = config.get("trainer") or {}
    if not isinstance(trainer, Mapping) or int(
        trainer.get("max_epochs", -1)
    ) != 40:
        raise RuntimeError("RU checkpoint must come from the 40-epoch run")
    return {
        "source_seed": 42,
        "source_semantic_mode": "repeatability_uniqueness_only",
        "source_semantic_weight": region_weight,
    }


def warm_start_query_semantic_model(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    """Load a historical RU checkpoint while allowing only new semantic keys.

    This is deliberately stricter than a general ``strict=False`` load: any
    missing legacy backbone, BoQ or RU-gate parameter aborts the experiment.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"initial checkpoint not found: {checkpoint_path}")
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    if expected_sha256 is not None:
        expected_sha256 = str(expected_sha256).lower()
        if (
            len(expected_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_sha256
            )
        ):
            raise ValueError("expected RU checkpoint SHA256 must be 64 lowercase hex")
        if checkpoint_sha256 != expected_sha256:
            raise RuntimeError(
                "RU checkpoint SHA256 mismatch: expected "
                f"{expected_sha256}, found {checkpoint_sha256}"
            )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("initial checkpoint must be a mapping")
    provenance = _validate_ru_checkpoint_config(checkpoint)
    raw_state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(raw_state, Mapping):
        raise TypeError("initial checkpoint contains no state_dict mapping")
    state = _canonical_checkpoint_state(raw_state)
    for prefix in ("backbone.", "aggregator.", "semantic_region_gate."):
        if not any(key.startswith(prefix) for key in state):
            raise RuntimeError(
                f"RU warm start requires checkpoint weights with prefix {prefix!r}"
            )

    expected_new_keys = {
        key
        for key in model.state_dict()
        if key.startswith("aggregator.semantic_head.")
        or (
            key.startswith("aggregator.boqs.")
            and ".semantic_query_proj." in key
        )
    }
    present_new_keys = expected_new_keys.intersection(state)
    if present_new_keys:
        raise RuntimeError(
            "RU warm start must not contain query-semantic parameters: "
            f"{sorted(present_new_keys)}"
        )
    try:
        incompatible = model.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(
            f"RU checkpoint is incompatible with the configured base model: {exc}"
        ) from exc
    allowed_missing = []
    illegal_missing = []
    for key in incompatible.missing_keys:
        if key.startswith("aggregator.semantic_head.") or (
            key.startswith("aggregator.boqs.")
            and ".semantic_query_proj." in key
        ):
            allowed_missing.append(key)
        else:
            illegal_missing.append(key)
    if illegal_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "unsafe RU warm start: missing legacy keys="
            f"{illegal_missing}, unexpected keys={list(incompatible.unexpected_keys)}"
        )
    if set(allowed_missing) != expected_new_keys or not allowed_missing:
        raise RuntimeError(
            "RU warm start must initialize the complete query-semantic branch; "
            f"expected={sorted(expected_new_keys)}, missing={sorted(allowed_missing)}"
        )
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "loaded_keys": len(state) - len(incompatible.unexpected_keys),
        "new_keys": tuple(sorted(allowed_missing)),
        **provenance,
    }


def freeze_for_query_semantic_screen(model: torch.nn.Module) -> tuple[str, ...]:
    """Freeze RU and leave only the BoQ semantic branch trainable."""
    if getattr(model, "semantic_region_gate", None) is None:
        raise RuntimeError("frozen query-semantic screen requires the RU gate")
    aggregator = getattr(model, "aggregator", None)
    if aggregator is None or not hasattr(aggregator, "semantic_parameters"):
        raise RuntimeError("configured aggregator has no query-semantic parameters")
    model.requires_grad_(False)
    semantic_parameters = list(aggregator.semantic_parameters())
    if not semantic_parameters:
        raise RuntimeError("configured aggregator has an empty semantic branch")
    for parameter in semantic_parameters:
        parameter.requires_grad_(True)
    model._query_semantic_base_frozen = True
    trainable = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if not trainable or any(
        not (
            name.startswith("aggregator.semantic_head.")
            or (
                name.startswith("aggregator.boqs.")
                and ".semantic_query_proj." in name
            )
        )
        for name in trainable
    ):
        raise RuntimeError(
            f"unexpected trainable parameters after frozen-base setup: {trainable}"
        )
    return trainable


def _affine_patch_parameters(
    patch_count: int,
    cache_index: int,
    seed: int,
) -> tuple[int, int]:
    """Return stable affine-permutation parameters for one cached image.

    An affine permutation is inexpensive, batch-order independent and preserves
    the exact joint histogram of labels and confidences.  The multiplier is
    chosen coprime to ``patch_count`` so every patch occurs exactly once.
    """
    if patch_count < 2:
        raise ValueError("random semantic control requires at least two patches")
    candidate = 3 + 2 * ((int(cache_index) + int(seed)) % patch_count)
    candidate %= patch_count
    if candidate < 2:
        candidate += 3
    attempts = 0
    while math.gcd(candidate, patch_count) != 1 or candidate % patch_count == 1:
        candidate = (candidate + 2) % patch_count
        if candidate < 2:
            candidate += 3
        attempts += 1
        if attempts > patch_count:
            raise RuntimeError("could not construct a random patch permutation")
    offset = (
        int(cache_index) * 2_654_435_761 + int(seed) * 97_531
    ) % patch_count
    return candidate, offset


def _affine_patch_permutation(
    patch_count: int,
    cache_index: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a stable non-identity permutation for one cached image."""
    candidate, offset = _affine_patch_parameters(
        patch_count, cache_index, seed
    )
    positions = torch.arange(patch_count, device=device, dtype=torch.long)
    return (candidate * positions + offset) % patch_count


def spatially_randomize_targets(
    labels: torch.Tensor,
    confidence: torch.Tensor,
    cache_indices: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically permute label/confidence pairs within every image."""
    if labels.ndim != 3 or confidence.shape != labels.shape:
        raise ValueError("labels and confidence must have shape (B,H,W)")
    if cache_indices.ndim != 1 or cache_indices.numel() != labels.shape[0]:
        raise ValueError("cache_indices must have shape (B,)")
    flat_labels = labels.flatten(1)
    flat_confidence = confidence.flatten(1)
    parameters = [
        _affine_patch_parameters(flat_labels.shape[1], cache_index, seed)
        for cache_index in cache_indices.detach().cpu().tolist()
    ]
    multipliers = torch.tensor(
        [value[0] for value in parameters],
        device=labels.device,
        dtype=torch.long,
    )
    offsets = torch.tensor(
        [value[1] for value in parameters],
        device=labels.device,
        dtype=torch.long,
    )
    positions = torch.arange(
        flat_labels.shape[1], device=labels.device, dtype=torch.long
    )
    permutations = (
        multipliers[:, None] * positions[None] + offsets[:, None]
    ) % flat_labels.shape[1]
    random_labels = torch.gather(flat_labels, dim=1, index=permutations)
    random_confidence = torch.gather(
        flat_confidence, dim=1, index=permutations
    )
    return random_labels.view_as(labels), random_confidence.view_as(confidence)


class QuerySemanticTarget:
    """Confidence-filtered cached segmentation supervision."""

    def __init__(
        self,
        mode: str,
        num_classes: int,
        min_confidence: float = 0.5,
        ignore_index: int = 255,
        random_seed: int = 42,
    ) -> None:
        mode = str(mode).lower()
        if mode not in QUERY_SEMANTIC_MODES:
            raise ValueError(
                f"query semantic mode must be one of {sorted(QUERY_SEMANTIC_MODES)}"
            )
        if num_classes < 2 or num_classes > 255:
            raise ValueError("num_classes must be in [2, 255]")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if not 0 <= ignore_index <= 255:
            raise ValueError("ignore_index must be in [0, 255]")
        self.mode = mode
        self.num_classes = int(num_classes)
        self.min_confidence = float(min_confidence)
        self.ignore_index = int(ignore_index)
        self.random_seed = int(random_seed)

    @staticmethod
    def _confidence_to_float(confidence: torch.Tensor) -> torch.Tensor:
        if confidence.dtype == torch.uint8:
            return confidence.float().div_(255.0)
        confidence = confidence.float()
        if not bool(torch.isfinite(confidence).all()):
            raise ValueError("query semantic confidence must be finite")
        if bool(((confidence < 0) | (confidence > 1)).any()):
            raise ValueError("query semantic confidence must be in [0, 1]")
        return confidence

    def __call__(
        self,
        semantic_logits: torch.Tensor,
        labels: torch.Tensor,
        confidence: torch.Tensor,
        cache_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if semantic_logits.ndim != 4:
            raise ValueError("semantic_logits must have shape (B,C,H,W)")
        batch_size, channels, height, width = semantic_logits.shape
        if channels != self.num_classes:
            raise ValueError(
                f"semantic logits have {channels} classes, expected {self.num_classes}"
            )
        expected = (batch_size, height, width)
        if tuple(labels.shape) != expected or tuple(confidence.shape) != expected:
            raise ValueError(
                "cached semantic target grid must exactly match logits: "
                f"expected {expected}, got labels={tuple(labels.shape)}, "
                f"confidence={tuple(confidence.shape)}"
            )
        if cache_indices.ndim != 1 or cache_indices.numel() != batch_size:
            raise ValueError("cache_indices must have shape (B,)")
        if not bool(torch.isfinite(semantic_logits).all()):
            raise ValueError("semantic_logits must be finite")

        labels = labels.long()
        confidence = self._confidence_to_float(confidence)
        if self.mode == "random":
            labels, confidence = spatially_randomize_targets(
                labels, confidence, cache_indices.long(), self.random_seed
            )

        label_is_valid = (labels >= 0) & (labels < self.num_classes)
        not_ignored = labels != self.ignore_index
        valid = (
            label_is_valid
            & not_ignored
            & (confidence >= self.min_confidence)
        )
        safe_labels = labels.masked_fill(~label_is_valid, 0)
        per_patch = F.cross_entropy(
            semantic_logits.float(), safe_labels, reduction="none"
        )
        weights = confidence * valid.float()
        weight_sum = weights.sum()
        if bool(valid.any()):
            loss = (per_patch * weights).sum() / weight_sum.clamp_min(1e-12)
            accuracy = (
                (semantic_logits.argmax(dim=1) == safe_labels).float() * valid
            ).sum() / valid.sum()
            valid_confidence = confidence[valid].mean()
        else:
            # Keep a differentiable scalar so DDP sees the semantic head even
            # when a rare batch has no patch above the confidence threshold.
            loss = semantic_logits.sum() * 0.0
            accuracy = semantic_logits.new_zeros(())
            valid_confidence = semantic_logits.new_zeros(())

        probabilities = semantic_logits.float().softmax(dim=1)
        entropy = -(
            probabilities.clamp_min(1e-12)
            * probabilities.clamp_min(1e-12).log()
        ).sum(dim=1)
        entropy = entropy.mean() / math.log(self.num_classes)
        stats = {
            "query_semantic_valid_frac": valid.float().mean(),
            "query_semantic_valid_confidence": valid_confidence,
            "query_semantic_accuracy": accuracy,
            "query_semantic_entropy_norm": entropy,
        }
        return loss, stats
