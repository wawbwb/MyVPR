"""Crop-CLS local-semantic feature modulation for DINOv2-BoQ.

This module implements the small inference-time student used by the
SemVPR-inspired screen.  A frozen CLIP teacher is optional and is owned by the
plain-Python target object, so teacher weights never enter training
checkpoints.  The FiLM projection is zero-initialised: inserting the module in
an RU model leaves the first forward bit-for-bit unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.models.query_semantic import (
    _canonical_checkpoint_state,
    _file_sha256,
    _validate_ru_checkpoint_config,
)


CROP_SEMANTIC_MODES = frozenset(
    {"aligned", "wrong_region", "wrong_place"}
)
CROP_SEMANTIC_NEW_KEY_PREFIX = "backbone.crop_semantic_film."


class CropSemanticFiLM(nn.Module):
    """Predict continuous local semantics and modulate DINO patch channels.

    The caller supplies patch tokens only (no CLS token) with shape ``B,N,C``.
    The returned semantic tokens are deliberately not L2-normalised; the
    target pools a complete 2x2 image region before applying cosine loss.

    ``channel_scale`` has exactly zero weight and bias at construction, so
    ``modulated_patch_tokens`` initially equals ``patch_tokens`` exactly.
    """

    def __init__(
        self,
        in_channels: int = 768,
        hidden_dim: int = 128,
        semantic_dim: int = 512,
        alpha: float = 0.1,
    ) -> None:
        super().__init__()
        if int(in_channels) <= 0:
            raise ValueError("in_channels must be positive")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(semantic_dim) <= 0:
            raise ValueError("semantic_dim must be positive")
        if not math.isfinite(float(alpha)) or not 0.0 < float(alpha) <= 0.2:
            raise ValueError("alpha must be finite and in (0, 0.2]")

        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.semantic_dim = int(semantic_dim)
        self.alpha = float(alpha)

        self.bottleneck = nn.Linear(self.in_channels, self.hidden_dim)
        self.activation = nn.GELU()
        self.semantic_projection = nn.Linear(
            self.hidden_dim, self.semantic_dim
        )
        self.channel_scale = nn.Linear(self.hidden_dim, self.in_channels)
        nn.init.zeros_(self.channel_scale.weight)
        nn.init.zeros_(self.channel_scale.bias)

        self._bypass_depth = 0
        self._last_diagnostics: dict[str, torch.Tensor] = {}

    @property
    def bypassed(self) -> bool:
        return self._bypass_depth > 0

    @contextmanager
    def bypass(self):
        """Temporarily disable feature modulation without changing weights."""

        self._bypass_depth += 1
        try:
            yield self
        finally:
            self._bypass_depth -= 1
            if self._bypass_depth < 0:  # defensive guard against misuse
                self._bypass_depth = 0
                raise RuntimeError("CropSemanticFiLM bypass depth underflow")

    def forward(
        self,
        patch_tokens: torch.Tensor,
        *,
        return_semantic: bool = True,
        semantic_batch_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if patch_tokens.ndim != 3:
            raise ValueError("patch_tokens must have shape (B,N,C)")
        if patch_tokens.shape[-1] != self.in_channels:
            raise ValueError(
                "patch token width does not match in_channels: "
                f"expected {self.in_channels}, found {patch_tokens.shape[-1]}"
            )
        if not patch_tokens.is_floating_point():
            raise TypeError("patch_tokens must be floating point")

        hidden = self.activation(self.bottleneck(patch_tokens))
        if semantic_batch_indices is not None:
            if not return_semantic:
                raise ValueError(
                    "semantic_batch_indices requires return_semantic=true"
                )
            if (
                semantic_batch_indices.ndim != 1
                or not semantic_batch_indices.numel()
            ):
                raise ValueError(
                    "semantic_batch_indices must be a non-empty 1D tensor"
                )
            semantic_batch_indices = semantic_batch_indices.to(
                device=hidden.device, dtype=torch.long
            )
            if bool(
                (
                    (semantic_batch_indices < 0)
                    | (semantic_batch_indices >= hidden.shape[0])
                ).any()
            ):
                raise ValueError("semantic_batch_indices are out of range")
            semantic_hidden = hidden.index_select(
                0, semantic_batch_indices
            )
        else:
            semantic_hidden = hidden
        semantic_tokens = (
            self.semantic_projection(semantic_hidden)
            if return_semantic
            else None
        )
        raw_scale = self.channel_scale(hidden)
        modulation = self.alpha * torch.tanh(raw_scale.float())
        modulation = modulation.to(dtype=patch_tokens.dtype)
        if self.bypassed:
            modulated = patch_tokens
            applied_modulation = torch.zeros_like(modulation)
        else:
            modulated = patch_tokens * (1.0 + modulation)
            applied_modulation = modulation

        with torch.no_grad():
            applied_fp32 = applied_modulation.detach().float()
            raw_scale_fp32 = raw_scale.detach().float()
            self._last_diagnostics = {
                "crop_film_raw_scale_std": raw_scale_fp32.std(
                    unbiased=False
                ),
                "crop_film_modulation_rms": applied_fp32.square().mean().sqrt(),
                "crop_film_modulation_abs_max": applied_fp32.abs().amax(),
                "crop_film_bypassed": raw_scale_fp32.new_tensor(
                    float(self.bypassed)
                ),
            }
            if semantic_tokens is not None:
                semantic_fp32 = semantic_tokens.detach().float()
                self._last_diagnostics.update(
                    {
                        "crop_semantic_token_std": semantic_fp32.std(
                            unbiased=False
                        ),
                        "crop_semantic_token_norm": semantic_fp32.norm(
                            dim=-1
                        ).mean(),
                    }
                )
        return modulated, semantic_tokens, raw_scale

    def diagnostics(self) -> dict[str, torch.Tensor]:
        """Return detached scalar diagnostics from the most recent forward."""

        return {
            name: value.detach()
            for name, value in self._last_diagnostics.items()
        }


class CropCLSSemanticTarget:
    """Region-pooled cosine target backed by one clean CLIP crop per place.

    This is intentionally not an ``nn.Module``.  Its optional teacher is
    frozen, lazy, absent from the student state dict, and may be injected in
    tests.  A training step selects one view per place and one shared quadrant
    ``global_step % 4``.  ``wrong_region`` compares the same teacher embedding
    with the opposite student quadrant; ``wrong_place`` rolls teacher
    embeddings by one place while preserving the quadrant.
    """

    def __init__(
        self,
        mode: str = "aligned",
        teacher: Any | None = None,
        teacher_factory: Callable[[], Any] | None = None,
        teacher_model_name: str = "ViT-B-16",
        teacher_pretrained: str = "openai",
        teacher_hf_mirror: str | None = "https://hf-mirror.com",
        teacher_chunk_size: int = 20,
        expected_teacher_image_size: tuple[int, int] = (280, 280),
    ) -> None:
        mode = str(mode).lower()
        if mode not in CROP_SEMANTIC_MODES:
            raise ValueError(
                f"mode must be one of {sorted(CROP_SEMANTIC_MODES)}"
            )
        if teacher is not None and teacher_factory is not None:
            raise ValueError("provide teacher or teacher_factory, not both")
        self.mode = mode
        self._teacher = teacher
        self._teacher_factory = teacher_factory
        self.teacher_model_name = str(teacher_model_name)
        self.teacher_pretrained = str(teacher_pretrained)
        self.teacher_hf_mirror = teacher_hf_mirror
        if isinstance(teacher_chunk_size, bool) or int(teacher_chunk_size) < 1:
            raise ValueError("teacher_chunk_size must be a positive integer")
        self.teacher_chunk_size = int(teacher_chunk_size)
        if (
            len(expected_teacher_image_size) != 2
            or any(int(value) <= 0 for value in expected_teacher_image_size)
        ):
            raise ValueError(
                "expected_teacher_image_size must contain two positive integers"
            )
        self.expected_teacher_image_size = tuple(
            int(value) for value in expected_teacher_image_size
        )
        if teacher is not None:
            self._freeze_teacher(teacher)

    @staticmethod
    def _freeze_teacher(teacher: Any) -> None:
        if isinstance(teacher, nn.Module):
            teacher.requires_grad_(False)
            teacher.eval()

    def _get_teacher(self, device: torch.device) -> Any:
        if self._teacher is None:
            if self._teacher_factory is not None:
                self._teacher = self._teacher_factory()
            else:
                from src.models.clip_teacher import CLIPTeacherEncoder

                self._teacher = CLIPTeacherEncoder(
                    model_name=self.teacher_model_name,
                    pretrained=self.teacher_pretrained,
                    hf_mirror=self.teacher_hf_mirror,
                )
            self._freeze_teacher(self._teacher)
        if isinstance(self._teacher, nn.Module):
            self._teacher.to(device)
            self._teacher.eval()
        return self._teacher

    def prepare_teacher(self, device: torch.device | str = "cpu") -> Any:
        """Construct the frozen teacher before the final experiment reseed.

        The target remains a plain Python object, so the teacher is still
        excluded from the Lightning checkpoint and optimiser.
        """

        return self._get_teacher(torch.device(device))

    @staticmethod
    def region_indices(
        place_count: int, global_step: int, device: torch.device
    ) -> torch.Tensor:
        """Use one shared quadrant per step; four steps cover all quadrants."""

        if int(place_count) <= 0:
            raise ValueError("place_count must be positive")
        quadrant = int(global_step) % 4
        return torch.full(
            (int(place_count),), quadrant, device=device, dtype=torch.long
        )

    @staticmethod
    def _validate_quadrants(
        quadrants: torch.Tensor, place_count: int
    ) -> torch.Tensor:
        if quadrants.ndim != 1 or quadrants.numel() != place_count:
            raise ValueError("region_indices must have shape (P,)")
        quadrants = quadrants.long()
        if bool(((quadrants < 0) | (quadrants > 3)).any()):
            raise ValueError("region_indices values must be in [0, 3]")
        return quadrants

    @staticmethod
    def _select_views(
        value: torch.Tensor,
        view_indices: torch.Tensor | None,
        value_name: str,
    ) -> torch.Tensor:
        if value.ndim not in (4, 5):
            return value
        # Student tokens: P,K,N,D (4D). Teacher images: P,K,C,H,W (5D).
        place_count, view_count = value.shape[:2]
        if view_indices is None:
            view_indices = torch.zeros(
                place_count, device=value.device, dtype=torch.long
            )
        if view_indices.ndim != 1 or view_indices.numel() != place_count:
            raise ValueError(f"{value_name} view_indices must have shape (P,)")
        view_indices = view_indices.to(device=value.device, dtype=torch.long)
        if bool(((view_indices < 0) | (view_indices >= view_count)).any()):
            raise ValueError(
                f"{value_name} view_indices must be in [0, {view_count - 1}]"
            )
        place_indices = torch.arange(place_count, device=value.device)
        return value[place_indices, view_indices]

    @classmethod
    def _prepare_student_tokens(
        cls,
        student_tokens: torch.Tensor,
        view_indices: torch.Tensor | None,
        place_count: int | None,
        views_per_place: int | None,
    ) -> torch.Tensor:
        if student_tokens.ndim == 4:
            student_tokens = cls._select_views(
                student_tokens, view_indices, "student_tokens"
            )
        elif student_tokens.ndim == 3:
            if place_count is not None or views_per_place is not None:
                if place_count is None or views_per_place is None:
                    raise ValueError(
                        "place_count and views_per_place must be supplied together"
                    )
                expected = int(place_count) * int(views_per_place)
                if student_tokens.shape[0] != expected:
                    raise ValueError(
                        "flattened student batch does not match P*K: "
                        f"expected {expected}, found {student_tokens.shape[0]}"
                    )
                student_tokens = student_tokens.reshape(
                    int(place_count),
                    int(views_per_place),
                    student_tokens.shape[1],
                    student_tokens.shape[2],
                )
                student_tokens = cls._select_views(
                    student_tokens, view_indices, "student_tokens"
                )
        else:
            raise ValueError(
                "student_tokens must have shape (P,N,D), (P,K,N,D), "
                "or flattened (P*K,N,D)"
            )
        if not student_tokens.is_floating_point():
            raise TypeError("student_tokens must be floating point")
        return student_tokens

    @staticmethod
    def _square_grid_size(patch_count: int) -> int:
        side = math.isqrt(int(patch_count))
        if side * side != int(patch_count) or side % 2:
            raise ValueError(
                "student patch count must form an even square grid"
            )
        return side

    @classmethod
    def _pool_regions(
        cls, student_tokens: torch.Tensor, quadrants: torch.Tensor
    ) -> torch.Tensor:
        place_count, patch_count, feature_dim = student_tokens.shape
        side = cls._square_grid_size(patch_count)
        grid = student_tokens.reshape(
            place_count, side, side, feature_dim
        )
        half = side // 2
        pooled = []
        for place_index, quadrant in enumerate(quadrants.tolist()):
            row = quadrant // 2
            column = quadrant % 2
            region = grid[
                place_index,
                row * half : (row + 1) * half,
                column * half : (column + 1) * half,
            ]
            pooled.append(region.mean(dim=(0, 1)))
        return torch.stack(pooled)

    def _crop_teacher_regions(
        self, teacher_images: torch.Tensor, quadrants: torch.Tensor
    ) -> torch.Tensor:
        if teacher_images.ndim != 4 or teacher_images.shape[1] != 3:
            raise ValueError("teacher_images must have shape (P,3,H,W)")
        if teacher_images.shape[0] != quadrants.numel():
            raise ValueError("teacher image count must equal place count")
        height, width = teacher_images.shape[-2:]
        if (height, width) != self.expected_teacher_image_size:
            raise ValueError(
                "teacher image size does not match the registered crop "
                f"protocol: expected {self.expected_teacher_image_size}, "
                f"found {(height, width)}"
            )
        if height < 2 or width < 2 or height % 2 or width % 2:
            raise ValueError("teacher image height and width must be even")
        half_height, half_width = height // 2, width // 2
        crops = []
        for place_index, quadrant in enumerate(quadrants.tolist()):
            row = quadrant // 2
            column = quadrant % 2
            crops.append(
                teacher_images[
                    place_index,
                    :,
                    row * half_height : (row + 1) * half_height,
                    column * half_width : (column + 1) * half_width,
                ]
            )
        return torch.stack(crops)

    @staticmethod
    def _teacher_output_tensor(output: Any) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)) and output:
            if torch.is_tensor(output[0]):
                return output[0]
        if isinstance(output, Mapping):
            for key in ("t_global", "global", "image_features"):
                value = output.get(key)
                if torch.is_tensor(value):
                    return value
        raise TypeError(
            "teacher must return a tensor, a tuple starting with a tensor, "
            "or a mapping containing t_global/global/image_features"
        )

    def _encode_teacher(self, crops: torch.Tensor) -> torch.Tensor:
        teacher = self._get_teacher(crops.device)
        chunks = []
        with torch.no_grad():
            for crop_chunk in crops.split(self.teacher_chunk_size, dim=0):
                output = teacher(crop_chunk)
                chunks.append(
                    self._teacher_output_tensor(output).detach().float()
                )
        embeddings = torch.cat(chunks, dim=0)
        if embeddings.ndim != 2 or embeddings.shape[0] != crops.shape[0]:
            raise ValueError("teacher embeddings must have shape (P,D)")
        if not bool(torch.isfinite(embeddings).all()):
            raise ValueError("teacher embeddings must be finite")
        return embeddings

    def __call__(
        self,
        student_tokens: torch.Tensor,
        *,
        teacher_images: torch.Tensor | None = None,
        teacher_embeddings: torch.Tensor | None = None,
        region_indices: torch.Tensor | None = None,
        view_indices: torch.Tensor | None = None,
        global_step: int = 0,
        place_count: int | None = None,
        views_per_place: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        student_tokens = self._prepare_student_tokens(
            student_tokens,
            view_indices,
            place_count,
            views_per_place,
        )
        place_count = student_tokens.shape[0]
        if place_count <= 0:
            raise ValueError("semantic target requires at least one place")
        if region_indices is None:
            region_indices = self.region_indices(
                place_count, global_step, student_tokens.device
            )
        else:
            region_indices = region_indices.to(student_tokens.device)
        region_indices = self._validate_quadrants(
            region_indices, place_count
        )

        if (teacher_images is None) == (teacher_embeddings is None):
            raise ValueError(
                "provide exactly one of teacher_images or teacher_embeddings"
            )
        if teacher_embeddings is None:
            if teacher_images.ndim == 5:
                teacher_images = self._select_views(
                    teacher_images, view_indices, "teacher_images"
                )
            teacher_images = teacher_images.to(student_tokens.device)
            crops = self._crop_teacher_regions(
                teacher_images, region_indices
            )
            teacher_embeddings = self._encode_teacher(crops)
        else:
            teacher_embeddings = teacher_embeddings.to(
                device=student_tokens.device, dtype=torch.float32
            )
            if teacher_embeddings.ndim != 2 or teacher_embeddings.shape[0] != (
                place_count
            ):
                raise ValueError("teacher_embeddings must have shape (P,D)")
            if not bool(torch.isfinite(teacher_embeddings).all()):
                raise ValueError("teacher_embeddings must be finite")

        if teacher_embeddings.shape[1] != student_tokens.shape[2]:
            raise ValueError(
                "teacher/student semantic dimensions differ: "
                f"{teacher_embeddings.shape[1]} vs {student_tokens.shape[2]}"
            )

        # Quadrants use row-major indices: 0=TL, 1=TR, 2=BL, 3=BR.
        # ``3 - q`` is therefore the true diagonal control (TL<->BR,
        # TR<->BL); ``(q + 2) % 4`` would only swap rows.
        opposite_regions = 3 - region_indices
        aligned_student = self._pool_regions(
            student_tokens, region_indices
        )
        wrong_region_student = self._pool_regions(
            student_tokens, opposite_regions
        )
        aligned_student = F.normalize(aligned_student.float(), dim=-1)
        wrong_region_student = F.normalize(
            wrong_region_student.float(), dim=-1
        )
        teacher_embeddings = F.normalize(
            teacher_embeddings.float(), dim=-1
        )

        aligned_cosine = (aligned_student * teacher_embeddings).sum(dim=-1)
        wrong_region_cosine = (
            wrong_region_student * teacher_embeddings
        ).sum(dim=-1)
        if place_count >= 2:
            wrong_place_teacher = teacher_embeddings.roll(1, dims=0)
            wrong_place_cosine = (
                aligned_student * wrong_place_teacher
            ).sum(dim=-1)
            wrong_place_valid = aligned_cosine.new_ones(())
        else:
            wrong_place_cosine = aligned_cosine.new_zeros(
                aligned_cosine.shape
            )
            wrong_place_valid = aligned_cosine.new_zeros(())

        selected_cosine = {
            "aligned": aligned_cosine,
            "wrong_region": wrong_region_cosine,
            "wrong_place": wrong_place_cosine,
        }[self.mode]
        if self.mode == "wrong_place" and place_count < 2:
            loss = student_tokens.sum() * 0.0
        else:
            loss = 1.0 - selected_cosine.mean()

        aligned_mean = aligned_cosine.mean()
        wrong_region_mean = wrong_region_cosine.mean()
        wrong_place_mean = wrong_place_cosine.mean()
        stats = {
            "crop_semantic_cosine": selected_cosine.mean(),
            "crop_semantic_aligned_cosine": aligned_mean,
            "crop_semantic_wrong_region_cosine": wrong_region_mean,
            "crop_semantic_wrong_place_cosine": wrong_place_mean,
            "crop_semantic_aligned_minus_wrong_region": (
                aligned_mean - wrong_region_mean
            ),
            "crop_semantic_aligned_minus_wrong_place": (
                aligned_mean - wrong_place_mean
            ),
            "crop_semantic_wrong_place_valid": wrong_place_valid,
            "crop_semantic_quadrant": aligned_mean.new_tensor(
                float(int(region_indices[0]))
            ),
        }
        return loss, stats


def warm_start_crop_semantic_film_model(
    model: nn.Module,
    checkpoint_path: str | Path,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    """Strictly warm-start RU while allowing exactly the new FiLM branch.

    Any missing historical backbone/BoQ/RU-gate key, any unexpected key, or a
    checkpoint that already contains even part of the new branch aborts.
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
        if key.startswith(CROP_SEMANTIC_NEW_KEY_PREFIX)
    }
    if not expected_new_keys:
        raise RuntimeError(
            "configured model has no backbone.crop_semantic_film parameters"
        )
    present_new_keys = expected_new_keys.intersection(state)
    if present_new_keys:
        raise RuntimeError(
            "RU warm start must not contain crop-semantic FiLM parameters: "
            f"{sorted(present_new_keys)}"
        )

    try:
        incompatible = model.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(
            f"RU checkpoint is incompatible with the configured base model: {exc}"
        ) from exc
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing != expected_new_keys or unexpected:
        illegal_missing = sorted(missing - expected_new_keys)
        absent_new = sorted(expected_new_keys - missing)
        raise RuntimeError(
            "unsafe RU crop-semantic warm start: missing legacy keys="
            f"{illegal_missing}, new keys not reported missing={absent_new}, "
            f"unexpected keys={sorted(unexpected)}"
        )

    film = getattr(getattr(model, "backbone", None), "crop_semantic_film", None)
    if film is None or not hasattr(film, "channel_scale"):
        raise RuntimeError("configured model has no usable crop-semantic FiLM")
    if torch.count_nonzero(film.channel_scale.weight).item() or torch.count_nonzero(
        film.channel_scale.bias
    ).item():
        raise RuntimeError(
            "crop-semantic FiLM channel_scale must remain exactly zero after "
            "RU warm start"
        )

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "loaded_keys": len(state),
        "new_keys": tuple(sorted(expected_new_keys)),
        "film_zero_initialized": True,
        **provenance,
    }
