"""DINO-anchored residual CLIP fusion for the Phase-A VPR screen.

The trainable fusion branch is part of the DINO backbone so its parameters are
saved with the VPR checkpoint.  The frozen OpenCLIP encoder is deliberately
owned by a plain Python provider: it is reconstructed from the pinned config
and never duplicated inside Lightning checkpoints.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
import math
from numbers import Real
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


RESIDUAL_CLIP_MODES = frozenset(
    {"aligned", "global_only", "wrong_region", "wrong_place"}
)
# Runtime interventions deliberately exclude ``wrong_place``.  Evaluation
# batches are ordinary image sequences rather than the P x K place batches
# used for training, so a diagnostic script must supply donor CLIP features
# explicitly and fuse them through the aligned path.
RESIDUAL_CLIP_INTERVENTION_MODES = frozenset(
    {"aligned", "global_only", "wrong_region"}
)
RESIDUAL_CLIP_NEW_KEY_PREFIX = "backbone.residual_clip_fusion."


class ResidualCLIPFusion(nn.Module):
    """Inject frozen CLIP patch features as a zero-start DINO residual.

    ``dino_features`` are the final DINO local features immediately before the
    pretrained RU gate.  CLIP local/global features are already projected into
    CLIP's 512-D joint space and L2 normalised by the provider.  The module
    aligns local CLIP features to the DINO grid, learns a CLIP-to-DINO
    projection, subtracts a *fixed* normalised DINO anchor, and applies a
    zero-initialised residual adapter::

        Z = D_raw + W_zero(P_C(C_norm) - norm(D_raw))

    There is intentionally no separate trainable DINO-side projection, which
    would add another easy DINO-only path.  ``W_zero`` can still exploit the
    fixed ``-norm(D_raw)`` term, so semantic attribution must come from the
    matched global/wrong-region/wrong-place controls, not RU improvement alone.

    Offline intervention audits may set ``semantic_gamma`` together with an
    explicit ``intervention_mode``.  This applies
    ``R0 + gamma * (R_variant - R0)``, where ``R0`` is the realised zero-CLIP
    residual including the adapter bias.  Omitting the argument preserves the
    historical training and inference path exactly.
    """

    def __init__(
        self,
        in_channels: int = 768,
        clip_dim: int = 512,
        clip_grid_size: tuple[int, int] | list[int] = (14, 14),
        mode: str = "aligned",
        views_per_place: int = 4,
    ) -> None:
        super().__init__()
        if int(in_channels) <= 0:
            raise ValueError("in_channels must be positive")
        if int(clip_dim) <= 0:
            raise ValueError("clip_dim must be positive")
        if len(clip_grid_size) != 2 or any(
            int(value) <= 0 for value in clip_grid_size
        ):
            raise ValueError("clip_grid_size must contain two positive integers")
        mode = str(mode).lower()
        if mode not in RESIDUAL_CLIP_MODES:
            raise ValueError(f"mode must be one of {sorted(RESIDUAL_CLIP_MODES)}")
        if int(views_per_place) < 1:
            raise ValueError("views_per_place must be positive")

        self.in_channels = int(in_channels)
        self.clip_dim = int(clip_dim)
        self.clip_grid_size = tuple(int(value) for value in clip_grid_size)
        self.mode = mode
        self.views_per_place = int(views_per_place)

        # No bias: P_C cannot manufacture an image-independent semantic token.
        self.clip_projection = nn.Linear(
            self.clip_dim, self.in_channels, bias=False
        )
        self.residual_adapter = nn.Linear(
            self.in_channels, self.in_channels, bias=True
        )
        nn.init.zeros_(self.residual_adapter.weight)
        nn.init.zeros_(self.residual_adapter.bias)

        self._bypass_depth = 0
        self._last_diagnostics: dict[str, torch.Tensor] = {}

    @property
    def bypassed(self) -> bool:
        return self._bypass_depth > 0

    @contextmanager
    def bypass(self):
        """Temporarily return the exact unfused DINO tensor."""

        self._bypass_depth += 1
        try:
            yield self
        finally:
            self._bypass_depth -= 1
            if self._bypass_depth < 0:
                self._bypass_depth = 0
                raise RuntimeError("ResidualCLIPFusion bypass depth underflow")

    def _aligned_local_tokens(
        self,
        clip_patch_features: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if clip_patch_features.ndim != 3:
            raise ValueError(
                "clip_patch_features must have shape (B,N,clip_dim)"
            )
        batch_size, patch_count, feature_dim = clip_patch_features.shape
        if feature_dim != self.clip_dim:
            raise ValueError(
                "CLIP patch width does not match clip_dim: expected "
                f"{self.clip_dim}, found {feature_dim}"
            )
        grid_height, grid_width = self.clip_grid_size
        if patch_count != grid_height * grid_width:
            raise ValueError(
                "CLIP patch count does not match clip_grid_size: expected "
                f"{grid_height * grid_width}, found {patch_count}"
            )
        if not clip_patch_features.is_floating_point():
            raise TypeError("clip_patch_features must be floating point")

        local = F.normalize(clip_patch_features.float(), dim=-1)
        local = local.reshape(
            batch_size, grid_height, grid_width, self.clip_dim
        ).permute(0, 3, 1, 2)
        local = F.interpolate(
            local,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        local = local.permute(0, 2, 3, 1).reshape(
            batch_size, output_size[0] * output_size[1], self.clip_dim
        )
        # Bilinear interpolation no longer preserves unit norm.
        return F.normalize(local, dim=-1)

    @staticmethod
    def _wrong_region_permutation(
        patch_count: int, device: torch.device
    ) -> torch.Tensor:
        if patch_count < 2 or patch_count % 2:
            raise ValueError(
                "wrong_region requires an even patch count of at least two"
            )
        positions = torch.arange(patch_count, device=device)
        # A half-grid cyclic shift is bijective and has no fixed points.
        return (positions + patch_count // 2) % patch_count

    def _select_control_tokens(
        self,
        aligned_local: torch.Tensor,
        clip_global_features: torch.Tensor,
        *,
        apply_training_control: bool,
        intervention_mode: str | None = None,
    ) -> tuple[torch.Tensor, bool]:
        batch_size, patch_count, _ = aligned_local.shape
        if clip_global_features.ndim != 2 or tuple(
            clip_global_features.shape
        ) != (batch_size, self.clip_dim):
            raise ValueError(
                "clip_global_features must have shape "
                f"({batch_size},{self.clip_dim})"
            )
        if not clip_global_features.is_floating_point():
            raise TypeError("clip_global_features must be floating point")

        if intervention_mode is None:
            effective_mode = self.mode
            apply_control = bool(apply_training_control)
        else:
            effective_mode = str(intervention_mode).lower()
            if effective_mode not in RESIDUAL_CLIP_INTERVENTION_MODES:
                if effective_mode == "wrong_place":
                    raise ValueError(
                        "wrong_place is not a batch-local eval intervention; "
                        "supply deterministic donor CLIP features and use "
                        "intervention_mode='aligned'"
                    )
                raise ValueError(
                    "intervention_mode must be one of "
                    f"{sorted(RESIDUAL_CLIP_INTERVENTION_MODES)}"
                )
            # An explicit intervention is independent of module.train/eval.
            apply_control = True

        # With no explicit intervention, preserve the registered protocol:
        # every formal checkpoint evaluates with its own aligned CLIP grid,
        # while corruptions apply only during its matched training run.
        if effective_mode == "aligned" or not apply_control:
            return aligned_local, False
        if effective_mode == "global_only":
            global_features = F.normalize(
                clip_global_features.float(), dim=-1
            )
            return global_features[:, None].expand(-1, patch_count, -1), True
        if effective_mode == "wrong_region":
            permutation = self._wrong_region_permutation(
                patch_count, aligned_local.device
            )
            return aligned_local.index_select(1, permutation), True

        # A wrong-place representation cannot be a deployable image function
        # if its donor is chosen from the current evaluation batch.  The
        # shared training-only policy above avoids that batch dependence.
        if batch_size % self.views_per_place:
            raise ValueError(
                "wrong_place training batch must be divisible by "
                f"views_per_place={self.views_per_place}"
            )
        place_count = batch_size // self.views_per_place
        if place_count < 2:
            raise ValueError("wrong_place requires at least two places")
        rolled = aligned_local.reshape(
            place_count,
            self.views_per_place,
            patch_count,
            self.clip_dim,
        ).roll(1, dims=0)
        return rolled.reshape_as(aligned_local), True

    def forward(
        self,
        dino_features: torch.Tensor,
        clip_patch_features: torch.Tensor,
        clip_global_features: torch.Tensor,
        *,
        apply_training_control: bool = False,
        intervention_mode: str | None = None,
        semantic_gamma: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma_value: float | None = None
        if semantic_gamma is not None:
            if intervention_mode is None:
                raise ValueError(
                    "semantic_gamma requires an explicit intervention_mode"
                )
            if isinstance(semantic_gamma, bool) or not isinstance(
                semantic_gamma, Real
            ):
                raise TypeError("semantic_gamma must be a finite real scalar")
            gamma_value = float(semantic_gamma)
            if not math.isfinite(gamma_value) or not 0.0 <= gamma_value <= 8.0:
                raise ValueError("semantic_gamma must be finite and in [0, 8]")
        if dino_features.ndim != 4:
            raise ValueError("dino_features must have shape (B,C,H,W)")
        if dino_features.shape[1] != self.in_channels:
            raise ValueError(
                "DINO channels do not match in_channels: expected "
                f"{self.in_channels}, found {dino_features.shape[1]}"
            )
        if not dino_features.is_floating_point():
            raise TypeError("dino_features must be floating point")
        if clip_patch_features.shape[0] != dino_features.shape[0]:
            raise ValueError("DINO and CLIP batch sizes must match")

        batch_size, channels, height, width = dino_features.shape
        aligned_local = self._aligned_local_tokens(
            clip_patch_features, (height, width)
        )
        selected_clip, control_applied = self._select_control_tokens(
            aligned_local,
            clip_global_features,
            apply_training_control=bool(apply_training_control),
            intervention_mode=intervention_mode,
        )

        dino_tokens = dino_features.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channels
        )
        selected_clip = selected_clip.to(dtype=dino_tokens.dtype)
        projected_clip = self.clip_projection(selected_clip)
        dino_anchor = F.normalize(dino_tokens.float(), dim=-1).to(
            dtype=dino_tokens.dtype
        )
        difference = projected_clip - dino_anchor
        variant_residual = self.residual_adapter(difference)

        base_residual = None
        semantic_residual_fp32 = None
        if gamma_value is None:
            # Keep the historical train/inference path tensor-for-tensor
            # unchanged when no diagnostic decomposition was requested.
            raw_residual = variant_residual
        else:
            # R0 is the *actual* zero-CLIP path, including projection and
            # residual-adapter biases.  Computing P(0) explicitly keeps this
            # decomposition exact if the projection architecture changes.
            zero_projected = self.clip_projection(
                torch.zeros_like(selected_clip)
            )
            base_difference = zero_projected - dino_anchor
            base_residual = self.residual_adapter(base_difference)
            semantic_residual_fp32 = (
                variant_residual.float() - base_residual.float()
            )

            # Preserve exact realised AMP endpoints.  A uniform lerp would
            # introduce an avoidable subtract/add rounding at gamma 0 or 1.
            if gamma_value == 0.0:
                raw_residual = base_residual
            elif gamma_value == 1.0:
                raw_residual = variant_residual
            else:
                gamma_tensor = semantic_residual_fp32.new_tensor(gamma_value)
                raw_residual = (
                    base_residual.float()
                    + gamma_tensor * semantic_residual_fp32
                ).to(dtype=variant_residual.dtype)

        if self.bypassed:
            fused_tokens = dino_tokens
            applied_residual = torch.zeros_like(raw_residual)
        else:
            fused_tokens = dino_tokens + raw_residual
            applied_residual = raw_residual
        fused = fused_tokens.reshape(
            batch_size, height, width, channels
        ).permute(0, 3, 1, 2).contiguous()

        with torch.no_grad():
            residual_fp32 = applied_residual.detach().float()
            difference_fp32 = difference.detach().float()
            clip_fp32 = selected_clip.detach().float()
            projected_fp32 = projected_clip.detach().float()
            diagnostics = {
                "residual_clip_residual_rms": residual_fp32.square()
                .mean()
                .sqrt(),
                "residual_clip_residual_max_abs": residual_fp32.abs().amax(),
                "residual_clip_difference_rms": difference_fp32.square()
                .mean()
                .sqrt(),
                "residual_clip_token_norm_mean": clip_fp32.norm(dim=-1).mean(),
                "residual_clip_projected_norm_mean": projected_fp32.norm(
                    dim=-1
                ).mean(),
                "residual_clip_control_applied": residual_fp32.new_tensor(
                    float(control_applied)
                ),
                "residual_clip_bypassed": residual_fp32.new_tensor(
                    float(self.bypassed)
                ),
            }
            if gamma_value is not None:
                if base_residual is None or semantic_residual_fp32 is None:
                    raise AssertionError(
                        "semantic decomposition tensors were not constructed"
                    )
                base_fp32 = base_residual.detach().float()
                semantic_fp32 = semantic_residual_fp32.detach().float()
                variant_fp32 = variant_residual.detach().float()
                closure = base_fp32 + semantic_fp32 - variant_fp32
                diagnostics.update(
                    {
                        "residual_clip_semantic_gamma": (
                            residual_fp32.new_tensor(gamma_value)
                        ),
                        "residual_clip_base_residual_rms": (
                            base_fp32.square().mean().sqrt()
                        ),
                        "residual_clip_base_residual_max_abs": (
                            base_fp32.abs().amax()
                        ),
                        "residual_clip_semantic_residual_rms": (
                            semantic_fp32.square().mean().sqrt()
                        ),
                        "residual_clip_semantic_residual_max_abs": (
                            semantic_fp32.abs().amax()
                        ),
                        "residual_clip_variant_residual_rms": (
                            variant_fp32.square().mean().sqrt()
                        ),
                        "residual_clip_decomposition_closure_max_abs": (
                            closure.abs().amax()
                        ),
                    }
                )
            self._last_diagnostics = diagnostics
        return fused, raw_residual

    def diagnostics(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach()
            for name, value in self._last_diagnostics.items()
        }


class ResidualCLIPFeatureProvider:
    """Lazy frozen OpenCLIP provider kept outside the student state dict."""

    def __init__(
        self,
        *,
        encoder: Any | None = None,
        encoder_factory: Callable[[], Any] | None = None,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        hf_mirror: str | None = "https://hf-mirror.com",
        chunk_size: int = 20,
        expected_clip_dim: int = 512,
        expected_patch_count: int = 196,
    ) -> None:
        if encoder is not None and encoder_factory is not None:
            raise ValueError("provide encoder or encoder_factory, not both")
        if isinstance(chunk_size, bool) or int(chunk_size) < 1:
            raise ValueError("chunk_size must be a positive integer")
        if int(expected_clip_dim) <= 0 or int(expected_patch_count) <= 0:
            raise ValueError("expected CLIP dimensions must be positive")
        self._encoder = encoder
        self._encoder_factory = encoder_factory
        self.model_name = str(model_name)
        self.pretrained = str(pretrained)
        self.hf_mirror = hf_mirror
        self.chunk_size = int(chunk_size)
        self.expected_clip_dim = int(expected_clip_dim)
        self.expected_patch_count = int(expected_patch_count)
        if encoder is not None:
            self._freeze_encoder(encoder)

    @staticmethod
    def _freeze_encoder(encoder: Any) -> None:
        if isinstance(encoder, nn.Module):
            encoder.requires_grad_(False)
            encoder.eval()

    def _get_encoder(self, device: torch.device) -> Any:
        if self._encoder is None:
            if self._encoder_factory is not None:
                self._encoder = self._encoder_factory()
            else:
                from src.models.clip_teacher import CLIPTeacherEncoder

                self._encoder = CLIPTeacherEncoder(
                    model_name=self.model_name,
                    pretrained=self.pretrained,
                    hf_mirror=self.hf_mirror,
                )
            self._freeze_encoder(self._encoder)
        if isinstance(self._encoder, nn.Module):
            self._encoder.to(device)
            self._encoder.eval()
        return self._encoder

    def prepare(self, device: torch.device | str = "cpu") -> Any:
        return self._get_encoder(torch.device(device))

    def encode(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (B,3,H,W)")
        if not images.is_floating_point():
            raise TypeError("images must be floating point")
        encoder = self._get_encoder(images.device)
        global_chunks = []
        patch_chunks = []
        with torch.no_grad():
            for image_chunk in images.split(self.chunk_size, dim=0):
                output = encoder(image_chunk)
                if not isinstance(output, (tuple, list)) or len(output) < 2:
                    raise TypeError(
                        "CLIP encoder must return (global, raw_patch_tokens)"
                    )
                global_features, raw_patch_tokens = output[:2]
                if not hasattr(encoder, "project_patch_tokens"):
                    raise TypeError(
                        "CLIP encoder must expose project_patch_tokens"
                    )
                patch_features = encoder.project_patch_tokens(raw_patch_tokens)
                global_chunks.append(global_features.detach())
                patch_chunks.append(patch_features.detach())

        global_features = torch.cat(global_chunks, dim=0)
        patch_features = torch.cat(patch_chunks, dim=0)
        expected_global = (images.shape[0], self.expected_clip_dim)
        expected_patches = (
            images.shape[0],
            self.expected_patch_count,
            self.expected_clip_dim,
        )
        if tuple(global_features.shape) != expected_global:
            raise ValueError(
                "CLIP global features have the wrong shape: expected "
                f"{expected_global}, found {tuple(global_features.shape)}"
            )
        if tuple(patch_features.shape) != expected_patches:
            raise ValueError(
                "CLIP patch features have the wrong shape: expected "
                f"{expected_patches}, found {tuple(patch_features.shape)}"
            )
        if not bool(torch.isfinite(global_features).all()) or not bool(
            torch.isfinite(patch_features).all()
        ):
            raise ValueError("CLIP features must be finite")
        return global_features, patch_features

    def audit(self) -> dict[str, object]:
        encoder = self._encoder
        trainable = 0
        training = False
        if isinstance(encoder, nn.Module):
            trainable = sum(
                parameter.numel()
                for parameter in encoder.parameters()
                if parameter.requires_grad
            )
            training = bool(encoder.training)
        return {
            "prepared": encoder is not None,
            "trainable_parameters": trainable,
            "training": training,
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "chunk_size": self.chunk_size,
        }


def warm_start_residual_clip_model(
    model: nn.Module,
    checkpoint_path: str | Path,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    """Strictly load RU while allowing exactly the new residual branch."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"initial checkpoint not found: {checkpoint_path}")
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    if expected_sha256 is not None:
        expected_sha256 = str(expected_sha256).lower()
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in expected_sha256
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
        if key.startswith(RESIDUAL_CLIP_NEW_KEY_PREFIX)
    }
    if not expected_new_keys:
        raise RuntimeError(
            "configured model has no backbone.residual_clip_fusion parameters"
        )
    present_new_keys = expected_new_keys.intersection(state)
    if present_new_keys:
        raise RuntimeError(
            "RU warm start must not contain residual-CLIP parameters: "
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
        raise RuntimeError(
            "unsafe RU residual-CLIP warm start: missing legacy keys="
            f"{sorted(missing - expected_new_keys)}, new keys not reported "
            f"missing={sorted(expected_new_keys - missing)}, unexpected "
            f"keys={sorted(unexpected)}"
        )

    fusion = getattr(
        getattr(model, "backbone", None), "residual_clip_fusion", None
    )
    if fusion is None or not hasattr(fusion, "residual_adapter"):
        raise RuntimeError("configured model has no usable residual-CLIP branch")
    if torch.count_nonzero(fusion.residual_adapter.weight).item() or (
        torch.count_nonzero(fusion.residual_adapter.bias).item()
    ):
        raise RuntimeError(
            "residual adapter must remain exactly zero after RU warm start"
        )
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "loaded_keys": len(state),
        "new_keys": tuple(sorted(expected_new_keys)),
        "residual_zero_initialized": True,
        **provenance,
    }


def freeze_for_residual_clip_screen(model: nn.Module) -> tuple[str, ...]:
    """Freeze DINO/RU/BoQ and leave only P_C/W_zero trainable."""

    if getattr(model, "semantic_region_gate", None) is None:
        raise RuntimeError("residual-CLIP screen requires the pretrained RU gate")
    fusion = getattr(
        getattr(model, "backbone", None), "residual_clip_fusion", None
    )
    if fusion is None:
        raise RuntimeError("configured backbone has no residual-CLIP branch")
    model.requires_grad_(False)
    fusion.requires_grad_(True)
    model._residual_clip_base_frozen = True
    trainable = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if not trainable or any(
        not name.startswith(RESIDUAL_CLIP_NEW_KEY_PREFIX) for name in trainable
    ):
        raise RuntimeError(
            f"unexpected trainable parameters after residual freeze: {trainable}"
        )
    return trainable
