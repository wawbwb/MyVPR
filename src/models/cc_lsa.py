"""SemVPR-style Local Semantic Alignment encoder for CC-LSA Gate A.

``LocalSemanticAlignmentEncoder`` starts from the pinned OpenCLIP visual
encoder and fine-tunes only its last transformer blocks plus the final
projection.  A separate frozen ``CLIPTeacherEncoder`` supplies crop CLS
targets.  Only the trainable delta is stored in the checkpoint; reconstruction
therefore remains bound to the pinned base-model fingerprint.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


CC_LSA_TEACHER_SCHEMA = "openvpr_cc_lsa_teacher"
CC_LSA_TEACHER_VERSION = 1


class LocalSemanticAlignmentEncoder(nn.Module):
    """Trainable local CLIP map initialised from a frozen OpenCLIP model."""

    def __init__(
        self,
        *,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        hf_mirror: str | None = "https://hf-mirror.com",
        trainable_last_n_blocks: int = 2,
    ) -> None:
        super().__init__()
        if isinstance(trainable_last_n_blocks, bool) or int(
            trainable_last_n_blocks
        ) < 1:
            raise ValueError("trainable_last_n_blocks must be positive")
        from src.models.clip_teacher import CLIPTeacherEncoder

        base = CLIPTeacherEncoder(
            model_name=model_name,
            pretrained=pretrained,
            hf_mirror=hf_mirror,
        )
        self.visual = base.visual
        self.model_name = str(model_name)
        self.pretrained = str(pretrained)
        self.hf_mirror = hf_mirror
        self.trainable_last_n_blocks = int(trainable_last_n_blocks)
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
            persistent=False,
        )
        self._configure_trainable_parameters()
        del base

    @property
    def native_grid_size(self) -> tuple[int, int]:
        image_size = getattr(self.visual, "image_size", 224)
        patch_size = getattr(self.visual, "patch_size", 16)
        if isinstance(image_size, (tuple, list)):
            image_height, image_width = (int(value) for value in image_size)
        else:
            image_height = image_width = int(image_size)
        if isinstance(patch_size, (tuple, list)):
            patch_height, patch_width = (int(value) for value in patch_size)
        else:
            patch_height = patch_width = int(patch_size)
        return image_height // patch_height, image_width // patch_width

    @property
    def semantic_dim(self) -> int:
        projection = getattr(self.visual, "proj", None)
        if projection is not None:
            return int(projection.shape[1])
        return int(self.visual.ln_post.normalized_shape[0])

    def _configure_trainable_parameters(self) -> None:
        self.visual.requires_grad_(False)
        blocks = self.visual.transformer.resblocks
        if self.trainable_last_n_blocks > len(blocks):
            raise ValueError(
                "trainable_last_n_blocks exceeds OpenCLIP transformer depth"
            )
        for block in blocks[-self.trainable_last_n_blocks :]:
            block.requires_grad_(True)
        self.visual.ln_post.requires_grad_(True)
        projection = getattr(self.visual, "proj", None)
        if isinstance(projection, nn.Parameter):
            projection.requires_grad_(True)

    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        )

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        state = self.state_dict()
        names = set(self.trainable_parameter_names())
        return {name: state[name].detach().cpu() for name in sorted(names)}

    def load_trainable_state_dict(
        self, state: Mapping[str, torch.Tensor]
    ) -> None:
        expected = set(self.trainable_parameter_names())
        found = set(state)
        if found != expected:
            raise ValueError(
                "LSA trainable delta keys differ from the configured model: "
                f"missing={sorted(expected - found)[:5]}, "
                f"unexpected={sorted(found - expected)[:5]}"
            )
        current = self.state_dict()
        for name, value in state.items():
            if (
                not torch.is_tensor(value)
                or value.shape != current[name].shape
                or value.dtype != current[name].dtype
            ):
                raise ValueError(f"invalid LSA delta tensor for {name}")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"non-finite LSA delta tensor for {name}")
            current[name] = value
        self.load_state_dict(current, strict=True)

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (B,3,H,W)")
        images = images * self.imagenet_std + self.imagenet_mean
        images = images.clamp(0.0, 1.0)
        image_size = getattr(self.visual, "image_size", (224, 224))
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        images = F.interpolate(
            images,
            size=tuple(int(value) for value in image_size),
            mode="bicubic",
            align_corners=False,
        )
        return (images - self.clip_mean) / self.clip_std

    def _tokens(self, images: torch.Tensor) -> torch.Tensor:
        visual = self.visual
        tokens = visual.conv1(images)
        tokens = tokens.reshape(tokens.shape[0], tokens.shape[1], -1).permute(
            0, 2, 1
        )
        cls_token = visual.class_embedding.to(tokens.dtype) + torch.zeros(
            tokens.shape[0],
            1,
            tokens.shape[-1],
            dtype=tokens.dtype,
            device=tokens.device,
        )
        tokens = torch.cat((cls_token, tokens), dim=1)
        tokens = tokens + visual.positional_embedding.to(tokens.dtype)
        if hasattr(visual, "patch_dropout"):
            tokens = visual.patch_dropout(tokens)
        tokens = visual.ln_pre(tokens)
        return visual.transformer(tokens)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised native-grid patch descriptors ``(B,N,D)``."""

        tokens = self._tokens(self._preprocess(images))[:, 1:]
        tokens = self.visual.ln_post(tokens)
        projection = getattr(self.visual, "proj", None)
        if projection is not None:
            tokens = tokens @ projection
        return F.normalize(tokens.float(), dim=-1)


def pool_region_grid(
    local_tokens: torch.Tensor, *, rows: int = 2, columns: int = 2
) -> torch.Tensor:
    """Average a square token map into a fixed non-overlapping region grid."""

    if local_tokens.ndim != 3 or not local_tokens.is_floating_point():
        raise ValueError("local_tokens must be floating point with shape (B,N,D)")
    if rows < 1 or columns < 1:
        raise ValueError("region rows/columns must be positive")
    side = math.isqrt(int(local_tokens.shape[1]))
    if side * side != local_tokens.shape[1]:
        raise ValueError("local token count must form a square grid")
    if side % rows or side % columns:
        raise ValueError("local token grid must be divisible by region grid")
    grid = local_tokens.reshape(
        local_tokens.shape[0], side, side, local_tokens.shape[-1]
    )
    pooled = []
    height = side // rows
    width = side // columns
    for row in range(rows):
        for column in range(columns):
            region = grid[
                :,
                row * height : (row + 1) * height,
                column * width : (column + 1) * width,
            ]
            pooled.append(region.mean(dim=(1, 2)))
    return F.normalize(torch.stack(pooled, dim=1).float(), dim=-1)


def crop_image_grid(
    images: torch.Tensor, *, rows: int = 2, columns: int = 2
) -> torch.Tensor:
    """Return row-major non-overlapping image crops as ``(B*R*C,3,h,w)``."""

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape (B,3,H,W)")
    height, width = images.shape[-2:]
    if height % rows or width % columns:
        raise ValueError("image dimensions must be divisible by crop grid")
    crop_height, crop_width = height // rows, width // columns
    crops = []
    for row in range(rows):
        for column in range(columns):
            crops.append(
                images[
                    :,
                    :,
                    row * crop_height : (row + 1) * crop_height,
                    column * crop_width : (column + 1) * crop_width,
                ]
            )
    return torch.stack(crops, dim=1).flatten(0, 1)


def lsa_cosine_loss(
    local_tokens: torch.Tensor,
    crop_cls_targets: torch.Tensor,
    *,
    rows: int = 2,
    columns: int = 2,
) -> torch.Tensor:
    targets = F.normalize(crop_cls_targets.detach().float(), dim=-1)
    predicted = pool_region_grid(local_tokens, rows=rows, columns=columns)
    if predicted.shape != targets.shape:
        raise ValueError(
            f"LSA prediction/target shapes differ: {predicted.shape} vs {targets.shape}"
        )
    loss = 1.0 - (predicted * targets).sum(dim=-1)
    if not bool(torch.isfinite(loss).all()):
        raise RuntimeError("non-finite LSA cosine loss")
    return loss.mean()


def module_state_sha256(module: nn.Module) -> str:
    return tensor_mapping_sha256(module.state_dict())


def tensor_mapping_sha256(values: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes and bytes in a stable order."""

    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise TypeError("tensor mapping must contain string -> Tensor entries")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_lsa_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    require_contract_pass: bool = True,
) -> tuple[LocalSemanticAlignmentEncoder, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("LSA checkpoint root must be a mapping")
    if checkpoint.get("schema") != CC_LSA_TEACHER_SCHEMA:
        raise ValueError("unsupported LSA checkpoint schema")
    if checkpoint.get("version") != CC_LSA_TEACHER_VERSION:
        raise ValueError("unsupported LSA checkpoint version")
    config = checkpoint.get("model")
    expected_model_keys = {
        "model_name",
        "pretrained",
        "hf_mirror",
        "trainable_last_n_blocks",
    }
    if not isinstance(config, dict) or set(config) != expected_model_keys:
        raise ValueError("LSA checkpoint has no model config")
    model = LocalSemanticAlignmentEncoder(
        model_name=str(config["model_name"]),
        pretrained=str(config["pretrained"]),
        hf_mirror=config.get("hf_mirror"),
        trainable_last_n_blocks=int(config["trainable_last_n_blocks"]),
    )
    base_hash = module_state_sha256(model.visual)
    if checkpoint.get("base_model_state_sha256") != base_hash:
        raise ValueError("LSA checkpoint base OpenCLIP fingerprint differs")
    delta = checkpoint.get("trainable_state_dict")
    if not isinstance(delta, Mapping):
        raise ValueError("LSA checkpoint has no trainable delta")
    if checkpoint.get("trainable_state_sha256") != tensor_mapping_sha256(delta):
        raise ValueError("LSA checkpoint trainable delta fingerprint differs")
    model.load_trainable_state_dict(delta)
    contract = checkpoint.get("teacher_contract")
    if not isinstance(contract, dict):
        raise ValueError("LSA checkpoint has no teacher contract")
    if require_contract_pass and contract.get("verdict") != "PASS":
        raise ValueError("LSA teacher contract did not pass")
    model.to(map_location)
    model.eval()
    return model, checkpoint
