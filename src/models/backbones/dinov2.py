from collections.abc import Mapping

import torch
import torch.nn as nn


class DinoV2(nn.Module):
    AVAILABLE_MODELS = [
        'dinov2_vits14',
        'dinov2_vitb14',
        'dinov2_vitl14',
        'dinov2_vitg14'
    ]
    
    def __init__(
        self,
        backbone_name="dinov2_vitb14",
        num_unfrozen_blocks=2,
        return_cls_token=False,
        crop_semantic_film=None,
        residual_clip_fusion=None,
    ):
        """DinoV2 backbone with the ability to keep only the last num_unfrozen_blocks trainable.

        Args:
            backbone_name (str, optional): DinoV2 variant. Defaults to "dinov2_vitb14".
            num_unfrozen_blocks (int, optional): number of blocks to unfreeze. Defaults to 2.

        Raises:
            ValueError: if the backbone_name is not in the available models.
        """
        super().__init__()
        
        self.backbone_name = backbone_name
        self.num_unfrozen_blocks = num_unfrozen_blocks
        self.return_cls_token = return_cls_token
        # make sure the backbone_name is in the available models
        if self.backbone_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Backbone {self.backbone_name} is not recognized!" 
                             f"Supported backbones are: {self.AVAILABLE_MODELS}")
                             
                
        # Keep the explicit ref used by every recorded DINOv2 experiment.
        # Besides making the source checkpoint path reproducible, this lets
        # torch.hub reuse ``facebookresearch_dinov2_main`` without first
        # contacting GitHub merely to discover the repository default branch.
        self.dino = torch.hub.load(
            'facebookresearch/dinov2:main', self.backbone_name
        )
        
        # freeze the patch embedding and positional encoding
        self.dino.patch_embed.requires_grad_(False)
        self.dino.pos_embed.requires_grad_(False)
        
        # freeze the first blocks, keep only the last num_unfrozen_blocks trainable
        for i in range(len(self.dino.blocks) - self.num_unfrozen_blocks):
            self.dino.blocks[i].requires_grad_(False)
        
        self.out_channels = self.dino.embed_dim

        self.crop_semantic_film = None
        crop_semantic_film = crop_semantic_film or {}
        if not isinstance(crop_semantic_film, Mapping):
            raise TypeError("crop_semantic_film must be a mapping or null")
        if bool(crop_semantic_film.get("enabled", False)):
            insert_before = int(
                crop_semantic_film.get(
                    "insert_before_last_n_blocks", self.num_unfrozen_blocks
                )
            )
            if insert_before != self.num_unfrozen_blocks:
                raise ValueError(
                    "crop_semantic_film must be inserted immediately before "
                    "all unfrozen DINO blocks: insert_before_last_n_blocks="
                    f"{insert_before}, num_unfrozen_blocks="
                    f"{self.num_unfrozen_blocks}"
                )
            if not 0 < self.num_unfrozen_blocks < len(self.dino.blocks):
                raise ValueError(
                    "crop_semantic_film requires at least one frozen and one "
                    "unfrozen DINO block"
                )
            from src.models.crop_semantic_film import CropSemanticFiLM

            self.crop_semantic_film = CropSemanticFiLM(
                in_channels=self.dino.embed_dim,
                hidden_dim=int(crop_semantic_film.get("hidden_dim", 128)),
                semantic_dim=int(
                    crop_semantic_film.get("semantic_dim", 512)
                ),
                alpha=float(crop_semantic_film.get("alpha", 0.1)),
            )

        self.residual_clip_fusion = None
        self._residual_clip_provider = None
        residual_clip_fusion = residual_clip_fusion or {}
        if not isinstance(residual_clip_fusion, Mapping):
            raise TypeError("residual_clip_fusion must be a mapping or null")
        if bool(residual_clip_fusion.get("enabled", False)):
            if self.crop_semantic_film is not None:
                raise ValueError(
                    "crop_semantic_film and residual_clip_fusion are mutually "
                    "exclusive"
                )
            from src.models.residual_clip_fusion import (
                ResidualCLIPFeatureProvider,
                ResidualCLIPFusion,
            )

            self.residual_clip_fusion = ResidualCLIPFusion(
                in_channels=self.dino.embed_dim,
                clip_dim=int(residual_clip_fusion.get("clip_dim", 512)),
                clip_grid_size=residual_clip_fusion.get(
                    "clip_grid_size", (14, 14)
                ),
                mode=str(residual_clip_fusion.get("mode", "aligned")),
                views_per_place=int(
                    residual_clip_fusion.get("views_per_place", 4)
                ),
            )
            encoder_cfg = residual_clip_fusion.get("encoder", {}) or {}
            if not isinstance(encoder_cfg, Mapping):
                raise TypeError(
                    "residual_clip_fusion.encoder must be a mapping or null"
                )
            # This provider is intentionally not an nn.Module.  Its frozen
            # OpenCLIP weights are reconstructed from the pinned config and
            # stay out of every Lightning checkpoint.
            self._residual_clip_provider = ResidualCLIPFeatureProvider(
                model_name=str(encoder_cfg.get("model_name", "ViT-B-16")),
                pretrained=str(encoder_cfg.get("pretrained", "openai")),
                hf_mirror=encoder_cfg.get(
                    "hf_mirror", "https://hf-mirror.com"
                ),
                chunk_size=int(
                    residual_clip_fusion.get("clip_chunk_size", 20)
                ),
                expected_clip_dim=int(
                    residual_clip_fusion.get("clip_dim", 512)
                ),
                expected_patch_count=(
                    int(residual_clip_fusion.get("clip_grid_size", (14, 14))[0])
                    * int(
                        residual_clip_fusion.get(
                            "clip_grid_size", (14, 14)
                        )[1]
                    )
                ),
            )

    def _forward_impl(
        self,
        x,
        *,
        return_crop_semantics=False,
        semantic_batch_indices=None,
    ):
        B, _, H, W = x.shape
        input_images = x
        # No need to compute gradients for frozen layers
        with torch.no_grad():
            x = self.dino.prepare_tokens_with_masks(x)
            for blk in self.dino.blocks[ : -self.num_unfrozen_blocks]:
                x = blk(x)
        x = x.detach()

        crop_semantic_tokens = None
        crop_semantic_raw_scale = None
        if self.crop_semantic_film is not None:
            cls_token = x[:, :1]
            patch_tokens, crop_semantic_tokens, crop_semantic_raw_scale = (
                self.crop_semantic_film(
                    x[:, 1:],
                    return_semantic=return_crop_semantics,
                    semantic_batch_indices=semantic_batch_indices,
                )
            )
            x = torch.cat((cls_token, patch_tokens), dim=1)

        # Last blocks are trained
        for blk in self.dino.blocks[-self.num_unfrozen_blocks : ]:
            x = blk(x)
            
        x_cls = x[:, 0]
        x = x[:, 1:] # remove the [CLS] token
        
        # reshape the output tensor to B, C, H, W
        _, _, C = x.shape # we know C == self.dino.embed_dim, but still...
        x = x.permute(0, 2, 1).contiguous().view(B, C, H//14, W//14)

        if self.residual_clip_fusion is not None:
            if self._residual_clip_provider is None:
                raise RuntimeError(
                    "residual CLIP fusion has no frozen feature provider"
                )
            clip_global, clip_patches = self._residual_clip_provider.encode(
                input_images
            )
            x, _ = self.residual_clip_fusion(
                x,
                clip_patches,
                clip_global,
                # The VPR framework keeps frozen DINO in eval mode while
                # explicitly leaving this small branch in train mode.
                apply_training_control=self.residual_clip_fusion.training,
            )
        
        backbone_output = (x, x_cls) if self.return_cls_token else x
        if return_crop_semantics:
            if self.crop_semantic_film is None:
                raise RuntimeError(
                    "return_crop_semantics requested without an enabled "
                    "crop_semantic_film"
                )
            return (
                backbone_output,
                crop_semantic_tokens,
                crop_semantic_raw_scale,
            )
        return backbone_output

    def forward(self, x):
        """Inference path: apply FiLM but skip the teacher-only 512D head."""

        return self._forward_impl(x, return_crop_semantics=False)

    def forward_with_crop_semantics(
        self, x, *, semantic_batch_indices=None
    ):
        """Training path returning the local semantic student tokens."""

        return self._forward_impl(
            x,
            return_crop_semantics=True,
            semantic_batch_indices=semantic_batch_indices,
        )

    def crop_semantic_diagnostics(self):
        if self.crop_semantic_film is None:
            return {}
        return self.crop_semantic_film.diagnostics()

    def residual_clip_diagnostics(self):
        if self.residual_clip_fusion is None:
            return {}
        return self.residual_clip_fusion.diagnostics()

    def prepare_residual_clip_provider(
        self, device: torch.device | str = "cpu"
    ):
        if self._residual_clip_provider is None:
            raise RuntimeError("residual CLIP provider is not enabled")
        return self._residual_clip_provider.prepare(device)

    def residual_clip_provider_audit(self):
        if self._residual_clip_provider is None:
            return {
                "prepared": False,
                "trainable_parameters": 0,
                "training": False,
            }
        return self._residual_clip_provider.audit()
