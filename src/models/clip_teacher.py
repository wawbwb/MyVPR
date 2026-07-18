# ----------------------------------------------------------------------------
# CLIP Teacher Encoder for knowledge distillation in VPR.
# Wraps a frozen CLIP ViT model (via open_clip) and exposes both the
# normalised global descriptor and the raw patch tokens.
# ----------------------------------------------------------------------------

from importlib.metadata import PackageNotFoundError, version

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPTeacherEncoder(nn.Module):
    """Frozen CLIP visual encoder that returns:
        t_global  – L2-normalised CLS-projected global vector  (B, D_global)
        t_tokens  – raw ViT patch tokens                       (B, N, D_token)
    The module automatically converts ImageNet-normalised inputs to CLIP
    preprocessing (resize-224, CLIP mean/std) so callers do not need to
    change the existing data pipeline.
    """

    def __init__(
        self,
        model_name="ViT-B-16",
        pretrained="openai",
        hf_mirror="https://hf-mirror.com",
        dynamic_categories: list[str] | None = None,
    ):
        super().__init__()
        import os

        try:
            installed_version = version("open_clip_torch")
        except PackageNotFoundError as exc:
            raise ImportError(
                "Phase C requires open_clip_torch==2.26.1. "
                "Install the pinned project environment first."
            ) from exc
        if installed_version != "2.26.1":
            raise RuntimeError(
                "Phase C attention extraction is validated against "
                "open_clip_torch==2.26.1, but found "
                f"{installed_version}. Install the pinned version to keep "
                "the teacher target reproducible."
            )

        import open_clip

        # 使用镜像源（如未手动指定则自动设置）
        if hf_mirror and not os.environ.get("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = hf_mirror

        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.visual = model.visual

        # Freeze everything
        self.visual.eval()
        for p in self.visual.parameters():
            p.requires_grad_(False)

        # ── Semantic gate: encode text prompts for dynamic/unstable categories ──
        if dynamic_categories is not None:
            tokenizer = open_clip.get_tokenizer(model_name)
            prompts = [f"a photo of a {c}" for c in dynamic_categories]
            tokens = tokenizer(prompts)
            with torch.no_grad():
                text_feats = model.encode_text(tokens)  # (K, D_global)
                text_feats = F.normalize(text_feats, dim=-1)
            self.register_buffer("dynamic_text_feats", text_feats)  # (K, D_global)
        else:
            self.dynamic_text_feats = None

        # Discard the full model (keep only visual)
        del model

        # Dimension bookkeeping
        self.token_dim = self.visual.ln_post.normalized_shape[0]
        if hasattr(self.visual, "proj") and self.visual.proj is not None:
            self.global_dim = self.visual.proj.shape[1]
        else:
            self.global_dim = self.token_dim

        # Normalisation buffers: ImageNet → [0,1] → CLIP
        self.register_buffer(
            "imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
        )

    # ------------------------------------------------------------------
    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """ImageNet-normalised (B,3,H,W) → CLIP-normalised (B,3,224,224)."""
        x = x * self.imagenet_std + self.imagenet_mean  # → [0, 1]
        x = x.clamp(0.0, 1.0)
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        x = (x - self.clip_mean) / self.clip_std
        return x

    # ------------------------------------------------------------------
    @staticmethod
    def _cls_to_patch_attention(blk, block_input, batch_first: bool):
        """Recompute only the final block's CLS attention row.

        The actual residual block still runs through its original ``forward``
        method. This side computation therefore cannot change the teacher
        descriptor or patch tokens.
        """
        attn = getattr(blk, "attn", None)
        if (
            attn is None
            or not hasattr(attn, "in_proj_weight")
            or attn.in_proj_weight is None
            or not hasattr(attn, "num_heads")
            or not hasattr(blk, "ln_1")
        ):
            raise RuntimeError(
                "Unsupported OpenCLIP visual attention block. Phase C expects "
                "the standard ViT block from open_clip_torch==2.26.1."
            )

        normalised = blk.ln_1(block_input)
        if not batch_first:
            normalised = normalised.transpose(0, 1)

        qkv = F.linear(
            normalised,
            attn.in_proj_weight,
            getattr(attn, "in_proj_bias", None),
        )
        query, key, _ = qkv.chunk(3, dim=-1)
        batch_size, sequence_length, embed_dim = query.shape
        num_heads = attn.num_heads
        if embed_dim % num_heads != 0:
            raise RuntimeError(
                f"Attention width {embed_dim} is not divisible by {num_heads} heads"
            )
        head_dim = embed_dim // num_heads

        query_cls = query[:, 0].reshape(batch_size, num_heads, head_dim)
        key = key.reshape(
            batch_size, sequence_length, num_heads, head_dim
        ).permute(0, 2, 1, 3)
        logits = torch.einsum("bhd,bhld->bhl", query_cls, key)
        logits = logits * (head_dim ** -0.5)
        weights = logits.float().softmax(dim=-1)

        # Drop CLS self-attention and condition the remainder on patch tokens.
        # This is already a probability distribution: never softmax it again.
        patch_attention = weights[:, :, 1:].mean(dim=1)
        return patch_attention / patch_attention.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

    # ------------------------------------------------------------------
    def _encode(self, x: torch.Tensor, return_attn: bool = False):
        v = self.visual

        # Patch embedding
        x = v.conv1(x)  # (B, D, gh, gw)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B, N, D)

        # Prepend CLS token
        cls_tok = v.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_tok, x], dim=1)  # (B, N+1, D)
        x = x + v.positional_embedding.to(x.dtype)

        if hasattr(v, "patch_dropout"):
            x = v.patch_dropout(x)

        x = v.ln_pre(x)

        cls_attn = None
        if not return_attn:
            # Transformer — open_clip handles NLD↔LND transposing internally
            # based on v.transformer.batch_first; do NOT manually permute.
            x = v.transformer(x)  # (B, N+1, D)
        else:
            # Manually iterate the residual blocks so we can capture the
            # CLS-to-patch attention from the last layer. We replicate the
            # transformer's own NLD↔LND handling to stay layout-consistent.
            transformer = v.transformer
            batch_first = getattr(transformer, "batch_first", True)
            if not batch_first:
                x = x.transpose(0, 1)  # (B,N,D) → (N,B,D)
            resblocks = transformer.resblocks
            for blk in resblocks[:-1]:
                x = blk(x)
            last_block_input = x
            x = resblocks[-1](x)
            cls_attn = self._cls_to_patch_attention(
                resblocks[-1], last_block_input, batch_first
            )
            if not batch_first:
                x = x.transpose(0, 1)  # → (B,N,D)

        # ln_post is only on CLS in the original CLIP
        cls_out = v.ln_post(x[:, 0])
        patch_tokens = x[:, 1:]  # (B, N, D_token)

        # Global descriptor via projection head
        if hasattr(v, "proj") and v.proj is not None:
            t_global = cls_out @ v.proj  # (B, D_global)
        else:
            t_global = cls_out

        if return_attn:
            return t_global, patch_tokens, cls_attn
        return t_global, patch_tokens

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, x: torch.Tensor, return_attn: bool = False):
        self.visual.eval()
        x = self._preprocess(x)
        if return_attn:
            t_global, t_tokens, cls_attn = self._encode(x, return_attn=True)
            t_global = F.normalize(t_global, dim=-1)
            return t_global, t_tokens, cls_attn
        t_global, t_tokens = self._encode(x)
        t_global = F.normalize(t_global, dim=-1)
        return t_global, t_tokens

    # ------------------------------------------------------------------
    @torch.no_grad()
    def project_patch_tokens(self, t_tokens: torch.Tensor) -> torch.Tensor:
        """Project raw ViT patch tokens into CLIP's joint semantic space.

        The returned patch embeddings are L2-normalised and detached.  They
        are used only to construct training targets; CLIP is never part of the
        student inference path.
        """
        self.visual.eval()
        patch_proj = self.visual.ln_post(t_tokens)
        if hasattr(self.visual, "proj") and self.visual.proj is not None:
            patch_proj = patch_proj @ self.visual.proj

        # Normalise in fp32 so small positive-vs-negative reliability margins
        # are not lost under mixed precision, then release the large tensor in
        # its original dtype. Pairwise matching promotes chunks back to fp32.
        output_dtype = patch_proj.dtype
        patch_proj = F.normalize(patch_proj.float(), dim=-1)
        return patch_proj.to(output_dtype)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_semantic_mask(self, t_tokens: torch.Tensor) -> torch.Tensor:
        """Compute per-patch suppression mask based on dynamic category similarity.

        For each patch token, project through ln_post + proj to get a CLIP-space
        embedding, then compute max cosine similarity to the dynamic category
        text embeddings. High similarity → dynamic/unstable → low gate value.

        Uses per-image z-score normalization to overcome CLIP's tight similarity
        range across patches, then sigmoid to produce the gate.

        Args:
            t_tokens: (B, N, D_token) raw patch tokens from _encode

        Returns:
            mask: (B, N) in [0, 1], where 1 = stable (keep), 0 = dynamic (suppress)
            Returns all-ones if dynamic_text_feats is None.
        """
        if self.dynamic_text_feats is None:
            return torch.ones(
                t_tokens.shape[0], t_tokens.shape[1],
                device=t_tokens.device, dtype=t_tokens.dtype,
            )

        patch_proj = self.project_patch_tokens(t_tokens)

        # Cosine similarity to each dynamic category: (B, N, K)
        sim = torch.einsum("bnd,kd->bnk", patch_proj, self.dynamic_text_feats)
        # Max similarity across categories: (B, N)
        max_sim = sim.max(dim=-1).values  # higher = more likely dynamic

        # Per-image z-score normalization: patches that are *relatively* more
        # similar to dynamic categories (within this image) get suppressed.
        mu = max_sim.mean(dim=-1, keepdim=True)
        sigma = max_sim.std(dim=-1, keepdim=True).clamp(min=1e-6)
        z = (max_sim - mu) / sigma  # (B, N), mean=0, std=1

        # sigmoid(-z): high z (more dynamic than average) → low mask
        # scale=2 gives moderate discrimination: z=+1 → mask≈0.12, z=-1 → mask≈0.88
        mask = torch.sigmoid(-2.0 * z)
        return mask

    # Always keep teacher in eval ---------------------------------
    def train(self, mode=True):
        super().train(mode)
        self.visual.eval()
        return self
