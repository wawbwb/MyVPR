# ----------------------------------------------------------------------------
# SG-SA distillation module: trainable projections, attention weights,
# region vectors, and loss computation.
# ----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.semantic_reliability import VPRSemanticReliabilityTarget


class SpatialAttentionHead(nn.Module):
    """Lightweight student-only spatial gate used by phase C.

    The head predicts one or more probability distributions over the backbone
    grid.  Their mean is converted to a residual multiplicative gate whose
    spatial mean is one.  Zero-initialising the projection makes the initial
    gate exactly the identity, so enabling the module does not perturb a
    pretrained VPR model before it has learned a useful attention map.
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 1,
        gate_strength: float = 1.0,
    ):
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be at least 1")
        if not 0.0 <= gate_strength <= 1.0:
            raise ValueError("gate_strength must be in [0, 1]")

        self.num_heads = num_heads
        self.gate_strength = float(gate_strength)
        self.proj = nn.Conv2d(in_channels, num_heads, kernel_size=1)

        # Uniform attention -> unit gate at initialisation.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self, featmap: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(weighted_featmap, attention_probabilities)``.

        Args:
            featmap: ``(B, C, H, W)`` student backbone features.

        Returns:
            weighted_featmap: ``(B, C, H, W)`` features passed to MixVPR.
            attention: ``(B, num_heads, H*W)``; every head sums to one.
        """
        if featmap.ndim != 4:
            raise ValueError(
                f"featmap must have shape (B,C,H,W), got {tuple(featmap.shape)}"
            )

        logits = self.proj(featmap).flatten(2)
        # Keep probabilities in fp32 under mixed precision. Casting them back
        # here can underflow small cells to zero and remove their KL gradient.
        attention = F.softmax(logits.float(), dim=-1)

        spatial_size = attention.shape[-1]
        mean_attention = attention.mean(dim=1).view(
            featmap.shape[0], 1, featmap.shape[-2], featmap.shape[-1]
        )
        gate = (
            (1.0 - self.gate_strength)
            + self.gate_strength * spatial_size * mean_attention
        ).to(featmap.dtype)
        return featmap * gate, attention


class DistillationModule(nn.Module):
    """Distillation head implementing SG-SA weighted region & global distillation.

    Trainable components (receive gradients):
      * token_proj         – maps teacher tokens  D_token → proj_dim
      * global_attn_proj   – maps teacher global  D_global → proj_dim  (for w0)
      * student_global_proj– maps student descriptor → D_global          (for L_g)
      * student_feat_proj  – maps student feat channels → proj_dim      (only if proj_dim ≠ C)

    Ablation modes (``distill_mode``):
      * ``global_only``         – only L_g
      * ``region_no_gate``      – L_g + L_r  (w = softmax attention, no gate)
      * ``region_gate``         – L_g + L_r  (w = gated SG-SA, needs img_aug)
      * ``region_semantic_gate``– L_g + L_r  (w = SG-SA × semantic suppression mask)
    """

    def __init__(
        self,
        teacher: nn.Module,
        teacher_token_dim: int,
        teacher_global_dim: int,
        student_feat_channels: int,
        student_global_dim: int,
        proj_dim: int | None = None,
        tau: float = 0.07,
        distill_mode: str = "region_gate",
        attention_target: str = "clip_attention",
        reliability_temperature: float = 0.1,
        reliability_negative_topk: int = 1,
        reliability_positive_weight: float = 1.0,
        reliability_negative_weight: float = 1.0,
        reliability_pair_chunk_size: int = 32,
    ):
        super().__init__()
        self.teacher = teacher
        self.tau = tau
        self.distill_mode = distill_mode
        self.use_gate = distill_mode == "region_gate"
        self.use_semantic_gate = distill_mode == "region_semantic_gate"
        self.use_region = distill_mode != "global_only"

        target_aliases = {
            "cls_attention": "clip_attention",
            "clip_attention": "clip_attention",
            "vpr_reliability": "semantic_reliability",
            "vpr_semantic_reliability": "semantic_reliability",
            "semantic_reliability": "semantic_reliability",
        }
        try:
            self.attention_target = target_aliases[attention_target]
        except KeyError as exc:
            raise ValueError(
                "attention_target must be one of: clip_attention, "
                "semantic_reliability"
            ) from exc

        self.semantic_reliability_target = None
        if self.attention_target == "semantic_reliability":
            self.semantic_reliability_target = VPRSemanticReliabilityTarget(
                temperature=reliability_temperature,
                negative_topk=reliability_negative_topk,
                positive_weight=reliability_positive_weight,
                negative_weight=reliability_negative_weight,
                pair_chunk_size=reliability_pair_chunk_size,
            )

        if proj_dim is None:
            proj_dim = student_feat_channels
        self.proj_dim = proj_dim

        # ---- projections ----
        self.student_global_proj = nn.Linear(student_global_dim, teacher_global_dim)

        if self.use_region:
            self.token_proj = nn.Linear(teacher_token_dim, proj_dim)
            self.global_attn_proj = nn.Linear(teacher_global_dim, proj_dim)
            if proj_dim != student_feat_channels:
                self.student_feat_proj = nn.Linear(student_feat_channels, proj_dim)
            else:
                self.student_feat_proj = None

    # ------------------------------------------------------------------
    # SG-SA weights
    # ------------------------------------------------------------------
    def compute_sgsa_weights(
        self,
        t_tokens_proj: torch.Tensor,
        t_global: torch.Tensor,
        t_tokens: torch.Tensor | None = None,
        t_tokens_aug: torch.Tensor | None = None,
        semantic_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return normalised spatial weights  (B, N)."""
        # w0 = softmax( cos(proj(token_i), attn_proj(t_global)) / tau )
        t_global_proj = self.global_attn_proj(t_global)  # (B, proj_dim)
        sim = F.cosine_similarity(
            t_tokens_proj, t_global_proj.unsqueeze(1), dim=-1
        )  # (B, N)
        w0 = F.softmax(sim / self.tau, dim=-1)  # (B, N)

        if self.use_gate and t_tokens is not None and t_tokens_aug is not None:
            gate = F.cosine_similarity(t_tokens, t_tokens_aug, dim=-1)  # (B, N)
            gate = (gate + 1.0) / 2.0  # map [-1,1] → [0,1]
            w = w0 * gate
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
        elif self.use_semantic_gate and semantic_mask is not None:
            w = w0 * semantic_mask
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            w = w0

        return w

    # ------------------------------------------------------------------
    # Region vectors
    # ------------------------------------------------------------------
    def compute_regions(
        self,
        w: torch.Tensor,
        student_featmap: torch.Tensor,
        t_tokens_proj: torch.Tensor,
    ):
        """Compute s_region and t_region.

        Args:
            w              : (B, N)
            student_featmap: (B, C, Hs, Ws)
            t_tokens_proj  : (B, N, proj_dim)
        Returns:
            s_region, t_region – both (B, proj_dim), L2-normalised
        """
        B, C, Hs, Ws = student_featmap.shape
        N = t_tokens_proj.shape[1]

        # Reshape weights to (B,1, √N, √N) and bilinear-resize to (B,1,Hs,Ws)
        side = int(N**0.5)
        w_2d = w.view(B, 1, side, side)
        w_resized = F.interpolate(
            w_2d, size=(Hs, Ws), mode="bilinear", align_corners=False
        )

        # Student region
        if self.student_feat_proj is not None:
            sfeat = student_featmap.flatten(2).permute(0, 2, 1)  # (B, Hs*Ws, C)
            sfeat = self.student_feat_proj(sfeat)                # (B, Hs*Ws, proj_dim)
            sfeat = sfeat.permute(0, 2, 1).view(B, self.proj_dim, Hs, Ws)
        else:
            sfeat = student_featmap

        s_region = (w_resized * sfeat).sum(dim=[2, 3])  # (B, proj_dim)
        s_region = F.normalize(s_region, dim=-1)

        # Teacher region
        t_region = (w.unsqueeze(-1) * t_tokens_proj).sum(dim=1)  # (B, proj_dim)
        t_region = F.normalize(t_region, dim=-1)

        return s_region, t_region

    # ------------------------------------------------------------------
    # Phase-C attention distillation
    # ------------------------------------------------------------------
    @staticmethod
    def attention_kl_loss(
        student_attn: torch.Tensor,
        teacher_cls_attn: torch.Tensor,
        student_hw: tuple[int, int],
        eps: float = 1e-8,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """KL(target || student) on the student's spatial grid.

        ``teacher_cls_attn`` is retained as the parameter name for backwards
        compatibility, but it may now contain either CLIP CLS attention or a
        VPR-conditioned semantic reliability distribution. Both inputs are
        already probabilities, so applying softmax here would flatten them.
        """
        if student_attn.ndim != 3:
            raise ValueError(
                "student_attn must have shape (B,num_heads,H*W), "
                f"got {tuple(student_attn.shape)}"
            )
        if teacher_cls_attn.ndim != 2:
            raise ValueError(
                "teacher_cls_attn must have shape (B,num_patches), "
                f"got {tuple(teacher_cls_attn.shape)}"
            )
        if student_attn.shape[0] != teacher_cls_attn.shape[0]:
            raise ValueError("student and teacher attention batch sizes differ")
        if valid_mask is not None:
            valid_mask = valid_mask.reshape(-1)
            if valid_mask.numel() != student_attn.shape[0]:
                raise ValueError("valid_mask must have one value per batch item")

        teacher_patches = teacher_cls_attn.shape[-1]
        teacher_side = int(teacher_patches**0.5)
        if teacher_side * teacher_side != teacher_patches:
            raise ValueError(
                "teacher patch count must form a square grid, "
                f"got {teacher_patches}"
            )

        target_h, target_w = student_hw
        if student_attn.shape[-1] != target_h * target_w:
            raise ValueError(
                "student attention size does not match student_hw: "
                f"{student_attn.shape[-1]} vs {target_h}x{target_w}"
            )

        teacher = teacher_cls_attn.float().clamp_min(eps)
        teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(eps)
        teacher = teacher.view(-1, 1, teacher_side, teacher_side)
        teacher = F.interpolate(
            teacher,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).flatten(1)
        teacher = teacher.clamp_min(eps)
        teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(eps)

        student = student_attn.float().mean(dim=1).clamp_min(eps)
        student = student / student.sum(dim=-1, keepdim=True).clamp_min(eps)

        per_sample = (
            teacher * (teacher.log() - student.log())
        ).sum(dim=-1)
        if valid_mask is None:
            return per_sample.mean()

        weights = valid_mask.to(per_sample.dtype)
        # A short final batch can contain only one place. Return a graph-
        # connected zero rather than mean(empty)=NaN in that case.
        return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        images_aug: torch.Tensor | None,
        student_featmap: torch.Tensor,
        student_global: torch.Tensor,
        student_attn: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        compute_global: bool = True,
        compute_region: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images          : (B,3,H,W) ImageNet-normalised
            images_aug      : (B,3,H,W) augmented view (can be None)
            student_featmap : (B,C,Hs,Ws) backbone feature map
            student_global  : (B,D_s) aggregator descriptor
            student_attn    : optional (B,num_heads,Hs*Ws) phase-C map
            labels          : place labels, required by semantic reliability
            compute_global  : skip the global branch when its lambda is zero
            compute_region  : skip the region branch when its lambda is zero

        Returns:
            dict with ``loss_global``, ``loss_region`` and ``loss_attn``.
        """
        # ---- teacher (frozen) ----
        reliability_valid = None
        reliability_stats: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            self.teacher.eval()
            needs_cls_attention = (
                student_attn is not None
                and self.attention_target == "clip_attention"
            )
            if needs_cls_attention:
                t_global, t_tokens, teacher_cls_attn = self.teacher(
                    images, return_attn=True
                )
            else:
                t_global, t_tokens = self.teacher(images)
                teacher_cls_attn = None

            if (
                student_attn is not None
                and self.attention_target == "semantic_reliability"
            ):
                if labels is None:
                    raise ValueError(
                        "labels are required for semantic reliability attention"
                    )
                if not hasattr(self.teacher, "project_patch_tokens"):
                    raise RuntimeError(
                        "semantic reliability requires a teacher with "
                        "project_patch_tokens()"
                    )
                patch_embeddings = self.teacher.project_patch_tokens(t_tokens)
                teacher_cls_attn, reliability_valid, reliability_stats = (
                    self.semantic_reliability_target(
                        patch_embeddings=patch_embeddings,
                        global_embeddings=t_global,
                        labels=labels,
                    )
                )
            t_tokens_aug = None
            semantic_mask = None
            if compute_region and self.use_gate and images_aug is not None:
                _, t_tokens_aug = self.teacher(images_aug)
            if compute_region and self.use_semantic_gate:
                semantic_mask = self.teacher.compute_semantic_mask(t_tokens)

        result: dict[str, torch.Tensor] = {}
        result.update(reliability_stats)

        # ---- global distillation ----
        if compute_global:
            s_global_proj = F.normalize(
                self.student_global_proj(student_global), dim=-1
            )
            result["loss_global"] = (
                s_global_proj - t_global
            ).pow(2).sum(dim=-1).mean()
        else:
            result["loss_global"] = student_global.new_zeros(())

        # ---- region distillation ----
        if self.use_region and compute_region:
            t_tokens_proj = self.token_proj(t_tokens)  # (B, N, proj_dim)
            w = self.compute_sgsa_weights(
                t_tokens_proj, t_global, t_tokens, t_tokens_aug, semantic_mask
            )
            s_region, t_region = self.compute_regions(
                w, student_featmap, t_tokens_proj
            )
            result["loss_region"] = (
                1.0 - F.cosine_similarity(s_region, t_region, dim=-1)
            ).mean()
        else:
            result["loss_region"] = torch.tensor(
                0.0, device=student_global.device
            )

        if student_attn is not None:
            result["loss_attn"] = self.attention_kl_loss(
                student_attn=student_attn,
                teacher_cls_attn=teacher_cls_attn,
                student_hw=student_featmap.shape[-2:],
                valid_mask=reliability_valid,
            )
        else:
            result["loss_attn"] = student_global.new_zeros(())

        return result

    # Always keep teacher in eval
    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self
