# ----------------------------------------------------------------------------
# SG-SA distillation module: trainable projections, attention weights,
# region vectors, and loss computation.
# ----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    ):
        super().__init__()
        self.teacher = teacher
        self.tau = tau
        self.distill_mode = distill_mode
        self.use_gate = distill_mode == "region_gate"
        self.use_semantic_gate = distill_mode == "region_semantic_gate"
        self.use_region = distill_mode != "global_only"

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
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        images_aug: torch.Tensor | None,
        student_featmap: torch.Tensor,
        student_global: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images          : (B,3,H,W) ImageNet-normalised
            images_aug      : (B,3,H,W) augmented view (can be None)
            student_featmap : (B,C,Hs,Ws) backbone feature map
            student_global  : (B,D_s) aggregator descriptor

        Returns:
            dict  with keys ``loss_global`` and ``loss_region``
        """
        # ---- teacher (frozen) ----
        with torch.no_grad():
            self.teacher.eval()
            t_global, t_tokens = self.teacher(images)
            t_tokens_aug = None
            semantic_mask = None
            if self.use_gate and images_aug is not None:
                _, t_tokens_aug = self.teacher(images_aug)
            if self.use_semantic_gate:
                semantic_mask = self.teacher.compute_semantic_mask(t_tokens)

        result: dict[str, torch.Tensor] = {}

        # ---- global distillation ----
        s_global_proj = F.normalize(
            self.student_global_proj(student_global), dim=-1
        )
        result["loss_global"] = (s_global_proj - t_global).pow(2).sum(dim=-1).mean()

        # ---- region distillation ----
        if self.use_region:
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

        return result

    # Always keep teacher in eval
    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self
