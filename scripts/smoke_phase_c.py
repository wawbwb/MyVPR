"""Fast server-side smoke test for phase C (no dataset required).

This instantiates the pinned OpenCLIP teacher, verifies that requesting C1
attention does not alter its normal outputs, and checks that the C2 KL loss
produces gradients for the student spatial head.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.clip_teacher import CLIPTeacherEncoder
from src.models.distillation import DistillationModule, SpatialAttentionHead


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = CLIPTeacherEncoder(
        model_name="ViT-B-16",
        pretrained="openai",
    ).to(device)
    teacher.eval()

    # The training dataloader supplies ImageNet-normalised tensors.
    images = torch.randn(2, 3, 320, 320, device=device)
    with torch.no_grad():
        global_normal, tokens_normal = teacher(images)
        global_attn, tokens_attn, cls_attn = teacher(images, return_attn=True)

    torch.testing.assert_close(global_attn, global_normal, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(tokens_attn, tokens_normal, rtol=1e-5, atol=1e-6)
    if cls_attn.shape != (2, 196):
        raise AssertionError(f"expected CLIP attention (2,196), got {cls_attn.shape}")
    torch.testing.assert_close(
        cls_attn.sum(dim=-1),
        torch.ones(2, device=device),
        rtol=1e-5,
        atol=1e-6,
    )

    head = SpatialAttentionHead(in_channels=1024).to(device)
    featmap = torch.randn(2, 1024, 20, 20, device=device)
    weighted, student_attn = head(featmap)
    torch.testing.assert_close(weighted, featmap, rtol=1e-5, atol=1e-6)

    loss = DistillationModule.attention_kl_loss(
        student_attn=student_attn,
        teacher_cls_attn=cls_attn,
        student_hw=(20, 20),
    )
    loss.backward()
    grad = head.proj.weight.grad
    if (
        grad is None
        or not torch.isfinite(grad).all().item()
        or grad.abs().sum().item() == 0
    ):
        raise AssertionError("attention KL did not produce a finite non-zero gradient")

    print(
        "Phase C smoke test passed: "
        f"device={device}, attention_shape={tuple(cls_attn.shape)}, "
        f"kl={loss.item():.6f}, grad_norm={grad.norm().item():.6f}"
    )


if __name__ == "__main__":
    main()
