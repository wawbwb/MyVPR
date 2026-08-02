import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.models.distillation import DistillationModule
from src.models.semantic_positive import CLIPSemanticPositiveLoss


def _angle_embeddings(degrees: list[float]) -> torch.Tensor:
    angles = torch.deg2rad(torch.tensor(degrees, dtype=torch.float32))
    return torch.stack((angles.cos(), angles.sin()), dim=1)


def _controlled_batch(
    *, requires_grad: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two three-view places with an unambiguous CLIP-hard pair each."""
    labels = torch.tensor([10, 10, 10, 20, 20, 20])
    clip_embeddings = _angle_embeddings(
        [0.0, 30.0, 120.0, 0.0, 60.0, 180.0]
    )
    student_descriptors = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.6, 0.8],
            [1.0, 0.0],
            [0.0, 1.0],
            [-0.8, 0.6],
        ]
    )
    if requires_grad:
        student_descriptors.requires_grad_()
        clip_embeddings.requires_grad_()
    return student_descriptors, clip_embeddings, labels


class _RecordingTeacher(nn.Module):
    """Small frozen-teacher stand-in used to verify clean-view routing."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.last_images: torch.Tensor | None = None
        self.register_buffer(
            "global_embeddings",
            F.normalize(
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]]
                ),
                dim=-1,
            ),
        )

    def forward(self, images: torch.Tensor, return_attn: bool = False):
        assert not return_attn
        self.last_images = images.detach().clone()
        batch_size = images.shape[0]
        globals_ = self.global_embeddings[:batch_size] * self.scale
        tokens = torch.zeros(batch_size, 1, 2, device=images.device)
        return globals_, tokens


def test_distillation_module_uses_clean_view_without_teacher_gradient() -> None:
    teacher = _RecordingTeacher()
    module = DistillationModule(
        teacher=teacher,
        teacher_token_dim=2,
        teacher_global_dim=2,
        student_feat_channels=1,
        student_global_dim=2,
        distill_mode="global_only",
        semantic_positive_enabled=True,
        semantic_positive_selection="clip",
    )
    module.train()
    assert not teacher.training

    student_images = torch.zeros(2, 3, 8, 8)
    clean_teacher_images = torch.ones(2, 3, 6, 6)
    student_descriptors = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]], requires_grad=True
    )
    output = module(
        images=student_images,
        images_aug=None,
        student_featmap=torch.randn(2, 1, 2, 2),
        student_global=student_descriptors,
        labels=torch.tensor([5, 5]),
        teacher_images=clean_teacher_images,
        compute_global=False,
        compute_region=False,
    )

    assert teacher.last_images is not None
    torch.testing.assert_close(teacher.last_images, clean_teacher_images)
    torch.testing.assert_close(output["loss_positive"], torch.tensor(1.0))
    assert "semantic_positive_selected_clip_sim" in output

    output["loss_positive"].backward()
    assert student_descriptors.grad is not None
    assert student_descriptors.grad.abs().sum() > 0
    assert teacher.scale.grad is None


def test_clip_selection_loss_gradient_and_metadata_stats() -> None:
    student, clip_embeddings, labels = _controlled_batch(requires_grad=True)
    objective = CLIPSemanticPositiveLoss(selection="clip", positive_topk=1)

    years = torch.tensor([2018, 2019, 2022, 2020, 2021, 2024])
    months = torch.tensor([1, 2, 7, 3, 4, 11])
    headings = torch.tensor([350, 90, 10, 30, 120, 300])
    loss, stats = objective(
        student,
        clip_embeddings,
        labels,
        years=years,
        months=months,
        headings=headings,
    )

    # CLIP selects (0,2) and (3,5), whose student cosine similarities are
    # 0.6 and -0.8. The optimized objective is mean(1 - cosine) = 1.1.
    selected_pairs = torch.tensor([[0, 2], [3, 5]])
    normalized_student = F.normalize(student.detach(), dim=-1)
    expected_student_similarity = torch.stack(
        [
            normalized_student[first] @ normalized_student[second]
            for first, second in selected_pairs.tolist()
        ]
    )
    expected_loss = (1.0 - expected_student_similarity).mean()
    torch.testing.assert_close(loss, expected_loss)

    normalized_clip = F.normalize(clip_embeddings.detach(), dim=-1)
    expected_clip_similarity = torch.stack(
        [
            normalized_clip[first] @ normalized_clip[second]
            for first, second in selected_pairs.tolist()
        ]
    ).mean()
    torch.testing.assert_close(
        stats["semantic_positive_selected_clip_sim"],
        expected_clip_similarity,
    )
    torch.testing.assert_close(
        stats["semantic_positive_selected_clip_disagreement"],
        1.0 - expected_clip_similarity,
    )
    torch.testing.assert_close(
        stats["semantic_positive_selected_student_sim"],
        expected_student_similarity.mean(),
    )
    torch.testing.assert_close(
        stats["semantic_positive_candidate_pair_count"], torch.tensor(6.0)
    )
    torch.testing.assert_close(
        stats["semantic_positive_candidates_per_place"], torch.tensor(3.0)
    )
    torch.testing.assert_close(
        stats["semantic_positive_selected_pair_count"], torch.tensor(2.0)
    )
    torch.testing.assert_close(
        stats["semantic_positive_valid_place_frac"], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        stats["semantic_positive_valid_view_frac"], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        stats["semantic_positive_view_coverage_frac"], torch.tensor(4.0 / 6.0)
    )
    # Only the second selected CLIP pair is also the student-hardest pair.
    torch.testing.assert_close(
        stats["semantic_positive_student_hard_overlap_frac"],
        torch.tensor(0.5),
    )

    # Metadata diagnostics use circular month/heading distance.
    torch.testing.assert_close(
        stats["semantic_positive_selected_year_gap"], torch.tensor(4.0)
    )
    torch.testing.assert_close(
        stats["semantic_positive_selected_month_gap"], torch.tensor(5.0)
    )
    torch.testing.assert_close(
        stats["semantic_positive_selected_heading_gap_deg"],
        torch.tensor(55.0),
    )
    torch.testing.assert_close(
        stats["semantic_positive_metadata_pair_frac"], torch.tensor(1.0)
    )
    for control_name in (
        "semantic_positive_random_control",
        "semantic_positive_shuffled_control",
        "semantic_positive_student_control",
    ):
        torch.testing.assert_close(stats[control_name], torch.zeros(()))

    assert all(key.startswith("semantic_positive_") for key in stats)
    assert all(value.ndim == 0 for value in stats.values())
    assert all(torch.isfinite(value) for value in stats.values())
    assert all(not value.requires_grad for value in stats.values())

    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert student.grad[[0, 2, 3, 5]].abs().sum() > 0
    torch.testing.assert_close(student.grad[[1, 4]], torch.zeros(2, 2))
    # CLIP is a detached selector, never a regression target.
    assert clip_embeddings.grad is None


def test_student_control_selects_the_hardest_student_positive() -> None:
    labels = torch.tensor([7, 7, 7])
    clip_embeddings = _angle_embeddings([0.0, 10.0, 100.0])
    student = _angle_embeddings([0.0, 120.0, 10.0]).requires_grad_()

    student_loss, student_stats = CLIPSemanticPositiveLoss(
        selection="student"
    )(student, clip_embeddings, labels)
    clip_loss, _ = CLIPSemanticPositiveLoss(selection="clip")(
        student, clip_embeddings, labels
    )

    # The student-hard pair is (0,1), with cosine cos(120 degrees) = -0.5.
    torch.testing.assert_close(student_loss, torch.tensor(1.5))
    torch.testing.assert_close(
        student_stats["semantic_positive_selected_clip_sim"],
        torch.cos(torch.deg2rad(torch.tensor(10.0))),
    )
    torch.testing.assert_close(
        student_stats["semantic_positive_student_hard_overlap_frac"],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        student_stats["semantic_positive_student_control"], torch.tensor(1.0)
    )
    assert student_loss > clip_loss

    student_loss.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0


def test_random_control_is_deterministic_and_does_not_consume_global_rng() -> None:
    student, clip_embeddings, labels = _controlled_batch()
    objective = CLIPSemanticPositiveLoss(selection="random", positive_topk=2)

    torch.manual_seed(12345)
    rng_before = torch.random.get_rng_state().clone()
    first_loss, first_stats = objective(student, clip_embeddings, labels)
    rng_after = torch.random.get_rng_state()
    torch.testing.assert_close(rng_after, rng_before)

    torch.manual_seed(9876)
    second_loss, second_stats = objective(student, clip_embeddings, labels)
    torch.testing.assert_close(first_loss, second_loss)
    assert first_stats.keys() == second_stats.keys()
    for key in first_stats:
        torch.testing.assert_close(first_stats[key], second_stats[key])

    torch.testing.assert_close(
        first_stats["semantic_positive_random_control"], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        first_stats["semantic_positive_selected_pair_count"], torch.tensor(4.0)
    )


def test_shuffled_control_is_fixed_and_breaks_clip_pair_correspondence() -> None:
    student, clip_embeddings, labels = _controlled_batch()
    objective = CLIPSemanticPositiveLoss(selection="shuffled", positive_topk=1)

    loss, stats = objective(student, clip_embeddings, labels)
    repeated_loss, repeated_stats = objective(student, clip_embeddings, labels)

    # The stable batch-level score permutation selects (0,2) and (3,4) for
    # this controlled batch. Their student similarities are 0.6 and 0.0.
    torch.testing.assert_close(loss, torch.tensor(0.7))
    torch.testing.assert_close(
        stats["semantic_positive_selected_clip_sim"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        stats["semantic_positive_shuffled_control"], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        stats["semantic_positive_random_control"], torch.tensor(0.0)
    )
    torch.testing.assert_close(repeated_loss, loss)
    for key in stats:
        torch.testing.assert_close(repeated_stats[key], stats[key])

    clip_loss, clip_stats = CLIPSemanticPositiveLoss(selection="clip")(
        student, clip_embeddings, labels
    )
    assert not torch.isclose(loss, clip_loss)
    assert not torch.isclose(
        stats["semantic_positive_selected_clip_sim"],
        clip_stats["semantic_positive_selected_clip_sim"],
    )


def test_clip_mining_is_invariant_to_noncontiguous_batch_layout() -> None:
    student, clip_embeddings, labels = _controlled_batch()
    years = torch.tensor([2018, 2019, 2022, 2020, 2021, 2024])
    months = torch.tensor([1, 2, 7, 3, 4, 11])
    headings = torch.tensor([350, 90, 10, 30, 120, 300])
    objective = CLIPSemanticPositiveLoss(selection="clip")
    expected_loss, expected_stats = objective(
        student,
        clip_embeddings,
        labels,
        years=years,
        months=months,
        headings=headings,
    )

    permutation = torch.tensor([4, 0, 5, 2, 3, 1])
    actual_loss, actual_stats = objective(
        student[permutation],
        clip_embeddings[permutation],
        labels[permutation],
        years=years[permutation],
        months=months[permutation],
        headings=headings[permutation],
    )

    torch.testing.assert_close(actual_loss, expected_loss)
    assert actual_stats.keys() == expected_stats.keys()
    for key in actual_stats:
        torch.testing.assert_close(actual_stats[key], expected_stats[key])


def test_no_same_place_pair_returns_graph_connected_zero() -> None:
    student = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    clip_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([10, 20])

    loss, stats = CLIPSemanticPositiveLoss()(
        student, clip_embeddings, labels
    )

    torch.testing.assert_close(loss, torch.zeros(()))
    assert loss.grad_fn is not None
    torch.testing.assert_close(
        stats["semantic_positive_candidate_pair_count"], torch.zeros(())
    )
    torch.testing.assert_close(
        stats["semantic_positive_selected_pair_count"], torch.zeros(())
    )
    torch.testing.assert_close(
        stats["semantic_positive_valid_place_frac"], torch.zeros(())
    )
    torch.testing.assert_close(
        stats["semantic_positive_metadata_pair_frac"], torch.zeros(())
    )
    assert all(value.ndim == 0 and torch.isfinite(value) for value in stats.values())

    loss.backward()
    torch.testing.assert_close(student.grad, torch.zeros_like(student))


def test_nonfinite_clip_view_is_excluded_from_candidates() -> None:
    student = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    clip_embeddings = torch.tensor([[1.0, 0.0], [float("nan"), 1.0]])
    labels = torch.tensor([5, 5])

    loss, stats = CLIPSemanticPositiveLoss()(
        student, clip_embeddings, labels
    )

    torch.testing.assert_close(loss, torch.zeros(()))
    torch.testing.assert_close(
        stats["semantic_positive_valid_view_frac"], torch.tensor(0.5)
    )
    torch.testing.assert_close(
        stats["semantic_positive_candidate_pair_count"], torch.zeros(())
    )
    assert torch.isfinite(loss)


def test_positive_topk_is_clamped_to_available_pairs() -> None:
    student = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    clip_embeddings = _angle_embeddings([0.0, 60.0, 180.0])
    labels = torch.tensor([9, 9, 9])

    loss, stats = CLIPSemanticPositiveLoss(
        selection="random", positive_topk=99
    )(student, clip_embeddings, labels)

    # All C(3,2)=3 pairs are selected; student cosines are 0, -1 and 0.
    torch.testing.assert_close(loss, torch.tensor(4.0 / 3.0))
    torch.testing.assert_close(
        stats["semantic_positive_candidate_pair_count"], torch.tensor(3.0)
    )
    torch.testing.assert_close(
        stats["semantic_positive_selected_pair_count"], torch.tensor(3.0)
    )
    torch.testing.assert_close(
        stats["semantic_positive_view_coverage_frac"], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        stats["semantic_positive_student_hard_overlap_frac"],
        torch.tensor(1.0),
    )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"selection": "negative"}, ValueError),
        ({"positive_topk": 0}, ValueError),
        ({"positive_topk": 1.5}, TypeError),
        ({"positive_topk": True}, TypeError),
    ],
)
def test_constructor_rejects_invalid_settings(
    kwargs: dict, error: type[Exception]
) -> None:
    with pytest.raises(error):
        CLIPSemanticPositiveLoss(**kwargs)


def test_forward_rejects_malformed_inputs_and_metadata() -> None:
    objective = CLIPSemanticPositiveLoss()
    student = torch.randn(2, 3)
    clip_embeddings = torch.randn(2, 4)
    labels = torch.tensor([0, 0])

    with pytest.raises(ValueError, match="student_descriptors"):
        objective(student.unsqueeze(0), clip_embeddings, labels)
    with pytest.raises(ValueError, match="clip_embeddings"):
        objective(student, clip_embeddings.unsqueeze(0), labels)
    with pytest.raises(TypeError, match="student_descriptors"):
        objective(student.long(), clip_embeddings, labels)
    with pytest.raises(TypeError, match="clip_embeddings"):
        objective(student, clip_embeddings.long(), labels)
    with pytest.raises(TypeError, match="labels"):
        objective(student, clip_embeddings, labels.float())
    with pytest.raises(ValueError, match="same batch size"):
        objective(student, clip_embeddings[:1], labels)
    with pytest.raises(ValueError, match="years"):
        objective(student, clip_embeddings, labels, years=torch.tensor([2020]))
    with pytest.raises(ValueError, match="months"):
        objective(student, clip_embeddings, labels, months=torch.tensor([1]))
    with pytest.raises(ValueError, match="headings"):
        objective(student, clip_embeddings, labels, headings=torch.tensor([0]))


def test_partial_or_invalid_metadata_is_safely_summarized() -> None:
    student, clip_embeddings, labels = _controlled_batch()
    objective = CLIPSemanticPositiveLoss(selection="clip")

    years = torch.tensor([float("nan"), 2019, 2022, 2020, 2021, 2024])
    _, partial_stats = objective(
        student, clip_embeddings, labels, years=years
    )
    # The valid selected pair (3,5) has a four-year gap. Missing month and
    # heading metadata make the all-fields-valid fraction zero.
    torch.testing.assert_close(
        partial_stats["semantic_positive_selected_year_gap"],
        torch.tensor(4.0),
    )
    torch.testing.assert_close(
        partial_stats["semantic_positive_selected_month_gap"], torch.zeros(())
    )
    torch.testing.assert_close(
        partial_stats["semantic_positive_selected_heading_gap_deg"],
        torch.zeros(()),
    )
    torch.testing.assert_close(
        partial_stats["semantic_positive_metadata_pair_frac"], torch.zeros(())
    )

    months = torch.tensor([0, 2, 7, 3, 4, 11])
    headings = torch.tensor([350, 90, 10, 30, 120, 300])
    _, invalid_stats = objective(
        student,
        clip_embeddings,
        labels,
        years=torch.tensor([2018, 2019, 2022, 2020, 2021, 2024]),
        months=months,
        headings=headings,
    )
    # One of the two selected pairs has an invalid month, while every other
    # supplied field is valid.
    torch.testing.assert_close(
        invalid_stats["semantic_positive_metadata_pair_frac"],
        torch.tensor(0.5),
    )
    torch.testing.assert_close(
        invalid_stats["semantic_positive_selected_month_gap"], torch.tensor(4.0)
    )
