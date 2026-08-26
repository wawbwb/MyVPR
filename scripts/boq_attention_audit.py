"""Pure helpers for auditing BoQ cross-attention against spatial masks.

The functions in this module do not load datasets or checkpoints.  Keeping the
tensor accounting here side-effect free makes the metric definitions easy to
unit test without FAISS, Lightning, or an MSLS installation.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator, Sequence
from typing import Any

import torch


@contextlib.contextmanager
def force_per_head_cross_attention(aggregator: torch.nn.Module) -> Iterator[None]:
    """Make every BoQ cross-attention return per-head weights temporarily.

    PyTorch's ``MultiheadAttention`` averages returned weights across heads by
    default.  The average does not affect the attention output or descriptor,
    but it hides head specialisation.  A pre-hook changes only the format of
    the returned diagnostic weights and is always removed by this context
    manager.
    """

    blocks = getattr(aggregator, "boqs", None)
    if blocks is None or len(blocks) == 0:
        raise TypeError("aggregator has no BoQ blocks")

    handles = []

    def _force_weights(
        _module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        updated = dict(kwargs)
        updated["need_weights"] = True
        updated["average_attn_weights"] = False
        return args, updated

    try:
        for block in blocks:
            cross_attention = getattr(block, "cross_attn", None)
            if not isinstance(cross_attention, torch.nn.MultiheadAttention):
                raise TypeError("BoQ block has no torch MultiheadAttention cross_attn")
            handles.append(
                cross_attention.register_forward_pre_hook(
                    _force_weights,
                    with_kwargs=True,
                )
            )
        yield
    finally:
        for handle in handles:
            handle.remove()


def fc_energy_slot_weights(
    fc_weight: torch.Tensor,
    *,
    num_layers: int,
    num_queries: int,
) -> torch.Tensor:
    """Return a non-negative layer/query weighting derived from final FC energy.

    This is only a descriptor-aware proxy.  It cannot account for values,
    signed mixing, output projections, LayerNorm, or activations.
    """

    if fc_weight.ndim != 2:
        raise ValueError("BoQ fc weight must be two-dimensional")
    expected_slots = int(num_layers) * int(num_queries)
    if fc_weight.shape[1] != expected_slots:
        raise ValueError(
            "BoQ fc input slots do not match layer/query count: "
            f"{fc_weight.shape[1]} vs {expected_slots}"
        )
    energy = fc_weight.detach().float().square().sum(dim=0)
    total = energy.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0.0:
        raise ValueError("BoQ fc slot energy is not finite and positive")
    return (energy / total).reshape(num_layers, num_queries)


def _validate_attention_stack(
    attentions: Sequence[torch.Tensor],
    *,
    grid_size: tuple[int, int],
    normalisation_atol: float,
) -> torch.Tensor:
    if not attentions:
        raise ValueError("at least one BoQ attention layer is required")
    if len(grid_size) != 2 or min(grid_size) <= 0:
        raise ValueError(f"invalid grid size: {grid_size}")

    expected_tokens = int(grid_size[0]) * int(grid_size[1])
    first_shape = tuple(attentions[0].shape)
    if len(first_shape) != 4:
        raise ValueError(
            "per-head BoQ attention must have shape (B,H,Q,N), got "
            f"{first_shape}"
        )
    if first_shape[-1] != expected_tokens:
        raise ValueError(
            f"attention token count {first_shape[-1]} != grid {grid_size}"
        )

    layers = []
    for layer_index, attention in enumerate(attentions):
        if tuple(attention.shape) != first_shape:
            raise ValueError(
                f"BoQ attention layer {layer_index} shape {tuple(attention.shape)} "
                f"does not match {first_shape}"
            )
        values = attention.detach().float()
        if not bool(torch.isfinite(values).all()):
            raise ValueError("BoQ attention contains non-finite values")
        if bool((values < -normalisation_atol).any()):
            raise ValueError("BoQ attention contains negative probabilities")
        sums = values.sum(dim=-1)
        if not torch.allclose(
            sums,
            torch.ones_like(sums),
            rtol=0.0,
            atol=normalisation_atol,
        ):
            max_error = float((sums - 1.0).abs().max())
            raise ValueError(
                "BoQ attention is not normalised across spatial keys; "
                f"maximum error is {max_error:.6g}"
            )
        layers.append(values)
    return torch.stack(layers, dim=1)


def _map_focus_statistics(
    maps: torch.Tensor,
    *,
    top_fractions: Sequence[float],
) -> dict[str, torch.Tensor]:
    """Calculate concentration statistics for ``(B,C,N)`` probability maps."""

    num_tokens = maps.shape[-1]
    eps = torch.finfo(maps.dtype).tiny
    entropy_nats = -(maps.clamp_min(eps) * maps.clamp_min(eps).log()).sum(dim=-1)
    result: dict[str, torch.Tensor] = {
        "normalised_entropy": entropy_nats / math.log(num_tokens),
        "effective_patch_fraction": entropy_nats.exp() / num_tokens,
        "peak_attention": maps.max(dim=-1).values,
        "peak_density": num_tokens * maps.max(dim=-1).values,
    }
    for fraction in top_fractions:
        if not math.isfinite(float(fraction)) or not 0.0 < float(fraction) <= 1.0:
            raise ValueError("top fractions must lie in (0, 1]")
        count = max(1, int(math.ceil(num_tokens * float(fraction))))
        name = f"top_{int(round(100 * float(fraction)))}pct_attention_mass"
        result[name] = maps.topk(count, dim=-1).values.sum(dim=-1)
    return result


def compute_attention_components(
    attentions: Sequence[torch.Tensor],
    *,
    grid_size: tuple[int, int],
    fc_slot_weights: torch.Tensor,
    top_fractions: Sequence[float] = (0.1, 0.2),
    normalisation_atol: float = 2e-3,
) -> dict[str, Any]:
    """Reduce per-head BoQ weights into auditable spatial components.

    ``attentions`` contains one ``(B,H,Q,N)`` probability tensor per BoQ
    layer.  The primary maps are equal means over heads and learned queries.
    A final equal mean over layers is named ``consensus_raw``.  The optional
    ``fc_energy_proxy`` uses squared final-FC column norms and remains a proxy,
    not a causal attribution.
    """

    stack = _validate_attention_stack(
        attentions,
        grid_size=grid_size,
        normalisation_atol=normalisation_atol,
    )
    batch_size, num_layers, num_heads, num_queries, num_tokens = stack.shape
    expected_fc_shape = (num_layers, num_queries)
    if tuple(fc_slot_weights.shape) != expected_fc_shape:
        raise ValueError(
            f"fc slot weights must have shape {expected_fc_shape}, got "
            f"{tuple(fc_slot_weights.shape)}"
        )
    fc_weights = fc_slot_weights.to(device=stack.device, dtype=stack.dtype)
    if not bool(torch.isfinite(fc_weights).all()) or bool((fc_weights < 0).any()):
        raise ValueError("fc slot weights must be finite and non-negative")
    if not torch.allclose(
        fc_weights.sum(),
        torch.ones((), device=stack.device, dtype=stack.dtype),
        rtol=0.0,
        atol=1e-5,
    ):
        raise ValueError("fc slot weights must sum to one")

    head_maps = stack.mean(dim=3)  # (B,L,H,N), learned-query mean
    query_maps = stack.mean(dim=2)  # (B,L,Q,N), head mean
    layer_maps = stack.mean(dim=(2, 3))  # (B,L,N)
    consensus = layer_maps.mean(dim=1)
    fc_proxy = (query_maps * fc_weights[None, :, :, None]).sum(dim=(1, 2))
    component_names = [
        *(f"layer_{index + 1}" for index in range(num_layers)),
        "consensus_raw",
        "fc_energy_proxy",
    ]
    component_maps = torch.cat(
        (layer_maps, consensus[:, None, :], fc_proxy[:, None, :]),
        dim=1,
    )

    return {
        "stack": stack,
        "head_maps": head_maps,
        "query_maps": query_maps,
        "layer_maps": layer_maps,
        "consensus_map": consensus,
        "fc_proxy_map": fc_proxy,
        "component_names": component_names,
        "component_maps": component_maps,
        "focus": _map_focus_statistics(
            component_maps,
            top_fractions=top_fractions,
        ),
        "dimensions": {
            "batch_size": batch_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_queries": num_queries,
            "num_tokens": num_tokens,
        },
    }


def compute_mask_overlap(
    maps: torch.Tensor,
    masks: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Measure soft and support overlap between attention maps and masks.

    Args:
        maps: ``(B,C,N)`` spatial probability maps.
        masks: ``(B,V,H,W)`` or ``(B,V,N)`` mask area fractions in ``[0,1]``.

    Returns:
        Areas have shape ``(B,V)`` and masses/enrichment ``(B,V,C)``.
        Empty masks receive NaN enrichment instead of a fabricated value.
    """

    if maps.ndim != 3:
        raise ValueError("attention maps must have shape (B,C,N)")
    if masks.ndim == 4:
        masks = masks.flatten(2)
    if masks.ndim != 3:
        raise ValueError("masks must have shape (B,V,N) or (B,V,H,W)")
    if maps.shape[0] != masks.shape[0] or maps.shape[-1] != masks.shape[-1]:
        raise ValueError("attention maps and masks do not share batch/token axes")

    values = masks.to(device=maps.device, dtype=maps.dtype)
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()) or bool(
        (values > 1).any()
    ):
        raise ValueError("mask fractions must be finite and lie in [0,1]")
    area = values.mean(dim=-1)
    mass = torch.einsum("bcn,bvn->bvc", maps, values)
    enrichment = torch.full_like(mass, float("nan"))
    eligible = area > 0
    enrichment[eligible] = mass[eligible] / area[eligible].unsqueeze(-1)

    support = values > 0
    support_area = support.float().mean(dim=-1)
    support_mass = torch.einsum("bcn,bvn->bvc", maps, support.float())
    support_enrichment = torch.full_like(support_mass, float("nan"))
    support_eligible = support_area > 0
    support_enrichment[support_eligible] = (
        support_mass[support_eligible]
        / support_area[support_eligible].unsqueeze(-1)
    )
    return {
        "area": area,
        "mass": mass,
        "excess": mass - area[:, :, None],
        "enrichment": enrichment,
        "eligible": eligible,
        "support_area": support_area,
        "support_mass": support_mass,
        "support_enrichment": support_enrichment,
    }
