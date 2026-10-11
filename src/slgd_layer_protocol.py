"""Exploratory frozen layer selection, never selected on evaluation outcomes."""
import hashlib
import numpy as np
import torch
from torch.nn import functional as F

from src.slgd_hard_protocol import rerank_indices


LAYER_POLICY = dict(revision=1, layers=[4, 6, 8, 10, 12], grids=[10, 20],
                    fine_queries_per_partition=192, min_mutual_matches=3,
                    selection='Calibration only: positive hard net vs L12 and nonnegative all net; hard net/all net/lower layer tie break',
                    fine_selection='All frozen-RU hard queries then SHA256 ordered ordinary queries; same subset for every layer/grid',
                    checkpoint='Original RU, including its fine-tuned DINO; not fresh pretrained DINO',
                    features='Raw post-block patches, no added final LayerNorm; pool then L2 normalize',
                    precision='Unit tokens quantized float16, re-normalized for float32 matching',
                    scope='Exploratory reuse of exposed GSV splits; not a fresh confirmatory test or MSLS/Pitts result')


class LayerCapture:
    """Hook the actual legacy forward, so L12 has identical semantics to RU."""
    def __init__(self, backbone, layers):
        self.backbone, self.layers = backbone, tuple(layers)
        if len(set(self.layers)) != len(self.layers) or not self.layers:
            raise ValueError('Layers must be nonempty and unique')
        blocks = backbone.dino.blocks
        if len(blocks) != 12 or any(type(x) is not int or not 1 <= x <= 12 for x in self.layers):
            raise ValueError('Expected 12-block DINOv2 ViT-B, using one-based layer numbers')
        if getattr(backbone, 'crop_semantic_film', None) is not None or getattr(backbone, 'residual_clip_fusion', None) is not None:
            raise ValueError('Expected original RU without additional feature branches')
        self.values = {}
        self.handles = []
        for layer in self.layers:
            def hook(module, inputs, output, layer=layer):
                if not isinstance(output, torch.Tensor) or output.ndim != 3:
                    raise ValueError('Expected B,N,C block output')
                self.values[layer] = output.detach().clone()
            self.handles.append(blocks[layer-1].register_forward_hook(hook))

    def __call__(self, images):
        self.values.clear()
        result = self.backbone(images)
        final = result[0] if isinstance(result, tuple) else result
        height, width = final.shape[-2:]
        features = {}
        for layer in self.layers:
            tokens = self.values[layer][:, 1:]
            if tokens.shape[1] != height*width:
                raise ValueError('Unexpected register tokens/grid')
            features[layer] = tokens.transpose(1, 2).reshape(final.shape[0], -1, height, width)
        if 12 in features and not torch.equal(features[12], final):
            raise ValueError('Hooked L12 does not reproduce legacy RU feature map exactly')
        self.values.clear()
        return features, final

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.values.clear()


def quantized_tokens(features, side):
    if side not in LAYER_POLICY['grids'] or tuple(features.shape[-2:]) != (20, 20):
        raise ValueError('Expected original 20x20 grid and preregistered grid size')
    tokens = F.adaptive_avg_pool2d(features, (side, side)).flatten(2).transpose(1, 2)
    return F.normalize(tokens.float(), dim=-1).cpu().numpy().astype(np.float16)


def fine_query_ids(groups, hard, calibration_places, cap=192):
    """No ground truth/raw/teacher score enters subset selection."""
    groups, hard = np.asarray(groups), np.asarray(hard, bool)
    if groups.shape != hard.shape or cap < 1:
        raise ValueError('Invalid subset inputs')
    selected = []
    for mask in (groups < calibration_places, groups >= calibration_places):
        ids = np.flatnonzero(mask).tolist()
        difficult = [i for i in ids if hard[i]]
        if len(difficult) > cap:
            raise ValueError('Fine subset cap smaller than frozen-RU hard stratum; do not silently drop cases')
        ordinary = sorted((i for i in ids if not hard[i]),
                          key=lambda i: hashlib.sha256(f'slgd-layer-fine:{i}'.encode()).hexdigest())
        selected.extend(difficult+ordinary[:max(0, cap-len(difficult))])
    return sorted(selected)


def summarize_layers(candidates, positive, groups, gap, scores, valid, cutoff,
                     reference='L12_g10'):
    candidates, positive, groups, gap = map(np.asarray, (candidates, positive, groups, gap))
    if candidates.ndim != 2 or candidates.shape[0] != len(groups) or positive.shape != groups.shape or gap.shape != groups.shape:
        raise ValueError('Query identities differ')
    if reference not in scores or set(scores) != set(valid):
        raise ValueError('Missing reference or different variant keys')
    truth = candidates == positive[:, None]
    ru, reachable = truth[:, 0], truth.any(1)
    hard = ~ru | (gap <= .02)
    correct, predictions = {}, {}
    for name, values in scores.items():
        if values.shape != candidates.shape or valid[name].shape != candidates.shape:
            raise ValueError('Variant candidate shape differs')
        rank = rerank_indices(values, valid[name])
        predictions[name] = candidates[np.arange(len(groups)), rank].tolist()
        correct[name] = truth[np.arange(len(groups)), rank]
    summary = {}
    for role, mask in [('calibration', groups < cutoff), ('evaluation', groups >= cutoff)]:
        n, hn = int(mask.sum()), int((mask & hard).sum())
        if not n:
            raise ValueError('Both partitions must be present')
        metrics = {}
        for name, values in scores.items():
            ok = correct[name]
            fixes = mask & ~ru & ok
            damage = mask & ru & ~ok
            h = mask & hard
            reachable_mask = mask & reachable
            # Positive vs strongest negative, never used for inference scores.
            posmask = truth & valid[name]
            negmask = ~truth & valid[name]
            available = reachable_mask & posmask.any(1) & negmask.any(1)
            pos_score = np.where(posmask, values, -np.inf).max(1)
            neg_score = np.where(negmask, values, -np.inf).max(1)
            margins = pos_score[available]-neg_score[available]
            metrics[name] = dict(correct=int(ok[mask].sum()), r1=float(ok[mask].mean()),
                corrections_vs_ru=int(fixes.sum()), regressions_vs_ru=int(damage.sum()),
                correction_places=len(np.unique(groups[fixes])),
                corrections_vs_last=int((mask & ~correct[reference] & ok).sum()),
                regressions_vs_last=int((mask & correct[reference] & ~ok).sum()),
                net_vs_last=int(ok[mask].sum())-int(correct[reference][mask].sum()),
                hard_correct=int(ok[h].sum()),
                hard_net_vs_last=int(ok[h].sum())-int(correct[reference][h].sum()),
                reachable_ru_errors_corrected=int((fixes & reachable).sum()),
                valid_candidate_fraction=float(valid[name][mask].mean()),
                margin_queries=int(available.sum()),
                mean_positive_negative_margin=float(margins.mean()) if len(margins) else None,
                positive_negative_margin_win_fraction=float((margins > 0).mean()) if len(margins) else None)
        summary[role] = dict(queries=n, hard_queries=hn, ru_correct=int(ru[mask].sum()),
                             reachable_ru_errors=int((mask & ~ru & reachable).sum()), variants=metrics)
    return summary, dict(correct={k:v.tolist() for k,v in correct.items()}, predictions=predictions,
                         ru=ru.tolist(), hard=hard.tolist(), reachable=reachable.tolist())


def choose_layer(calibration, layers=(4, 6, 8, 10, 12)):
    """Accept calibration only, not an object containing evaluation data."""
    variants = calibration['variants']
    eligible = [layer for layer in layers if layer != 12 and
                variants[f'L{layer}_g10']['hard_net_vs_last'] > 0 and
                variants[f'L{layer}_g10']['net_vs_last'] >= 0]
    if not eligible:
        return dict(layer=12, reason='NO_CALIBRATION_COMPLEMENT; keep reference, not method failure')
    best = max(eligible, key=lambda layer: (variants[f'L{layer}_g10']['hard_net_vs_last'],
                                           variants[f'L{layer}_g10']['net_vs_last'], -layer))
    return dict(layer=best, reason='CALIBRATION_SELECTED; evaluation cannot change selection')
