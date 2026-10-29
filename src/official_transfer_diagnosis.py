"""Five fixed checks; no outcome-driven sweep, new seed, or budget extension."""
import numpy as np
import torch

from src.models.conv_residual_ablation import ResidualRuleNeck
from src.slgd_hard_protocol import distance_metres

EXPERIMENTS = {
    'e1_implementation': dict(kind='audit', rho=.1, lr=1e-4, negatives='historical'),
    'e2_attenuate': dict(kind='inference_only', rho=.025, lr=None, negatives='historical'),
    'e3_low_rho': dict(kind='train', rho=.025, lr=1e-4, negatives='historical'),
    'e4_low_lr': dict(kind='train', rho=.1, lr=1e-5, negatives='historical'),
    'e5_native_negatives': dict(kind='train', rho=.1, lr=1e-4, negatives='official'),
}
POLICY = dict(revision=2, experiments=EXPERIMENTS, seed=42, max_experiments=5,
    new_training_runs=3, steps=2048, train_size=280, primary_eval_size=322,
    secondary_eval_size=280, views=4, microbatch=4, weight_decay=0., clip=1.,
    selection='Exactly five registered items; fixed last2048, no best epoch/size/seed or additional coefficient',
    mining='Same train pool/views/anchors/batch counts; official mean four-view descriptors; cosine-hard distinct negatives >=25m from their anchor; index tie-break',
    scope='Single-seed controlled cause screening on exposed validation sets; not exhaustive causality, stable-gain, patentability or Pitts-test proof')


def gradient_comparison(left, right, tolerance):
    """Reject nonfinite/zero reference gradients; do not relax a failed limit."""
    if left.shape != right.shape or tolerance <= 0:
        raise ValueError('Invalid gradient comparison')
    norm = float(left.norm())
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    relative = float((left-right).norm()/left.norm().clamp_min(1e-12)) if finite else None
    return dict(relative_l2=relative, reference_norm=norm, candidate_norm=float(right.norm()),
        max_abs=float((left-right).abs().max()) if finite else None,
        tolerance=tolerance, passed=finite and norm>0 and relative<=tolerance)


class ScaledRMSNeck(ResidualRuleNeck):
    """Keep all existing names/weights; change only the output amplitude."""
    def __init__(self, original, rho):
        if not 0 < rho <= .1:
            raise ValueError('Unsupported registered residual cap')
        super().__init__(original, 'rms_tanh')
        self.output_scale = float(rho / .1)

    def forward(self, layers, auxiliary=False):
        _, aux, residual = super().forward(layers, auxiliary)
        residual = self.output_scale * residual
        return layers[-1] + residual, aux, residual


def choose_negative_batch(anchors, scores, coordinates, places_per_batch):
    """Change only negatives, retaining anchor order and exact batch size."""
    anchors = list(anchors)
    scores, coordinates = np.asarray(scores), np.asarray(coordinates)
    if len(anchors) != 8 or len(set(anchors)) != 8:
        raise ValueError('Expected eight distinct registered anchors')
    if scores.shape != (8, len(coordinates)) or coordinates.shape != (len(coordinates), 2):
        raise ValueError('Score/GPS population differs')
    if not np.isfinite(scores).all() or not np.isfinite(coordinates).all():
        raise ValueError('Nonfinite mining inputs')
    if any(type(a) is not int or a < 0 or a >= len(coordinates) for a in anchors):
        raise ValueError('Invalid anchor indices')
    if not 8 <= places_per_batch <= 16:
        raise ValueError('Unregistered batch capacity')
    allowed = distance_metres(coordinates[anchors], coordinates) >= 25.
    chosen = list(anchors); selections = []
    indices = np.arange(len(coordinates))
    for row, anchor in enumerate(anchors):
        if len(chosen) == places_per_batch:
            break
        ranking = np.lexsort((indices, -scores[row]))
        candidate = next((int(i) for i in ranking if allowed[row,i] and int(i) not in chosen), None)
        if candidate is None:
            raise ValueError('Insufficient official hard negatives; no silent fallback')
        chosen.append(candidate)
        selections.append(dict(anchor=anchor, negative=candidate, similarity=float(scores[row,candidate])))
    if len(chosen) != places_per_batch:
        raise ValueError('Batch size changed')
    return chosen, selections


def paired_summary(reference, candidate):
    reference, candidate = np.asarray(reference,bool), np.asarray(candidate,bool)
    if reference.ndim != 1 or reference.shape != candidate.shape or len(reference)==0:
        raise ValueError('Unmatched nonempty outcome populations')
    fixes = np.flatnonzero(~reference & candidate).tolist()
    regressions = np.flatnonzero(reference & ~candidate).tolist()
    return dict(corrections=fixes, regressions=regressions, net=len(fixes)-len(regressions),
                delta_r1_pp=100*(float(candidate.mean())-float(reference.mean())))
