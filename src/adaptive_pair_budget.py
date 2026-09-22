"""Causal two-stage Pair-VPR budget rule: consumes only the first20 scores."""
import hashlib
import numpy as np


def prefix_margin(prefix):
    x = np.asarray(prefix)
    if x.ndim != 2 or x.shape[1] != 20 or not np.isfinite(x).all():
        raise ValueError('Policy must receive exactly 20 finite observed scores')
    ordered = np.sort(x, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def decide(prefix, threshold):
    if not np.isfinite(threshold): raise ValueError('Invalid threshold')
    return np.where(prefix_margin(prefix) <= threshold, 44, 20)


def fixed_random(ids, fraction):
    return np.array([44 if int(hashlib.sha256(('adaptive-budget42:'+str(i)).encode()).hexdigest()[:16], 16)/2**64 < fraction
                     else 20 for i in ids])


def outcomes(scores, labels, budgets):
    scores = np.asarray(scores); labels = np.asarray(labels); budgets = np.asarray(budgets)
    if scores.ndim != 2 or scores.shape[1] != 44 or labels.shape != scores.shape or labels.dtype != bool:
        raise ValueError('Invalid score/GT arrays')
    if not np.isfinite(scores).all() or budgets.shape != (len(scores),) or budgets.dtype.kind not in 'iu' or ((budgets < 20) | (budgets > 44)).any():
        raise ValueError('Invalid budget')
    selected = np.argmax(np.where(np.arange(44)[None, :] < budgets[:, None], scores, -np.inf), axis=1)
    return labels[np.arange(len(scores)), selected], selected
