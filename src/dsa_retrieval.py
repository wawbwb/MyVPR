"""Place-clustered cross-view retrieval; no learned calibration."""
import numpy as np
import torch


def cross_view(x):
    x = torch.as_tensor(x, dtype=torch.float64)
    if x.ndim != 3 or x.shape[0] < 2 or x.shape[1] != 4 or not torch.isfinite(x).all():
        raise ValueError('Expected finite P x 4 x D vectors')
    if (x.norm(dim=-1) <= 1e-12).any(): raise ValueError('Zero descriptor')
    x = torch.nn.functional.normalize(x, dim=-1)
    ids = torch.arange(len(x)); hits=[]; ranks=[]
    for q in range(4):
        for d in range(4):
            if q == d: continue
            scores = x[:,q] @ x[:,d].T
            positive = scores.diagonal()[:,None]
            # Deterministic tie break: smaller database index first.
            rank = 1+(scores > positive).sum(1)+((scores == positive)&(ids[None,:] < ids[:,None])).sum(1)
            ranks.append(rank.numpy()); hits.append((rank == 1).numpy())
    return np.stack(hits,axis=1), np.stack(ranks,axis=1)


def paired(a, b):
    if a.shape != b.shape or a.ndim != 2: raise ValueError('Unmatched paired outcomes')
    delta = (a.astype(float)-b.astype(float)).mean(1)
    rng = np.random.default_rng(42030)
    means = np.array([delta[rng.integers(len(delta),size=len(delta))].mean() for _ in range(2000)])
    return dict(delta_r1_pp=float(delta.mean()*100), place_bootstrap_95ci_pp=(np.quantile(means,[.025,.975])*100).tolist(),
                corrections=int((a&~b).sum()), regressions=int((~a&b).sum()),
                scope='12 ordered view pairs per place, bootstrap places not individual pairs; unadjusted exploratory interval')
