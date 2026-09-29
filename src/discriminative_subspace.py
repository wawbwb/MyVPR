"""Train-only equal-place scatter estimation and fixed subspace controls."""
import torch

MODES = ('pca', 'shuffled_fisher', 'place_fisher')


def scatter(x):
    x = torch.as_tensor(x, dtype=torch.float64)
    if x.ndim != 3 or min(x.shape[:2]) < 2 or not torch.isfinite(x).all():
        raise ValueError('Expected finite places x views x channels')
    means = x.mean(1)
    within = (x-means[:, None]).flatten(0, 1)
    between = means-means.mean(0)
    return within.T@within/len(within), between.T@between/len(between)


def fit(x, rank=16, fisher=True):
    sw, sb = scatter(x)
    dim = len(sw)
    if not 0 < rank < dim: raise ValueError('Invalid rank')
    tau = sw.trace()/dim
    if tau <= 0: raise ValueError('Degenerate within-place scatter')
    if fisher:
        # Fixed shrinkage, never selected with development outcomes.
        regular = .9*sw + (.1+1e-6)*tau*torch.eye(dim, dtype=sw.dtype)
        chol = torch.linalg.cholesky(regular)
        left = torch.linalg.solve_triangular(chol, sb, upper=False)
        matrix = torch.linalg.solve_triangular(chol, left.T, upper=False).T
        values, vectors = torch.linalg.eigh((matrix+matrix.T)*.5)
        raw = torch.linalg.solve_triangular(chol.T, vectors[:, -rank:], upper=True)
        condition = float(torch.linalg.cond(regular))
    else:
        values, vectors = torch.linalg.eigh(sw+sb)
        raw, condition = vectors[:, -rank:], None
    basis = torch.linalg.qr(raw, mode='reduced').Q
    return basis, dict(top_eigenvalues=values[-rank:].flip(0).tolist(), regularized_within_condition=condition,
                      orthogonality_error=float((basis.T@basis-torch.eye(rank)).abs().max()))


def shuffle_places(x, seed=42029):
    """Same number of groups and four views, destroy actual place grouping."""
    gen = torch.Generator().manual_seed(seed)
    flat = x.flatten(0, 1)
    indices = torch.randperm(len(flat), generator=gen)
    return flat[indices].reshape_as(x)


def ratio(x, basis):
    sw, sb = scatter(torch.as_tensor(x, dtype=torch.float64)@basis.double())
    w, b = float(sw.trace()), float(sb.trace())
    if w <= 0: raise ValueError('Degenerate projected variance')
    return dict(within=w, between=b, ratio=b/w)


def overlap(a, b):
    return float((a.double().T@b.double()).square().sum()/a.shape[1])


def assess(metrics, overlap_with_shuffle):
    t, p, s = (metrics[m] for m in ('place_fisher', 'pca', 'shuffled_fisher'))
    checks = dict(beats_pca_5pct=t['ratio'] > 1.05*p['ratio'],
                  beats_shuffled_5pct=t['ratio'] > 1.05*s['ratio'],
                  retains_between_variance=t['between'] >= .25*p['between'],
                  distinct_from_shuffled=overlap_with_shuffle < .95)
    return dict(checks=checks, pass_gate=all(checks.values()))
