"""Paired replication summaries; seed repeats never multiply query sample size."""
import numpy as np


def positive_components(ground_truth):
    """Connected query groups sharing any positive reference, deterministic."""
    parent = np.arange(len(ground_truth))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    first = {}
    for i, positives in enumerate(ground_truth):
        if not len(positives):
            raise ValueError('Empty ground truth')
        for reference in positives:
            reference = int(reference)
            if reference in first:
                a, b = find(i), find(first[reference])
                parent[max(a, b)] = min(a, b)
            else:
                first[reference] = i
    _, groups = np.unique([find(i) for i in range(len(parent))], return_inverse=True)
    return groups


def paired_interval(delta, groups, repetitions=2000, seed=20261026):
    delta, groups = np.asarray(delta, float), np.asarray(groups)
    if delta.ndim != 1 or groups.shape != delta.shape or not len(delta) or repetitions < 100:
        raise ValueError('Invalid paired bootstrap inputs')
    _, labels = np.unique(groups, return_inverse=True)
    sums = np.bincount(labels, weights=delta)
    sizes = np.bincount(labels)
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, repetitions, 64):
        ids = rng.integers(0, len(sizes), size=(min(64, repetitions-start), len(sizes)))
        estimates.extend((100*sums[ids].sum(1)/sizes[ids].sum(1)).tolist())
    return dict(delta_r1_pp=float(100*delta.mean()),
        percentile_95_ci_pp=np.quantile(estimates, [.025, .975]).tolist(),
        resampling_units=len(sizes), largest_unit_queries=int(sizes.max()),
        largest_unit_fraction=float(sizes.max()/len(delta)),
        adequate_unit_count=len(sizes) >= 20 and sizes.max()/len(delta) <= .5,
        repetitions=repetitions, seed=seed)


def contrast(candidate, control, clusters, repetitions=2000, seed=20261026):
    candidate, control = np.asarray(candidate, bool), np.asarray(control, bool)
    if candidate.shape != control.shape or candidate.ndim != 2 or candidate.shape[0] < 2:
        raise ValueError('Matched seed-by-query matrices required')
    delta = candidate.astype(float)-control.astype(float)
    # Average over seeds first; each sampled query carries ALL seed outcomes.
    per_query = delta.mean(0)
    counts = delta.sum(1)
    return dict(seed_net_queries=counts.astype(int).tolist(),
        mean_net_queries=float(counts.mean()), std_net_queries=float(counts.std(ddof=1)),
        positive_seeds=int((counts > 0).sum()), zero_seeds=int((counts == 0).sum()),
        negative_seeds=int((counts < 0).sum()),
        query_paired=paired_interval(per_query, np.arange(len(per_query)), repetitions, seed),
        positive_component_paired=paired_interval(per_query, clusters, repetitions, seed))
