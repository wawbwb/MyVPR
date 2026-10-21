"""Pure fixed-checkpoint diagnostic statistics (not benchmark selection)."""
import numpy as np


def stats(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(x):
        return dict(n=0, mean=None, p50=None, p95=None, max=None)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite diagnostic statistic')
    return dict(n=len(x), mean=float(x.mean()), p50=float(np.median(x)),
                p95=float(np.quantile(x, .95)), max=float(x.max()))


def selected_queries(report, limit=64):
    """All baseline errors + all observed arm disagreements, then fixed corrects.

    Label-informed explanatory sample; never an unbiased performance estimate.
    """
    variants = report['variants']
    base = np.asarray(variants['innovation1']['outcomes'], bool)
    must = set(np.flatnonzero(~base).tolist())
    for arm in ('separate', 'aligned', 'permuted'):
        must.update(np.flatnonzero(base != np.asarray(variants[arm]['outcomes'], bool)).tolist())
    if len(must) > limit:
        raise ValueError('Mandatory explanatory cases exceed registered limit')
    for q in np.random.default_rng(42).permutation(len(base)):
        if len(must) == limit:
            break
        must.add(int(q))
    return sorted(must)


def candidate_union(report, queries, ground_truth):
    result = set()
    for q in queries:
        for arm in ('innovation1', 'separate', 'aligned', 'permuted'):
            result.update(map(int, report['variants'][arm]['predictions'][q]))
        result.update(map(int, ground_truth[q]))
    return sorted(result)


def margin_report(scores, references, queries, ground_truth, baseline_top1):
    """Margins in fixed candidate union INCLUDING ALL ground-truth positives.

    Counterfactual winners are conditional on this union, NOT full-DB recall.
    """
    scores = np.asarray(scores)
    if scores.shape != (len(queries), len(references)) or not np.isfinite(scores).all():
        raise ValueError('Invalid diagnostic scores')
    top1, margins, positive_minus_old = [], [], []
    for row, q in zip(scores, queries):
        positive = np.isin(references, ground_truth[q])
        if not positive.any() or positive.all():
            raise ValueError('Need both positive and negative candidates')
        best = float(row[positive].max())
        top1.append(int(references[int(row.argmax())]))
        margins.append(best - float(row[~positive].max()))
        old = np.flatnonzero(np.asarray(references) == baseline_top1[q])
        if len(old) != 1:
            raise ValueError('Original winner not in fixed candidate union')
        positive_minus_old.append(best - float(row[old[0]]))
    return dict(top1=top1, positive_negative_margin=margins,
                positive_minus_baseline_top1=positive_minus_old,
                correct=[t in set(map(int, ground_truth[q])) for t, q in zip(top1, queries)])
