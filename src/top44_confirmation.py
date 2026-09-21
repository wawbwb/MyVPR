"""Fixed paired statistics and panorama grouping; no model dependencies."""
import math
from pathlib import PurePosixPath
import re
import numpy as np

N_RESAMPLES = 10000
BOOTSTRAP_SEED = 42


def panorama_group(path):
    p = PurePosixPath(path.replace('\\', '/'))
    match = re.fullmatch(r'(\d+)_pitch\d+_yaw\d+\.(?:jpg|jpeg|png)', p.name, re.IGNORECASE)
    if match is None: raise ValueError('Cannot infer panorama ID: '+path)
    return str(p.parent / match.group(1))


def partition(index, old_index):
    ids = index['query_indices']; old = old_index['query_indices']
    for item in (index, old_index):
        if not (len(item['query_indices']) == len(item['queries']) == len(item['gt'])):
            raise ValueError('Truncated index arrays')
    if (len(ids) != len(set(ids)) or set(ids) != set(range(index['all_queries']))
        or not set(old).issubset(ids) or len(old) != len(set(old))): raise ValueError('Invalid query partition')
    if index['database'] != old_index['database'] or index['index_sha256'] != old_index['index_sha256']:
        raise ValueError('Different dataset/index identities')
    mapping = {qid: (path, gt) for qid, path, gt in zip(ids, index['queries'], index['gt'])}
    for qid, path, gt in zip(old, old_index['queries'], old_index['gt']):
        if mapping[qid] != (path, gt): raise ValueError('Old query/GT changed')
    excluded = set(old)
    rows = [dict(query_index=qid, path=mapping[qid][0], gt=mapping[qid][1],
                 group=panorama_group(mapping[qid][0]), prior_query=qid in excluded) for qid in sorted(ids)]
    return rows


def exact_mcnemar(corrections, regressions):
    """Two-sided exact binomial McNemar p; ignores dependence between queries."""
    n = corrections+regressions
    if n == 0: return 1.0
    k = min(corrections, regressions)
    terms = [math.exp(math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)-n*math.log(2))
             for i in range(k+1)]
    return min(1., 2*math.fsum(terms))


def cluster_interval(delta, groups, repeats=N_RESAMPLES, seed=BOOTSTRAP_SEED):
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim != 1 or len(delta) != len(groups) or not np.isin(delta, [-1, 0, 1]).all():
        raise ValueError('Invalid paired differences/groups')
    if repeats < 1: raise ValueError('Invalid resample count')
    if len(delta) == 0: return None
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2: return None
    sums = np.bincount(inverse, weights=delta); counts = np.bincount(inverse)
    rng = np.random.default_rng(seed); values = []
    for offset in range(0, repeats, 128):
        draw = rng.integers(0, len(unique), size=(min(128, repeats-offset), len(unique)))
        values.extend((100*sums[draw].sum(1)/counts[draw].sum(1)).tolist())
    return np.quantile(values, [.025, .975]).tolist()


def paired_statistics(rows):
    if not rows: return dict(queries=0, interpretation='NO_QUERIES')
    ids = [r['query_index'] for r in rows]
    if len(set(ids)) != len(ids): raise ValueError('Repeated query IDs')
    old = np.array([r['old_correct'] for r in rows], bool); new = np.array([r['new_correct'] for r in rows], bool)
    delta = new.astype(int)-old.astype(int); groups = [r['group'] for r in rows]
    fixes = [ids[i] for i in np.flatnonzero(delta == 1)]; regress = [ids[i] for i in np.flatnonzero(delta == -1)]
    interval = cluster_interval(delta, groups)
    net = len(fixes)-len(regress)
    verdict = ('SUPPORTED_ON_THIS_SAMPLE' if net > 0 and interval is not None and interval[0] > 0
               else 'OBSERVED_DEGRADATION' if net < 0 else 'INCONCLUSIVE')
    return dict(queries=len(rows), panorama_groups=len(set(groups)), top20_correct=int(old.sum()),
        top44_correct=int(new.sum()), top20_r1=float(old.mean()), top44_r1=float(new.mean()),
        corrections=fixes, regressions=regress, net=net, delta_r1_pp=float(100*delta.mean()),
        panorama_cluster_bootstrap_95ci_pp=interval,
        exact_mcnemar_two_sided_p_query_iid=exact_mcnemar(len(fixes), len(regress)),
        query_iid_test_caution='Auxiliary only: queries within panoramas/nearby places are dependent; no correction for historical method selection',
        interpretation=verdict,
        resampling='10000 uniform panorama-cluster draws, retaining paired outcomes and all views; percentile interval; seed42',
        limitation='Panorama grouping does not remove correlation between nearby panoramas or historical development exposure')
