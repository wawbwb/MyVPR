"""Preregistered larger-gallery SLGD diagnostic, independent of teacher scores."""
import numpy as np


HARD_POLICY = dict(revision=2, query_places=2048, distractor_places=8192,
                   calibration_places=1024, query_views=[0, 1, 2], reference_view=3,
                   distractor_view=0, sampling_epoch=99, candidate_k=20,
                   difficulty_margin=.02, negative_exclusion_metres=25.,
                   min_hard_queries=128, min_reachable_error_places=20,
                   min_raw_headroom_pp=1., min_teacher_gain_pp=1.,
                   min_teacher_hard_net=5, max_all_net_regression=0,
                   min_candidate_valid_fraction=.95,
                   cache_precision='RU float32; normalized raw/teacher tokens float16; scoring float32',
                   scope='Exploratory GSV exact-place-ID/GPS-filtered diagnostic; not standard MSLS/Pitts or stable benefit',
                   selection='Re-evaluate frozen v1 last teacher; no retraining, score-based sample selection, or threshold sweep')


def make_plan(place_ids, teacher_contract, smoke=False):
    """Keep original train identities excluded; split before any score is read."""
    ids = [str(x) for x in place_ids]
    if len(ids) != len(set(ids)):
        raise ValueError('Place identities must be unique')
    training = set(teacher_contract['train_places'])
    old_holdout = list(teacher_contract['holdout_places'])
    if len(old_holdout) != len(set(old_holdout)):
        raise ValueError('Teacher holdout identities must be unique')
    if training & set(old_holdout) or not (training | set(old_holdout)).issubset(set(ids)):
        raise ValueError('Teacher split is not compatible with dataset')
    import hashlib
    ordered = sorted(ids, key=lambda x: hashlib.sha256(f'slgd-partition42:{x}'.encode()).hexdigest())
    old_holdout_set = set(old_holdout)
    eligible = old_holdout + [x for x in ordered if x not in training and x not in old_holdout_set]
    nq, nd = (16, 32) if smoke else (HARD_POLICY['query_places'], HARD_POLICY['distractor_places'])
    if len(eligible) < nq+nd:
        raise ValueError('Insufficient nontraining places')
    queries, distractors = eligible[:nq], eligible[nq:nq+nd]
    if (set(queries) | set(distractors)) & training or set(queries) & set(distractors):
        raise ValueError('Train/query/gallery split overlaps')
    lookup = {x: i for i, x in enumerate(ids)}
    return dict(query_places=queries, distractor_places=distractors,
                dataset_indices=[lookup[x] for x in queries+distractors],
                calibration_places=nq//2 if smoke else HARD_POLICY['calibration_places'],
                rows=nq*4+nd, query_rows=[4*i+j for i in range(nq) for j in (0, 1, 2)],
                gallery_rows=[4*i+3 for i in range(nq)]+list(range(nq*4, nq*4+nd)),
                query_place_groups=np.repeat(np.arange(nq), 3).tolist(),
                positive_gallery_indices=np.repeat(np.arange(nq), 3).tolist())


def distance_metres(left, right):
    """Broadcast pairwise haversine distances; inputs are latitude/longitude."""
    left, right = np.asarray(left, float), np.asarray(right, float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != 2 or right.shape[1] != 2:
        raise ValueError('Expected N,2 latitude/longitude arrays')
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Coordinates must be finite')
    if (np.abs(left[:, 0]) > 90).any() or (np.abs(right[:, 0]) > 90).any():
        raise ValueError('Latitude outside valid range')
    l, r = np.deg2rad(left), np.deg2rad(right)
    d = np.sin((l[:, None, 0]-r[None, :, 0])/2)**2
    d += np.cos(l[:, None, 0])*np.cos(r[None, :, 0])*np.sin((l[:, None, 1]-r[None, :, 1])/2)**2
    return 6371000.*2*np.arcsin(np.sqrt(np.clip(d, 0, 1)))


def rerank_indices(scores, valid):
    """Stable RU-order tie break. All-invalid queries abstain to RU rank one."""
    scores, valid = np.asarray(scores), np.asarray(valid, bool)
    if scores.ndim != 2 or scores.shape != valid.shape or not np.isfinite(scores).all():
        raise ValueError('Invalid score/valid arrays')
    masked = np.where(valid, scores, -np.inf)
    prediction = masked.argmax(1)
    prediction[~valid.any(1)] = 0
    return prediction


def diagnostic_summary(candidates, scores, teacher_valid, raw_valid,
                       positive_gallery, groups, ru_gap, calibration_places,
                       policy=HARD_POLICY):
    """Only frozen RU defines difficulty; report aggregate and hard strata."""
    candidates = np.asarray(candidates)
    positive_gallery = np.asarray(positive_gallery)
    groups = np.asarray(groups)
    ru_gap = np.asarray(ru_gap)
    if candidates.ndim != 2 or len(candidates) != len(groups) or len(positive_gallery) != len(groups):
        raise ValueError('Query/candidate shapes differ')
    truth = candidates == positive_gallery[:, None]
    reachable = truth.any(1)
    ru = truth[:, 0]
    raw = truth[np.arange(len(truth)), rerank_indices(scores['raw'], raw_valid)]
    teacher = truth[np.arange(len(truth)), rerank_indices(scores['teacher'], teacher_valid)]
    hard = ~ru | (ru_gap <= policy['difficulty_margin'])
    partitions = {}
    for name, mask in [('calibration', groups < calibration_places), ('evaluation', groups >= calibration_places)]:
        h = mask & hard
        n = int(h.sum())
        error_places = len(np.unique(groups[mask & ~ru & reachable]))
        headroom = 100.*int((h & reachable & ~raw).sum())/n if n else 0.
        corrections = int((h & ~raw & teacher).sum())
        regressions = int((h & raw & ~teacher).sum())
        all_net = int(teacher[mask].sum())-int(raw[mask].sum())
        valid_fraction = float(teacher_valid[h].mean()) if n else 0.
        sufficiency = dict(hard_queries=n >= policy['min_hard_queries'],
                           reachable_error_places=error_places >= policy['min_reachable_error_places'],
                           raw_headroom=headroom+1e-10 >= policy['min_raw_headroom_pp'])
        gain = 100.*(corrections-regressions)/n if n else 0.
        efficacy = dict(hard_gain=gain+1e-10 >= policy['min_teacher_gain_pp'],
                        hard_net=corrections-regressions >= policy['min_teacher_hard_net'],
                        all_not_worse=all_net >= -policy['max_all_net_regression'],
                        candidate_valid_fraction=valid_fraction >= policy['min_candidate_valid_fraction'])
        partitions[name] = dict(queries=int(mask.sum()), hard_queries=n,
                                ru_correct=int(ru[mask].sum()), raw_correct=int(raw[mask].sum()),
                                teacher_correct=int(teacher[mask].sum()), reachable_errors=int((mask & ~ru & reachable).sum()),
                                reachable_error_places=error_places, unreachable_errors=int((mask & ~reachable).sum()),
                                hard_raw_correct=int(raw[h].sum()), hard_teacher_correct=int(teacher[h].sum()),
                                hard_oracle_correct=int(reachable[h].sum()), raw_headroom_pp=headroom,
                                hard_corrections=corrections, hard_regressions=regressions,
                                hard_net=corrections-regressions, hard_gain_pp=gain, all_net=all_net,
                                teacher_candidate_valid_fraction=valid_fraction,
                                sufficiency_checks=sufficiency, efficacy_checks=efficacy)
    enough = all(all(v['sufficiency_checks'].values()) for v in partitions.values())
    passed = all(all(v['efficacy_checks'].values()) for v in partitions.values())
    verdict = 'INSUFFICIENT_HARD_CASES' if not enough else ('PASS' if passed else 'NO_TEACHER_ADVANTAGE')
    return dict(verdict=verdict, partitions=partitions, scope=policy['scope']), dict(
        ru=ru.tolist(), raw=raw.tolist(), teacher=teacher.tolist(), reachable=reachable.tolist(), hard=hard.tolist())
