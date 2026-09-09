"""CPU-only descriptive affine-consistency audit of saved Pair-VPR MNNs.

No model, OpenCV, image loading, ranking changes, or GT-based parameter selection.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np


GRID = 23
THRESHOLD = 1.5  # patch units (21 pixels in the 322-pixel input)
TRIALS = 256
NULL_REPEATS = 5


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str, parts)).encode()).digest()[:8], 'little')


def coords(ids):
    return np.column_stack((ids % GRID + .5, ids // GRID + .5)).astype(float)


def coverage(points):
    if not len(points):
        return {'occupied_4x4_fraction': 0., 'bbox_area_fraction': 0.}
    cells = np.minimum((points / GRID * 4).astype(int), 3)
    return {'occupied_4x4_fraction': len(np.unique(cells, axis=0)) / 16,
            'bbox_area_fraction': float(np.prod(np.ptp(points, axis=0)) / GRID**2)}


def fit_affine(x, y):
    design = np.column_stack((x, np.ones(len(x))))
    if len(x) < 3 or np.linalg.matrix_rank(design) < 3:
        return None
    matrix = np.linalg.lstsq(design, y, rcond=None)[0]
    linear = matrix[:2]
    if not np.isfinite(matrix).all() or np.linalg.cond(linear) > 100:
        return None
    return matrix


def residuals(x, y, matrix):
    forward = np.linalg.norm(x @ matrix[:2] + matrix[2] - y, axis=1)
    inverse = np.linalg.inv(matrix[:2])
    backward = np.linalg.norm((y - matrix[2]) @ inverse - x, axis=1)
    return np.maximum(forward, backward)


def consensus(x, y, seed):
    rng = np.random.default_rng(seed)
    best = np.zeros(len(x), dtype=bool)
    best_key, best_matrix = (-1, -float('inf')), None
    for _ in range(TRIALS if len(x) >= 3 else 0):
        pick = rng.choice(len(x), 3, replace=False)
        matrix = fit_affine(x[pick], y[pick])
        if matrix is None:
            continue
        err = residuals(x, y, matrix)
        mask = err <= THRESHOLD
        # Refine without accepting a reduction in the objective.
        for _ in range(2):
            updated = fit_affine(x[mask], y[mask])
            if updated is None:
                break
            updated_err = residuals(x, y, updated)
            updated_mask = updated_err <= THRESHOLD
            old_key = (int(mask.sum()), -float(np.median(err[mask])) if mask.any() else -float('inf'))
            new_key = (int(updated_mask.sum()), -float(np.median(updated_err[updated_mask])) if updated_mask.any() else -float('inf'))
            if new_key < old_key:
                break
            matrix, err, mask = updated, updated_err, updated_mask
        key = (int(mask.sum()), -float(np.median(err[mask])) if mask.any() else -float('inf'))
        if key > best_key:
            best_key, best, best_matrix = key, mask, matrix
    error = residuals(x, y, best_matrix) if best_matrix is not None else None
    return {'inliers': int(best.sum()), 'inlier_fraction': float(best.mean()) if len(best) else 0.,
            'median_inlier_error_patches': float(np.median(error[best])) if best.any() else None,
            'q_coverage': coverage(x[best]), 'db_coverage': coverage(y[best]),
            'affine_matrix': None if best_matrix is None else best_matrix.tolist()}, best


def diagram(x, y, mask):
    # Native vector diagnostic, not an attention map or edited photograph.
    elements = ['<svg viewBox="0 0 520 240"><rect width="230" height="230" fill="#eee"/>',
                '<rect x="280" width="230" height="230" fill="#eee"/>']
    for i, (a, b) in enumerate(zip(x, y)):
        color = '#167a3c' if mask[i] else '#b6b6b6'
        opacity = '.7' if mask[i] else '.12'
        elements.append(f'<line x1="{a[0]*10}" y1="{a[1]*10}" x2="{280+b[0]*10}" y2="{b[1]*10}" stroke="{color}" opacity="{opacity}"/>')
    elements.append('</svg>')
    return ''.join(elements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Choose a new output directory')
    completion = json.loads((args.input / 'completed.json').read_text(encoding='utf8'))
    rows = json.loads((args.input / 'results.json').read_text(encoding='utf8'))
    require(completion.get('complete') and len(rows) == completion['pairs'] and len(rows) > 0, 'Incomplete input')
    # Validate the complete input before creating output files.
    seen = set()
    for row in rows:
        key = (row['query'], row['db'])
        require(key not in seen, f'Duplicate pair: {key}')
        seen.add(key)
        m = np.asarray(row['matches'], dtype=float).reshape(-1, 3)
        require(len(m) == row['mnn_count'] and np.isfinite(m).all(), f'Invalid matches: {key}')
        require(((m[:, :2] >= 0) & (m[:, :2] < GRID**2)).all()
                and (m[:, :2] == np.floor(m[:, :2])).all(), f'Invalid token IDs: {key}')
        require(len(np.unique(m[:, 0])) == len(m) and len(np.unique(m[:, 1])) == len(m), f'Not one-to-one MNN: {key}')
    args.output.mkdir(parents=True)
    reports, sections = [], {}
    for row in rows:
        qi, di = row['query'], row['db']
        matches = np.asarray(row['matches'], dtype=float).reshape(-1, 3)
        x, y = coords(matches[:, 0].astype(int)), coords(matches[:, 1].astype(int))
        result, mask = consensus(x, y, seed_for(42, qi, di, 'fit'))
        null = []
        for repeat in range(NULL_REPEATS):
            shuffled = np.random.default_rng(seed_for(42, qi, di, repeat, 'shuffle')).permutation(len(y))
            control, _ = consensus(x, y[shuffled], seed_for(42, qi, di, repeat, 'null_fit'))
            null.append(control['inliers'])
        record = {k: v for k, v in row.items() if k != 'matches'}
        record.update({'geometry': result, 'all_q_coverage': coverage(x), 'all_db_coverage': coverage(y),
                       'null_inlier_counts': null, 'null_mean': float(np.mean(null)),
                       'excess_inliers_over_null_mean': result['inliers'] - float(np.mean(null)),
                       'inlier_match_indices': np.flatnonzero(mask).tolist()})
        reports.append(record)
        label = f'db={di}; GT={row["gt"]}; saved rank={row["refined_rank"]}; inliers={result["inliers"]}/{len(x)}; shuffled={null}'
        sections.setdefault(qi, []).append('<h2>'+html.escape(label)+'</h2>'+diagram(x, y, mask))
        print(f'Query {qi}, db {di}: inliers={result["inliers"]}/{len(x)}, null={null}', flush=True)
    paired = []
    for qi in sections:
        subset = [r for r in reports if r['query'] == qi]
        top = [r for r in subset if r['refined_rank'] == 1]
        positives = [r for r in subset if r['gt']]
        require(len(top) == 1 and positives, f'Expected top1 and GT for {qi}')
        # The reference GT is selected by the EXISTING model, never by geometry.
        best_gt = max(positives, key=lambda r: r['score_sum'])
        def compact(r):
            return {'db': r['db'], 'gt': r['gt'], 'score': r['score_sum'], 'mnn': r['mnn_count'],
                    'inliers': r['geometry']['inliers'], 'inlier_fraction': r['geometry']['inlier_fraction'],
                    'excess': r['excess_inliers_over_null_mean'],
                    'q_coverage': r['geometry']['q_coverage'], 'db_coverage': r['geometry']['db_coverage']}
        paired.append({'query': qi, 'group': subset[0]['group'], 'top1': compact(top[0]),
                       'best_gt_by_existing_score': compact(best_gt),
                       'all_gt': [compact(r) for r in positives],
                       'all_non_gt': [compact(r) for r in subset if not r['gt']]})
        (args.output / f'q_{qi:04d}.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px sans-serif;margin:24px}svg{width:800px;max-width:100%}</style>'
            + f'<h1>Query {qi}: affine consensus</h1><p>Green: fitted affine inliers; gray: other MNNs. No image semantics, GT verification, or attention explanation. Use the original image review alongside these coordinate diagrams.</p>'
            + ''.join(sections[qi]), encoding='utf8')
    (args.output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Descriptive geometry audit</h1>'
        + ''.join(f'<p><a href="q_{q:04d}.html">Query {q}</a></p>' for q in sections), encoding='utf8')
    for name, value in [('pairs.json', reports), ('summary.json', paired)]:
        (args.output / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')
    manifest = {'complete': True, 'pairs': len(reports), 'queries': len(sections),
                'grid': GRID, 'threshold_patches': THRESHOLD, 'trials': TRIALS,
                'null_repeats': NULL_REPEATS, 'seed': 42, 'numpy': np.__version__,
                'input_sha256': sha(args.input / 'results.json'),
                'input_completion_sha256': sha(args.input / 'completed.json'),
                'script_sha256': sha(Path(__file__)),
                'output_sha256': {n: sha(args.output / n) for n in ('pairs.json', 'summary.json')},
                'limitations': ['Single affine model is not valid for every 3D scene/viewpoint.',
                    'RANSAC is approximate; 5 shuffled fits are a descriptive null, not a significance test.',
                    'No GT-selected geometry thresholds, no reranking, no claimed recall gain.',
                    'Patch-center coordinates and globally contextualized tokens are not precise keypoints.',
                    'Input completion did not hash results.json; this run fingerprints the received file, not its transfer integrity.']}
    (args.output / 'completed.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print('Complete:', args.output.resolve())


if __name__ == '__main__':
    main()
