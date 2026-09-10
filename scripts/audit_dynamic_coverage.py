"""Read-only, CPU dynamic-coverage audit of an existing frozen-prior screen."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np

VARIANTS = ('baseline', 'zero_bias', 'aligned', 'shuffled', 'random')
GROUPS = ('zero', '(0,5%)', '[5%,15%)', '[15%,30%)', '[30%,100%]')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def group(value):
    require(np.isfinite(value) and 0 <= value <= 1, 'Invalid coverage')
    if value == 0: return GROUPS[0]
    if value < .05: return GROUPS[1]
    if value < .15: return GROUPS[2]
    if value < .30: return GROUPS[3]
    return GROUPS[4]


def paired(left, right, indices):
    indices = np.asarray(indices, dtype=int)
    l, r = np.asarray(left, dtype=bool)[indices], np.asarray(right, dtype=bool)[indices]
    return {'n': len(indices), 'left_correct': int(l.sum()), 'right_correct': int(r.sum()),
            'left_only_query_ids': indices[l & ~r].tolist(),
            'right_only_query_ids': indices[r & ~l].tolist(),
            'net': int(l.sum())-int(r.sum()),
            'delta_r1_pp': 100*(float(l.mean())-float(r.mean())) if len(l) else None}


def summarize(coverage, hits):
    labels = [group(v) for v in coverage]
    result = {}
    for label in ('all', *GROUPS):
        indices = list(range(len(labels))) if label == 'all' else [i for i, g in enumerate(labels) if g == label]
        result[label] = {'n': len(indices), 'small_group_below_30': len(indices) < 30,
            'query_ids': indices,
            'r1': {v: float(np.mean(np.asarray(hits[v])[indices])) if indices else None for v in VARIANTS},
            'vs_zero_bias': {v: paired(hits[v], hits['zero_bias'], indices) for v in VARIANTS},
            'aligned_vs_controls': {v: paired(hits['aligned'], hits[v], indices) for v in ('baseline', 'shuffled', 'random')}}
    return result


def sample_queries(paths, coverage, per_group):
    """Hash-order sampling depends only on image identity and coverage, never outcomes."""
    return {label: sorted([i for i, c in enumerate(coverage) if group(c) == label],
                         key=lambda i: hashlib.sha256(('42:'+paths[i]).encode()).hexdigest())[:per_group]
            for label in GROUPS}


def safe_image(root, relative):
    path = (root/relative).resolve()
    path.relative_to(root.resolve())
    require(path.is_file(), 'Image missing: '+str(path))
    return path


def save_preview(root, relative, mask, output):
    from PIL import Image
    with Image.open(safe_image(root, relative)) as im:
        rgb = im.convert('RGB').resize((400, 300), Image.Resampling.BICUBIC)
    heat = Image.fromarray(np.uint8(np.clip(mask, 0, 1)*255)).resize(rgb.size, Image.Resampling.NEAREST)
    pixels = np.asarray(rgb).astype(float)
    alpha = np.asarray(heat)[..., None]/255*.6
    overlay = Image.fromarray(np.uint8(pixels*(1-alpha)+np.array([255, 0, 0])*alpha))
    panel = Image.new('RGB', (800, 300))
    panel.paste(rgb, (0, 0)); panel.paste(overlay, (400, 0))
    panel.save(output, quality=88)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=Path('doc/dynamic_category_prior_screen_b0.5_full_db_condition_queries'))
    p.add_argument('--mask-cache', type=Path, default=Path('.cache/dynamic_prior/msls_val_full_db_condition_union_deeplabv3_mbv3_grid20.npz'))
    p.add_argument('--dataset-root', type=Path, default=Path('datasets/msls-val'))
    p.add_argument('--output', type=Path, default=Path('doc/dynamic_coverage_audit_v1'))
    p.add_argument('--samples-per-group', type=int, default=4)
    a = p.parse_args()
    require(not a.output.exists(), 'Choose a new output directory; never overwrite reports')
    require(0 <= a.samples_per_group <= 10, 'samples-per-group must be 0..10')
    require((a.run/'query_outcomes.csv').is_file(), 'Missing query_outcomes.csv: locate the original screen output on the training machine; aggregate recall cannot reconstruct individual outcomes. No extraction was started.')
    run = json.loads((a.run/'run.json').read_text(encoding='utf8'))
    require(run['schema_version'] == 3 and run['method'] == 'frozen_dynamic_category_negative_attention_prior', 'Need corrected full-db screen v3')
    require(sha(a.mask_cache) == run['mask_cache']['sha256'], 'Mask cache differs from original intervention')
    with np.load(a.mask_cache, allow_pickle=False) as z:
        paths = [str(v).replace('\\', '/') for v in z['image_paths']]
        masks = np.asarray(z['masks'], dtype=np.float32)
        ndb = int(z['num_references'])
    require(ndb == 18871 and len(paths) == len(set(paths)), 'Invalid cache identity')
    require(masks.shape == (len(paths), 20, 20) and np.isfinite(masks).all() and (masks >= 0).all() and (masks <= 1).all(), 'Invalid patch-fraction masks')
    db = [str(v).replace('\\', '/') for v in np.load(a.dataset_root/'msls_val_dbImages.npy', allow_pickle=False)]
    require(paths[:ndb] == db, 'Database path/order mismatch')
    lookup = {s: i for i, s in enumerate(paths)}
    cover = masks.mean((1, 2), dtype=np.float64)
    outcomes = read_csv(a.run/'query_outcomes.csv')
    aggregates = read_csv(a.run/'summary.csv')
    outcome_map = {}
    for row in outcomes:
        key = (row['dataset'], row['variant'], int(row['query_index']))
        require(key not in outcome_map, 'Duplicate outcome '+str(key))
        require(row['variant'] in VARIANTS and row['top1_correct'] in ('0', '1'), 'Invalid outcome')
        outcome_map[key] = row
    all_results, reviews, manifest_hashes = {}, [], {}
    expected_rows = 0
    for split in run['datasets']:
        name = split['name']; n = split['num_queries']; expected_rows += n*len(VARIANTS)
        files = {}
        for role, info in split['manifests'].items():
            path = a.dataset_root/Path(info['path']).name
            manifest_hashes[str(path)] = sha(path)
            require(manifest_hashes[str(path)] == info['sha256'], 'Manifest changed: '+str(path))
            files[role] = path
        qpaths = [str(v).replace('\\', '/') for v in np.load(files['queries'], allow_pickle=False)]
        gt = np.load(files['ground_truth'], allow_pickle=True)  # trusted local dataset GT only
        require(len(qpaths) == n == len(gt) and len(set(qpaths)) == n, 'Split length/duplicate query mismatch')
        require(all(s in lookup and lookup[s] >= ndb for s in qpaths), 'Query missing from union mask cache')
        gt_ids = []
        for g in gt:
            ids = np.asarray(g).reshape(-1)
            require(len(ids) > 0 and np.issubdtype(ids.dtype, np.integer) and (ids >= 0).all() and (ids < ndb).all(), 'Invalid ground truth')
            gt_ids.append(ids)
        hits, predictions = {}, {}
        for v in VARIANTS:
            h, pred = [], []
            for q in range(n):
                row = outcome_map[(name, v, q)]
                i = int(row['top1_reference_index'])
                require(0 <= i < ndb and row['query_path'].replace('\\', '/') == qpaths[q] and row['top1_reference_path'].replace('\\', '/') == db[i], 'Outcome path/index mismatch')
                correct = int(i in gt_ids[q])
                require(correct == int(row['top1_correct']), 'GT does not reproduce saved hit')
                h.append(correct); pred.append(i)
            expected = [r for r in aggregates if r['dataset'] == name and r['variant'] == v]
            require(len(expected) == 1 and int(expected[0]['num_queries']) == n and abs(np.mean(h)-float(expected[0]['r@1'])) < 1e-10, 'Aggregate recall not reproduced')
            hits[v], predictions[v] = h, pred
        coverage = [float(cover[lookup[s]]) for s in qpaths]
        reference = [int(np.min(g)) for g in gt_ids]  # fixed GT index, NOT best scoring/image
        all_results[name] = {'protocol': split['protocol'], 'query_coverage_strata': summarize(coverage, hits),
            'per_query': [dict(query_index=q, query_path=s, coverage=coverage[q], group=group(coverage[q]),
                hits={v: hits[v][q] for v in VARIANTS}, top1_reference_indices={v: predictions[v][q] for v in VARIANTS},
                fixed_gt_reference_index=reference[q], fixed_gt_coverage=float(cover[reference[q]]),
                absolute_coverage_difference_to_fixed_gt=abs(coverage[q]-float(cover[reference[q]]))) for q, s in enumerate(qpaths)]}
        for label, chosen in sample_queries(qpaths, coverage, a.samples_per_group).items():
            for q in chosen:
                reviews.append({'dataset': name, 'query_index': q, 'group': label,
                    'images': [('query', lookup[qpaths[q]]), ('fixed_GT_min_index', reference[q]), ('baseline_top1', predictions['baseline'][q])],
                    'manual_review': {'missed_dynamic_objects': None, 'static_regions_mislabelled': None,
                        'boundary_mixing': None, 'visible_overlap': None, 'apparent_dynamic_change': None,
                        'notes': ''}})
    require(len(outcome_map) == expected_rows, 'Unexpected/missing outcome rows')
    a.output.mkdir(parents=True)
    (a.output/'images').mkdir()
    rendered = {}
    pages = ['<!doctype html><meta charset="utf-8"><title>Dynamic coverage audit</title>',
        '<h1>动态掩码固定抽样核验</h1><p>每图左侧原图，右侧20×20面积比例红色叠加。不是原生像素分割；红色越深代表动态覆盖越高。</p>',
        '<p>按覆盖率和路径哈希固定选样，不按检索正误选样。GT固定选最小数据库索引；同地点GT不保证可见重叠或跨时车辆变化。请填写review_samples.json的人工核验项。</p>']
    for review in reviews:
        pages.append('<h2>'+html.escape(f"{review['dataset']} query {review['query_index']} / {review['group']}")+'</h2>')
        for role, idx in review['images']:
            if idx not in rendered:
                filename = f'images/{idx:06d}.jpg'
                save_preview(a.dataset_root, paths[idx], masks[idx], a.output/filename)
                rendered[idx] = filename
            pages.append('<p>'+html.escape(f'{role}: {paths[idx]} coverage={cover[idx]:.2%}')+'</p><img width="800" src="'+rendered[idx]+'">')
        review['images'] = [{'role': role, 'cache_index': idx, 'path': paths[idx], 'coverage': float(cover[idx])} for role, idx in review['images']]
    scope = ('Post-hoc descriptive audit; fixed query coverage bins, no parameter tuning or significance claims. '
             'Conditions overlap standard queries and are not independent datasets. Coverage is not motion, '
             'occlusion severity, actual temporal change, or a causal attention attribution. Fixed-GT coverage '
             'difference is only a proxy, not an aligned object-change measurement. Independent benchmark '
             'and controlled dynamic-change intervention are NOT completed by this report.')
    compact = {name: {'protocol': value['protocol'], 'query_coverage_strata': value['query_coverage_strata']} for name, value in all_results.items()}
    input_hashes = {str(path): sha(path) for path in (a.run/'run.json', a.run/'summary.csv', a.run/'query_outcomes.csv', a.mask_cache, a.dataset_root/'msls_val_dbImages.npy')}
    for filename, value in (('summary.json', {'scope': scope, 'datasets': compact}), ('per_query.json', all_results),
        ('review_samples.json', reviews), ('provenance.json', {'input_sha256': input_hashes, 'manifest_sha256': manifest_hashes,
            'script_sha256': sha(__file__), 'bins': GROUPS, 'sampling': 'sha256(42:path), independent of outcomes',
            'samples_per_group': a.samples_per_group})):
        (a.output/filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    (a.output/'index.html').write_text('\n'.join(pages), encoding='utf8')
    (a.output/'completed.json').write_text(json.dumps({'complete': True}), encoding='utf8')
    for name, value in compact.items():
        print(name)
        for label, result in value['query_coverage_strata'].items():
            r = result['vs_zero_bias']['aligned']
            print(f"  {label:15s} n={r['n']:4d} aligned/zero={r['left_correct']}/{r['right_correct']} "
                  f"corrections={len(r['left_only_query_ids'])} regressions={len(r['right_only_query_ids'])} net={r['net']:+d}")
    print('Saved:', a.output)


if __name__ == '__main__':
    main()
