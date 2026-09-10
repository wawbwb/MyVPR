#!/usr/bin/env python
"""Pinned 17Places AnyLoc/SegVLAD paired audit using trusted official artifacts."""
import argparse
import json
import os
from pathlib import Path
import pickle
import sys

if __package__:
    from .segvlad_official import COMMIT, git, literal_config, sha, vocabulary_assets
else:
    from segvlad_official import COMMIT, git, literal_config, sha, vocabulary_assets


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def recall_counts(predictions, gt, k=5):
    if len(predictions) != len(gt) or not gt:
        raise ValueError('Predictions/GT length mismatch or empty GT')
    if any(not row for row in gt):
        raise ValueError('Empty GT row: denominator must be explicit')
    return [sum(bool(set(p[:i]) & set(g)) for p, g in zip(predictions, gt))
            for i in range(1, k + 1)]


def compare_predictions(baseline, variant, gt):
    recall_counts(baseline, gt)
    recall_counts(variant, gt)
    a = [bool(set(p[:1]) & set(g)) for p, g in zip(baseline, gt)]
    b = [bool(set(p[:1]) & set(g)) for p, g in zip(variant, gt)]
    corrections = [i for i, (x, y) in enumerate(zip(a, b)) if not x and y]
    regressions = [i for i, (x, y) in enumerate(zip(a, b)) if x and not y]
    return {'correction_query_ids': corrections, 'regression_query_ids': regressions,
            'corrections': len(corrections), 'regressions': len(regressions),
            'net': len(corrections) - len(regressions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--seg-results', type=Path, required=True,
                        help='Existing official map/17places native result directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    repo, data, seg, out = (p.resolve() for p in
                            (args.repo, args.data_root, args.seg_results, args.output))
    if out.exists():
        parser.error('Output exists; select a new directory (no overwrites)')
    if git(repo, 'rev-parse', 'HEAD') != COMMIT or git(repo, 'status', '--porcelain', '--untracked-files=no'):
        parser.error('Expected clean pinned upstream checkout')
    cfg, exp = literal_config(repo)
    base = data / '17places'
    pca, centers, _ = vocabulary_assets(repo, base, cfg, exp, 'map')
    candidates = list(seg.glob('17places_matches_sims_domain_17places__*.pkl'))
    if len(candidates) != 1:
        parser.error('Expected exactly one map/17places matches_sims pickle')
    match_file = candidates[0]
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ.setdefault('WANDB_MODE', 'disabled')
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    # Set upstream argv before imports: some dependencies parse command-line options.
    sys.argv = [str(repo / 'place_rec_main.py')]
    import numpy as np
    import h5py
    import torch
    from natsort import natsorted
    from tqdm import tqdm
    import func_vpr
    from gt import get_gt
    from utilities import VLAD

    out.mkdir(parents=True)
    names = [natsorted(os.listdir(base / cfg[key])) for key in
             ('data_subpath1_r', 'data_subpath2_q')]
    if [len(x) for x in names] != [406, 406]:
        raise ValueError('Expected full 17Places: 406 references and 406 queries')
    assets = [pca, centers, match_file]
    orders = {}
    for side, images in zip(('r', 'q'), names):
        for kind in ('dino', 'masks'):
            path = base / 'out' / cfg[f'{kind}_h5_filename_{side}']
            assets.append(path)
            with h5py.File(path, 'r') as handle:
                keys = natsorted(handle.keys())
                orders[f'{kind}_{side}'] = keys == images
    gt = get_gt(dataset='17places', cfg=cfg['cfg'], workdir_data=str(data),
                ims1_r=names[0], ims2_q=names[1], func_vpr_module=func_vpr)
    gt = [[int(x) for x in row] for row in gt]
    expected_gt = [list(range(i - 15, i + 16)) for i in range(406)]
    audit = {'upstream_commit': COMMIT, 'references': names[0], 'queries': names[1],
             'cache_order_matches_images': orders, 'official_gt': gt,
             'official_gt_is_index_window_15': gt == expected_gt,
             'gt_note': 'Official index-window GT is retained, including out-of-range edge IDs; '
                        'packaged NPY annotations are not substituted or certified equivalent.',
             'source_sha256': {n: sha(repo / n) for n in ('func_vpr.py', 'gt.py',
                                'utilities.py', 'place_rec_main.py', 'place_rec_global_config.py')},
             'packaged_annotations': {}}
    for name in ('ground_truth_new.npy', 'my_ground_truth_new.npy'):
        path = base / name
        if path.exists():
            # Inspect metadata only; object payload is not needed for official scoring.
            with path.open('rb') as f:
                version = np.lib.format.read_magic(f)
                reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                shape, _, dtype = reader(f)
            audit['packaged_annotations'][name] = {'shape': list(shape), 'dtype': str(dtype),
                                                   'sha256': sha(path)}
    readme = base / 'ReadMe.txt'
    if readme.exists():
        audit['dataset_readme'] = readme.read_text(errors='replace')
    write_json(out / 'protocol_audit.json', audit)
    if not all(orders.values()) or gt != expected_gt:
        raise ValueError('Order/GT contract failed; inspect protocol_audit.json')

    # Reconstruct segment-to-image indices with exactly the official mask loader.
    ids, ranges = [], []
    for side, images in zip(('r', 'q'), names):
        per_image = []
        with h5py.File(base / 'out' / cfg[f'masks_h5_filename_{side}'], 'r') as handle:
            for i, name in enumerate(tqdm(images, desc=f'Audit {side} segment indices')):
                masks = func_vpr.preload_masks(handle, name)
                image_ids, _, _ = func_vpr.getIdxSingleFast(i, masks, minArea=exp['minArea'])
                if not len(image_ids):
                    raise ValueError(f'Empty official segment set: {side}/{name}')
                per_image.append(image_ids)
        flat = np.concatenate(per_image).astype(int)
        ids.append(flat)
        ranges.append([np.where(flat == i)[0] for i in range(len(images))])
    # Trusted author/local output pickle only; never use untrusted downloaded pickle.
    with match_file.open('rb') as f:
        saved = pickle.load(f)
    matches, distances = saved['matches'], saved['sims']
    if matches.shape != distances.shape or matches.shape != (len(ids[1]), 200):
        raise ValueError('Saved segment result shape mismatch')
    if not np.isfinite(distances).all() or matches.min() < 0 or matches.max() >= len(ids[0]):
        raise ValueError('Invalid saved distances/segment indices')
    pred_seg = func_vpr.get_matches(matches[:, :50], gt, 2 - distances[:, :50],
                                    ranges[1], ids[0], n=5, method='max_seg_topk_wt_borda_Im')
    pred_seg = [[int(x) for x in row] for row in pred_seg]
    counts_seg = recall_counts(pred_seg, gt)
    write_json(out / 'segvlad_reconstruction.json', {'counts': counts_seg,
                'expected_counts': [387, 394, 395, 397, 399], 'predictions': pred_seg})
    if counts_seg != [387, 394, 395, 397, 399]:
        raise ValueError('Existing SegVLAD recall not reproduced; stop before baseline')
    print('PASS: order, official GT rule, and SegVLAD 387/406 reconstruction', flush=True)
    if args.audit_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no baseline started')
    write_json(out / 'provenance.json', {'asset_sha256': {str(p): sha(p) for p in assets},
               'sklearn': __import__('sklearn').__version__, 'torch': torch.__version__,
               'numpy': np.__version__, 'vocabulary': 'map/17places'})
    vlad = VLAD(32, desc_dim=None, cache_dir=str(centers.parent))
    vlad.fit(None)
    features = []
    for side in ('r', 'q'):
        print(f'Official AnyLoc full-image VLAD: {side}', flush=True)
        features.append(func_vpr.aggFt(str(base / 'out' / cfg[f'dino_h5_filename_{side}']),
                                      None, None, cfg['cfg'], 'vlad', vlad, upsample=True))
    # Use the exact upstream normalization and KDTree retrieval, not a substitute scorer.
    native_recalls, match_info = func_vpr.get_recall(func_vpr.normalizeFeat(features[0]),
                                                  func_vpr.normalizeFeat(features[1]), gt, k=5)
    pred_base = [[int(x) for x in row['img_id_r']] for row in match_info]
    counts_base = recall_counts(pred_base, gt)
    if not np.allclose(np.asarray(native_recalls), np.asarray(counts_base) / 406 * 100):
        raise ValueError('AnyLoc recall reconstruction mismatch')
    np.savez_compressed(out / 'anyloc_descriptors.npz', reference=features[0], query=features[1])
    paired = compare_predictions(pred_base, pred_seg, gt)
    summary = {'queries': 406, 'anyloc_correct_at_1_to_5': counts_base,
               'segvlad_correct_at_1_to_5': counts_seg, **paired,
               'delta_r1_pp': paired['net'] / 406 * 100,
               'scope': 'Paired official method comparison, NOT pure semantic-mask ablation: '
                        'AnyLoc is full-image/no PCA; SegVLAD uses segments, neighbor aggregation, '
                        'PCA1024 and voting. Not an MSLS comparison.'}
    write_json(out / 'per_query.json', [{'query_id': i, 'query': name, 'gt': gt[i],
               'anyloc_top5': pred_base[i], 'segvlad_top5': pred_seg[i]}
               for i, name in enumerate(names[1])])
    write_json(out / 'summary.json', summary)
    write_json(out / 'completed.json', {'complete': True})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
