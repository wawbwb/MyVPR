#!/usr/bin/env python
"""Frozen RU + SAM/SLIC/grid region VLAD, GSV-only vocabulary/PCA fitting.

No model training; MSLS is exploratory evaluation only. See doc/REGION_VLAD_SCREEN.md.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from src.region_vlad import (grid_masks, select_masks, neighbour_union, patch_membership,
                             region_vlad, image_vote, equal_budget_union)

MODES = ('sam', 'slic', 'grid', 'shifted_sam')
INDEX_FILES = ('msls_val_dbImages.npy', 'msls_val_qImages.npy', 'msls_val_gt_25m.npy')
LEGACY_SCRIPT_SHA = '64a8e8a98e34c0829448d0014a8b3ba22462021dbd0d391bd9483b046dbd71be'


def compatible_contract(old, new):
    """Only whitelist fa3de35's stop-on-empty implementation; never waive other checks."""
    if old == new:
        return True
    candidate = json.loads(json.dumps(old))
    impl = candidate.get('implementation', {})
    if impl.get('scripts/region_vlad_screen.py') != LEGACY_SCRIPT_SHA:
        return False
    if 'empty_region_policy' in candidate:
        return False
    impl['scripts/region_vlad_screen.py'] = new['implementation']['scripts/region_vlad_screen.py']
    candidate['empty_region_policy'] = new['empty_region_policy']
    return candidate == new


def empty_regions(raw_count):
    memberships = {m: np.zeros((0, 400), dtype=bool) for m in MODES}
    stats = {m: {'count': 0, 'raw_count': raw_count if m == 'sam' else 0,
                 'base_area_mean': 0., 'super_area_mean': 0., 'token_area_mean': 0.,
                 'unique_super_masks': 0, 'delaunay_identity_fallback': False} for m in MODES}
    return memberships, stats


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    os.replace(tmp, path)


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def implementation():
    return {p: sha(ROOT/p) for p in ('scripts/region_vlad_screen.py', 'src/region_vlad.py',
                                   'src/cc_lsa_features.py', 'scripts/eval_condition_robustness.py',
                                   'src/models/backbones/dinov2.py', 'src/models/aggregators/boq.py',
                                   'src/models/semantic_region_gate.py')}


def versions():
    result = {}
    for name in ('torch', 'numpy', 'scipy', 'scikit-learn', 'scikit-image', 'segment-anything'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = 'not-installed'
    return result


def contract(args):
    return {'schema': 1, 'feature': 'RU trained DINO final pre-gate normalized 20x20 tokens',
            'checkpoint_sha256': sha(args.checkpoint), 'sam_sha256': sha(args.sam_checkpoint),
            'sam_type': args.sam_type, 'image_size': 280, 'max_regions': args.max_regions,
            'neighbour_order': args.neighbour_order, 'seed': args.seed,
            'sam_points_per_side': 16, 'sam_pred_iou_thresh': 0.88,
            'sam_stability_score_thresh': 0.95, 'sam_crop_n_layers': 0,
            'slic_compactness': 10, 'minimum_mask_pixels': 196,
            'modes': list(MODES), 'implementation': implementation(), 'versions': versions(),
            'empty_region_policy': 'all_four_zero_regions_keep_ru_query_abstains_v1'}


class Extractor:
    def __init__(self, args):
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        from scripts.eval_condition_robustness import load_inference_model_from_ckpt, build_transform
        self.device = torch.device(args.device)
        self.ru = load_inference_model_from_ckpt(args.checkpoint, self.device).eval()
        sam = sam_model_registry[args.sam_type](checkpoint=str(args.sam_checkpoint)).to(self.device).eval()
        self.sam = SamAutomaticMaskGenerator(sam, points_per_side=16, points_per_batch=32,
                                            pred_iou_thresh=0.88, stability_score_thresh=0.95,
                                            crop_n_layers=0, min_mask_region_area=0)
        self.transform = build_transform((280, 280))
        self.args = args
        self.checked = False

    @torch.inference_mode()
    def __call__(self, path):
        from skimage.segmentation import slic
        from src.cc_lsa_features import extract_ru_descriptor_and_local
        with Image.open(path) as im:
            im = im.convert('RGB')
            # Tensor resize exactly matches RU evaluation; SAM geometry is same 280 square.
            tensor = self.transform(im).unsqueeze(0).to(self.device)
            rgb = np.asarray(im.resize((280, 280), Image.Resampling.BICUBIC))
        descriptor, tokens = extract_ru_descriptor_and_local(self.ru, tensor, output_grid=(20, 20))
        if not self.checked:
            actual = self.ru(tensor)
            if isinstance(actual, (list, tuple)):
                actual = actual[0]
            actual = torch.nn.functional.normalize(actual.float(), dim=-1)
            require(torch.allclose(actual, descriptor, atol=2e-5, rtol=2e-4), 'RU extraction parity failed')
            self.checked = True
        raw = self.sam.generate(rgb)
        sam_masks = [r['segmentation'] for r in raw if r['area'] >= 196]
        if not sam_masks:
            memberships, stats = empty_regions(len(raw))
            tqdm.write(f'NO_REGION: {path}; SAM raw={len(raw)}, eligible=0; keep RU, all four region branches abstain')
            return tokens[0].cpu().numpy(), descriptor[0].cpu().numpy(), memberships, stats
        lab = slic(rgb, n_segments=self.args.max_regions, compactness=10, start_label=0)
        slic_masks = [lab == i for i in np.unique(lab)]
        count = min(self.args.max_regions, len(sam_masks), len(slic_masks))
        masks = {'sam': select_masks(sam_masks, count), 'slic': select_masks(slic_masks, count),
                 'grid': grid_masks(count)}
        masks['shifted_sam'] = np.roll(masks['sam'], shift=(140, 140), axis=(1, 2))
        memberships, stats = {}, {}
        for mode in MODES:
            merged, fallback = neighbour_union(masks[mode], self.args.neighbour_order)
            # Shift the already formed SuperSegments, preserving exact area/overlap.
            if mode == 'sam':
                sam_merged, sam_fallback = merged, fallback
            if mode == 'shifted_sam':
                merged, fallback = np.roll(sam_merged, (140, 140), (1, 2)), sam_fallback
            memberships[mode] = patch_membership(merged)
            stats[mode] = {'count': count, 'raw_count': len(raw) if mode == 'sam' else len(masks[mode]),
                           'base_area_mean': float(masks[mode].mean()),
                           'super_area_mean': float(merged.mean()),
                           'token_area_mean': float(memberships[mode].mean()),
                           'unique_super_masks': len(np.unique(memberships[mode], axis=0)),
                           'delaunay_identity_fallback': fallback}
        return tokens[0].cpu().numpy(), descriptor[0].cpu().numpy(), memberships, stats


def safe_image(root, relative):
    p = (root / str(relative)).resolve()
    require(p.is_relative_to(root.resolve()) and p.is_file(), f'Missing/unsafe image: {relative}')
    return p


def fit(args):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA
    from src.dataloaders.train.cc_lsa import discover_gsv_place_views
    require(not args.output.exists(), 'Fit output exists; choose a new directory')
    require(args.train_images >= 32, 'Fit needs at least 32 GSV images')
    c = contract(args)
    print('Discovering GSV train places and checking image paths...', flush=True)
    records, source = discover_gsv_place_views(args.dataset_root, seed=args.seed)
    records = [r for r in records if r.split == 0]
    rng = np.random.default_rng(args.seed)
    require(len(records) >= args.train_images, 'Insufficient train places')
    chosen = [records[i] for i in rng.choice(len(records), args.train_images, replace=False)]
    paths = [r.relative_path for r in chosen]
    extractor = Extractor(args)
    samples, masks = [], []
    for p in tqdm(paths, desc='Fit: GSV train-only masks/features', dynamic_ncols=True):
        x, _, m, _ = extractor(safe_image(args.dataset_root, p))
        require(len(m['sam']) > 0, f'No fit regions: {p}; do not silently change PCA training sample')
        samples.append(x)
        masks.append(m)
    x = np.concatenate(samples)
    print('Fit shared 32-cluster vocabulary', flush=True)
    km = MiniBatchKMeans(n_clusters=32, batch_size=4096, n_init=3, random_state=args.seed)
    km.fit(x)
    centers = torch.tensor(km.cluster_centers_, device=args.device)
    blocks = {mode: [] for mode in MODES}
    for token, m in tqdm(zip(samples, masks), total=len(samples), desc='Fit: region VLAD', dynamic_ncols=True):
        token = torch.tensor(token, device=args.device)
        for mode in MODES:
            blocks[mode].append(region_vlad(token, centers, torch.tensor(m[mode], device=args.device)).cpu().numpy())
    # Balance representation families before PCA. Never fit on validation images.
    balanced = []
    per_mode = min(args.pca_samples//len(MODES), min(sum(len(b) for b in blocks[m]) for m in MODES))
    require(per_mode*len(MODES) > args.pca_dim, 'Insufficient PCA samples')
    for mode in MODES:
        b = np.concatenate(blocks[mode])
        balanced.append(b[rng.choice(len(b), per_mode, replace=False)])
    print(f'Fit shared PCA: {per_mode*len(MODES)} rows -> {args.pca_dim} dims (CPU)', flush=True)
    pca = PCA(n_components=args.pca_dim, svd_solver='randomized', whiten=False, random_state=args.seed)
    pca.fit(np.concatenate(balanced))
    args.output.mkdir(parents=True)
    np.savez(args.output/'model.npz', centers=km.cluster_centers_.astype('float32'),
             mean=pca.mean_.astype('float32'), components=pca.components_.astype('float32'))
    save(args.output/'manifest.json', {'complete': True, 'contract': c, 'gsv_csv': source,
                                     'train_paths': paths, 'train_images': len(paths),
                                     'pca_samples': per_mode*len(MODES), 'pca_dim': args.pca_dim,
                                     'pca_explained_variance': float(pca.explained_variance_ratio_.sum()),
                                     'model_sha256': sha(args.output/'model.npz')})
    print(f'Fit complete: {args.output}', flush=True)


def cache(args):
    manifest = read(args.model/'manifest.json')
    require(manifest['complete'], 'Incomplete fitted model')
    require(sha(args.model/'model.npz') == manifest['model_sha256'], 'Fitted model changed')
    c = contract(args)
    require(compatible_contract(manifest['contract'], c), 'Feature/mask/code/environment contract differs from fit')
    db = np.load(args.dataset_root/INDEX_FILES[0]).astype(str).tolist()
    queries = np.load(args.dataset_root/INDEX_FILES[1]).astype(str).tolist()
    paths = db+queries
    require(len(db) == 18871 and len(queries) == 740 and len(set(paths)) == len(paths), 'Unexpected MSLS split')
    if args.limit_images:
        paths = paths[:args.limit_images]
    expected = {'contract': c, 'model_sha256': manifest['model_sha256'],
                'index_sha256': {n: sha(args.dataset_root/n) for n in INDEX_FILES},
                'paths': paths, 'references': len(db), 'queries': len(queries),
                'partial': len(paths) != len(db)+len(queries)}
    if args.output.exists():
        require(args.resume, 'Cache exists; use --resume with identical settings')
        previous = read(args.output/'contract.json')
        checked = dict(previous)
        require(compatible_contract(previous['contract'], c), 'Cache implementation contract mismatch')
        checked['contract'] = c
        require(checked == expected, 'Cache resume contract mismatch')
        if previous != expected:
            migration = args.output/'empty_region_migration.json'
            if not migration.exists():
                save(migration, {'original_contract': previous, 'new_contract': expected,
                                 'reason': 'fa3de35 valid nonempty shards unchanged; empty samples now abstain'})
            save(args.output/'contract.json', expected)
            print('Migrated known fa3de35 cache; existing nonempty shards retained.', flush=True)
    else:
        args.output.mkdir(parents=True)
        save(args.output/'contract.json', expected)
    (args.output/'shards').mkdir(exist_ok=True)
    params = np.load(args.model/'model.npz')
    centers, mean, components = [torch.tensor(params[k], device=args.device) for k in ('centers', 'mean', 'components')]
    extractor = None
    shard_hashes = []
    for i, p in enumerate(tqdm(paths, desc='Cache RU + four matched region variants', dynamic_ncols=True)):
        target = args.output/'shards'/f'{i:06d}.npz'
        if target.exists():
            with np.load(target) as z:
                require(str(z['path']) == p, 'Shard path mismatch')
                require(all(np.isfinite(z[m]).all() and z[m].shape[1] == len(components) for m in MODES), 'Corrupt shard')
                require(np.isfinite(z['ru']).all(), 'Nonfinite cached RU descriptor')
        else:
            if extractor is None:
                extractor = Extractor(args)
            start = time.monotonic()
            tokens, ru, masks, stats = extractor(safe_image(args.dataset_root, p))
            tokens = torch.tensor(tokens, device=args.device)
            result = {}
            for mode in MODES:
                if len(masks[mode]) == 0:
                    result[mode] = np.empty((0, len(components)), dtype=np.float32)
                    continue
                raw = region_vlad(tokens, centers, torch.tensor(masks[mode], device=args.device))
                projected = (raw-mean) @ components.T
                require(torch.isfinite(projected).all() and (projected.norm(dim=1) > 1e-8).all(), 'Invalid PCA output')
                result[mode] = torch.nn.functional.normalize(projected, dim=1).cpu().numpy()
            tmp = target.with_suffix('.tmp')
            with tmp.open('wb') as f:
                np.savez_compressed(f, path=p, ru=ru, stats=json.dumps(stats),
                                    seconds=time.monotonic()-start, **result)
            os.replace(tmp, target)
        shard_hashes.append(sha(target))
    summaries = {mode: [] for mode in MODES}
    for i in range(len(paths)):
        with np.load(args.output/'shards'/f'{i:06d}.npz') as z:
            info = json.loads(str(z['stats']))
            for mode in MODES:
                summaries[mode].append(info[mode])
    audit = {mode: {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
             for mode, rows in summaries.items()}
    save(args.output/'mask_audit.json', audit)
    empty_ids = [i for i, row in enumerate(summaries['sam']) if row['count'] == 0]
    save(args.output/'empty_regions.json', {'count': len(empty_ids),
                                          'images': [{'index': i, 'path': paths[i]} for i in empty_ids],
                                          'policy': c['empty_region_policy']})
    save(args.output/'completed.json', {'complete': True, 'partial': expected['partial'],
                                      'contract_sha256': sha(args.output/'contract.json'),
                                      'shard_sha256': shard_hashes})
    print(json.dumps(audit, indent=2), flush=True)
    print('Cache complete; partial=', expected['partial'], flush=True)


def measure(predictions, gt, baseline):
    hit = np.array([np.isin(row[:1], p).any() for row, p in zip(predictions, gt)])
    result = {'correct': int(hit.sum()), 'r1': float(hit.mean()),
              'corrections': np.flatnonzero(hit & ~baseline).tolist(),
              'regressions': np.flatnonzero(~hit & baseline).tolist()}
    for k in (5, 10, 20):
        result[f'r{k}'] = float(np.mean([np.isin(row[:k], p).any() for row, p in zip(predictions, gt)]))
    result['net'] = int(hit.sum()-baseline.sum())
    return result


def evaluate(args):
    require(not args.output.exists(), 'Evaluation output exists; choose a new directory')
    done = read(args.cache/'completed.json')
    c = read(args.cache/'contract.json')
    require(done['complete'] and not done['partial'], 'Only full completed cache can be evaluated')
    require(sha(args.cache/'contract.json') == done['contract_sha256'], 'Cache contract changed')
    require(c['contract']['implementation'] == implementation(), 'Implementation changed since cache creation')
    for n, digest in c['index_sha256'].items():
        require(sha(args.dataset_root/n) == digest, 'MSLS index/GT differs')
    ndb, nq = c['references'], c['queries']
    shards = [args.cache/'shards'/f'{i:06d}.npz' for i in range(ndb+nq)]
    require(len(done['shard_sha256']) == len(shards), 'Incomplete shard list')
    descriptors = []
    stats = {m: [] for m in MODES}
    extraction_seconds = 0.
    for p, digest in tqdm(zip(shards, done['shard_sha256']), total=len(shards), desc='Verify cache'):
        require(sha(p) == digest, f'Corrupt shard: {p}')
        with np.load(p) as z:
            descriptors.append(z['ru'])
            info = json.loads(str(z['stats']))
            extraction_seconds += float(z['seconds'])
            for m in MODES:
                stats[m].append(info[m])
    ru = torch.tensor(np.stack(descriptors), device=args.device)
    predictions = {'ru': torch.cat([(q @ ru[:ndb].T).topk(20, dim=1).indices.cpu()
                                   for q in ru[ndb:].split(32)]).numpy()}
    del ru
    # Predictions are computed without GT. GT is used below for reporting only.
    runtime, region_bytes = {}, {}
    for mode in MODES:
        start = time.monotonic()
        refs, query_regions, owners = [], [], []
        for i, p in enumerate(shards):
            with np.load(p) as z:
                v = z[mode].copy()
            if i < ndb:
                refs.append(v)
                owners.extend([i]*len(v))
            else:
                query_regions.append(v)
        reference = torch.tensor(np.concatenate(refs), device=args.device)
        require(len(reference) > 0, 'No database regions at all; stop evaluation')
        region_bytes[mode] = reference.numel()*reference.element_size()
        owners = np.asarray(owners)
        nearest, similarities = [], []
        for q in tqdm(query_regions, desc=f'Full database region search: {mode}', dynamic_ncols=True):
            score, idx = (torch.tensor(q, device=args.device) @ reference.T).topk(min(args.region_top_k, len(reference)), dim=1)
            nearest.append(idx.cpu().numpy())
            similarities.append(score.cpu().numpy())
        nonempty = [v for v in similarities if v.size]
        require(len(nonempty) > 0, 'No query regions at all; stop evaluation')
        low, high = min(float(v.min()) for v in nonempty), max(float(v.max()) for v in nonempty)
        predictions[mode] = np.stack([image_vote(i, s, owners, low, high) if s.size
                                     else np.full(20, -1, dtype=np.int64)
                                     for i, s in zip(nearest, similarities)])
        runtime[mode] = time.monotonic()-start
        del reference
    gt = np.load(args.dataset_root/INDEX_FILES[2], allow_pickle=True)
    require(len(gt) == nq and all(len(p) and np.all((np.asarray(p) >= 0) & (np.asarray(p) < ndb)) for p in gt), 'Invalid GT')
    baseline = np.array([row[0] in p for row, p in zip(predictions['ru'], gt)])
    require(int(baseline.sum()) == 675, f'RU not reproduced: {baseline.sum()}/740; stop comparison')
    measurements = {m: measure(v, gt, baseline) for m, v in predictions.items()}
    unions = {}
    ru20 = np.array([np.isin(row, p).any() for row, p in zip(predictions['ru'], gt)])
    for mode in MODES:
        ids = np.stack([equal_budget_union(u, r) for u, r in zip(predictions['ru'], predictions[mode])])
        predictions[mode+'_union20'] = ids
        hit = np.array([np.isin(row, p).any() for row, p in zip(ids, gt)])
        unions[mode] = {'oracle_count': int(hit.sum()), 'newly_reachable': np.flatnonzero(hit & ~ru20).tolist(),
                        'lost_reachable': np.flatnonzero(~hit & ru20).tolist(),
                        'net_oracle': int(hit.sum()-ru20.sum())}
    mask_summary = {}
    for m, rows in stats.items():
        mask_summary[m] = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
    report = {'complete': True, 'exploratory': True, 'queries': nq, 'references': ndb,
              'actual_retrieval': measurements, 'equal_budget_union20_ORACLE_ONLY': unions,
              'mask_statistics_mean': mask_summary, 'search_including_shard_load_seconds': runtime,
              'database_region_tensor_bytes': region_bytes, 'total_extraction_seconds': extraction_seconds,
              'cache_contract_sha256': done['contract_sha256'], 'region_top_k': args.region_top_k,
              'empty_regions': {'database_ids': [i for i, r in enumerate(stats['sam'][:ndb]) if r['count'] == 0],
                                'query_ids': [i for i, r in enumerate(stats['sam'][ndb:]) if r['count'] == 0],
                                'policy': 'All four regional branches abstain. RU retained. All 740 queries remain in denominator; no hidden RU fallback in regional R1.'},
              'warning': 'SAM is class-agnostic objectness. This RU-B/280 adaptation is NOT official SegVLAD reproduction. Union oracle is not achieved R1.'}
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'predictions.npz', **predictions)
    report['predictions_sha256'] = sha(args.output/'predictions.npz')
    save(args.output/'summary.json', report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for command in ('fit', 'cache', 'eval'):
        s = sub.add_parser(command)
        s.add_argument('--dataset-root', type=Path, required=True)
        s.add_argument('--output', type=Path, required=True)
        s.add_argument('--device', default='cuda:0')
        if command != 'eval':
            s.add_argument('--checkpoint', type=Path, required=True)
            s.add_argument('--sam-checkpoint', type=Path, required=True)
            s.add_argument('--sam-type', choices=('vit_b', 'vit_l', 'vit_h'), default='vit_b')
            s.add_argument('--max-regions', type=int, default=16)
            s.add_argument('--neighbour-order', type=int, choices=(0, 1, 2, 3), default=1)
            s.add_argument('--seed', type=int, default=42)
        if command == 'fit':
            s.add_argument('--train-images', type=int, default=256)
            s.add_argument('--pca-dim', type=int, default=256)
            s.add_argument('--pca-samples', type=int, default=2048)
        if command == 'cache':
            s.add_argument('--model', type=Path, required=True)
            s.add_argument('--resume', action='store_true')
            s.add_argument('--limit-images', type=int, default=0, help='Engineering smoke only; never evaluated')
        if command == 'eval':
            s.add_argument('--cache', type=Path, required=True)
            s.add_argument('--region-top-k', type=int, default=50)
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(args, 'seed'):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        require(1 <= args.max_regions <= 64, 'max-regions must be 1..64')
    if args.command == 'cache':
        require(args.limit_images >= 0, 'limit-images cannot be negative')
    if args.command == 'fit':
        require(1 <= args.pca_dim < 32*768 and args.pca_samples > args.pca_dim, 'Invalid PCA dimensions/samples')
    if args.command == 'eval':
        require(1 <= args.region_top_k <= 1000, 'Invalid region search budget')
    {'fit': fit, 'cache': cache, 'eval': evaluate}[args.command](args)


if __name__ == '__main__':
    main()
