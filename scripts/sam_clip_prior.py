"""Frozen SAM -> crop CLIP prior -> weighted SuperSegments -> candidate retrieval.

Independent exploratory reranker. Never trains/changes BoQ or innovation point 1.
Reuse legacy 280px RU native-token cache; do not silently extract another one.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.sam_clip_prior import (ARMS, CLASSES, TEMPLATES, fragment_risk,
    interventions, weighted_vlad, pair_score, fixed_ranking)
from src.semantic_region_protocol import read, write, sha, save_npz, owned_run

CLIP_SHA = '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
FILES = ('scripts/sam_clip_prior.py', 'src/sam_clip_prior.py',
         'src/region_vlad.py', 'src/semantic_region_protocol.py', 'config/sam_clip_prior.json')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def image_path(root, name):
    p = (root/name).resolve()
    require(p.is_relative_to(root.resolve()) and p.is_file(), f'Missing/unsafe image: {name}')
    return p


def source(a):
    summary = read(a.source/'summary.json')
    require(a.checkpoint.is_file(), 'Need RU checkpoint file, not an empty environment variable')
    require(summary['ru_sha256'] == sha(a.checkpoint), 'Native cache/RU mismatch')
    names = ['msls_val_dbImages.npy', 'msls_val_qImages.npy']
    db, query = [np.load(a.dataset_root/n, allow_pickle=False).astype(str).tolist() for n in names]
    require((len(db), len(query)) == (18871, 740), 'Need original MSLS 18871/740 split')
    require(len(set(db+query)) == len(db+query), 'Duplicate image paths')
    with np.load(a.source/'per_query.npz', allow_pickle=False) as z:
        candidates, scores = z['candidates'].copy(), z['ru_scores'].copy()
    require(candidates.shape == scores.shape == (740, 20), 'Need frozen top20 candidates')
    require(np.issubdtype(candidates.dtype, np.integer) and candidates.min() >= 0
        and candidates.max() < len(db) and np.isfinite(scores).all(), 'Invalid candidates/scores')
    local = np.load(a.source/'local.npy', mmap_mode='r', allow_pickle=False)
    desc = np.load(a.source/'descriptors.npy', mmap_mode='r', allow_pickle=False)
    require(local.shape == (19611, 400, 768) and desc.shape == (19611, 12288), 'Wrong native cache layout')
    for q, row in enumerate(candidates):
        require(len(np.unique(row)) == 20 and (np.diff(scores[q]) <= 1e-6).all(), 'Candidate order mismatch')
        require(np.allclose(np.asarray(desc[row]) @ np.asarray(desc[len(db)+q]), scores[q],
            atol=2e-5, rtol=2e-4), 'Candidate/global descriptor mismatch')
    return db+query, candidates, scores, local


def crop_views(rgb, mask, padding):
    """Both representations have identical padded boxes; only background differs."""
    yy, xx = np.where(mask)
    require(len(xx) > 0, 'Empty raw fragment')
    h, w = mask.shape
    pad = max(1, int(round(max(xx.max()-xx.min()+1, yy.max()-yy.min()+1)*padding)))
    x0, x1 = max(0, xx.min()-pad), min(w, xx.max()+1+pad)
    y0, y1 = max(0, yy.min()-pad), min(h, yy.max()+1+pad)
    neutral = np.rint(np.array([.48145466, .4578275, .40821073])*255).astype(np.uint8)
    masked = np.where(mask[:, :, None], rgb, neutral[None, None, :])
    # Square letterbox avoids a second centre-crop cutting the object off.
    result = []
    for view in (rgb, masked):
        patch = Image.fromarray(view[y0:y1, x0:x1])
        size = max(patch.size)
        canvas = Image.new('RGB', (size, size), tuple(int(v) for v in neutral))
        canvas.paste(patch, ((size-patch.width)//2, (size-patch.height)//2))
        result.append(canvas)
    return result


class Extractor:
    def __init__(self, a, c):
        require(a.clip_checkpoint.is_file() and sha(a.clip_checkpoint) == CLIP_SHA,
            'Need pinned local OpenAI ViT-B-16.pt; no downloads')
        require(a.sam_checkpoint.is_file(), 'Missing SAM-B checkpoint')
        require(__import__('os').environ.get('CUDA_VISIBLE_DEVICES') == '1' and a.device == 'cuda:0',
            'Expose physical GPU1 only: CUDA_VISIBLE_DEVICES=1; use --device cuda:0')
        torch.cuda.set_per_process_memory_fraction(c['memory_fraction'], 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        from open_clip.openai import load_openai_model
        import open_clip
        from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        self.clip = load_openai_model(str(a.clip_checkpoint), device='cpu', precision='fp32').eval().requires_grad_(False).to(a.device)
        self.preprocess = Compose([Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            ToTensor(), Normalize((.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711))])
        tokenizer = open_clip.get_tokenizer('ViT-B-16')
        with torch.inference_mode():
            text = self.clip.encode_text(tokenizer([t.format(k) for k in CLASSES for t in TEMPLATES]).to(a.device)).float()
            text = torch.nn.functional.normalize(text, dim=-1).reshape(len(CLASSES), len(TEMPLATES), -1).mean(1)
            self.text = torch.nn.functional.normalize(text, dim=-1)
        self.scale = float(self.clip.logit_scale.exp().clamp(max=100).item())
        sam = sam_model_registry['vit_b'](checkpoint=str(a.sam_checkpoint)).eval().requires_grad_(False).to(a.device)
        self.sam = SamAutomaticMaskGenerator(sam, points_per_side=16, points_per_batch=32,
            pred_iou_thresh=.88, stability_score_thresh=.95, crop_n_layers=0)
        self.a, self.c = a, c

    @torch.inference_mode()
    def __call__(self, path):
        c, a = self.c, self.a
        with Image.open(path) as im:
            rgb = np.asarray(im.convert('RGB').resize((280, 280), Image.Resampling.BICUBIC))
        raw = self.sam.generate(rgb)
        selected = [r for r in raw if c['minimum_pixels'] <= r['area'] <= c['maximum_area_fraction']*280**2]
        selected.sort(key=lambda r: (-r['area'], r['bbox'][1], r['bbox'][0]))
        require(len(selected) <= c['max_fragments'],
            f'{len(selected)} eligible raw fragments exceeds fixed budget; no silent truncation: {path}')
        masks = np.stack([r['segmentation'] for r in selected]) if selected else np.empty((0, 280, 280), bool)
        views = [v for m in masks for v in crop_views(rgb, m, c['crop_padding'])]
        scores = []
        for i in range(0, len(views), c['clip_batch']):
            x = torch.stack([self.preprocess(v) for v in views[i:i+c['clip_batch']]]).to(a.device)
            z = torch.nn.functional.normalize(self.clip.encode_image(x).float(), dim=-1)
            scores.append((z @ self.text.T).cpu().numpy())
        cosine = np.concatenate(scores).reshape(-1, 2, len(CLASSES)) if scores else np.empty((0, 2, len(CLASSES)), np.float32)
        risks = fragment_risk(cosine, self.scale, c['confidence'], c['cosine_margin'])
        return masks, cosine, risks, self.scale, len(raw)


def validate_shard(path, image_sha, dim, c=None):
    c = c or {'confidence': .6, 'cosine_margin': .02}
    with np.load(path, allow_pickle=False) as z:
        require(str(z['image_sha']) == image_sha, 'Image changed since shard creation')
        n = len(z['risks'])
        require(z['cosines'].shape == (n, 2, len(CLASSES)), 'Wrong CLIP score dimensions')
        require(z['masks_packbits'].shape == (n, 9800), 'Wrong packed mask dimensions')
        # Force CRC checks of all stored members, including packed masks/stats.
        for name in z.files:
            value = z[name]
            if np.issubdtype(value.dtype, np.number):
                require(np.isfinite(value).all(), f'Nonfinite shard member: {name}')
        for arm in ARMS:
            d, r = z[arm], z[arm+'_reliability']
            require(d.shape == (n, dim) and r.shape == (n,), 'Wrong descriptor/reliability dimensions')
            require(np.isfinite(d).all() and np.isfinite(r).all() and ((r >= 0) & (r <= 1)).all(), 'Nonfinite/invalid shard')
            if n:
                require(np.allclose(np.linalg.norm(d.astype(np.float32), axis=1), 1, atol=.002), 'Nonunit descriptor')
        require(np.array_equal(fragment_risk(z['cosines'], float(z['scale']),
            c['confidence'], c['cosine_margin']), z['risks']), 'Risk reconstruction mismatch')


def cache(a):
    c = read(ROOT/'config/sam_clip_prior.json')
    paths, candidates, _, local = source(a)
    manifest = read(a.model/'manifest.json')
    require(manifest['complete'] and manifest['contract']['checkpoint_sha256'] == sha(a.checkpoint), 'VLAD/RU fit mismatch')
    require(manifest['contract']['image_size'] == 280 and manifest['contract']['feature'] ==
        'RU trained DINO final pre-gate normalized 20x20 tokens', 'Wrong vocabulary feature convention')
    require(sha(a.model/'model.npz') == manifest['model_sha256'], 'Fitted VLAD/PCA changed')
    with np.load(a.model/'model.npz', allow_pickle=False) as z:
        centers, mean, components = [torch.tensor(z[k].copy()) for k in ('centers', 'mean', 'components')]
    require(centers.shape == (32, 768) and components.shape[1] == len(mean) == 32*768, 'Unexpected shared fit')
    ids = np.unique(np.r_[candidates.ravel(), np.arange(18871, 19611)])
    required = ids[:a.limit_images] if a.limit_images else ids
    import importlib.metadata
    contract = {'schema': c['schema'], 'config': c, 'source': str(a.source.resolve()),
        'dataset_root': str(a.dataset_root.resolve()), 'required': ids.tolist(),
        'inputs': {n: sha(a.source/n) for n in ('summary.json', 'per_query.npz', 'local.npy', 'descriptors.npy')},
        'index': {n: sha(a.dataset_root/n) for n in ('msls_val_dbImages.npy', 'msls_val_qImages.npy')},
        'fit': {'manifest': sha(a.model/'manifest.json'), 'model': manifest['model_sha256']},
        'weights': {'ru': sha(a.checkpoint), 'sam': sha(a.sam_checkpoint), 'clip': sha(a.clip_checkpoint)},
        'classes': CLASSES, 'templates': TEMPLATES,
        'code': {n: sha(ROOT/n) for n in FILES},
        'versions': {n: importlib.metadata.version(n) for n in ('numpy', 'torch', 'open_clip_torch', 'segment-anything', 'scipy')},
        'scope': 'Exploratory top20 reranker, legacy RU 280; not official SegVLAD, not single-stage BoQ training'}
    # JSON roundtrip normalizes tuples before exact resume comparisons.
    contract = json.loads(json.dumps(contract))
    worst = len(ids)*c['max_fragments']*(280**2/8+len(ARMS)*(components.shape[0]*2+4))/1024**3
    print(f'Reuse native tokens; {len(ids)} candidate images. Uncompressed region upper bound ~{worst:.2f} GiB, no DINO re-extraction.', flush=True)
    with owned_run(a.output, contract, a.resume):
        if (a.output/'completed.json').exists():
            print('Already complete; use eval. No GPU extraction started.'); return
        (a.output/'shards').mkdir(exist_ok=True)
        extractor = None
        shard_hashes = {}
        counts, dynamic, pixel_means, shuffle_means, crop_agreements = [], [], [], [], []
        for i in tqdm(required, desc='Raw SAM / CLIP / weighted SuperSegments'):
            path = image_path(a.dataset_root, paths[i])
            image_sha = sha(path)
            target = a.output/'shards'/f'{i:06d}.npz'
            if not target.exists():
                if extractor is None:
                    extractor = Extractor(a, c)
                start = time.monotonic()
                masks, cosine, risks, scale, raw_count = extractor(path)
                arrays = {'image_sha': image_sha, 'path': paths[i], 'cosines': cosine, 'risks': risks,
                    'scale': scale, 'raw_count': raw_count,
                    'masks_packbits': np.packbits(masks.reshape(len(masks), 280**2), axis=1)}
                stats = {'region_count': 0, 'dynamic_fragments': 0}
                if len(masks):
                    arms, stats = interventions(masks, risks, f"{c['seed']}:{paths[i]}", c['strength'], c['neighbour_order'], c['grid'])
                    tokens = torch.tensor(np.asarray(local[i], np.float32))
                    require(torch.isfinite(tokens).all(), 'Invalid cached DINO tokens')
                    for arm, (membership, reliability) in arms.items():
                        with torch.inference_mode():
                            d = (weighted_vlad(tokens, centers, membership)-mean) @ components.T
                            require(torch.isfinite(d).all() and (d.norm(dim=1) > 1e-8).all(), 'Degenerate PCA result')
                            arrays[arm] = torch.nn.functional.normalize(d, dim=1).numpy().astype(np.float16)
                        arrays[arm+'_reliability'] = reliability
                else:
                    for arm in ARMS:
                        arrays[arm] = np.empty((0, len(components)), np.float16)
                        arrays[arm+'_reliability'] = np.empty(0, np.float32)
                arrays['stats'] = json.dumps(stats)
                arrays['seconds'] = time.monotonic()-start
                save_npz(target, **arrays)
            # Corrupt file is an explicit stop, never skipped/reported complete.
            validate_shard(target, image_sha, len(components), c)
            with np.load(target, allow_pickle=False) as z:
                stats = json.loads(str(z['stats']))
                counts.append(stats['region_count'])
                dynamic.append(stats['dynamic_fragments'])
                pixel_means.append(stats.get('pixel_risk_mean', 0.))
                shuffle_means.append(stats.get('shuffled_pixel_risk_mean', 0.))
                crop_agreements.extend((z['cosines'].argmax(-1)[:, 0] == z['cosines'].argmax(-1)[:, 1]).tolist())
            shard_hashes[target.name] = sha(target)
            write(a.output/'progress.json', {'phase': 'cache', 'checked': len(shard_hashes),
                'total': len(ids), 'partial': len(required) != len(ids)})
        write(a.output/'coverage.json', {'images': len(counts),
            'empty_region_images': int(sum(n == 0 for n in counts)),
            'eligible_fragments': int(sum(counts)), 'gated_dynamic_fragments': int(sum(dynamic)),
            'images_with_dynamic_prior': int(sum(n > 0 for n in dynamic)),
            'mean_pixel_risk': float(np.mean(pixel_means)),
            'mean_shuffled_pixel_risk': float(np.mean(shuffle_means)),
            'two_view_class_agreement': float(np.mean(crop_agreements)) if crop_agreements else None,
            'note': 'Descriptive only; no extra semantic gate or automatic threshold tuning'})
        if len(required) != len(ids):
            print('SMOKE CACHE ONLY. Repeat without --limit-images and with --resume to finish.'); return
        write(a.output/'completed.json', {'complete': True, 'schema': c['schema'],
            'contract_sha256': sha(a.output/'contract.json'), 'shards': shard_hashes,
            'coverage_sha256': sha(a.output/'coverage.json')})


def evaluate(a):
    c = read(ROOT/'config/sam_clip_prior.json')
    done, contract = read(a.cache/'completed.json'), read(a.cache/'contract.json')
    require(done.get('complete') is True and done['schema'] == c['schema'], 'Need complete cache')
    require(sha(a.cache/'contract.json') == done['contract_sha256'] and contract['config'] == c, 'Cache/config changed')
    require(sha(a.cache/'coverage.json') == done['coverage_sha256'], 'Coverage report changed')
    require(contract['code'] == {n: sha(ROOT/n) for n in FILES}, 'Cache/scorer implementation changed')
    require(str(a.source.resolve()) == contract['source'], 'Source path changed')
    for n, digest in contract['inputs'].items():
        require(sha(a.source/n) == digest, f'Source changed: {n}')
    for n, digest in contract['index'].items():
        require(sha(a.dataset_root/n) == digest, f'Dataset index changed: {n}')
    with np.load(a.source/'per_query.npz', allow_pickle=False) as z:
        candidates, scores = z['candidates'].copy(), z['ru_scores'].copy()
    require(not a.output.exists(), 'Choose new evaluation directory; no overwrites')
    expected = {f'{i:06d}.npz' for i in contract['required']}
    require(set(done['shards']) == expected, 'Incomplete region shard inventory')
    data = {}
    for name, digest in tqdm(done['shards'].items(), desc='Verify compact descriptors'):
        path = a.cache/'shards'/name
        require(sha(path) == digest, f'Damaged shard: {path}')
        with np.load(path, allow_pickle=False) as z:
            data[int(path.stem)] = {k: z[k].copy() for arm in ARMS for k in (arm, arm+'_reliability')}
    predictions = {'ru': candidates.copy()}
    all_scores = {}
    for arm in ARMS:
        regional = np.zeros_like(scores)
        for q, row in enumerate(tqdm(candidates, desc=f'Match {arm}')):
            query = data[18871+q]
            # A query with no regions abstains; missing candidate regions score 0.
            for j, i in enumerate(row):
                d = data[int(i)]
                regional[q, j] = pair_score(query[arm], d[arm], query[arm+'_reliability'], d[arm+'_reliability'])
        predictions[arm] = np.stack([row[fixed_ranking(g, r, c['rerank_alpha'])]
            for row, g, r in zip(candidates, scores, regional)])
        all_scores[arm] = regional
    # GT is loaded only AFTER all predictions are fixed.
    gt_path = a.dataset_root/'msls_val_gt_25m.npy'
    gt = np.load(gt_path, allow_pickle=True)
    require(len(gt) == 740, 'Wrong GT length')
    baseline = np.array([row[0] in g for row, g in zip(candidates, gt)])
    require(baseline.sum() == 675, 'Original RU 675/740 was not reproduced')
    report = {}
    for arm, pred in predictions.items():
        hit = np.array([row[0] in g for row, g in zip(pred, gt)])
        report[arm] = {'correct': int(hit.sum()), 'r1': float(hit.mean()),
            'corrections': np.flatnonzero(hit & ~baseline).tolist(),
            'regressions': np.flatnonzero(~hit & baseline).tolist(),
            'top1_changed': int((pred[:, 0] != candidates[:, 0]).sum()),
            'r5': float(np.mean([np.isin(row[:5], g).any() for row, g in zip(pred, gt)])),
            'r10': float(np.mean([np.isin(row[:10], g).any() for row, g in zip(pred, gt)]))}
    a.output.mkdir(parents=True)
    save_npz(a.output/'predictions.npz', **predictions)
    save_npz(a.output/'region_scores.npz', **all_scores)
    write(a.output/'summary.json', report)
    write(a.output/'provenance.json', {'cache_completed_sha256': sha(a.cache/'completed.json'),
        'gt_sha256': sha(gt_path), 'scope': contract['scope'],
        'candidate_reachable_errors': int(sum(not b and np.isin(row, g).any() for b, row, g in zip(baseline, candidates, gt)))})
    write(a.output/'completed.json', {'complete': True,
        'files': {p.name: sha(p) for p in sorted(a.output.iterdir())}})
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['cache', 'eval'])
    p.add_argument('--source', type=Path, default=Path('doc/visual_pair_msls_hard_mix_v2'))
    p.add_argument('--checkpoint', type=Path, default=Path('MISSING_RU_CHECKPOINT'))
    p.add_argument('--sam-checkpoint', type=Path, default=Path('.cache/sam/sam_vit_b_01ec64.pth'))
    p.add_argument('--clip-checkpoint', type=Path, default=Path('/home/wt/.cache/clip/ViT-B-16.pt'))
    p.add_argument('--dataset-root', type=Path, default=Path('datasets/msls-val'))
    p.add_argument('--model', type=Path, default=Path('.cache/region_vlad/fit_gsv_v1'))
    p.add_argument('--cache', type=Path, default=Path('.cache/sam_clip_prior_v1'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--limit-images', type=int, default=0)
    a = p.parse_args()
    require(a.limit_images >= 0, 'Negative image limit')
    torch.set_num_threads(4)
    {'cache': cache, 'eval': evaluate}[a.command](a)


if __name__ == '__main__':
    main()
