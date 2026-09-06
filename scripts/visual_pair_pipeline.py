#!/usr/bin/env python
"""Visual-first screen: cache GSV pairs, train/select/calibrate, then evaluate MSLS.

All outputs are new directories. No failed LSA teacher is needed.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.visual_pair_verifier import VisualPairVerifier, pair_edges, anchored_score
from src.cc_lsa_gate_a import file_sha256
from src.cc_lsa_features import extract_ru_descriptor_and_local
from src.dataloaders.train.cc_lsa import gsv_image_name


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def implementation():
    return {name: file_sha256(ROOT/name) for name in ['scripts/visual_pair_pipeline.py', 'src/visual_pair_verifier.py', 'src/cc_lsa_features.py']}


class Images(Dataset):
    def __init__(self, root, paths):
        from scripts.eval_condition_robustness import build_transform
        self.root, self.paths = root, paths
        self.transform = build_transform((280, 280))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.root / self.paths[i]) as image:
            return self.transform(image.convert('RGB')), i


def extract(args, paths):
    from scripts.eval_condition_robustness import load_inference_model_from_ckpt
    estimate = len(paths) * (400 * 768 * 2 + 12288 * 4)
    if shutil.disk_usage(args.output).free < estimate * 1.2:
        raise RuntimeError('Insufficient free disk space for native token cache')
    model = load_inference_model_from_ckpt(args.checkpoint, torch.device(args.device)).eval()
    loader = DataLoader(Images(args.dataset_root, paths), batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False)
    local = desc = None
    with torch.inference_mode():
        for images, indices in tqdm(loader, desc='Cache frozen RU/native DINO', dynamic_ncols=True):
            d, z = extract_ru_descriptor_and_local(model, images.to(args.device), output_grid=(20, 20))
            if local is None:
                local = np.lib.format.open_memmap(args.output/'local.npy', mode='w+', dtype='float16', shape=(len(paths), *z.shape[1:]))
                desc = np.lib.format.open_memmap(args.output/'descriptors.npy', mode='w+', dtype='float32', shape=(len(paths), d.shape[1]))
                # Verify the shared extraction helper reproduces actual inference.
                actual = model(images.to(args.device))
                if isinstance(actual, (tuple, list)):
                    actual = actual[0]
                actual = torch.nn.functional.normalize(actual.float(), dim=-1)
                if not torch.allclose(d, actual, atol=2e-5, rtol=2e-4):
                    raise RuntimeError('RU helper differs from checkpoint forward')
            local[indices.numpy()] = z.cpu().numpy()
            desc[indices.numpy()] = d.cpu().numpy()
    local.flush()
    desc.flush()
    del model
    return local, desc


def build_pairs(args, local, desc, queries, database):
    candidates, scores, edges = [], [], []
    db = torch.tensor(np.asarray(desc[database]), device=args.device)
    with torch.inference_mode():
        for q in tqdm(queries, desc='Native visual candidate pairs', dynamic_ncols=True):
            similarity = torch.tensor(np.asarray(desc[q]), device=args.device) @ db.T
            values, positions = similarity.topk(min(args.top_k, len(database)))
            ids = np.asarray(database)[positions.cpu().numpy()]
            blocks = []
            for start in range(0, len(ids), args.batch_size):
                selected = ids[start:start+args.batch_size]
                qtokens = torch.tensor(np.asarray(local[q]), device=args.device).unsqueeze(0)
                dtokens = torch.tensor(np.asarray(local[selected]), device=args.device)
                blocks.append(pair_edges(qtokens, dtokens).cpu().numpy())
            candidates.append(ids)
            scores.append(values.cpu().numpy())
            edges.append(np.concatenate(blocks))
    return np.asarray(candidates), np.asarray(scores), np.asarray(edges, dtype=np.float32)


def cache(args):
    import pandas as pd
    groups = [[], [], []]
    source = {}
    for csv in sorted((args.dataset_root/'Dataframes').glob('*.csv')):
        source[csv.name] = file_sha256(csv)
        frame = pd.read_csv(csv)
        for place, rows in frame.groupby('place_id', sort=True):
            if len(rows) < 4:
                continue
            key = f'{csv.stem}:{place}'
            h = digest(f'{args.seed}:{key}')
            bucket = int(h[:8], 16) % 10
            split = 1 if bucket == 0 else 2 if bucket == 1 else 0
            names = sorted({f'Images/{csv.stem}/{gsv_image_name(row)}' for _, row in rows.iterrows()}, key=lambda p: digest(f'{args.seed}:{p}'))
            if len(names) >= 2:
                groups[split].append((h, key, names[:2]))
    paths, records = [], []
    for split, limit in enumerate([args.train_places, args.holdout_places, args.holdout_places]):
        selected = sorted(groups[split])[:limit]
        if len(selected) < args.top_k:
            raise ValueError('Each split needs at least top-k places')
        for _, key, names in selected:
            records.append({'place': key, 'split': split, 'rows': [len(paths), len(paths)+1]})
            paths.extend(names)
    write_json(args.output/'index.json', {'paths': paths, 'places': records})
    local, desc = extract(args, paths)
    for split, name in enumerate(['train', 'select', 'calibrate']):
        selected = [r for r in records if r['split'] == split]
        q, db = [r['rows'][0] for r in selected], [r['rows'][1] for r in selected]
        candidates, scores, edges = build_pairs(args, local, desc, q, db)
        labels = candidates == np.asarray(db)[:, None]
        # Training receives an explicit positive if not retrieved. Evaluation never does.
        if split == 0:
            for i in np.flatnonzero(~labels.any(1)):
                candidates[i, -1] = db[i]
                scores[i, -1] = desc[q[i]] @ desc[db[i]]
                edges[i, -1] = pair_edges(torch.tensor(np.asarray(local[q[i]])[None], device=args.device), torch.tensor(np.asarray(local[db[i]])[None], device=args.device)).cpu().numpy()[0]
                labels[i, -1] = True
        np.savez(args.output/f'{name}.npz', edges=edges, scores=scores, labels=labels)
    write_json(args.output/'manifest.json', {'complete': True, 'schema': 'visual_pair_v1', 'implementation': implementation(), 'ru_sha256': file_sha256(args.checkpoint), 'seed': args.seed, 'top_k': args.top_k, 'source_csv_sha256': source, 'split_note': 'place-disjoint for new head; RU has seen GSV', 'hashes': {p.name: file_sha256(p) for p in args.output.iterdir() if p.is_file()}})


def predictions(model, data, device):
    output = []
    with torch.no_grad():
        for e, s in zip(data['edges'], data['scores']):
            output.append(model(torch.tensor(e, device=device), torch.tensor(s, device=device)).cpu().numpy())
    return np.asarray(output)


def counts(data, residual, alpha):
    score = data['scores'] + alpha * np.tanh(residual)
    order = np.argsort(-score, axis=1, kind='stable')
    hits = np.take_along_axis(data['labels'], order, axis=1)
    base = data['labels'][:, 0]
    return {'queries': len(base), 'baseline_correct': int(base.sum()), 'correct': int(hits[:, 0].sum()), 'corrections': int((~base & hits[:, 0]).sum()), 'regressions': int((base & ~hits[:, 0]).sum()), 'net': int(hits[:, 0].sum()-base.sum()), 'r1': float(hits[:, 0].mean()), 'r5': float(hits[:, :5].any(1).mean()), 'r10': float(hits[:, :10].any(1).mean())}


def train(args):
    manifest = json.loads((args.cache/'manifest.json').read_text())
    if not manifest['complete'] or manifest['schema'] != 'visual_pair_v1':
        raise ValueError('Incomplete/unsupported cache')
    if manifest['implementation'] != implementation():
        raise ValueError('Code changed since caching; review and regenerate cache')
    for name in ['train.npz', 'select.npz', 'calibrate.npz', 'index.json']:
        if file_sha256(args.cache/name) != manifest['hashes'][name]:
            raise ValueError(f'Cache hash mismatch: {name}')
    data = {}
    for name in ['train', 'select', 'calibrate']:
        with np.load(args.cache/f'{name}.npz') as archive:
            data[name] = {key: archive[key] for key in archive.files}
    model = VisualPairVerifier().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = np.random.default_rng(args.seed)
    best_loss = float('inf')
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        order = generator.permutation(len(data['train']['scores']))
        if args.smoke_test:
            order = order[:1]
        bar = tqdm(order, desc=f'Visual head epoch {epoch+1}/{args.epochs}', dynamic_ncols=True)
        for i in bar:
            e = torch.tensor(data['train']['edges'][i], device=args.device)
            s = torch.tensor(data['train']['scores'][i], device=args.device)
            target = torch.tensor(data['train']['labels'][i].argmax(), device=args.device).long()
            optimizer.zero_grad(set_to_none=True)
            score = anchored_score(s, model(e, s), 0.1) / 0.05
            loss = torch.nn.functional.cross_entropy(score[None], target[None])
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise RuntimeError('Non-finite FP32 loss/gradient')
            optimizer.step()
            losses.append(loss.item())
            bar.set_postfix(loss=f'{loss.item():.4f}')
        if args.smoke_test:
            print('SMOKE PASS: one finite update; no checkpoint written')
            return
        model.eval()
        residual = predictions(model, data['select'], args.device)
        # Selection by whole-list retrieval, with conservative tie breaking.
        metric = counts(data['select'], residual, 0.1)
        selection_loss = -metric['correct']
        history.append({'epoch': epoch+1, 'train_loss': float(np.mean(losses)), 'selection': metric})
        if selection_loss < best_loss:
            best_loss = selection_loss
            torch.save(model.state_dict(), args.output/'head.pt')
        print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(torch.load(args.output/'head.pt', map_location=args.device, weights_only=True))
    residual = predictions(model, data['calibrate'], args.device)
    sweep = [{'alpha': a, **counts(data['calibrate'], residual, a)} for a in [0., 0.01, 0.03, 0.1]]
    chosen = max(sweep, key=lambda r: (r['correct'], -r['alpha']))
    write_json(args.output/'run.json', {'complete': True, 'implementation': implementation(), 'ru_sha256': manifest['ru_sha256'], 'cache_manifest_sha256': file_sha256(args.cache/'manifest.json'), 'head_sha256': file_sha256(args.output/'head.pt'), 'top_k': manifest['top_k'], 'history': history, 'calibration': sweep, 'alpha': chosen['alpha'], 'verdict': 'CANDIDATE' if chosen['net'] > 0 else 'NO_GAIN', 'seed': args.seed})
    print('Calibration:', chosen, flush=True)


def evaluate(args):
    from src.dataloaders.valid.mapillary_sls import MapillarySLSDataset
    run = json.loads((args.run/'run.json').read_text())
    if run['implementation'] != implementation():
        raise ValueError('Code changed after training')
    if not run['complete'] or run['ru_sha256'] != file_sha256(args.checkpoint) or run['head_sha256'] != file_sha256(args.run/'head.pt'):
        raise ValueError('Checkpoint/run provenance mismatch')
    args.top_k = run['top_k']
    ds = MapillarySLSDataset(args.dataset_root)
    local, desc = extract(args, ds.image_paths.tolist())
    candidates, scores, edges = build_pairs(args, local, desc, range(ds.num_references, len(ds)), range(ds.num_references))
    labels = np.asarray([np.isin(row, gt) for row, gt in zip(candidates, ds.ground_truth)])
    data = {'edges': edges, 'scores': scores, 'labels': labels}
    model = VisualPairVerifier().to(args.device).eval()
    model.load_state_dict(torch.load(args.run/'head.pt', map_location=args.device, weights_only=True))
    residual = predictions(model, data, args.device)
    result = counts(data, residual, run['alpha'])
    if ds.num_queries != 740 or ds.num_references != 18871 or result['baseline_correct'] != 675:
        raise RuntimeError(f'Frozen RU/standard split not reproduced: {result}; expected 675/740 and 18871 DB')
    np.savez(args.output/'per_query.npz', candidates=candidates, ru_scores=scores, residual=residual, labels=labels)
    write_json(args.output/'summary.json', {'alpha': run['alpha'], 'metrics': result, 'exploratory': True, 'run_sha256': file_sha256(args.run/'run.json'), 'ru_sha256': run['ru_sha256']})
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['cache', 'train', 'eval'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--cache', type=Path)
    p.add_argument('--run', type=Path)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--train-places', type=int, default=10000)
    p.add_argument('--holdout-places', type=int, default=1000)
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--smoke-test', action='store_true')
    args = p.parse_args()
    if min(args.batch_size, args.train_places, args.holdout_places, args.top_k, args.epochs) < 1 or args.num_workers < 0:
        p.error('Invalid nonpositive argument')
    required = ['cache'] if args.stage == 'train' else ['dataset_root', 'checkpoint']
    if args.stage == 'eval':
        required.append('run')
    for key in required:
        if getattr(args, key) is None or not getattr(args, key).exists():
            p.error(f'Missing --{key.replace("_", "-")}')
    if args.smoke_test and args.stage != 'train':
        p.error('--smoke-test is train-only')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    {'cache': cache, 'train': train, 'eval': evaluate}[args.stage](args)


if __name__ == '__main__':
    main()
