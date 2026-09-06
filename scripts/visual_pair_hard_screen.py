#!/usr/bin/env python
"""V2 visual screen: reuse V1 tokens, bidirectional same-city mining.

No MSLS labels enter mining, training or calibration. V1 files are untouched.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from tqdm import tqdm
from scripts import visual_pair_pipeline as v1
from src.visual_pair_verifier import VisualPairVerifier, pair_edges, anchored_score
from src.cc_lsa_gate_a import file_sha256


def source_hash():
    return file_sha256(Path(__file__))


def hard_rows(data, margin=0.03):
    positive = np.where(data['labels'], data['scores'], -np.inf).max(1)
    negative = np.where(~data['labels'], data['scores'], -np.inf).max(1)
    return ~data['baseline_correct'] | ((positive - negative) < margin)


def training_order(data, rng, mode):
    """Fixed N updates for both modes. Hard mix changes sampling only."""
    n = len(data['scores'])
    if mode == 'uniform':
        return rng.permutation(n)
    hard = np.flatnonzero(hard_rows(data))
    easy = np.flatnonzero(~hard_rows(data))
    if not len(hard) or not len(easy):
        raise ValueError('Hard/easy partition empty; inspect mining audit')
    order = np.concatenate([rng.choice(hard, n//2, replace=True),
                            rng.choice(easy, n-n//2, replace=True)])
    rng.shuffle(order)
    return order


def report(data, residual, alpha):
    result = v1.counts(data, residual, alpha)
    hard = hard_rows(data)
    result['hard_queries'] = int(hard.sum())
    result['reachable_errors'] = int((~data['baseline_correct'] & data['labels'].any(1)).sum())
    if hard.any():
        subset = {k: data[k][hard] for k in ['scores', 'labels']}
        result['hard_subset'] = v1.counts(subset, residual[hard], alpha)
    return result


def remine(args):
    manifest = json.loads((args.source/'manifest.json').read_text())
    if not manifest.get('complete') or manifest['schema'] != 'visual_pair_v1':
        raise ValueError('Need completed V1 GSV cache')
    # Verify cached arrays, not just their filenames. Never rewrite old hashes.
    for name in ['local.npy', 'descriptors.npy', 'index.json']:
        if file_sha256(args.source/name) != manifest['hashes'][name]:
            raise ValueError(f'Source cache hash mismatch: {name}')
    if manifest['implementation'] != v1.implementation():
        raise ValueError('V1 extraction implementation changed')
    index = json.loads((args.source/'index.json').read_text())
    records = index['places']
    local = np.load(args.source/'local.npy', mmap_mode='r')
    desc = np.load(args.source/'descriptors.npy', mmap_mode='r')
    audits = {}
    for split, name in enumerate(['train', 'select', 'calibrate']):
        groups = {}
        for record in records:
            if record['split'] == split:
                groups.setdefault(record['place'].rsplit(':', 1)[0], []).append(record)
        all_edges, all_scores, all_labels = [], [], []
        qids, candidate_ids, baseline, injected = [], [], [], []
        omitted = []
        for city, places in sorted(groups.items()):
            database = np.asarray([i for r in places for i in r['rows']], dtype=np.int64)
            if len(database)-1 < args.top_k:
                omitted.append({'city': city, 'places': len(places), 'queries': len(database)})
                continue
            mate = {r['rows'][0]: r['rows'][1] for r in places}
            mate.update({r['rows'][1]: r['rows'][0] for r in places})
            db = torch.tensor(np.asarray(desc[database]), device=args.device)
            with torch.inference_mode():
                for q in tqdm(database, desc=f'{name}/{city}', dynamic_ncols=True):
                    sim = torch.tensor(np.asarray(desc[q]), device=args.device) @ db.T
                    sim[torch.tensor(database == q, device=args.device)] = -torch.inf
                    scores, positions = sim.topk(args.top_k)
                    ids = database[positions.cpu().numpy()]
                    original_correct = bool(ids[0] == mate[q])
                    inject = split == 0 and mate[q] not in ids
                    if inject:
                        ids[-1] = mate[q]
                        scores[-1] = float(desc[q] @ desc[mate[q]])
                    edges = []
                    for start in range(0, len(ids), args.batch_size):
                        qt = torch.tensor(np.asarray(local[q])[None], device=args.device)
                        dt = torch.tensor(np.asarray(local[ids[start:start+args.batch_size]]), device=args.device)
                        edges.append(pair_edges(qt, dt).cpu().numpy())
                    all_edges.append(np.concatenate(edges))
                    all_scores.append(scores.cpu().numpy())
                    all_labels.append(ids == mate[q])
                    qids.append(q)
                    candidate_ids.append(ids)
                    baseline.append(original_correct)
                    injected.append(inject)
        if not qids:
            raise ValueError(f'No eligible same-city queries: {name}')
        data = dict(edges=np.asarray(all_edges), scores=np.asarray(all_scores),
                    labels=np.asarray(all_labels), baseline_correct=np.asarray(baseline),
                    query_ids=np.asarray(qids), candidates=np.asarray(candidate_ids),
                    injected=np.asarray(injected))
        np.savez(args.output/f'{name}.npz', **data)
        audits[name] = {'queries': len(qids), 'ru_correct_before_injection': int(sum(baseline)),
                        'injected_positives': int(sum(injected)), 'hard_queries': int(hard_rows(data).sum()),
                        'omitted_small_cities': omitted}
        print(json.dumps(audits[name]), flush=True)
    v1.write_json(args.output/'manifest.json', {'complete': True, 'schema': 'visual_pair_hard_v2',
        'implementation': v1.implementation(), 'hard_source_sha256': source_hash(),
        'source_manifest_sha256': file_sha256(args.source/'manifest.json'),
        'ru_sha256': manifest['ru_sha256'], 'top_k': args.top_k, 'audit': audits,
        'hashes': {f'{n}.npz': file_sha256(args.output/f'{n}.npz') for n in ['train', 'select', 'calibrate']}})


def train(args):
    manifest = json.loads((args.cache/'manifest.json').read_text())
    if not manifest.get('complete') or manifest['schema'] != 'visual_pair_hard_v2':
        raise ValueError('Expected complete V2 mined cache')
    if manifest['hard_source_sha256'] != source_hash() or manifest['implementation'] != v1.implementation():
        raise ValueError('Mining/training code changed')
    data = {}
    for name in ['train', 'select', 'calibrate']:
        if file_sha256(args.cache/f'{name}.npz') != manifest['hashes'][f'{name}.npz']:
            raise ValueError('Pair cache hash mismatch')
        with np.load(args.cache/f'{name}.npz') as archive:
            data[name] = {k: archive[k] for k in archive.files}
    model = VisualPairVerifier().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    best, history = -1, []
    for epoch in range(args.epochs):
        model.train()
        order = training_order(data['train'], rng, args.mode)
        if args.smoke_test:
            order = order[:1]
        losses = []
        for i in tqdm(order, desc=f'{args.mode} epoch {epoch+1}/{args.epochs}', dynamic_ncols=True):
            e = torch.tensor(data['train']['edges'][i], device=args.device)
            s = torch.tensor(data['train']['scores'][i], device=args.device)
            target = torch.tensor(int(data['train']['labels'][i].argmax()), device=args.device)
            optimizer.zero_grad(set_to_none=True)
            score = anchored_score(s, model(e, s), 0.1) / 0.05
            loss = torch.nn.functional.cross_entropy(score[None], target[None])
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise RuntimeError('Non-finite loss/gradient')
            optimizer.step()
            losses.append(loss.item())
        if args.smoke_test:
            print('SMOKE PASS: one finite update, no checkpoint')
            return
        model.eval()
        residual = v1.predictions(model, data['select'], args.device)
        metric = report(data['select'], residual, 0.1)
        history.append({'epoch': epoch+1, 'steps': len(order), 'loss': float(np.mean(losses)), 'selection': metric})
        if metric['correct'] > best:
            best = metric['correct']
            torch.save(model.state_dict(), args.output/'head.pt')
        print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(torch.load(args.output/'head.pt', map_location=args.device, weights_only=True))
    residual = v1.predictions(model, data['calibrate'], args.device)
    sweep = [{'alpha': a, **report(data['calibrate'], residual, a)} for a in [0., .01, .03, .1]]
    chosen = max(sweep, key=lambda r: (r['correct'], -r['alpha']))
    v1.write_json(args.output/'run.json', {'complete': True, 'implementation': v1.implementation(),
        'hard_source_sha256': source_hash(), 'ru_sha256': manifest['ru_sha256'],
        'head_sha256': file_sha256(args.output/'head.pt'), 'cache_manifest_sha256': file_sha256(args.cache/'manifest.json'),
        'mode': args.mode, 'seed': args.seed, 'top_k': manifest['top_k'], 'history': history,
        'calibration': sweep, 'alpha': chosen['alpha'], 'verdict': 'CANDIDATE' if chosen['net'] > 0 else 'NO_GAIN'})
    print('Calibration:', json.dumps(chosen), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['remine', 'train', 'eval'])
    p.add_argument('--output', type=Path, required=True)
    for name in ['source', 'cache', 'run', 'checkpoint', 'dataset-root']:
        p.add_argument('--'+name, type=Path)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mode', choices=['uniform', 'hard_mix'], default='uniform')
    p.add_argument('--smoke-test', action='store_true')
    args = p.parse_args()
    if min(args.batch_size, args.epochs) < 1 or args.top_k < 2 or args.num_workers < 0:
        p.error('Invalid numeric arguments')
    required = {'remine': ['source'], 'train': ['cache'], 'eval': ['run', 'checkpoint', 'dataset_root']}[args.stage]
    for name in required:
        value = getattr(args, name)
        if value is None or not value.exists():
            p.error(f'Missing --{name.replace("_", "-")}')
    if args.smoke_test and args.stage != 'train':
        p.error('Smoke is train-only')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.stage == 'eval':
        run = json.loads((args.run/'run.json').read_text())
        if run.get('hard_source_sha256') != source_hash():
            raise ValueError('V2 training code mismatch')
        v1.evaluate(args)
    else:
        {'remine': remine, 'train': train}[args.stage](args)


if __name__ == '__main__':
    main()
