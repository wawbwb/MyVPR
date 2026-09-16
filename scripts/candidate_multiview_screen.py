"""Frozen query-crop candidate coverage screen. No pair scoring or training."""
import argparse
import hashlib
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import candidate_hard_exploratory as e
from scripts.candidate_set_screen import read, write, sha, complete, load_npz, npz

POLICY = dict(crop_width_fraction=.75, horizontal_positions=[0., .5, 1.],
    crop_height='full', transform='crop native RGB then bilinear 322, ImageNet normalization',
    per_view_topk=20, full_topk=80, union='full20 then left20 then center20 then right20, stable dedup, no refill',
    scope='Candidate oracle coverage only, NOT actual reranked R1; fixed before extraction, no GT crop selection')


def crop_boxes(width, height):
    if width < 4 or height < 1: raise ValueError('Invalid image dimensions')
    span = int(round(width*.75))
    return [(int(round((width-span)*t)), 0, int(round((width-span)*t))+span, height)
            for t in POLICY['horizontal_positions']]


def candidate_sets(full_scores, crop_scores):
    full_scores = np.asarray(full_scores); crop_scores = np.asarray(crop_scores)
    if full_scores.ndim != 1 or len(full_scores) < 80 or crop_scores.shape != (3, len(full_scores)):
        raise ValueError('Invalid retrieval dimensions')
    if not np.isfinite(full_scores).all() or not np.isfinite(crop_scores).all(): raise ValueError('Nonfinite scores')
    full = np.argsort(-full_scores, kind='stable')[:80]
    views = np.argsort(-crop_scores, axis=1, kind='stable')[:, :20]
    union = np.asarray(list(dict.fromkeys(np.concatenate([full[:20], views.ravel()]).tolist())), dtype=np.int64)
    return dict(full20=full[:20], full80=full, crop_top20=views, union=union)


def outcome(qi, sets, positives, frozen_correct):
    hits = {k: bool(set(sets[k].tolist()) & set(positives)) for k in ('full20', 'full80', 'union')}
    return dict(query_index=qi, frozen_correct=bool(frozen_correct), **hits,
        union_size=len(sets['union']), new_pairs=len(set(sets['union'].tolist())-set(sets['full20'].tolist())))


def summarize(rows):
    n = len(rows)
    return dict(queries=n, reachable={k: sum(r[k] for r in rows) for k in ('full20', 'full80', 'union')},
        crop_unique_gain_ids=[r['query_index'] for r in rows if r['union'] and not r['full80']],
        full80_unique_gain_ids=[r['query_index'] for r in rows if r['full80'] and not r['union']],
        newly_reachable_ids=[r['query_index'] for r in rows if r['union'] and not r['full20']],
        full80_newly_reachable_ids=[r['query_index'] for r in rows if r['full80'] and not r['full20']],
        frozen_errors=sum(not r['frozen_correct'] for r in rows),
        originally_unreachable=sum(not r['full20'] for r in rows),
        union_size_mean=float(np.mean([r['union_size'] for r in rows])),
        additional_pair_count=sum(r['new_pairs'] for r in rows), actual_pair_scoring=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=['train', 'dev'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--plan', type=Path, default=Path('doc/candidate_hard_plan_v1'))
    p.add_argument('--cache-root', type=Path, default=Path('.cache/candidate_hard_v1'))
    p.add_argument('--gsv-root', type=Path, default=Path('datasets/gsv_cities'))
    p.add_argument('--official-repo', type=Path, default=Path('/home/wt/workspace/Pair-VPR-official'))
    p.add_argument('--audit', type=Path, default=Path('doc/pairvpr_official_paired_audit_v1'))
    a = p.parse_args()
    print('Verifying frozen plan/cache hashes; no database extraction...', flush=True)
    plan = e.verify(a.plan, e.h.codes()); e.h.ensure_disjoint(plan['plan'])
    cache = a.cache_root/a.split; c = e.verify(cache, e.h.codes())
    part = plan['plan'][a.split]; database, queries = part['database'], part['queries']
    nd, nq = len(database), len(queries)
    if (c['plan_sha256'] != sha(a.plan/'completed.json') or c['split'] != a.split
            or c['ndb'] != nd or c['queries'] != nq or c['topk'] != 20): raise ValueError('Cache identity mismatch')
    code = {**e.codes(), 'scripts/candidate_multiview_screen.py': hashlib.sha256(
        Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()}
    contract = dict(code=code, policy=POLICY, split=a.split, cache_sha256=sha(cache/'completed.json'),
        plan_sha256=sha(a.plan/'completed.json'), official=c['official'])
    if a.output.exists():
        if not a.resume or read(a.output/'contract.json') != contract: raise ValueError('Use fresh output or identical --resume')
        if (a.output/'completed.json').exists():
            e.verify(a.output, code); print('Already complete and verified'); return
    vectors = []; hashes = []
    for start in range(0, nd+nq, 128):
        shard = load_npz(cache/'global'/f'{start:07d}.npz')
        count = min(128, nd+nq-start)
        if shard['vectors'].shape != (count, 512) or len(shard['hashes']) != count: raise ValueError('Bad global shard')
        vectors.append(shard['vectors']); hashes.extend(shard['hashes'].tolist())
    vectors = np.concatenate(vectors)
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-4):
        raise ValueError('Invalid cached global vectors')
    import torch
    from PIL import Image
    from torchvision import transforms as T
    from tqdm import tqdm
    model, identity = e.h.original.official(a)
    if identity != c['official']: raise ValueError('Frozen model identity mismatch')
    transform = T.Compose([T.Resize((322, 322), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(),
        T.Normalize([.485, .456, .406], [.229, .224, .225])])
    # Full-image re-encoding checks model/preprocessing against the cached query.
    with torch.inference_mode():
        x, _ = e.h.original.image(a.gsv_root, queries[0], hashes[nd])
        _, z = model(x[None].cuda(), None, 'global')
        if not np.allclose(z.cpu().numpy()[0], vectors[nd], atol=2e-5, rtol=2e-4):
            raise ValueError('Full-query descriptor does not reproduce cached pipeline')
    if not a.output.exists():
        a.output.mkdir(parents=True); write(a.output/'contract.json', contract); (a.output/'queries').mkdir()
    db = torch.from_numpy(vectors[:nd]).cuda(); rows = []; elapsed = 0.; extracted = 0
    labels = [r['label'] for r in database]
    places = {}
    for i, label in enumerate(labels): places.setdefault(label, []).append(i)
    with torch.inference_mode():
        for qi, q in enumerate(tqdm(queries, desc=f'{a.split}: crop retrieval only')):
            path = (a.gsv_root/q['path']).resolve()
            if not path.is_relative_to(a.gsv_root.resolve()) or sha(path) != hashes[nd+qi]: raise ValueError('Query image changed')
            file = a.output/'queries'/f'{qi:06d}.npz'
            if file.exists() and file.with_suffix('.sha.json').exists():
                saved = load_npz(file); crop_z = saved['crop_vectors']
            else:
                torch.cuda.synchronize(); begin = time.perf_counter()
                with Image.open(path) as im:
                    im = im.convert('RGB'); boxes = crop_boxes(*im.size)
                    batch = torch.stack([transform(im.crop(box)) for box in boxes]).cuda()
                _, z = model(batch, None, 'global'); crop_z = z.cpu().numpy()
                torch.cuda.synchronize(); elapsed += time.perf_counter()-begin; extracted += 1
            if crop_z.shape != (3, 512) or not np.isfinite(crop_z).all() or not np.allclose(np.linalg.norm(crop_z, axis=1), 1, atol=2e-4):
                raise ValueError('Invalid crop vectors')
            full_s = (torch.from_numpy(vectors[nd+qi]).cuda()@db.T).cpu().numpy()
            crop_s = (torch.from_numpy(crop_z).cuda()@db.T).cpu().numpy()
            sets = candidate_sets(full_s, crop_s)
            old = load_npz(cache/'pairs'/f'{qi:06d}.npz')
            e.h.validate_row(old, places[q['label']], nd)
            if not np.array_equal(sets['full20'], old['candidates']): raise ValueError('Original top20 not reproduced')
            if file.exists() and file.with_suffix('.sha.json').exists():
                if any(not np.array_equal(saved[k], v) for k, v in sets.items()): raise ValueError('Resumed ranking changed')
            else: npz(file, crop_vectors=crop_z, **sets)
            rows.append(outcome(qi, sets, places[q['label']], old['labels'][np.argmax(old['base'])]))
    write(a.output/'summary.json', summarize(rows)); write(a.output/'per_query.json', rows)
    write(a.output/'timing.json', dict(new_queries_this_invocation=extracted, crop_encoding_seconds=elapsed,
        note='Includes decode/transform/transfer/encoding of newly computed crops only; excludes verification/retrieval and earlier interrupted invocations'))
    write(a.output/'query_mapping.json', queries); complete(a.output)
    print(summarize(rows)); print('Complete. Candidate coverage is NOT reranked accuracy:', a.output)


if __name__ == '__main__': main()
