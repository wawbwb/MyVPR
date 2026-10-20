"""v2: fixed original queries/geometry, full-split same-city hard negatives.

No student updates. Existing v1 verdict/files remain immutable. This is a
separate, result-aware development audit, not an independent blind test.
"""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from tqdm import tqdm

from scripts import clip_key_study as key
from scripts import cv_slot_teacher_audit as original
from scripts import clip_nuisance_representation as representation
from scripts.semantic_region_study import transform
from src.cv_slot_protocol import ARMS, correspondence, graph_score, decide, paired_interval
from src.cv_slot_hard_protocol import choose_negatives
from src.models.clip_dense_keys import DenseClipKeys
from src.models.official_boq import load_official
from src.semantic_region_protocol import owned_run, read, write, sha, state_hash, save_npz, seed_all

SCHEMA = 'cv_slot_teacher_hard_v2'
SOURCES = ('src/cv_slot_hard_protocol.py', 'scripts/cv_slot_teacher_hard.py',
    'config/cv_slot_teacher_hard.yaml', 'tests/test_cv_slot_teacher_hard.py')


def configuration(path):
    import yaml
    c = yaml.safe_load(path.read_text(encoding='utf-8'))
    locked = dict(schema=SCHEMA, seed=42, train_places=8192, places_per_batch=16, epochs=5,
        views=4, sample_per_split=128, student_size=322, grid=14, topk=8,
        dino_temperature=.1, clip_temperature=.2, negative_places=4,
        minimum_distance_m=25., device='cuda:0', cuda_memory_fraction=.2)
    thresholds = dict(min_geometry_places=128, min_real_places=120,
        min_probability_gain=.001, accuracy_tolerance=.002, real_pair_tolerance=0.)
    if any(c.get(k) != v for k, v in locked.items()) or c['thresholds'] != thresholds:
        raise ValueError('Expected locked v2 pool-only change; thresholds unchanged')
    return c


def inputs(c):
    old_root = key.path(c, 'original_audit')
    old = original.verify(old_root)
    if old['code'] != key.source_hashes(original.SOURCES):
        raise ValueError('Original sealed teacher code changed')
    config_old = original.configuration(ROOT/'config/cv_slot_teacher_audit.yaml')
    for field in ('seed', 'plan', 'student_size', 'sample_per_split', 'topk',
                  'dino_temperature', 'clip_temperature', 'thresholds'):
        if c[field] != config_old[field]:
            raise ValueError('v2 unexpectedly changes more than negative pool: '+field)
    old_cache = key.path(c, 'global_cache')
    features = representation.verified(old_cache)
    source = features['source']['base']
    current = key.identity(c)
    for field in ('plan_sha256', 'plan_complete_sha256', 'official', 'clip_sha256'):
        if source[field] != current[field] or old[field] != current[field]:
            raise ValueError('Global cache / v1 / current source mismatch: '+field)
    if source['config']['student_size'] != 322:
        raise ValueError('Global mining descriptors need the same image size')
    return old, current


def contract_for(c, old, current):
    return dict(**dict(current, schema=SCHEMA), stage='hard_pair_teacher_audit',
        hard_schema=SCHEMA, hard_code=key.source_hashes(SOURCES),
        frozen_helpers=key.source_hashes((*original.SOURCES, *representation.SOURCES)),
        original_completed_sha256=sha(key.path(c, 'original_audit')/'completed.json'),
        global_completed_sha256=sha(key.path(c, 'global_cache')/'completed.json'),
        selected=old['selected'],
        pool='Entire same split:8192 train/1024 holdout places; same city, GPS>=25m from both positive views',
        negative_selection='Top4 frozen official BoQ cosine; no CLIP/model-arm/outcome-dependent mining',
        geometry='Exact immutable v1 per-place geometry reused; no threshold change',
        scope='Result-aware GSV development follow-up, NOT independent/blind evidence')


def finish(output):
    files = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*'))
        if p.is_file() and p.name not in ('.lock', 'completed.json') and not p.name.endswith('.tmp')}
    write(output/'completed.json', dict(schema=SCHEMA, complete=True, files=files))


def verify(output):
    result = read(output/'completed.json')
    if result.get('schema') != SCHEMA or result.get('complete') is not True:
        raise ValueError('Incomplete hard-pair teacher audit')
    for name, digest in result['files'].items():
        p = (output/name).resolve()
        if not p.is_relative_to(output.resolve()) or not p.is_file() or sha(p) != digest:
            raise ValueError('Hard audit checksum mismatch: '+str(p))
    return read(output/'contract.json')


def mine(c, plan, old, output):
    globals_ = np.load(key.path(c, 'global_cache')/'global_full.npy', mmap_mode='r', allow_pickle=False)
    if globals_.shape != (len(plan['images']), 12288) or globals_.dtype != np.float32:
        raise ValueError('Unexpected full-global cache population')
    # Independent row-order/encoder check against previously extracted v1 views.
    max_error = 0.
    for i in sum(old['selected'].values(), []):
        _, _, small = original.validate_shard(key.path(c, 'original_audit')/'shards'/f'{i:05d}.npz')
        rows = plan['places'][i]['rows'][:2]
        max_error = max(max_error, float(np.max(np.abs(globals_[rows]-small.astype(np.float32)))))
    if max_error > .0001:
        raise ValueError('Original query/positive descriptor reproduction differs')
    mining = {}
    with torch.inference_mode():
        for split, query_ids in old['selected'].items():
            pool = [i for i, p in enumerate(plan['places']) if p['split'] == split]
            rows = [plan['places'][i]['rows'][1] for i in pool]
            bank = torch.from_numpy(np.array(globals_[rows], copy=True)).to(c['device'])
            records = []
            for i in tqdm(query_ids, desc=f'Mine {split} from {len(pool)} places'):
                row0, row1 = plan['places'][i]['rows'][:2]
                q = torch.from_numpy(np.array(globals_[row0], copy=True)).to(c['device'])
                similarities = (q @ bank.T).cpu().numpy()
                negatives, eligible = choose_negatives(plan, i, pool, similarities,
                    c['negative_places'], c['minimum_distance_m'])
                if len(negatives) != c['negative_places']:
                    raise ValueError('Need four eligible same-city negative places for every query')
                scores = {j: float(similarities[k]) for k, j in enumerate(pool) if j in negatives}
                records.append(dict(query=i, negatives=negatives, eligible=eligible,
                    positive_global_similarity=float(globals_[row0] @ globals_[row1]),
                    negative_global_similarities=[scores[j] for j in negatives]))
            mining[split] = records
            del bank
    write(output/'mining.json', dict(partitions=mining, v1_global_max_abs_error=max_error))
    return mining


def original_rows(c, plan, old):
    return {row: (i, view) for i in sum(old['selected'].values(), [])
            for view, row in enumerate(plan['places'][i]['rows'][:2])}


def validate_image(path):
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {'dino', 'clip'}:
            raise ValueError('Unexpected image shard fields')
        d, s = z['dino'], z['clip']
    for value, shape in ((d, (196, 768)), (s, (196, 512))):
        if value.shape != shape or value.dtype != np.float16 or not np.isfinite(value).all():
            raise ValueError('Corrupt image shard')
        if np.max(np.abs(np.linalg.norm(value.astype(np.float32), axis=-1)-1)) > .003:
            raise ValueError('Nonunit image shard')
    return d, s


def extract(c, plan, old, mining, output):
    reused = original_rows(c, plan, old)
    rows = sorted({plan['places'][j]['rows'][1] for records in mining.values()
                   for record in records for j in record['negatives']} - set(reused))
    directory = output/'shards'
    directory.mkdir(exist_ok=True)
    hashes = read(output/'shard_hashes.json') if (output/'shard_hashes.json').exists() else {}
    if set(hashes)-set(map(str, rows)):
        raise ValueError('Unexpected cached image ID')
    missing = []
    for row in rows:
        p = directory/f'{row:06d}.npz'
        if str(row) in hashes:
            if not p.is_file() or sha(p) != hashes[str(row)]:
                raise ValueError('Sealed hard-negative shard changed')
            validate_image(p)
        else:
            missing.append(row)
    print('Reusing v1 queries/positives. New negative images:', len(missing), '/', len(rows), flush=True)
    if not missing:
        return
    seed_all(c['seed'])
    official = load_official(key.path(c, 'official_cache'), key.path(c, 'dino_source'), c['device'])
    clip = DenseClipKeys(key.path(c, 'clip_checkpoint'), c['device'])
    initial = dict(official=state_hash(official), clip=state_hash(clip))
    write(output/'frozen_initial.json', initial)
    student_transform, teacher_transform = transform(322), transform(224, teacher=True)
    root = key.path(c, 'dataset_root').resolve()
    with torch.inference_mode():
        for row in tqdm(missing, desc='Extract only new hard-negative images'):
            info = plan['images'][row]
            path = (root/info['path']).resolve()
            if not path.is_relative_to(root) or not path.is_file() or sha(path) != info['image_sha256']:
                raise ValueError('Original GSV image differs')
            with Image.open(path) as image:
                image = image.convert('RGB')
            f = official.backbone(student_transform(image)[None].to(c['device']))
            d = F.normalize(F.adaptive_avg_pool2d(f, (14, 14)).flatten(2).transpose(1, 2), dim=-1)
            s = clip(teacher_transform(image)[None].to(c['device'])).flatten(2).transpose(1, 2)
            p = directory/f'{row:06d}.npz'
            save_npz(p, dino=d[0].cpu().numpy().astype(np.float16), clip=s[0].cpu().numpy().astype(np.float16))
            validate_image(p)
            hashes[str(row)] = sha(p)
            write(output/'shard_hashes.json', hashes)
            write(output/'progress.json', dict(phase='extract', images=len(hashes), total=len(rows), student_updates=0))
    if dict(official=state_hash(official), clip=state_hash(clip)) != initial:
        raise ValueError('Frozen encoder state changed')
    write(output/'frozen_checks.json', dict(official_unchanged=True, clip_unchanged=True,
        new_images=len(rows), peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
    del official, clip
    torch.cuda.empty_cache()


def evaluate(c, plan, old, mining, output):
    reused = original_rows(c, plan, old)
    results = {}
    for split, records in mining.items():
        needed = {r for record in records for j in [record['query'], *record['negatives']]
                  for r in plan['places'][j]['rows'][:2] if r in reused}
        needed.update(plan['places'][j]['rows'][1] for record in records for j in record['negatives'])
        arrays = {}
        for row in needed:
            if row in reused:
                place, view = reused[row]
                d, s, _ = original.validate_shard(key.path(c, 'original_audit')/'shards'/f'{place:05d}.npz')
                arrays[row] = (d[view].copy(), s[view].copy())
            else:
                arrays[row] = validate_image(output/'shards'/f'{row:06d}.npz')
        def graph(a, b):
            da, ca = arrays[a]
            db, cb = arrays[b]
            return correspondence(*[torch.from_numpy(v.astype(np.float32)).to(c['device'])
                for v in (da, db, ca, cb)], topk=c['topk'], dino_temperature=c['dino_temperature'],
                clip_temperature=c['clip_temperature'], seed=c['seed']+a*100003+b*17)
        rows = []
        for record in tqdm(records, desc='Full-pool real pairs '+split):
            i = record['query']
            query_row = plan['places'][i]['rows'][0]
            scores = {arm: [] for arm in ARMS}
            for j in [i, *record['negatives']]:
                other = plan['places'][j]['rows'][1]
                ab, ba = graph(query_row, other), graph(other, query_row)
                for arm in ARMS:
                    scores[arm].append((graph_score(ab, arm)+graph_score(ba, arm))/2)
            row = dict(place=plan['places'][i]['place'], city=plan['places'][i]['city'], mining=record)
            for arm in ARMS:
                s = np.asarray(scores[arm])
                wins = (s[0] > s[1:]).astype(float)+.5*(s[0] == s[1:])
                row[arm] = dict(pairwise_win=float(wins.mean()), top1=float(s[0] > s[1:].max()),
                    margin=float(s[0]-s[1:].max()), scores=s.tolist())
            rows.append(row)
        geometry = read(key.path(c, 'original_audit')/f'{split}_geometry.json')
        report = decide(geometry, rows, c['thresholds'])
        report.update(negative_pool_places=sum(p['split'] == split for p in plan['places']),
            real_top1={arm: float(np.mean([x[arm]['top1'] for x in rows])) for arm in ARMS},
            real_pair_paired={arm: paired_interval([x['clip']['pairwise_win'] for x in rows],
                [x[arm]['pairwise_win'] for x in rows]) for arm in ('dino', 'shuffled')},
            minimum_eligible_same_city_negatives=min(x['eligible'] for x in records))
        write(output/f'{split}_real_pairs.json', rows)
        results[split] = report
        del arrays
    result = dict(schema=SCHEMA, verdict=results['holdout']['verdict'], partitions=results,
        thresholds=c['thresholds'], zero_student_updates=True,
        scope='Full-split hard-pool GSV development audit; geometry reused; no independent/blind claim')
    write(output/'summary.json', result)
    print(__import__('json').dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'config/cv_slot_teacher_hard.yaml')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    c = configuration(a.config)
    plan = key.open_plan(c)
    old, current = inputs(c)
    contract = contract_for(c, old, current)
    output = key.path(c, 'output')
    with owned_run(output, contract, a.resume):
        if (output/'completed.json').exists():
            verify(output)
            print('Already complete', flush=True)
            return
        key.gpu(c)
        start = time.time()
        mining = mine(c, plan, old, output)
        extract(c, plan, old, mining, output)
        evaluate(c, plan, old, mining, output)
        if key.source_hashes(key.PROTECTED) != contract['protected_sources']:
            raise ValueError('Innovation1 protected source changed')
        write(output/'progress.json', dict(phase='complete', verdict=read(output/'summary.json')['verdict'],
            elapsed_seconds=time.time()-start, student_updates=0))
        finish(output)


if __name__ == '__main__':
    main()
