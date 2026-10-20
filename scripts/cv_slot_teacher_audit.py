"""Small frozen DINO/SCLIP graph audit; no student updates, no benchmark access."""
import argparse
from collections import Counter
from importlib import metadata
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.nn import functional as F
from tqdm import tqdm

from scripts import clip_key_study as key
from scripts.semantic_region_study import transform
from src.models.clip_dense_keys import DenseClipKeys
from src.models.official_boq import load_official
from src.semantic_region_protocol import (owned_run, read, write, sha, state_hash,
    save_npz, seed_all)
from src.cv_slot_protocol import (ARMS, balanced_places, correspondence,
    crop_targets, geometry_metrics, graph_score, decide)

SCHEMA = 'cv_slot_teacher_audit_v1'
SOURCES = ('src/cv_slot_protocol.py', 'scripts/cv_slot_teacher_audit.py',
    'scripts/launch_cv_slot_teacher_audit.py', 'config/cv_slot_teacher_audit.yaml',
    'tests/test_cv_slot_teacher_audit.py')


def configuration(path):
    import yaml
    c = yaml.safe_load(path.read_text(encoding='utf-8'))
    if (c['schema'] != SCHEMA or c['seed'] != 42 or c['sample_per_split'] != 128
            or (c['grid'], c['crop_canvas'], c['crop_size'], c['crop_shift_pixels']) != (14, 256, 224, 32)
            or c['student_size'] != 322 or not 0 < c['cuda_memory_fraction'] <= .2
            or c['device'] != 'cuda:0' or c['negative_places'] != 4 or c['minimum_negatives'] != 2
            or c['topk'] != 8 or c['minimum_distance_m'] != 25.):
        raise ValueError('Expected locked small, native-grid teacher preflight')
    if c['thresholds'] != dict(min_geometry_places=128, min_real_places=120,
            min_probability_gain=.001, accuracy_tolerance=.002, real_pair_tolerance=0.):
        raise ValueError('Pre-extraction gate thresholds differ')
    return c


def identity(c, plan):
    import open_clip
    clip_root = Path(open_clip.__file__).parent
    result = key.identity(c)
    # Explicit helper source inventory, beyond original clip-key helper hashes.
    result.update(schema=SCHEMA, code=key.source_hashes(SOURCES),
        frozen_helpers=key.source_hashes(key.SOURCES),
        clip_code={p.relative_to(clip_root).as_posix(): sha(p) for p in sorted(clip_root.rglob('*.py'))},
        versions={n: metadata.version(n) for n in ('torch', 'torchvision', 'numpy', 'pillow', 'open_clip_torch')},
        selected={split: balanced_places(plan, split, c['sample_per_split'], c['seed'])
                  for split in ('train', 'holdout')},
        geometry='256 canvas; crops (0,0,224,224)/(32,32,256,256); second brightness0.85 contrast0.9',
        real='view0 query/view1 positive; up to4 same-city GPS>=25m negatives selected by official BoQ',
        comparison='identical DINO mutual-top8 support; only CLIP edge weights changed/permuted',
        stop_rule='No full student training unless every registered holdout teacher check passes',
        scope='Reused GSV development holdout, not blind generalization; zero model updates')
    return result


def finish(output):
    files = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*'))
        if p.is_file() and p.name not in ('.lock', 'completed.json') and not p.name.endswith('.tmp')}
    write(output/'completed.json', dict(schema=SCHEMA, complete=True, files=files))


def verify(output):
    value = read(output/'completed.json')
    if value.get('schema') != SCHEMA or value.get('complete') is not True:
        raise ValueError('Incomplete teacher audit')
    for name, digest in value['files'].items():
        p = (output/name).resolve()
        if not p.is_relative_to(output.resolve()) or not p.is_file() or sha(p) != digest:
            raise ValueError('Audit artifact checksum mismatch: '+str(p))
    return read(output/'contract.json')


def place_rows(c, plan, index):
    p = plan['places'][index]
    rows = [plan['images'][i] for i in p['rows'][:2]]
    result = []
    root = key.path(c, 'dataset_root').resolve()
    for row in rows:
        name = (root/row['path']).resolve()
        if not name.is_relative_to(root) or not name.is_file() or sha(name) != row['image_sha256']:
            raise ValueError('Original source image changed/missing: '+str(name))
        with Image.open(name) as image:
            result.append(image.convert('RGB'))
    canvas = result[0].resize((256, 256), Image.Resampling.BICUBIC)
    first = canvas.crop((0, 0, 224, 224))
    second = ImageEnhance.Contrast(ImageEnhance.Brightness(
        canvas.crop((32, 32, 256, 256))).enhance(.85)).enhance(.9)
    return [*result, first, second]


def validate_shard(path):
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {'dino', 'clip', 'global_descriptor'}:
            raise ValueError('Unexpected teacher shard fields')
        d, c, g = [z[k] for k in ('dino', 'clip', 'global_descriptor')]
    for value, shape in ((d, (4, 196, 768)), (c, (4, 196, 512)), (g, (2, 12288))):
        if value.shape != shape or value.dtype != np.float16 or not np.isfinite(value).all():
            raise ValueError('Invalid teacher shard shape/dtype/finiteness')
        if np.max(np.abs(np.linalg.norm(value.astype(np.float32), axis=-1)-1)) > .003:
            raise ValueError('Nonunit teacher shard')
    return d, c, g


def cached(c, plan, output, contract):
    indices = sum(contract['selected'].values(), [])
    directory = output/'shards'
    directory.mkdir(exist_ok=True)
    manifest_path = output/'shard_hashes.json'
    manifest = read(manifest_path) if manifest_path.exists() else {}
    unknown = set(manifest)-set(map(str, indices))
    if unknown:
        raise ValueError('Unexpected place in cache manifest')
    missing = []
    for index in indices:
        path = directory/f'{index:05d}.npz'
        if str(index) in manifest:
            if not path.is_file() or sha(path) != manifest[str(index)]:
                raise ValueError('Sealed shard missing/corrupt: '+str(path))
            validate_shard(path)
        else:
            # A crash after atomic shard write but before manifest update is safe:
            # regenerate this one uncommitted shard, never trust it implicitly.
            missing.append(index)
    print(f'Valid shards reused: {len(indices)-len(missing)}; new places: {len(missing)}', flush=True)
    if not missing:
        return
    key.gpu(c)
    seed_all(c['seed'])
    official = load_official(key.path(c, 'official_cache'), key.path(c, 'dino_source'), c['device'])
    clip = DenseClipKeys(key.path(c, 'clip_checkpoint'), c['device'])
    initial = dict(official=state_hash(official), clip=state_hash(clip))
    if (output/'frozen_initial.json').exists() and read(output/'frozen_initial.json') != initial:
        raise ValueError('Frozen teacher initialization changed')
    write(output/'frozen_initial.json', initial)
    student_transform, teacher_transform = transform(322), transform(224, teacher=True)
    with torch.inference_mode():
        for index in tqdm(missing, desc='Small frozen teacher sample'):
            views = place_rows(c, plan, index)
            dino, semantics, global_descriptor = [], [], []
            # Single image microbatches keep two frozen encoders under ~4.8GiB.
            for view_index, image in enumerate(views):
                features = official.backbone(student_transform(image)[None].to(c['device']))
                d = F.normalize(F.adaptive_avg_pool2d(features.float(), (14, 14)).flatten(2).transpose(1, 2), dim=-1)
                semantic = clip(teacher_transform(image)[None].to(c['device'])).flatten(2).transpose(1, 2)
                dino.append(d[0].cpu().numpy().astype(np.float16))
                semantics.append(semantic[0].cpu().numpy().astype(np.float16))
                if view_index < 2:
                    global_descriptor.append(official.aggregator(features)[0][0].cpu().numpy().astype(np.float16))
            path = directory/f'{index:05d}.npz'
            save_npz(path, dino=np.stack(dino), clip=np.stack(semantics),
                     global_descriptor=np.stack(global_descriptor))
            validate_shard(path)
            manifest[str(index)] = sha(path)
            write(manifest_path, manifest)
            write(output/'progress.json', dict(phase='extract', done=len(manifest), total=len(indices)))
    if initial != dict(official=state_hash(official), clip=state_hash(clip)):
        raise ValueError('Frozen teacher state changed during extraction')
    write(output/'frozen_checks.json', dict(official_unchanged=True, clip_unchanged=True,
        trainable_parameters=0, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
    del official, clip
    torch.cuda.empty_cache()


def metres(left, right):
    a, b = np.deg2rad(left), np.deg2rad(right)
    h = np.sin((a[0]-b[0])/2)**2 + np.cos(a[0])*np.cos(b[0])*np.sin((a[1]-b[1])/2)**2
    return float(6371000*2*np.arcsin(np.sqrt(np.clip(h, 0, 1))))


def evaluate(c, plan, output, contract):
    # Move only the tiny pair tensors to CUDA; cache stays on CPU.
    key.gpu(c)
    summaries = {}
    for split, indices in contract['selected'].items():
        arrays = {i: validate_shard(output/'shards'/f'{i:05d}.npz') for i in indices}
        geometry, real = [], []
        def graph(i, va, j, vb, seed):
            da, ca, _ = arrays[i]
            db, cb, _ = arrays[j]
            return correspondence(*[torch.from_numpy(v.astype(np.float32)).to(c['device'])
                for v in (da[va], db[vb], ca[va], cb[vb])], topk=c['topk'],
                dino_temperature=c['dino_temperature'], clip_temperature=c['clip_temperature'], seed=seed)
        excluded = []
        for i in tqdm(indices, desc=f'Known geometry + real pairs: {split}'):
            p = plan['places'][i]
            directions = [geometry_metrics(graph(i, 2, i, 3, c['seed']+i*11), *crop_targets()),
                geometry_metrics(graph(i, 3, i, 2, c['seed']+i*11+1), *crop_targets(reverse=True))]
            row = dict(place=p['place'], city=p['city'],
                candidate_gt_coverage=float(np.mean([x['candidate_gt_coverage'] for x in directions])),
                nonempty_rows=float(np.mean([x['nonempty_rows'] for x in directions])))
            row.update({arm: {metric: float(np.mean([x[arm][metric] for x in directions]))
                for metric in ('gt_probability', 'accuracy')} for arm in ARMS})
            geometry.append(row)
            # Filter by query AND positive view GPS to exclude near-place negatives.
            candidates = [j for j in indices if j != i and plan['places'][j]['city'] == p['city']
                and all(metres(plan['images'][r]['coordinates'], plan['images'][plan['places'][j]['rows'][1]]['coordinates'])
                        >= c['minimum_distance_m'] for r in p['rows'][:2])]
            if len(candidates) < c['minimum_negatives']:
                excluded.append(dict(place=p['place'], eligible_negatives=len(candidates)))
                continue
            q = arrays[i][2][0].astype(np.float32)
            candidates.sort(key=lambda j: (-float(q @ arrays[j][2][1].astype(np.float32)), j))
            candidates = candidates[:c['negative_places']]
            scores = {arm: [] for arm in ARMS}
            for j in [i, *candidates]:
                ga = graph(i, 0, j, 1, c['seed']+i*100003+j*17)
                gb = graph(j, 1, i, 0, c['seed']+i*100003+j*17+1)
                for arm in ARMS:
                    scores[arm].append((graph_score(ga, arm)+graph_score(gb, arm))/2)
            real_row = dict(place=p['place'], city=p['city'],
                negative_places=[plan['places'][j]['place'] for j in candidates])
            for arm in ARMS:
                s = np.asarray(scores[arm])
                wins = (s[0] > s[1:]).astype(float)+.5*(s[0] == s[1:])
                real_row[arm] = dict(pairwise_win=float(wins.mean()),
                    top1=float(s[0] > s[1:].max()), margin=float(s[0]-s[1:].max()), scores=s.tolist())
            real.append(real_row)
        write(output/f'{split}_geometry.json', geometry)
        write(output/f'{split}_real_pairs.json', real)
        summaries[split] = decide(geometry, real, c['thresholds'])
        summaries[split].update(city_counts=dict(Counter(plan['places'][i]['city'] for i in indices)),
            excluded_real_places=excluded,
            mean_candidate_gt_coverage=float(np.mean([x['candidate_gt_coverage'] for x in geometry])),
            mean_nonempty_rows=float(np.mean([x['nonempty_rows'] for x in geometry])))
    result = dict(schema=SCHEMA, verdict=summaries['holdout']['verdict'], partitions=summaries,
        thresholds=c['thresholds'], zero_student_updates=True,
        scope='Known-geometry check + real-view discrimination heuristic. This does not establish VPR gain.',
        next_action='Implement four matched student arms only after PASS; otherwise stop this teacher formulation.')
    write(output/'summary.json', result)
    print(__import__('json').dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'config/cv_slot_teacher_audit.yaml')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    c = configuration(a.config)
    plan = key.open_plan(c)
    contract = identity(c, plan)
    output = key.path(c, 'output')
    with owned_run(output, contract, a.resume):
        if (output/'completed.json').exists():
            verify(output)
            print('Already complete:', read(output/'summary.json')['verdict'], flush=True)
            return
        start = time.time()
        cached(c, plan, output, contract)
        evaluate(c, plan, output, contract)
        if key.source_hashes(key.PROTECTED) != contract['protected_sources']:
            raise ValueError('Innovation1 protected sources changed')
        write(output/'progress.json', dict(phase='complete', verdict=read(output/'summary.json')['verdict'],
            elapsed_seconds=time.time()-start))
        finish(output)


if __name__ == '__main__':
    main()
