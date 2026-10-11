"""Re-evaluate the frozen v1 teacher on a locked larger GSV gallery (no training)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid
from zipfile import BadZipFile

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_slgd import (RU_SHA, MatchedPlaces, seed, read, write, sha,
                               verify, code_hash)
from src.models.slgd import LocalProjector, local_tokens, mutual_score
from src.slgd_hard_protocol import HARD_POLICY, make_plan, distance_metres, diagnostic_summary

SOURCES = ('scripts/audit_slgd_hard.py', 'src/slgd_hard_protocol.py',
           'src/models/slgd.py', 'scripts/train_slgd.py',
           'src/dataloaders/train/gsv_cities.py', 'scripts/eval_condition_robustness.py',
           'src/models/backbones/dinov2.py', 'src/models/aggregators/boq.py')
SHAPES = dict(global_descriptors=(12288,), raw=(100, 768), teacher=(100, 128), coordinates=(2,))
DTYPES = dict(global_descriptors=np.float32, raw=np.float16, teacher=np.float16, coordinates=np.float64)


def save_npz(path, arrays):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)
    write(path.with_suffix('.json'), dict(sha256=sha(path)))


def validate_shard(path, indices):
    if sha(path) != read(path.with_suffix('.json'))['sha256']:
        raise ValueError('Shard hash mismatch; no automatic deletion: '+str(path))
    with np.load(path, allow_pickle=False) as stored:
        arrays = {k: stored[k] for k in stored.files}
    if set(arrays) != set(SHAPES) | {'indices'} or not np.array_equal(arrays['indices'], indices):
        raise ValueError('Shard identity differs: '+str(path))
    for name, dimensions in SHAPES.items():
        value = arrays[name]
        if value.shape != (len(indices),)+dimensions or value.dtype != DTYPES[name] or not np.isfinite(value).all():
            raise ValueError('Shard shape/dtype/non-finite value: '+name)
    for name in ('global_descriptors', 'raw', 'teacher'):
        norms = np.linalg.norm(arrays[name].astype(np.float32), axis=-1)
        if np.max(np.abs(norms-1)) > .002:
            raise ValueError('Non-unit descriptor/token: '+name)
    return arrays


class PlannedPlaces(Dataset):
    def __init__(self, dataset, plan, batch_start=0):
        self.data = MatchedPlaces(dataset, plan['dataset_indices'], HARD_POLICY['sampling_epoch'])
        self.plan = plan
        self.batch_start = batch_start

    def __len__(self):
        return len(self.data)-self.batch_start

    def __getitem__(self, index):
        index += self.batch_start
        images, _, metadata = self.data[index]
        nq = len(self.plan['query_places'])
        if index < nq:
            indices = torch.arange(index*4, index*4+4)
            return images, indices, metadata['coordinates']
        row = nq*4+index-nq
        return images[:1], torch.tensor([row]), metadata['coordinates'][:1]


def collate(items):
    return tuple(torch.cat([item[i] for item in items]) for i in range(3))


def shard_rows(plan, batch):
    nq = len(plan['query_places'])
    indices = []
    for place in range(batch*16, min((batch+1)*16, len(plan['dataset_indices']))):
        indices.extend(range(place*4, place*4+4) if place < nq else [nq*4+place-nq])
    return np.asarray(indices, dtype=np.int64)


def validate_teacher(contract, dataset_root):
    """Old training code remains frozen; v1 is not re-labelled as passing."""
    if contract['mode'] != 'teacher' or contract['stage'] != 'pilot' or contract['checkpoint_sha256'] != RU_SHA:
        raise ValueError('Expected completed v1 pilot teacher')
    for name, digest in contract['code'].items():
        if code_hash(ROOT/name) != digest:
            raise ValueError('Original teacher training source changed: '+name)
    metadata = {p.name: sha(p) for p in sorted((dataset_root/'Dataframes').glob('*.csv'))}
    if metadata != contract['metadata']:
        raise ValueError('GSV metadata differs from original teacher')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', type=Path, default=Path('logs/slgd_v1/teacher_pilot'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, default=Path('datasets/gsv_cities'))
    parser.add_argument('--output', type=Path, default=Path('logs/slgd_hard_v2'))
    parser.add_argument('--cache', type=Path, default=Path('.cache/slgd_hard_v2'))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    import fcntl
    from torchvision import transforms as T
    from tqdm import tqdm
    from src.dataloaders.train.gsv_cities import GSVCitiesDataset
    from scripts.eval_condition_robustness import load_inference_model_from_ckpt

    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Expose physical GPU1 only')
    if not args.checkpoint.is_file() or sha(args.checkpoint) != RU_SHA:
        raise ValueError('Original RU checkpoint missing/mismatched')
    teacher_contract = verify(args.teacher)
    validate_teacher(teacher_contract, args.dataset_root)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(.65, 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    lock = (args.cache.parent/(args.cache.name+'.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    seed(42)
    clean = T.Compose([T.Resize((280, 280), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(),
                       T.Normalize([.485,.456,.406],[.229,.224,.225])])
    cities = sorted(p.stem for p in (args.dataset_root/'Dataframes').glob('*.csv'))
    dataset = GSVCitiesDataset(dataset_path=args.dataset_root, cities=cities, img_per_place=4,
                              transform=clean, return_metadata=True)
    plan = make_plan(dataset.places_ids, teacher_contract, args.smoke)
    contract = dict(policy=HARD_POLICY, smoke=args.smoke, plan=plan,
                    teacher_completed_sha256=sha(args.teacher/'completed.json'),
                    teacher_checkpoint_sha256=sha(args.teacher/'last.pt'), ru_sha256=RU_SHA,
                    original_metadata=teacher_contract['metadata'],
                    code={name: code_hash(ROOT/name) for name in SOURCES},
                    versions=dict(torch=torch.__version__, numpy=np.__version__))
    for path in (args.output, args.cache):
        if path.exists():
            if not args.resume or read(path/'contract.json') != contract:
                raise ValueError('Output/cache exists or locked contract changed: '+str(path))
        else:
            path.mkdir()
            write(path/'contract.json', contract)
    if (args.output/'completed.json').is_file():
        verify(args.output)
        if read(args.output/'summary.json')['cache_manifest_sha256'] != sha(args.cache/'manifest.json'):
            raise ValueError('Completed output cache manifest changed')
        print('Already complete:', read(args.output/'summary.json')['verdict'], flush=True)
        return
    expected_bytes = plan['rows']*(12288*4+100*(768+128)*2+2*8)
    if shutil.disk_usage(args.cache).free < expected_bytes*1.2:
        raise ValueError('Insufficient disk space for compact audit cache')
    available_kib = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
    if available_kib*1024 < expected_bytes*1.5+1024**3:
        raise ValueError('Insufficient available RAM for compact token scoring; no extraction started')
    print(f"Locked plan: {len(plan['query_rows'])} queries, {len(plan['gallery_rows'])} references; "
          f"cache approximately {expected_bytes/1024**3:.2f} GiB (no full-resolution features)", flush=True)
    shards = args.cache/'shards'
    shards.mkdir(exist_ok=True)
    total_batches = (len(plan['dataset_indices'])+15)//16
    valid_batches = set()
    hashes = {}
    for batch in tqdm(range(total_batches), desc='Validate completed audit shards'):
        path = shards/f'{batch:05d}.npz'
        if path.is_file():
            try:
                validate_shard(path, shard_rows(plan, batch))
            except (OSError, EOFError, ValueError, KeyError, BadZipFile) as error:
                # A process kill between atomic npz and sidecar writes is recoverable.
                quarantine = args.cache/'quarantine'
                quarantine.mkdir(exist_ok=True)
                tag = uuid.uuid4().hex
                for source in (path, path.with_suffix('.json')):
                    if source.exists():
                        if not source.resolve().is_relative_to(args.cache.resolve()):
                            raise ValueError('Refusing to move a shard outside the cache')
                        source.replace(quarantine/(source.name+'.'+tag+'.bad'))
                print(f'Regenerate batch {batch}: {error}; original retained in quarantine', flush=True)
            else:
                valid_batches.add(batch)
                hashes[path.name] = sha(path)
    visual = None
    teacher = None
    loader = DataLoader(PlannedPlaces(dataset, plan), batch_size=16, num_workers=args.workers,
                        collate_fn=collate, persistent_workers=args.workers>0)
    for batch, (images, indices, coordinates) in enumerate(tqdm(loader, desc='Cache larger frozen GSV gallery')):
        write(args.output/'progress.json', dict(phase='cache', done=batch, total=total_batches,
                                               reused=len(valid_batches)))
        if batch in valid_batches:
            continue
        if visual is None:
            visual = load_inference_model_from_ckpt(args.checkpoint, 'cpu').cuda().eval().requires_grad_(False)
            teacher = LocalProjector().cuda().eval().requires_grad_(False)
            teacher.load_state_dict(torch.load(args.teacher/'last.pt', map_location='cpu', weights_only=True)['parameters'], strict=True)
        # Four views for query places, one view for distractor places, deterministically selected.
        rows = {name: [] for name in ('global_descriptors','raw','teacher')}
        with torch.inference_mode():
            for chunk in images.cuda().split(4):
                features = visual.backbone(chunk)
                if isinstance(features, tuple):
                    features = features[0]
                pooled = local_tokens(features)
                rows['raw'].append(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy().astype(np.float16))
                rows['teacher'].append(teacher(pooled).cpu().numpy().astype(np.float16))
                gated = features
                if visual.semantic_region_gate is not None:
                    gated = visual.semantic_region_gate(gated)[0]
                global_result = visual.aggregator(gated)
                if isinstance(global_result, tuple):
                    global_result = global_result[0]
                rows['global_descriptors'].append(global_result.cpu().numpy().astype(np.float32))
        arrays = {name: np.concatenate(value) for name, value in rows.items()}
        arrays.update(indices=indices.numpy().astype(np.int64), coordinates=coordinates.numpy().astype(np.float64))
        path = shards/f'{batch:05d}.npz'
        save_npz(path, arrays)
        validate_shard(path, shard_rows(plan, batch))
        hashes[path.name] = sha(path)
    write(args.cache/'manifest.json', dict(complete=True, contract_sha256=sha(args.cache/'contract.json'), files=hashes))
    del visual, teacher
    del loader
    torch.cuda.empty_cache()
    arrays = {name: np.empty((plan['rows'],)+shape, dtype=DTYPES[name]) for name, shape in SHAPES.items()}
    for batch in tqdm(range(total_batches), desc='Load verified compact cache'):
        stored = validate_shard(shards/f'{batch:05d}.npz', shard_rows(plan, batch))
        for name in arrays:
            arrays[name][stored['indices']] = stored[name]
    qrows = np.asarray(plan['query_rows'])
    grows = np.asarray(plan['gallery_rows'])
    gt = np.asarray(plan['positive_gallery_indices'])
    gallery = torch.from_numpy(arrays['global_descriptors'][grows]).cuda()
    candidates, gaps = [], []
    gps_exclusions = 0
    write(args.output/'progress.json', dict(phase='mine_candidates', queries=len(qrows)))
    for start in tqdm(range(0, len(qrows), 32), desc='Frozen RU top20 and difficulty'):
        query_rows = qrows[start:start+32]
        score = torch.from_numpy(arrays['global_descriptors'][query_rows]).cuda() @ gallery.T
        gps = distance_metres(arrays['coordinates'][query_rows], arrays['coordinates'][grows])
        allowed = gps >= HARD_POLICY['negative_exclusion_metres']
        allowed[np.arange(len(query_rows)), gt[start:start+32]] = True
        gps_exclusions += int((~allowed).sum())
        score.masked_fill_(~torch.from_numpy(allowed).cuda(), -torch.inf)
        order = torch.argsort(score, dim=1, descending=True, stable=True)[:, :HARD_POLICY['candidate_k']]
        values = score.gather(1, order)
        if not torch.isfinite(values).all():
            raise ValueError('Fewer than 20 valid GPS-filtered candidates')
        candidates.append(order.cpu().numpy())
        gaps.extend((values[:,0]-values[:,1]).cpu().tolist())
    candidates = np.concatenate(candidates)
    del gallery
    score_arrays = {name: np.empty(candidates.shape, np.float32) for name in ('raw','teacher')}
    valid_arrays = {name: np.empty(candidates.shape, bool) for name in ('raw','teacher')}
    write(args.output/'progress.json', dict(phase='local_candidate_matching', queries=len(qrows)))
    for start in tqdm(range(0, len(qrows), 16), desc='Matched raw-DINO / trained-teacher reranking'):
        query_rows = qrows[start:start+16]
        candidate_rows = grows[candidates[start:start+16]]
        for name in ('raw','teacher'):
            left = torch.from_numpy(arrays[name][np.repeat(query_rows, candidates.shape[1])].astype(np.float32)).cuda()
            right = torch.from_numpy(arrays[name][candidate_rows.flatten()].astype(np.float32)).cuda()
            with torch.inference_mode():
                left = torch.nn.functional.normalize(left, dim=-1)
                right = torch.nn.functional.normalize(right, dim=-1)
                score, count = mutual_score(left, right)
            score_arrays[name][start:start+len(query_rows)] = score.cpu().numpy().reshape(len(query_rows), -1)
            valid_arrays[name][start:start+len(query_rows)] = (count>=3).cpu().numpy().reshape(len(query_rows), -1)
    summary, outcomes = diagnostic_summary(candidates, score_arrays, valid_arrays['teacher'], valid_arrays['raw'],
                                           gt, np.asarray(plan['query_place_groups']), np.asarray(gaps), plan['calibration_places'])
    summary.update(smoke=args.smoke, queries=len(qrows), references=len(grows), gps_excluded_pairs=gps_exclusions,
                   cache_gib=expected_bytes/1024**3, teacher_retrained=False,
                   cache_manifest_sha256=sha(args.cache/'manifest.json'),
                   fp16_storage_scope='Matching is float32 on equally quantized unit raw/teacher tokens; not bitwise original FP32 reproduction')
    if args.smoke:
        summary['verdict'] = 'SMOKE_PASS'
        summary['scope'] = 'Mechanics only; no efficacy or sample sufficiency verdict'
    save_npz(args.output/'candidate_scores.npz', dict(candidates=candidates, raw_scores=score_arrays['raw'],
             teacher_scores=score_arrays['teacher'], raw_valid=valid_arrays['raw'], teacher_valid=valid_arrays['teacher'],
             positive_gallery=gt, groups=np.asarray(plan['query_place_groups']), ru_gap=np.asarray(gaps)))
    write(args.output/'outcomes.json', outcomes)
    write(args.output/'summary.json', summary)
    write(args.output/'progress.json', dict(phase='complete', verdict=summary['verdict']))
    write(args.output/'completed.json', dict(complete=True, files={p.name: sha(p) for p in args.output.iterdir()
          if p.is_file() and p.name != 'completed.json' and p.suffix != '.tmp'}))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print('No student/full/CLIP training was started. Inspect the locked diagnostic first.', flush=True)


if __name__ == '__main__':
    main()
