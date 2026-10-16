"""Independent GSV-only protocol and artifact helpers for the CLIP region study."""
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import random

import numpy as np
import torch


SCHEMA = 'semantic_spatial_pool_v1'
SOURCES = ('src/models/semantic_region_pool.py', 'src/models/sclip_region_teacher.py',
    'src/semantic_region_protocol.py', 'scripts/semantic_region_study.py',
    'config/semantic_region_study.yaml', 'scripts/run_semantic_region_study.sh',
    'tests/test_semantic_region_study.py', 'src/models/official_boq.py',
    'src/losses/vpr_losses.py', 'src/dataloaders/valid/mapillary_sls.py',
    'src/dataloaders/valid/pittsburgh.py')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()


def code_inventory(root):
    return {name: hashlib.sha256((root/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for name in SOURCES}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2,
                                  allow_nan=False)+'\n').encode())


def save_torch(path, value):
    # Stream large checkpoints to disk, avoiding an extra in-memory copy.
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_npz(path, **arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    atomic_bytes(path, stream.getvalue())


@contextmanager
def owned_run(output, contract, resume=False):
    """Linux process lock, exact immutable contract, explicit resume only."""
    import fcntl
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = output/'contract.json'
        if target.exists():
            if not resume:
                raise FileExistsError(f'Use --resume for existing run: {output}')
            if read(target) != contract:
                raise ValueError(f'Immutable run contract differs: {output}')
        else:
            if any(p.name != '.lock' for p in output.iterdir()):
                raise FileExistsError(f'Unowned output is not empty: {output}')
            write(target, contract)
        yield output


def seal(output):
    output = Path(output)
    files = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*'))
             if p.is_file() and p.name not in ('.lock', 'completed.json') and not p.name.endswith('.tmp')}
    write(output/'completed.json', dict(complete=True, schema=SCHEMA, files=files))


def verify(output):
    output = Path(output)
    result = read(output/'completed.json')
    if result.get('complete') is not True or result.get('schema') != SCHEMA:
        raise ValueError('Run is not complete')
    for name, expected in result['files'].items():
        path = (output/name).resolve()
        if not path.is_relative_to(output.resolve()) or not path.is_file() or sha(path) != expected:
            raise ValueError(f'Artifact checksum differs: {path}')
    return read(output/'contract.json')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)


def state_hash(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rng_state():
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=[n[0], n[1].tolist(), *n[2:]],
        torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(value):
    random.setstate(value['python'])
    n = value['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(value['torch'])
    if value['cuda']:
        torch.cuda.set_rng_state_all(value['cuda'])


def schedule(place_count, epochs, batch_places, seed):
    if place_count < batch_places or place_count % batch_places:
        raise ValueError('Train places must be divisible by places per batch')
    result = []
    for epoch in range(epochs):
        order = np.random.default_rng(seed+epoch).permutation(place_count).tolist()
        result.extend(order[i:i+batch_places] for i in range(0, place_count, batch_places))
    return result


def gps_retrieval(lossfn, descriptors, labels, coordinates):
    """MS loss with different-ID negatives within 25m excluded (all arms)."""
    ap, positive, an, negative = lossfn.miner(descriptors, labels)
    xy = torch.deg2rad(coordinates.double().to(descriptors.device))
    left, right = xy[an], xy[negative]
    a = torch.sin((left[:, 0]-right[:, 0])/2).square()
    a += torch.cos(left[:, 0])*torch.cos(right[:, 0])*torch.sin((left[:, 1]-right[:, 1])/2).square()
    metres = 6371000.*2*torch.asin(a.clamp(0, 1).sqrt())
    keep = metres >= 25.
    return lossfn.loss_fn(descriptors, labels, (ap, positive, an[keep], negative[keep]))


def paired(reference, candidate):
    a, b = np.asarray(reference, bool), np.asarray(candidate, bool)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError('Paired query populations differ')
    corrections = int((~a & b).sum())
    regressions = int((a & ~b).sum())
    return dict(queries=len(a), corrections=corrections, regressions=regressions,
                net=corrections-regressions, delta_r1_pp=100.*(b.mean()-a.mean()))
