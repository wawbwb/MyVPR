"""Explicit weight/source preparation only. Never constructs or trains a model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def download(url, target, expected=None):
    if target.is_file():
        if expected and sha(target) != expected:
            raise ValueError('Existing source digest mismatch: ' + str(target))
        return
    partial = target.with_suffix(target.suffix + '.part')
    # Interrupted downloads never become trusted complete files.
    with urllib.request.urlopen(url, timeout=60) as source, partial.open('wb') as dest:
        total = int(source.headers.get('Content-Length', 0))
        count = 0
        reported = 0
        while True:
            block = source.read(1024*1024)
            if not block:
                break
            dest.write(block)
            count += len(block)
            if count - reported >= 32*1024*1024:
                print(f'Downloaded {count/1024**2:.0f} MiB' +
                      (f' / {total/1024**2:.0f} MiB' if total else ''), flush=True)
                reported = count
        if not count or total and count != total:
            raise ValueError('Truncated official download')
    if expected and sha(partial) != expected:
        raise ValueError('Pinned author source SHA256 mismatch')
    partial.replace(target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=ROOT/'.cache/boq_resnet50_official_v1')
    args = p.parse_args()
    # Import constants from the local source AST, not torch/model dependencies.
    import ast
    tree = ast.parse((ROOT/'src/models/official_boq.py').read_text(encoding='utf-8'))
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('COMMIT', 'SOURCE_SHA'):
                constants[node.targets[0].id] = ast.literal_eval(node.value)
    commit, sources = constants['COMMIT'], constants['SOURCE_SHA']
    url = 'https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/resnet50_16384.pth'
    cache = args.cache.resolve()
    if cache.parent != (ROOT/'.cache').resolve() or not cache.name.startswith('boq_resnet50_official_'):
        p.error('Use an isolated .cache/boq_resnet50_official_<version> directory')
    cache.mkdir(parents=True, exist_ok=True)
    (cache/'source').mkdir(exist_ok=True)
    manifest = cache/'manifest.json'
    old = json.loads(manifest.read_text()) if manifest.exists() else None
    if old and (old['commit'] != commit or old['weight_url'] != url or old['source_sha256'] != sources):
        raise ValueError('Existing official manifest differs; use a new directory')
    for name, digest in sources.items():
        path = 'src/' + name if name in ('backbones.py', 'boq.py') else name
        previous = ROOT/'.cache/boq_official_v1/source'/name
        destination = cache/'source'/name
        if not destination.exists() and previous.is_file() and sha(previous)==digest:
            shutil.copyfile(previous,destination)
        download(f'https://raw.githubusercontent.com/amaralibey/Bag-of-Queries/{commit}/{path}',
                 destination, digest)
    weight = cache/'resnet50_16384.pth'
    if weight.exists() and not old:
        raise ValueError('Unmanifested existing weight: use a new cache directory; do not silently adopt unknown bytes')
    print('Preparing complete author ResNet50-BoQ checkpoint; no training', flush=True)
    download(url, weight, old['weight_sha256'] if old else None)
    if weight.stat().st_size < 1024*1024:
        raise ValueError('Official weight unexpectedly small')
    value = dict(commit=commit, weight_url=url, weight_sha256=sha(weight), source_sha256=sources,
        weight_digest_status='Locally recorded on first official-HTTPS retrieval; not claimed to be an author-published checksum')
    if old and old != value:
        raise ValueError('Existing official weight digest changed')
    if not old:
        temp = manifest.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
        temp.replace(manifest)
    print(json.dumps(value, indent=2), flush=True)


if __name__ == '__main__':
    main()
