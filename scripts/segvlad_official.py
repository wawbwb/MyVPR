#!/usr/bin/env python
"""Preflight/run pinned upstream SegVLAD with official 17Places artifacts.

Stdlib-only preflight. Never installs packages, edits upstream, or changes weights.
Only the explicitly selected run action executes upstream Python/pickle content.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

COMMIT = '628508d4a60c832961a2e2d8efa9eec724c2dc86'
EXPERIMENT = 'exp0_global_SegLoc_VLAD_PCA_o3'


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def literal_config(repo):
    tree = ast.parse((repo/'place_rec_global_config.py').read_text(encoding='utf8'))
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for name in node.targets:
                if isinstance(name, ast.Name) and name.id in ('datasets', 'experiments'):
                    result[name.id] = ast.literal_eval(node.value)
    return result['datasets']['17places'], result['experiments'][EXPERIMENT]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('check', 'run'))
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True, help='Parent containing 17places/ref, query, out')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', default='1', help='Physical GPU exposed before torch import')
    args = p.parse_args()
    repo, data, output = args.repo.resolve(), args.data_root.resolve(), args.output.resolve()
    if output.exists():
        p.error('Choose a new output directory; reports are never overwritten')
    if not (repo/'.git').exists():
        p.error('Clone the official repository first')
    head = git(repo, 'rev-parse', 'HEAD')
    if head != COMMIT or git(repo, 'status', '--porcelain', '--untracked-files=no'):
        p.error(f'Expected clean tracked upstream at {COMMIT}; do not patch official algorithms')
    cfg, exp = literal_config(repo)
    base = data/'17places'
    needed = [base/cfg[k] for k in ('data_subpath1_r', 'data_subpath2_q')]
    needed += [base/'out'/cfg[k] for k in ('masks_h5_filename_r', 'masks_h5_filename_q',
                                           'dino_h5_filename_r', 'dino_h5_filename_q')]
    needed += [base/'out'/('17places'+exp['pca_model_pkl']),
               repo/'cache/vocabulary/dinov2_vitg14/l31_value_c32/indoor/c_centers.pt']
    missing = [str(v) for v in needed if not v.exists()]
    # Probe real imports in a subprocess, using upstream path exactly as execution.
    probe = subprocess.run([sys.executable, '-c', 'import func_vpr; import utilities; import gt; import faiss; print("UPSTREAM_IMPORT_OK")'],
                           cwd=repo, capture_output=True, text=True)
    output.mkdir(parents=True)
    (output/'imports.txt').write_text(probe.stdout+'\n'+probe.stderr, encoding='utf8')
    report = {'upstream_commit': head, 'dataset': '17places', 'experiment': EXPERIMENT,
              'vocabulary': 'domain/indoor', 'dataset_config': cfg, 'experiment_config': exp,
              'python': sys.executable, 'missing_files': missing, 'imports_ok': probe.returncode == 0,
              'ready_to_attempt': not missing and probe.returncode == 0,
              'upstream_sources_sha256': {name: sha(repo/name) for name in
                  ('func_vpr.py', 'place_rec_main.py', 'place_rec_global_config.py', 'gt.py', 'utilities.py')},
              'scope': 'Official cached-feature retrieval check, NOT end-to-end extraction reproduction or MSLS comparison'}
    (output/'preflight.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2), flush=True)
    if probe.returncode:
        print('Import diagnostic:\n'+probe.stderr[-4000:], flush=True)
    if args.action == 'check':
        print('READY_TO_ATTEMPT' if report['ready_to_attempt'] else 'PREPARATION_INCOMPLETE: no evaluation started')
        return
    if not report['ready_to_attempt']:
        p.error(f'Preflight incomplete; read {output}/preflight.json and imports.txt')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['MPLBACKEND'] = 'Agg'
    import torch
    if not torch.cuda.is_available():
        p.error('CUDA unavailable; keep official artifacts and fix GPU state first')
    # The official program hardcodes a 1024-D FAISS index. Do not silently alter it.
    # Only load trusted author-provided pickle artifacts.
    import pickle
    with needed[-2].open('rb') as f:
        pca = pickle.load(f)
    if tuple(pca.components_.shape) != (1024, 49152):
        p.error(f'Official cached PCA incompatible: {pca.components_.shape}; do not silently change dimensions')
    # Record exact artifacts before executing unchanged upstream algorithms.
    report['artifact_sha256'] = {str(v): sha(v) for v in needed if v.is_file()}
    (output/'preflight.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    import place_rec_global_config as config
    config.workdir_data = str(data)  # path-only runtime override, no source edits
    sys.argv = ['place_rec_main.py', '--dataset', '17places', '--experiment', EXPERIMENT,
                '--vocab-vlad', 'domain', '--save-results']
    print('Running unchanged official entry point; native results go into 17places/out/results', flush=True)
    runpy.run_path(str(repo/'place_rec_main.py'), run_name='__main__')
    (output/'completed.json').write_text(json.dumps({'entrypoint_returned': True,
        'warning': 'Inspect native recall output; completion is not paper metric reproduction.'}, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
