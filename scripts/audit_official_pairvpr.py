#!/usr/bin/env python
"""Instrument official Pair-VPR evaluation without changing its scores/ranking.

Run inside the existing VPR environment; official repository stays unmodified.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

EXPECTED_WEIGHT = '18e7b95ba57d4d578fbf0a06a1846fc2dfb976c74da56bfd7d6a68afbcea8423'


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, data):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def paths_normalized(values):
    return [str(p.decode() if isinstance(p, bytes) else p).replace('\\', '/').removeprefix('train_val/') for p in values]


def as_indices(value, nqueries, ndb):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    a = np.asarray(value)
    if a.ndim != 2 or len(a) != nqueries or not np.issubdtype(a.dtype, np.integer):
        raise ValueError('Invalid prediction shape/dtype')
    if (a < 0).any() or (a >= ndb).any():
        raise ValueError('Invalid candidate index')
    if any(len(np.unique(row)) != len(row) for row in a):
        raise ValueError('Duplicate candidates in a prediction row')
    return a.astype(np.int64)


def comparison(reference, variant):
    return {'both_correct': int((reference & variant).sum()),
            'reference_only': int((reference & ~variant).sum()),
            'variant_only': int((~reference & variant).sum()),
            'both_wrong': int((~reference & ~variant).sum()),
            'corrections': int((~reference & variant).sum()),
            'regressions': int((reference & ~variant).sum()),
            'net': int(variant.sum()-reference.sum()),
            'oracle_union_count': int((reference | variant).sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--official-repo', type=Path, required=True)
    p.add_argument('--msls-path', type=Path, required=True)
    p.add_argument('--ru-audit', type=Path, required=True,
                   help='V1 MSLS audit folder containing summary.json and per_query.npz')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    repo, root = args.official_repo.resolve(), args.msls_path.resolve()
    sys.path.insert(0, str(repo))
    config_path = repo/'pairvpr/configs/pairvpr_speed_local.yaml'
    weight = repo/'trained_models/pairvpr-vitB.pth'
    dino = Path(torch.hub.get_dir())/'facebookresearch_dinov2_main'
    if not (dino/'hubconf.py').is_file():
        raise FileNotFoundError(dino/'hubconf.py')
    if sha(weight) != EXPECTED_WEIGHT:
        raise ValueError('Official weight SHA mismatch')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('Set CUDA_VISIBLE_DEVICES=1 to use physical GPU1')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one visible CUDA GPU')
    cfg = OmegaConf.load(config_path)
    assert cfg.augmentation.img_res == 322 and cfg.eval.refinetopcands == 100
    assert cfg.eval.memoryeffmode is True
    assert cfg.encoder.model_name == 'dinov2_vitb14_reg'
    from pairvpr.datasets.inference.mapillary_val_dataset import MapillaryValDataset
    dataset_root = Path(os.path.join(str(root.parent), str(cfg.dataset_locations.msls)))
    ds = MapillaryValDataset(str(dataset_root), str(repo))
    old_db = np.load(root/'msls_val_dbImages.npy')
    old_q = np.load(root/'msls_val_qImages.npy')
    old_gt = np.load(root/'msls_val_gt_25m.npy', allow_pickle=True)
    assert ds.num_references == 18871 and ds.num_queries == 740
    assert paths_normalized(ds.dbImages) == paths_normalized(old_db)
    assert paths_normalized(ds.qImages[ds.qIdx]) == paths_normalized(old_q)
    gt = [np.asarray(g, dtype=np.int64).reshape(-1) for g in ds.ground_truth]
    assert len(gt) == len(old_gt)
    assert all(np.array_equal(np.sort(a), np.sort(b)) for a, b in zip(gt, old_gt))
    assert all(len(g) and (g >= 0).all() and (g < 18871).all() for g in gt)
    for path in ds.images:
        if not (dataset_root/str(path)).is_file():
            raise FileNotFoundError(dataset_root/str(path))
    ru_summary = json.loads((args.ru_audit/'summary.json').read_text())
    with np.load(args.ru_audit/'per_query.npz') as ru:
        ru_candidates = as_indices(ru['candidates'], 740, 18871)
        ru_labels = np.asarray([np.isin(row, g) for row, g in zip(ru_candidates, gt)])
        if not np.array_equal(ru_labels, ru['labels']):
            raise ValueError('RU saved labels differ from current GT/index')
    ru_correct = ru_labels[:, 0]
    assert int(ru_correct.sum()) == 675
    assert ru_summary['metrics']['baseline_correct'] == 675
    assert 'ru_sha256' in ru_summary
    spec = importlib.util.spec_from_file_location('pairvpr_recorded_eval', repo/'pairvpr/eval/eval.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    original_report = evaluator.get_test_recalls
    signature = inspect.signature(original_report)
    for key in ['dataset_name', 'predictions', 'reranked_predictions', 'gt']:
        if key not in signature.parameters:
            raise RuntimeError(f'Official reporting interface changed: missing {key}')
    print('PREFLIGHT PASS: weight, config, images, GT, RU audit and reporting interface', flush=True)
    if args.preflight_only:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {
        'complete': False, 'official_repo': str(repo),
        'official_commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'official_diff': subprocess.check_output(['git', '-C', str(repo), 'diff', 'HEAD', '--', 'pairvpr'], text=True),
        'wrapper_sha256': sha(__file__), 'checkpoint_sha256': sha(weight),
        'config': OmegaConf.to_container(cfg, resolve=True), 'config_sha256': sha(config_path),
        'official_python_sha256': {str(f.relative_to(repo)): sha(f) for f in sorted((repo/'pairvpr').rglob('*.py'))},
        'local_dino_python_sha256': {str(f.relative_to(dino)): sha(f) for f in sorted(dino.rglob('*.py'))},
        'ru_audit_sha256': sha(args.ru_audit/'per_query.npz'), 'ru_summary': ru_summary,
        'index_sha256': {name: sha(root/name) for name in ['msls_val_dbImages.npy', 'msls_val_qImages.npy', 'msls_val_gt_25m.npy']},
        'versions': {name: importlib.metadata.version(name) for name in ['torch', 'torchvision', 'xformers', 'numpy', 'omegaconf']},
        'gpu': torch.cuda.get_device_name(0), 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
    }
    write_json(args.output/'provenance.json', provenance)
    original_hub = torch.hub.load

    def offline_hub(source, model, *a, **kw):
        if str(source).split(':')[0] == 'facebookresearch/dinov2':
            assert model == 'dinov2_vitb14_reg'
            kw.pop('source', None)
            kw['pretrained'] = False
            return original_hub(str(dino), model, *a, source='local', **kw)
        return original_hub(source, model, *a, **kw)

    torch.hub.load = offline_hub
    from pairvpr.models.pairvpr import PairVPRNet
    original_load = PairVPRNet.load_state_dict
    loaded = []

    def strict_load(self, state, *a, **kw):
        result = original_load(self, state, strict=True)
        loaded.append(True)
        print('PASS: complete official checkpoint loaded strictly', flush=True)
        return result

    PairVPRNet.load_state_dict = strict_load
    recorded = []

    def record(*a, **kw):
        bound = signature.bind(*a, **kw)
        bound.apply_defaults()
        values = bound.arguments
        assert loaded and values['dataset_name'].lower() == 'msls_val'
        assert len(values['gt']) == 740
        assert all(np.array_equal(np.sort(a), np.sort(b)) for a, b in zip(gt, values['gt']))
        global_ids = as_indices(values['predictions'], 740, 18871)
        refined_ids = as_indices(values['reranked_predictions'], 740, 18871)
        assert global_ids.shape[1] >= 100 and refined_ids.shape[1] == 100
        assert all(set(g[:100]) == set(r) for g, r in zip(global_ids, refined_ids))
        # Persist first; later errors in official pretty-printing won't lose ranks.
        np.savez(args.output/'predictions.npz', global_ids=global_ids,
                 refined_ids=refined_ids, ru_ids=ru_candidates)
        hits = {name: np.asarray([np.isin(row, g) for row, g in zip(ids, gt)])
                for name, ids in [('global', global_ids), ('refined', refined_ids)]}
        global_correct, refined_correct = hits['global'][:, 0], hits['refined'][:, 0]
        reachable = hits['global'][:, :100].any(1)
        summary = {'complete': True, 'queries': 740, 'references': 18871,
                   'global_vs_refined': comparison(global_correct, refined_correct),
                   'ru_vs_refined': comparison(ru_correct, refined_correct),
                   'global_correct': int(global_correct.sum()),
                   'refined_correct': int(refined_correct.sum()),
                   'ru_correct': int(ru_correct.sum()),
                   'top100_oracle': int(reachable.sum()),
                   'remaining_reachable_errors': int((reachable & ~refined_correct).sum()),
                   'remaining_unreachable_errors': int((~reachable).sum()),
                   'recall': {name: {str(k): float(h[:, :k].any(1).mean()) for k in [1, 5, 10, 100]} for name, h in hits.items()},
                   'exploratory': True,
                   'note': 'K>100 reranked recall is intentionally not reported; RU is a different system.'}
        with (args.output/'per_query.jsonl').open('x', encoding='utf8') as f:
            for i in range(740):
                row = {'query_index': i, 'query_path': str(old_q[i]),
                       'ru_correct': bool(ru_correct[i]), 'global_correct': bool(global_correct[i]),
                       'refined_correct': bool(refined_correct[i]), 'top100_reachable': bool(reachable[i]),
                       'ru_top1': str(old_db[ru_candidates[i, 0]]),
                       'global_top1': str(old_db[global_ids[i, 0]]),
                       'refined_top1': str(old_db[refined_ids[i, 0]])}
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
        summary['predictions_sha256'] = sha(args.output/'predictions.npz')
        write_json(args.output/'summary.json', summary)
        recorded.append(True)
        print(json.dumps(summary, indent=2), flush=True)
        return original_report(*a, **kw)

    evaluator.get_test_recalls = record
    parsed = evaluator.get_args_parser().parse_args([
        '--dsetroot', str(root.parent), '--config-file-eval', str(config_path),
        '--trained_ckpt', str(weight), '--val_datasets', 'MSLS_val'])
    try:
        evaluator.main(parsed)
        if len(recorded) != 1:
            raise RuntimeError('Official evaluation did not produce exactly one audit')
        write_json(args.output/'completed.json', {'complete': True, 'summary_sha256': sha(args.output/'summary.json')})
        print('AUDIT COMPLETE:', args.output, flush=True)
    finally:
        torch.hub.load = original_hub
        PairVPRNet.load_state_dict = original_load
        evaluator.get_test_recalls = original_report


if __name__ == '__main__':
    main()
