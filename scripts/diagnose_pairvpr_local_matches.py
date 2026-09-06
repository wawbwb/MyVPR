"""Small offline case diagnostic: official pair scores and encoder MNN, not attention."""
import argparse
import base64
import html
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from audit_official_pairvpr import EXPECTED_WEIGHT, sha, paths_normalized


def check(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-repo', type=Path, required=True)
    parser.add_argument('--msls-path', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(not args.output.exists(), 'Output exists; choose a fresh directory')
    check(os.environ.get('CUDA_VISIBLE_DEVICES') == '1' and torch.cuda.is_available()
          and torch.cuda.device_count() == 1, 'Use CUDA_VISIBLE_DEVICES=1')
    repo, root = args.official_repo.resolve(), args.msls_path.resolve()
    summary = json.loads((args.audit / 'summary.json').read_text())
    done = json.loads((args.audit / 'completed.json').read_text())
    prov = json.loads((args.audit / 'provenance.json').read_text())
    check(done['complete'] and summary['complete'], 'Incomplete audit')
    check(sha(args.audit / 'summary.json') == done['summary_sha256'], 'Summary hash mismatch')
    check(sha(args.audit / 'predictions.npz') == summary['predictions_sha256'], 'Prediction hash mismatch')
    for name, digest in prov['index_sha256'].items():
        check(sha(root / name) == digest, f'Index mismatch: {name}')
    db = np.load(root / 'msls_val_dbImages.npy')
    queries = np.load(root / 'msls_val_qImages.npy')
    gt = np.load(root / 'msls_val_gt_25m.npy', allow_pickle=True)
    with np.load(args.audit / 'predictions.npz') as data:
        global_ids, refined = data['global_ids'].copy(), data['refined_ids'].copy()
    check(global_ids.shape[0] == refined.shape[0] == len(queries), 'Ranking length mismatch')
    weight = repo / 'trained_models/pairvpr-vitB.pth'
    check(sha(weight) == EXPECTED_WEIGHT, 'Unexpected checkpoint')
    source = Path(torch.hub.get_dir()) / 'facebookresearch_dinov2_main'
    check((source / 'hubconf.py').is_file(), 'Local DINO source missing')
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location('official_pair_eval_diagnostic', repo / 'pairvpr/eval/eval.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    cfg = official.get_cfg_from_args_eval(SimpleNamespace(
        config_file_eval=str(repo / 'pairvpr/configs/pairvpr_speed_local.yaml'),
        trained_ckpt=str(weight), dsetroot=str(root.parent), val_datasets=['MSLS_val']))
    check(cfg.augmentation.img_res == 322 and cfg.encoder.model_name == 'dinov2_vitb14_reg', 'Unexpected configuration')
    original = torch.hub.load

    def local_hub(repository, model, *a, **kw):
        check(str(repository).split(':')[0] == 'facebookresearch/dinov2', 'Unexpected hub request')
        kw.pop('source', None)
        kw['pretrained'] = False
        return original(str(source), model, *a, source='local', **kw)

    torch.hub.load = local_hub
    try:
        model = official.PairVPRNet(cfg)
    finally:
        torch.hub.load = original
    state = torch.load(weight, map_location='cpu', weights_only=True)
    official.interpolate_pos_embed(cfg, model, state)
    model.load_state_dict(state, strict=True)
    del state
    model = model.cuda().eval()
    dataset, ndb, nq, official_gt = official.dataset_getter.get_test_dataset(
        cfg, str(root.parent), str(repo), 'MSLS_val', cfg.augmentation.img_res)
    check(ndb == len(db) and nq == len(queries), 'Official dataset size mismatch')
    check(paths_normalized(dataset.images) == paths_normalized(list(db) + list(queries)), 'Image order mismatch')
    check(all(np.array_equal(np.sort(a), np.sort(b)) for a, b in zip(gt, official_gt))
          and len(gt) == len(official_gt), 'Official GT mismatch')
    # Fixed descriptive sample; never use these GT-selected pairs as a performance test.
    errors = [1, 34, 126, 131, 163, 164, 439]
    controls = [49, 325, 565]
    cache, records, links = {}, [], []
    args.output.mkdir(parents=True)

    def extract(index):
        if index not in cache:
            tensor, actual = dataset[index]
            check(int(actual) == index, 'Dataset index mismatch')
            with torch.inference_mode():
                dense, _ = model(tensor.unsqueeze(0).cuda(), None, mode='global')
            check(dense.shape[1] == 529 and torch.isfinite(dense).all().item(), 'Invalid 23x23 features')
            cache[index] = dense.detach().cpu()
        return cache[index].cuda()

    def embedded(path):
        file = (root / path).resolve()
        check(file.is_relative_to(root), 'Unsafe image path')
        return 'data:image/jpeg;base64,' + base64.b64encode(file.read_bytes()).decode()

    for qi in errors + controls:
        positives = set(map(int, gt[qi]))
        candidates = list(map(int, global_ids[qi, :100]))
        check(set(candidates) == set(map(int, refined[qi])), 'Candidate set mismatch')
        check((int(refined[qi, 0]) in positives) == (qi in controls), 'Case group changed')
        # All positives IN the actual candidate set, not just one convenient GT.
        chosen = sorted(set([int(refined[qi, 0]), int(global_ids[qi, 0])]
                            + [j for j in candidates if j in positives]))
        check(any(j in positives for j in chosen), 'No reachable GT')
        qfeat = extract(ndb + qi)
        cards = []
        for di in chosen:
            dfeat = extract(di)
            with torch.inference_mode():
                forward = float(model(qfeat, dfeat, 'pairvpr').item())
                reverse = float(model(dfeat, qfeat, 'pairvpr').item())
                similarity = F.normalize(qfeat[0].float(), dim=-1) @ F.normalize(dfeat[0].float(), dim=-1).T
                best = similarity.argmax(1)
                mutual = best.new_tensor(range(529)) == similarity.argmax(0)[best]
                qidx = torch.where(mutual)[0]
                values = similarity[qidx, best[qidx]]
                order = values.argsort(descending=True)
                matches = [(int(qidx[k]), int(best[qidx[k]]), float(values[k])) for k in order]
            check(np.isfinite([forward, reverse]).all(), 'Nonfinite score')
            record = {'query': qi, 'group': 'far_error' if qi in errors else 'success_control',
                      'db': di, 'gt': di in positives, 'refined_rank': list(refined[qi]).index(di) + 1,
                      'forward_score': forward, 'reverse_score': reverse, 'score_sum': forward + reverse,
                      'mnn_count': len(matches), 'matches': matches}
            records.append(record)
            lines = []
            for k, (a, b, cosine) in enumerate(matches[:40]):
                x, y = ((a % 23 + .5) * 14, (a // 23 + .5) * 14)
                xx, yy = (342 + (b % 23 + .5) * 14, (b // 23 + .5) * 14)
                lines.append(f'<line x1="{x}" y1="{y}" x2="{xx}" y2="{yy}" stroke="hsl({k*137%360},90%,55%)" stroke-width="1"/>')
            svg = f'<svg viewBox="0 0 664 322"><image href="{embedded(paths_normalized([queries[qi]])[0])}" width="322" height="322" preserveAspectRatio="none"/><image href="{embedded(paths_normalized([db[di]])[0])}" x="342" width="322" height="322" preserveAspectRatio="none"/>{"".join(lines)}</svg>'
            cards.append(f'<h2>db {di}; GT={di in positives}; saved rank={record["refined_rank"]}; bidirectional score={forward+reverse:.5f}; MNN={len(matches)}</h2>{svg}')
        page = f'q_{qi:04d}.html'
        (args.output / page).write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px sans-serif;margin:24px}svg{max-width:1000px;width:100%}</style>'
            + f'<h1>Query {qi}</h1><p>Encoder mutual nearest neighbors, NOT decoder attention or verified geometric correspondences. Top 40 displayed; all matches in JSON. Higher official score ranks first. GT-selected descriptive cases only.</p>'
            + ''.join(cards), encoding='utf8')
        links.append(f'<li><a href="{page}">Query {qi}</a></li>')
        print(f'Query {qi}: {len(chosen)} pairs complete', flush=True)
    (args.output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Local correspondence diagnostic</h1><ul>'+''.join(links)+'</ul>', encoding='utf8')
    (args.output / 'results.json').write_text(json.dumps(records, indent=2), encoding='utf8')
    (args.output / 'completed.json').write_text(json.dumps({'complete': True, 'pairs': len(records),
        'checkpoint_sha256': sha(weight), 'script_sha256': sha(Path(__file__)),
        'audit_summary_sha256': sha(args.audit / 'summary.json'),
        'official_source_sha256': {str(p.relative_to(repo)): sha(p) for p in (repo/'pairvpr').rglob('*.py')},
        'config': str(cfg), 'torch': torch.__version__,
        'warning': 'No reranking changes. MNN is not attention/geometry validation; no gain estimate from GT-selected pairs.'}, indent=2), encoding='utf8')
    print('Done:', args.output)


if __name__ == '__main__':
    main()
