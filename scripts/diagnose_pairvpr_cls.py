"""Frozen last-CLS cross-attention intervention; fixed-pair margins, NOT recall."""
import argparse
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
from clip_token_drop import Cache, write
from pairvpr_cls_utils import remap_fraction, shuffled, selected_pairs, summarize


def check(ok, message):
    if not ok:
        raise ValueError(message)


class LastCLS:
    """Reuse original prefix; replace only last cross-attention CLS output."""
    def __init__(self, model):
        self.model = model
        self.block = model.dec_blocks[-1]
        self.saved = {}
        self.handles = [self.block.cross_attn.register_forward_hook(self.cross_hook),
                        self.block.norm3.register_forward_pre_hook(self.norm_hook)]

    def cross_hook(self, module, inputs, output):
        self.saved['cross'] = (inputs, output)

    def norm_hook(self, module, inputs):
        if inputs[0].shape[1] > 1:
            self.saved['residual'] = inputs[0]

    def score(self, fraction, alpha):
        a = self.block.cross_attn
        inputs, original = self.saved['cross']
        query, key, value = inputs[:3]
        b, n, c = key.shape
        h = a.num_heads
        q = a.projq(query[:, :1]).reshape(b, 1, h, c//h).transpose(1, 2)
        k = a.projk(key).reshape(b, n, h, c//h).transpose(1, 2)
        v = a.projv(value).reshape(b, n, h, c//h).transpose(1, 2)
        bias = torch.log1p(-alpha * fraction.reshape(1, 1, 1, n))
        weights = ((q @ k.transpose(-1, -2)) * a.scale + bias).softmax(-1)
        updated = a.proj_drop(a.proj((weights @ v).transpose(1, 2).reshape(b, 1, c)))
        x = self.saved['residual'][:, :1] + updated - original[:, :1]
        x = x + self.block.drop_path(self.block.mlp(self.block.norm3(x)))
        return self.model.classvprmodule(self.model.dec_norm(x)[:, 0])

    def close(self):
        for handle in self.handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-repo', type=Path, required=True)
    parser.add_argument('--msls-path', type=Path, default=Path('datasets/msls-val'))
    parser.add_argument('--audit', type=Path, default=Path('doc/pairvpr_official_paired_audit_v1'))
    parser.add_argument('--cache', type=Path, default=Path('.cache/clearclip_token_drop_msls_v1'))
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
    check(sha(repo/'pairvpr/configs/pairvpr_speed_local.yaml') == prov['config_sha256'],
          'Official config changed')
    source = Path(torch.hub.get_dir()) / 'facebookresearch_dinov2_main'
    check((source / 'hubconf.py').is_file(), 'Local DINO source missing')
    for name, digest in prov['official_python_sha256'].items():
        check(sha(repo / name) == digest, 'Official source changed: ' + name)
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
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cached = Cache(args.cache)
    identities = paths_normalized([r[0] for r in cached.records])
    expected = paths_normalized(list(db) + list(queries))
    check(len(set(identities)) == len(identities), 'Duplicate cache identity')
    lookup = {p: i for i, p in enumerate(identities)}
    check(set(expected) == set(identities), 'CLIP image coverage mismatch')
    # Both transforms must be direct full-image resize; no crop coordinates allowed.
    transform = str(dataset.input_transform)
    check('CenterCrop' not in transform and 'Random' not in transform and '322' in transform,
          'Unexpected official geometry: ' + transform)
    cases, excluded = selected_pairs(refined, gt)
    check(sum(c['group'] == 'reachable_error' for c in cases) == 31, 'Expected 31 reachable errors')
    check(sum(c['group'] == 'success' for c in cases) == 693, 'Expected 693 successes')
    args.output.mkdir(parents=True)
    write(args.output/'contract.json', {
        'alpha': .5, 'seeds': [11, 29, 47, 71, 101], 'cases': cases, 'excluded': excluded,
        'checkpoint_sha256': sha(weight), 'audit_sha256': sha(args.audit/'predictions.npz'),
        'cache_completed_sha256': cached.hash, 'transform': transform,
        'remap': 'area overlap of piecewise constant 20x20 coverage onto 23x23; not probabilities',
        'code': {p: sha(Path(__file__).parent/p) for p in
                 ['diagnose_pairvpr_cls.py', 'pairvpr_cls_utils.py']},
        'config': str(cfg), 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)})
    probe = LastCLS(model)
    maps = {}
    overlap = {}
    def extract(index):
        path = expected[index]
        ci = lookup[path]
        check(sha(root/path) == cached.image_hashes[ci], 'Image changed: ' + path)
        tensor, actual = dataset[index]
        check(int(actual) == index, 'Dataset order changed')
        dense, _ = model(tensor[None].cuda(), None, mode='global')
        check(dense.shape[1] == 529, 'Expected 23x23 features')
        if index not in maps:
            f = remap_fraction(cached.fractions[ci])
            variants = {'aligned': f}
            variants.update({f'shuffle_{s}': shuffled(f, path, s) for s in [11,29,47,71,101]})
            overlap[path] = {k: {'mean_absolute_difference': float(np.abs(v-f).mean()),
                                'soft_overlap': float(np.minimum(v,f).sum()/max(float(f.sum()),1e-12))}
                             for k,v in variants.items() if k != 'aligned'}
            maps[index] = {k: torch.tensor(v.flatten(), device='cuda', dtype=torch.float32)
                           for k, v in variants.items()}
        return dense
    records = []
    max_error = 0.
    try:
        with torch.inference_mode(), (args.output/'queries.jsonl').open('x', encoding='utf8') as stream:
            for ci, case in enumerate(cases):
                qi = case['query']; qindex = ndb + qi
                qfeat = extract(qindex)
                scores = {}
                for role in ['positive', 'negative']:
                    di = case[role]; dfeat = extract(di)
                    combined = {}
                    for first, second, memory in [(qfeat,dfeat,di),(dfeat,qfeat,qindex)]:
                        baseline = model(first, second, 'pairvpr')
                        restored = probe.score(maps[memory]['aligned'], 0.)
                        error = float((baseline-restored).abs().max())
                        max_error = max(max_error, error)
                        check(torch.allclose(baseline,restored,atol=2e-5,rtol=2e-5),
                              f'Zero intervention differs: {error}')
                        values = {'original': float(baseline.item())}
                        for name, fraction in maps[memory].items():
                            values[name] = float(probe.score(fraction,.5).item())
                        check(np.isfinite(list(values.values())).all(), 'Nonfinite scores')
                        for name, value in values.items():
                            combined[name] = combined.get(name,0.) + value
                    scores[role] = combined
                margins = {k: scores['positive'][k]-scores['negative'][k] for k in scores['positive']}
                check((margins['original'] >= -1e-4 if case['group']=='success'
                       else margins['original'] <= 1e-4), 'Original ordering differs from saved ranks')
                row = dict(case, scores=scores, margins=margins,
                           coverage={str(i): float(maps[i]['aligned'].mean()) for i in
                                     [qindex,case['positive'],case['negative']]})
                records.append(row)
                stream.write(json.dumps(row)+'\n'); stream.flush()
                if (ci+1)%10==0: print(f'Fixed pairs {ci+1}/{len(cases)}',flush=True)
    finally:
        probe.close()
    write(args.output/'summary.json', summarize(records))
    write(args.output/'mask_overlap.json', overlap)
    write(args.output/'completed.json', {'complete': True, 'queries':len(records),
          'max_zero_error':max_error, 'files':{p:sha(args.output/p) for p in
          ['contract.json','queries.jsonl','summary.json','mask_overlap.json']},
          'warning':'GT-selected fixed pairs; margin crossings are NOT R@1 gains.'})
    print('Saved diagnostic:',args.output)


if __name__ == '__main__':
    main()
