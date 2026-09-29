"""Post-hoc clean GSV training-side diagnostic; never optimize or select models."""
import argparse
import hashlib
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@torch.inference_mode()
def margins(features, batch_ids, fixed_pairs=None, chunk=64):
    """All four views are queries; exclude self, other-place negatives only."""
    n = len(features)
    if n % 4 or n < 8 or len(batch_ids) != n:
        raise ValueError('Need complete four-view places and batch IDs')
    if not torch.isfinite(features).all():
        raise ValueError('Nonfinite descriptors')
    labels = torch.arange(n, device=features.device) // 4
    batches = torch.as_tensor(batch_ids, device=features.device)
    rows_out = []
    for start in range(0, n, chunk):
        q = torch.arange(start, min(start+chunk, n), device=features.device)
        rows = torch.arange(len(q), device=features.device)
        scores = features[q] @ features.T
        scores[rows, q] = -torch.inf
        pos_mask = labels[q, None] == labels[None]
        pos, pi = scores.masked_fill(~pos_mask, -torch.inf).max(1)
        neg, ni = scores.masked_fill(pos_mask, -torch.inf).max(1)
        local_neg = scores.masked_fill(pos_mask | (batches[q, None] != batches[None]), -torch.inf).amax(1)
        if not torch.isfinite(local_neg).all():
            raise ValueError('Each batch must contain at least two places')
        pred = scores.argmax(1)
        fixed = pos-neg
        if fixed_pairs is not None:
            pairs = torch.as_tensor(fixed_pairs[start:start+len(q)], device=features.device)
            fixed = scores[rows, pairs[:, 0]]-scores[rows, pairs[:, 1]]
        for j in range(len(q)):
            rows_out.append(dict(query=int(q[j]), correct=bool(labels[pred[j]] == labels[q[j]]),
                positive=int(pi[j]), negative=int(ni[j]), margin=float(pos[j]-neg[j]),
                batch_margin=float(pos[j]-local_neg[j]), fixed_pair_margin=float(fixed[j]),
                hardest_negative_in_batch=bool(batches[ni[j]] == batches[q[j]])))
    return rows_out


def summarize(baseline, current, threshold=.02):
    if len(baseline) != len(current) or any(a['query'] != b['query'] for a, b in zip(baseline, current)):
        raise ValueError('Unmatched query order')
    groups = dict(all=list(range(len(baseline))), errors=[], low_margin_correct=[], ordinary_correct=[])
    for i, row in enumerate(baseline):
        name = 'errors' if not row['correct'] else ('low_margin_correct' if row['margin'] <= threshold else 'ordinary_correct')
        groups[name].append(i)
    result = {}
    for name, indices in groups.items():
        if not indices:
            result[name] = dict(queries=0, places=0)
            continue
        delta = np.array([current[i]['margin']-baseline[i]['margin'] for i in indices])
        fixed = np.array([current[i]['fixed_pair_margin']-baseline[i]['margin'] for i in indices])
        result[name] = dict(queries=len(indices), places=len({i//4 for i in indices}),
            baseline_correct=sum(baseline[i]['correct'] for i in indices),
            correct=sum(current[i]['correct'] for i in indices),
            corrections=sum(not baseline[i]['correct'] and current[i]['correct'] for i in indices),
            regressions=sum(baseline[i]['correct'] and not current[i]['correct'] for i in indices),
            margin_delta_mean=float(delta.mean()), margin_delta_median=float(np.median(delta)),
            margin_improved_over_1e6=int((delta > 1e-6).sum()),
            fixed_pair_delta_mean=float(fixed.mean()),
            baseline_hardest_negative_in_batch=sum(baseline[i]['hardest_negative_in_batch'] for i in indices),
            baseline_batch_margin_nonpositive=sum(baseline[i]['batch_margin'] <= 0 for i in indices))
    return result


def main():
    from torch.utils.data import DataLoader
    from torchvision.transforms import v2 as T
    from tqdm import tqdm
    from scripts.candidate_set_screen import read, write, sha, complete
    from scripts.adaptive_pair_budget import verified
    from scripts.train_depth_query import seed, MatchedPlaces, RU_SHA
    from scripts.eval_condition_robustness import load_inference_model_from_ckpt
    from src.models.depth_query import DepthQueryVPR
    from src.models.second_order_query import install, MODES
    from src.dataloaders.train.gsv_cities import GSVCitiesDataset
    import fcntl

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, default=Path('logs/second_order_query'))
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, default=Path('datasets/gsv_cities'))
    p.add_argument('--output', type=Path, default=Path('doc/second_order_training_audit_v1'))
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Expose physical GPU1 only')
    if not a.checkpoint.is_file() or sha(a.checkpoint) != RU_SHA:
        raise ValueError('Wrong RU checkpoint')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    lock = (a.output.parent/(a.output.name+'.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    contracts, payloads, sources = {}, {}, {}
    canonical = None
    for mode in MODES:
        run = a.runs/(mode+'_screen_v1')
        verified(run)
        c = read(run/'contract.json')
        shared = {k: v for k, v in c.items() if k != 'mode'}
        if c['mode'] != mode or c['smoke'] or (canonical is not None and shared != canonical):
            raise ValueError('Unmatched training contracts')
        canonical = shared
        for f, h in c['code'].items():
            if hashlib.sha256((ROOT/f).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != h:
                raise ValueError('Training code changed: '+f)
        for f, h in c['data'].items():
            if 'Dataframes' in Path(f).parts and sha(Path(f)) != h:
                raise ValueError('GSV metadata changed: '+f)
        payload = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
        if payload['state']['epoch'] != 3 or payload['state']['contract_sha256'] != sha(run/'contract.json'):
            raise ValueError('Wrong last checkpoint')
        contracts[mode], payloads[mode] = c, payload['parameters']
        sources[mode] = dict(contract=sha(run/'contract.json'), checkpoint=sha(run/'last.pt'), complete=sha(run/'completed.json'))
    contract = dict(sources=sources, ru=RU_SHA, low_margin_threshold=.02,
        view_sampling_epoch=0, checkpoint_epoch=3, scope='Post-hoc training-side clean four-view retrieval; not independent validation; clean batch exposure is NOT actual augmented miner replay',
        code=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    if a.output.exists():
        if not a.resume or read(a.output/'contract.json') != contract:
            raise ValueError('Use a fresh output or matching --resume')
        if (a.output/'completed.json').exists():
            verified(a.output)
            print('Already complete')
            return
    else:
        a.output.mkdir()
        write(a.output/'contract.json', contract)
    seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    clean = T.Compose([T.ToImage(), T.Resize((280, 280), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToDtype(torch.float32, scale=True), T.Normalize([.485, .456, .406], [.229, .224, .225])])
    dataset = GSVCitiesDataset(dataset_path=a.dataset_root, cities='all', img_per_place=4, transform=clean)
    mapping = {str(v): i for i, v in enumerate(dataset.places_ids)}
    places = canonical['train_places']
    if len(places) != 4096 or set(places) & set(canonical['dev_places']):
        raise ValueError('Invalid train/dev separation')
    indices = [mapping[v] for v in places]
    order = np.random.default_rng(42).permutation(indices).tolist()
    batch_lookup = {i: j//16 for j, i in enumerate(order)}
    batches = np.repeat([batch_lookup[i] for i in indices], 4)
    write(a.output/'places.json', dict(places=places, view_sampling_epoch=0))
    names = ('ru',)+tuple(MODES)
    if not all((a.output/(name+'.json')).exists() for name in names):
        visual = load_inference_model_from_ckpt(a.checkpoint, 'cpu').cuda().eval()
        model = DepthQueryVPR(visual, 'baseline').cuda().eval().requires_grad_(False)
        adapters = install(model.aggregator, MODES[0])
        model.cuda().eval()
        params = {k: v for k, v in model.named_parameters() if v.requires_grad}
        for mode, payload in payloads.items():
            if set(payload) != set(params) or any(payload[k].shape != params[k].shape or not torch.isfinite(payload[k]).all() for k in params):
                raise ValueError('Invalid branch payload: '+mode)
        features = {name: [] for name in names}
        loader = DataLoader(MatchedPlaces(dataset, indices, 0), batch_size=8, num_workers=4)
        with torch.inference_mode():
            for step, (images, _) in enumerate(tqdm(loader, desc='GSV train: shared frozen backbone, four heads')):
                layers = model.features(images.flatten(0, 1).cuda())
                for name in names:
                    for module in adapters:
                        module.enabled = name != 'ru'
                        if name != 'ru': module.mode = name
                    if name != 'ru':
                        for k, v in payloads[name].items(): params[k].copy_(v)
                    features[name].append(model.aggregate(layers).cpu())
                if step % 16 == 0:
                    write(a.output/'progress.json', dict(phase='extract', places=min((step+1)*8, len(indices)), total=len(indices)))
        del layers, model, visual, adapters, params
        torch.cuda.empty_cache()
        fixed_pairs = None
        for name in names:
            write(a.output/'progress.json', dict(phase='all_training_place_retrieval', mode=name))
            f = torch.cat(features.pop(name)).cuda()
            result = margins(f, batches, fixed_pairs)
            write(a.output/(name+'.json'), result)
            if name == 'ru': fixed_pairs = [[r['positive'], r['negative']] for r in result]
            del f
    baseline = read(a.output/'ru.json')
    results = {name: summarize(baseline, read(a.output/(name+'.json'))) for name in names}
    write(a.output/'summary.json', dict(scope=contract['scope'], arms=results,
        note='Groups fixed by RU only; margin improvement does not imply R@1 improvement. Views within a place are dependent. No new training, no semantic input, no MSLS/Pitts extraction.'))
    write(a.output/'progress.json', dict(phase='complete'))
    complete(a.output)
    print('COMPLETE', a.output, flush=True)


if __name__ == '__main__':
    main()
