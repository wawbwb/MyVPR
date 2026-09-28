"""Fixed-checkpoint, query-only mechanism diagnosis. No training or tuning."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.candidate_set_screen import read, write, sha, complete, npz, load_npz
from scripts.adaptive_pair_budget import verified
from scripts.train_depth_query import seed, RU_SHA
from scripts.eval_condition_robustness import load_inference_model_from_ckpt
from src.models.depth_query import DepthQueryVPR
from src.models.competitive_query import MODES, install
from src.dataloaders.valid.mapillary_sls import MapillarySLSDataset
from src.dataloaders.valid.pittsburgh import PittsburghDataset

VARIANTS = ('ru',) + MODES
METRICS = ('attention_overlap_cosine', 'normalized_attention_entropy', 'slot_effective_rank')


def groups(reference, outcomes):
    reference, outcomes = np.asarray(reference, dtype=bool), np.asarray(outcomes, dtype=bool)
    if reference.shape != outcomes.shape or reference.ndim != 1: raise ValueError('Outcome shapes differ')
    return dict(all=np.ones_like(reference), corrections=~reference & outcomes,
                regressions=reference & ~outcomes, stable_correct=reference & outcomes,
                stable_error=~reference & ~outcomes)


def describe(values):
    values = np.asarray(values)
    return dict(n=len(values), mean=float(values.mean()) if len(values) else None,
                median=float(np.median(values)) if len(values) else None)


def summarize(metrics, drift, masks):
    result = {}
    for group, mask in masks.items():
        entry = {}
        for v, variant in enumerate(VARIANTS):
            blocks = []
            for b in range(metrics.shape[2]):
                report = {}
                for k, name in enumerate(METRICS):
                    values = metrics[mask, v, b, k]
                    delta = values - metrics[mask, 0, b, k]
                    report[name] = dict(value=describe(values), delta_vs_ru=describe(delta))
                blocks.append(report)
            entry[variant] = dict(blocks=blocks, descriptor_l2_vs_ru=describe(drift[mask, v]))
        result[group] = entry
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--runs', type=Path, default=Path('logs/competitive_query'))
    p.add_argument('--dataset-root', type=Path, default=Path('datasets'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    import fcntl
    from torchvision.transforms import v2 as T
    from tqdm import tqdm
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Expose physical GPU1 only')
    if not a.checkpoint.is_file() or sha(a.checkpoint) != RU_SHA: raise ValueError('RU checkpoint mismatch')
    seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runs, payloads, hashes = {}, {}, {}
    matched = None
    for mode in MODES:
        run = a.runs/(mode+'_screen_v1')
        verified(run)
        c = read(run/'contract.json')
        if c['checkpoint_sha256'] != RU_SHA or c['smoke']: raise ValueError('Wrong training contract')
        for file, digest in c['code'].items():
            if hashlib.sha256((ROOT/file).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != digest:
                raise ValueError('Training source changed: '+file)
        for file, digest in c['data'].items():
            if sha(Path(file)) != digest: raise ValueError('Metadata changed: '+file)
        c.pop('mode')
        if matched is not None and matched != c: raise ValueError('Unmatched arms')
        matched = c
        s = read(run/'summary.json')
        payload = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
        if payload['state']['epoch'] != s['best_epoch']: raise ValueError('Selected epoch mismatch')
        if payload['state']['contract_sha256'] != sha(run/'contract.json'): raise ValueError('Checkpoint contract mismatch')
        runs[mode] = (run, s['best_epoch'])
        payloads[mode] = payload['parameters']
        hashes[mode] = dict(checkpoint_sha256=sha(run/'best.pt'), manifest_sha256=sha(run/'completed.json'), epoch=s['best_epoch'])
    contract = dict(runs=hashes, ru_sha256=RU_SHA, variants=VARIANTS, metrics=METRICS,
        code_sha256=sha(Path(__file__)), scope='All MSLS/Pitts query images only, no database re-extraction. '
        'Outcomes from verified original full retrieval. Post-hoc descriptive association, not a causal proof. '
        'Primary grouping uses competitive-vs-RU outcomes for ALL arms; own-arm grouping secondary. No optimizer.')
    # JSON converts tuples to lists; normalize before resume comparison.
    contract['variants'] = list(VARIANTS)
    contract['metrics'] = list(METRICS)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    lock = (a.output.parent/(a.output.name+'.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if a.output.exists():
        if not a.resume or read(a.output/'contract.json') != contract: raise ValueError('Existing output/contract mismatch')
        if (a.output/'completed.json').exists(): verified(a.output); print('Already complete'); return
    else:
        a.output.mkdir()
        write(a.output/'contract.json', contract)
    transform = T.Compose([T.ToImage(), T.Resize((280, 280), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                           T.ToDtype(torch.float32, scale=True), T.Normalize([.485, .456, .406], [.229, .224, .225])])
    datasets = [MapillarySLSDataset(a.dataset_root/'msls-val', transform), PittsburghDataset(a.dataset_root/'pitts30k-val', transform)]
    visual = load_inference_model_from_ckpt(a.checkpoint, 'cpu').cuda().eval()
    model = DepthQueryVPR(visual, 'baseline').cuda().eval().requires_grad_(False)
    sample = datasets[0][datasets[0].num_references][0][None].cuda()
    with torch.inference_mode(): expected = visual(sample)
    adapters = install(model.aggregator, 'competitive')
    model.cuda().eval().requires_grad_(False)
    names = [name for name, _ in model.named_parameters() if name.endswith('.strength')]
    for mode, state in payloads.items():
        if set(state) != set(names): raise ValueError('Unexpected checkpoint parameters')
        for m, name in zip(adapters, names):
            value = state[name]
            if value.shape != m.strength.shape or not torch.isfinite(value).all(): raise ValueError('Bad strengths')
    with torch.inference_mode(): error = float((model(sample)-expected).abs().max())
    if error > 2e-6: raise ValueError('Baseline descriptor not reproduced')
    write(a.output/'preflight.json', dict(ru_max_abs_error=error, optimizer_steps=0))
    for m in adapters: m.collect = True
    report = {}
    for ds in datasets:
        outcomes, reference = {}, None
        for mode, (run, epoch) in runs.items():
            initial = read(run/f'{ds.dataset_name}_epoch00.json')['correct']
            if reference is not None and initial != reference: raise ValueError('Baseline outcomes differ')
            reference = initial
            outcomes[mode] = read(run/f'{ds.dataset_name}_epoch{epoch:02d}.json')['correct']
            if len(outcomes[mode]) != ds.num_queries: raise ValueError('Query count mismatch')
        file = a.output/(ds.dataset_name+'.npz')
        if file.exists():
            data = load_npz(file)
        else:
            measurements, drifts = [], []
            queries = Subset(ds, range(ds.num_references, len(ds)))
            with torch.inference_mode():
                for images, indices in tqdm(DataLoader(queries, batch_size=16, num_workers=4), desc='Mechanism '+ds.dataset_name):
                    layers = model.features(images.cuda())
                    batch, delta, ru = [], [], None
                    for mode in VARIANTS:
                        for m, name in zip(adapters, names):
                            m.mode = 'competitive' if mode == 'ru' else mode
                            if mode == 'ru': m.strength.zero_()
                            else: m.strength.copy_(payloads[mode][name].cuda())
                        descriptor = model.aggregate(layers)
                        if mode == 'ru': ru = descriptor
                        batch.append(torch.stack([m.last_metrics for m in adapters], 1).numpy())
                        delta.append((descriptor-ru).norm(dim=-1).cpu().numpy())
                    measurements.append(np.stack(batch, 1))
                    drifts.append(np.stack(delta, 1))
                    write(a.output/'progress.json', dict(phase='extract', dataset=ds.dataset_name,
                        done=int(indices[-1])+1-ds.num_references, total=ds.num_queries))
            data = dict(metrics=np.concatenate(measurements), descriptor_l2=np.concatenate(drifts))
            if not all(np.isfinite(v).all() for v in data.values()): raise ValueError('Nonfinite diagnostic')
            npz(file, **data)
        metrics, drift = data['metrics'], data['descriptor_l2']
        if metrics.shape != (ds.num_queries, 4, len(adapters), 3): raise ValueError('Bad diagnostic shape')
        shared = groups(reference, outcomes['competitive'])
        entry = dict(queries=ds.num_queries, primary_competitive_groups=summarize(metrics, drift, shared),
                     query_ids={k: np.flatnonzero(v).tolist() for k, v in shared.items()}, secondary_own_groups={})
        for mode in MODES:
            own = groups(reference, outcomes[mode])
            stats = summarize(metrics, drift, own)
            entry['secondary_own_groups'][mode] = {g: stats[g][mode] for g in stats}
        report[ds.dataset_name] = entry
    write(a.output/'summary.json', dict(datasets=report, scope=contract['scope'], verdict='MECHANISM_REVIEW_REQUIRED'))
    write(a.output/'progress.json', dict(phase='complete'))
    complete(a.output)
    print('COMPLETE', a.output, flush=True)


if __name__ == '__main__': main()
