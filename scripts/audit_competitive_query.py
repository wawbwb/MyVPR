"""Read-only RU redundancy diagnosis. No optimization, no MSLS/Pitts tuning."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_depth_query import MatchedPlaces, seed, RU_SHA
from scripts.candidate_set_screen import write, read, sha, complete, npz, load_npz
from scripts.eval_condition_robustness import load_inference_model_from_ckpt
from src.dataloaders.train.gsv_cities import GSVCitiesDataset
from src.models.depth_query import DepthQueryVPR
from src.models.competitive_query import install


def describe(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {'n': 0, 'mean': None, 'median': None}
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
                p10=float(np.quantile(values, .1)), p90=float(np.quantile(values, .9)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, default=Path('datasets/gsv_cities'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    import fcntl
    from torchvision.transforms import v2 as T
    from tqdm import tqdm
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Expose physical GPU1 only')
    if not a.checkpoint.is_file() or sha(a.checkpoint) != RU_SHA:
        raise ValueError('Original RU checkpoint required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed(42)
    transform = T.Compose([T.ToImage(), T.Resize((280, 280), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                           T.ToDtype(torch.float32, scale=True), T.Normalize([.485, .456, .406], [.229, .224, .225])])
    dataset = GSVCitiesDataset(dataset_path=a.dataset_root, cities='all', img_per_place=4, transform=transform)
    order = sorted(range(len(dataset)), key=lambda i: hashlib.sha256(f'dsq-subset42:{dataset.places_ids[i]}'.encode()).hexdigest())
    if len(order) < 5120:
        raise ValueError('Insufficient eligible GSV places')
    partitions = dict(train_diagnostic=order[:1024], development=order[4096:5120])
    paths = ['src/models/competitive_query.py', 'src/models/depth_query.py', 'scripts/audit_competitive_query.py',
             'scripts/train_depth_query.py', 'src/dataloaders/train/gsv_cities.py',
             'src/models/aggregators/boq.py', 'scripts/eval_condition_robustness.py']
    contract = dict(checkpoint_sha256=RU_SHA, partitions={k: [str(dataset.places_ids[i]) for i in v] for k, v in partitions.items()},
                    metadata={str(f): sha(f) for f in sorted((a.dataset_root/'Dataframes').glob('*.csv'))},
                    code={f: hashlib.sha256((ROOT/f).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for f in paths},
                    protocol='Four deterministic views/place; each view queries remaining views within its partition; exclude self. '
                    'Development places excluded from DSQ subset, NOT guaranteed unseen during original RU training. '
                    'View-level outcomes are correlated; no iid significance claim. No weights updated.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    lock = (a.output.parent/(a.output.name+'.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if a.output.exists():
        if not a.resume or read(a.output/'contract.json') != contract:
            raise ValueError('Existing output or immutable contract differs')
        if (a.output/'completed.json').exists():
            for name, digest in read(a.output/'completed.json')['files'].items():
                if sha(a.output/name) != digest: raise ValueError('Completed artifact changed: '+name)
            print('Already complete', flush=True)
            return
    else:
        a.output.mkdir()
        write(a.output/'contract.json', contract)
    visual = load_inference_model_from_ckpt(a.checkpoint, 'cpu').cuda().eval()
    # Reuse only the verified frozen final-feature extraction; no DSQ branch runs.
    model = DepthQueryVPR(visual, 'baseline').cuda().eval().requires_grad_(False)
    sample = MatchedPlaces(dataset, [order[0]], 0)[0][0].cuda()
    with torch.inference_mode(): expected = visual(sample)
    modules = install(model.aggregator, 'competitive')
    model.cuda().eval()
    for module in modules: module.collect = True
    with torch.inference_mode():
        actual = model(sample)
        error = float((expected-actual).abs().max())
    if error > 2e-6 or not torch.isfinite(actual).all():
        raise ValueError(f'Zero-start real-image mismatch: {error}')
    write(a.output/'preflight.json', dict(descriptor_max_abs_error=error, dimension=actual.shape[1], optimizer_steps=0))
    del actual, expected, sample
    summary = {}
    for name, indices in partitions.items():
        folder = a.output/name
        folder.mkdir(exist_ok=True)
        features, metrics = [], []
        sampled = MatchedPlaces(dataset, indices, 0)
        for j in tqdm(range(len(indices)), desc='RU redundancy '+name):
            file = folder/f'{j:04d}.npz'
            if file.exists():
                data = load_npz(file)
            else:
                images, _ = sampled[j]
                with torch.inference_mode(): desc = model(images.cuda()).cpu().numpy()
                diagnostic = torch.stack([m.last_metrics for m in modules], 1).numpy()
                npz(file, descriptors=desc, metrics=diagnostic)
                data = dict(descriptors=desc, metrics=diagnostic)
            if (data['descriptors'].shape != (4, 12288) or data['metrics'].shape != (4, len(modules), 3)
                    or not all(np.isfinite(v).all() for v in data.values())):
                raise ValueError('Invalid shard: '+str(file))
            features.append(data['descriptors'])
            metrics.append(data['metrics'])
            if j % 32 == 0:
                write(a.output/'progress.json', dict(phase='extract', partition=name, done=j+1, total=len(indices)))
        features = torch.from_numpy(np.concatenate(features)).cuda()
        metric = np.concatenate(metrics)
        predictions = []
        for start in range(0, len(features), 64):
            scores = features[start:start+64] @ features.T
            rows = torch.arange(len(scores), device='cuda')
            scores[rows, rows+start] = -torch.inf
            predictions.extend(scores.argmax(1).cpu().tolist())
        predictions = np.asarray(predictions)
        correct = predictions//4 == np.arange(len(features))//4
        rows = []
        for i in range(len(features)):
            rows.append(dict(place_id=contract['partitions'][name][i//4], view=i%4,
                             predicted_place_id=contract['partitions'][name][int(predictions[i])//4],
                             correct=bool(correct[i]), by_block=metric[i].tolist()))
        write(a.output/f'{name}_per_view.json', rows)
        blocks = []
        for block in range(len(modules)):
            report = {}
            for k, label in enumerate(('attention_overlap_cosine', 'normalized_attention_entropy', 'slot_effective_rank')):
                values = metric[:, block, k]
                sorted_indices = np.argsort(values, kind='stable')
                quarters = np.array_split(sorted_indices, 4)
                report[label] = dict(all=describe(values), correct=describe(values[correct]), error=describe(values[~correct]),
                                     error_rate_by_ascending_quartile=[float((~correct[ix]).mean()) for ix in quarters])
            blocks.append(report)
        summary[name] = dict(places=len(indices), queries=len(correct), correct=int(correct.sum()), errors=int((~correct).sum()),
                             r1=float(correct.mean()), by_block=blocks)
        del features
    write(a.output/'summary.json', dict(partitions=summary, verdict='DIAGNOSTIC_ONLY_REVIEW_REQUIRED',
        metrics_order=['attention_overlap_cosine', 'normalized_attention_entropy', 'slot_effective_rank'],
        scope=contract['protocol'], decision='No automatic training gate; association is not causation. '
        'Inspect error counts and consistency across partitions before authorizing matched training.'))
    write(a.output/'progress.json', dict(phase='complete'))
    complete(a.output)
    print(summary, flush=True)


if __name__ == '__main__': main()
