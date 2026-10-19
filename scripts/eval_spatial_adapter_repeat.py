"""Per-seed fixed-final retrieval plus descriptive paired-seed aggregation."""
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scripts import spatial_adapter_repeat as study
from scripts import eval_clip_key_study as metrics
from scripts.semantic_region_study import transform
from src.semantic_region_protocol import owned_run, read, write, sha, state_hash

VARIANTS = ('official', 'spatial', 'channel')


class RGBImages(Dataset):
    def __init__(self, root, names, hashes):
        self.root = Path(root).resolve(); self.names = names; self.hashes = hashes
        self.transform = transform(322)
        if hashes is not None and len(hashes) != len(names): raise ValueError('Hash population differs')

    def __len__(self): return len(self.names)

    def __getitem__(self, i):
        p = (self.root/self.names[i]).resolve()
        if not p.is_relative_to(self.root) or not p.is_file(): raise ValueError('Missing/unsafe image')
        digest = sha(p)
        if self.hashes is not None and digest != self.hashes[i]: raise ValueError('Sealed GSV image changed')
        with Image.open(p) as image: pixels = self.transform(image.convert('RGB'))
        return pixels, i, digest


def comparisons(reports):
    if set(reports) != set(VARIANTS): raise ValueError('Unexpected variants')
    return dict(correct={a: r['correct'] for a, r in reports.items()},
        recall={a: r['recall'] for a, r in reports.items()},
        spatial_vs_official=metrics.paired_report(reports['official'], reports['spatial']),
        channel_vs_official=metrics.paired_report(reports['official'], reports['channel']),
        spatial_vs_channel=metrics.paired_report(reports['channel'], reports['spatial']))


def validate(report, d, contract):
    if report['tag'] != d['tag'] or report['dataset'] != contract['datasets'][d['tag']]:
        raise ValueError('Dataset provenance differs')
    if set(report['variants']) != set(VARIANTS): raise ValueError('Variant inventory differs')
    for arm, r in report['variants'].items():
        if metrics.retrieval_report(r['predictions'], d['ground_truth'], d['references']) != r:
            raise ValueError('Outcomes cannot reproduce GT/rankings: '+arm)
    if comparisons(report['variants']) != report['comparison']: raise ValueError('Paired counts differ')


def evaluate(c, a):
    seed = a.seed; plan = study.open_plan(c, seed); datasets = metrics.populations(c, plan)
    run = study.path(c, 'runs')/f'train_seed{seed}'
    if study.verified(run) != study.contract_for(c, seed, 'train') or not all(read(run/'summary.json')['checks'].values()):
        raise ValueError('Training is incomplete or differs')
    initial = read(run/'initial.json'); base = study.identity(c)
    contract = dict(stage='fixed_final_eval', seed=seed, **base,
        training_checkpoint_sha256=sha(run/'last.pt'), training_contract_sha256=sha(run/'contract.json'),
        datasets={d['tag']: dict(root=str(d['root'].resolve()), references=d['references'], queries=d['queries'],
            image_order_sha256=metrics.order_digest(d['names']), ground_truth_sha256=metrics.gt_digest(d['ground_truth']),
            metadata={p.name: sha(p) for p in sorted(d['root'].glob('*.npy'))}) for d in datasets},
        scope='Single-stage RGB retrieval. Fixed final epoch. No CLIP, no adversary, no blind test access.')
    output = study.path(c, 'runs')/f'eval_seed{seed}'
    with owned_run(output, contract, a.resume):
        if (output/'completed.json').exists(): study.verified(output); print('Already evaluated'); return
        study.key.gpu(c); models = study.build(c, seed); official = models['spatial'].visual
        frozen = state_hash(official)
        if frozen != initial['official']: raise ValueError('Official visual state differs')
        saved = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
        if saved['state']['step'] != len(plan['schedule']) or saved['state']['contract_sha256'] != contract['training_contract_sha256']:
            raise ValueError('Checkpoint is not registered final epoch')
        for arm, m in models.items():
            m.adapter.load_state_dict(saved['adapters'][arm], strict=True)
            if state_hash(m.adapter) != read(run/'summary.json')['adapter_sha256'][arm]: raise ValueError('Adapter state differs')
            m.requires_grad_(False).eval()
        for d in datasets:
            tag = d['tag']; target = output/(tag+'.json'); ready = output/(tag+'.ready.json')
            if ready.exists():
                if sha(target) != read(ready)['sha256']: raise ValueError('Corrupt report')
                validate(read(target), d, contract); continue
            started = time.monotonic()
            dataset = RGBImages(d['root'], d['names'], d['hashes'])
            loader = DataLoader(dataset, batch_size=16, num_workers=c['workers'], shuffle=False)
            arrays = {arm: np.empty((len(dataset), 12288), np.float32) for arm in VARIANTS}
            contents = []
            with torch.inference_mode():
                for batch, (images, ids, hashes) in enumerate(tqdm(loader, desc=f'seed{seed} '+tag)):
                    if ids.tolist() != list(range(batch*16, batch*16+len(images))): raise ValueError('Extraction order differs')
                    contents.extend(hashes)
                    for start in range(0, len(images), c['microbatch']):
                        stop = start+c['microbatch']; indices = ids[start:stop].numpy()
                        features = official.backbone(images[start:stop].to(c['device']))
                        values = dict(official=official.aggregator(features)[0],
                                      **{arm: m.describe(features)[0] for arm, m in models.items()})
                        for arm, value in values.items():
                            if not torch.isfinite(value).all() or float((value.norm(dim=1)-1).abs().max()) > 1e-4:
                                raise ValueError('Invalid descriptor')
                            arrays[arm][indices] = value.cpu().numpy()
                    write(output/'progress.json', dict(phase='extracting', seed=seed, dataset=tag,
                        images=int(ids[-1])+1, total=len(dataset), elapsed_seconds=time.monotonic()-started))
                reports = {}
                for arm in VARIANTS:
                    db = torch.from_numpy(arrays[arm][:d['references']]).to(c['device'])
                    queries = arrays[arm][d['references']:]
                    rankings = np.concatenate([(torch.from_numpy(queries[s:s+16]).to(c['device'])@db.T)
                        .topk(20, dim=1).indices.cpu().numpy() for s in range(0, len(queries), 16)])
                    reports[arm] = metrics.retrieval_report(rankings, d['ground_truth'], d['references']); del db
            report = dict(tag=tag, seed=seed, dataset=contract['datasets'][tag], variants=reports,
                comparison=comparisons(reports), image_content_sha256=metrics.order_digest(
                    [n+':'+v for n, v in zip(d['names'], contents)]), elapsed_seconds=time.monotonic()-started)
            validate(report, d, contract); write(target, report); write(ready, dict(sha256=sha(target)))
            print(tag, report['comparison']['correct'], flush=True); del arrays, loader, dataset
        checks = dict(official_unchanged=state_hash(official) == frozen,
            checkpoint_unchanged=sha(run/'last.pt') == contract['training_checkpoint_sha256'],
            innovation1_unchanged=study.key.source_hashes(study.key.PROTECTED) == base['protected_sources'])
        if not all(checks.values()): raise ValueError('Evaluation integrity failure')
        write(output/'summary.json', dict(seed=seed, results={d['tag']: read(output/(d['tag']+'.json'))['comparison'] for d in datasets},
            checks=checks, blind_test_used=False, scope=contract['scope']))
        write(output/'progress.json', dict(phase='complete')); study.finish(output)


def delta_summary(values):
    if len(values) != 3 or not all(np.isfinite(values)): raise ValueError('Need three finite paired seed differences')
    return dict(seed_delta_r1_pp=list(map(float, values)), mean_delta_r1_pp=statistics.mean(values),
        sample_std_r1_pp=statistics.stdev(values), min_delta_r1_pp=min(values), max_delta_r1_pp=max(values),
        positive_seeds=sum(v > 0 for v in values), tied_seeds=sum(v == 0 for v in values),
        negative_seeds=sum(v < 0 for v in values), all_three_positive=all(v > 0 for v in values))


def aggregate(c, a):
    root = study.path(c, 'runs'); inventory = {}; reports = {}
    for seed in c['seeds']:
        run = root/f'eval_seed{seed}'; contract = study.verified(run)
        if contract['seed'] != seed or contract['stage'] != 'fixed_final_eval' or contract['config'] != c:
            raise ValueError('Evaluation contract differs')
        if contract['code'] != study.key.source_hashes(study.SOURCES): raise ValueError('Evaluation sources differ')
        reports[seed] = read(run/'summary.json'); inventory[str(seed)] = sha(run/'completed.json')
        if not all(reports[seed]['checks'].values()): raise ValueError('Evaluation integrity check failed')
    contract = dict(stage='three_seed_aggregation', **study.identity(c), evaluation_completed_sha256=inventory,
        primary=c['primary_dataset'], policy='Report EVERY seed and dataset. No selecting best seed or treating repeated queries as independent.')
    output = root/'aggregate'
    with owned_run(output, contract, a.resume):
        if (output/'completed.json').exists(): study.verified(output); print('Aggregation complete'); return
        results = {}
        for tag in ('gsv-adaptation-holdout', 'msls-val', 'pitts30k-val'):
            raw = {str(seed): reports[seed]['results'][tag] for seed in c['seeds']}
            # Same images/GT/order and official baseline across seeds are mandatory.
            originals = [read(root/f'eval_seed{seed}'/(tag+'.json')) for seed in c['seeds']]
            reference = originals[0]
            if any(d['dataset'] != reference['dataset'] or d['image_content_sha256'] != reference['image_content_sha256']
                   or d['variants']['official'] != reference['variants']['official'] for d in originals[1:]):
                raise ValueError('Baseline/data are not comparable across seeds')
            results[tag] = dict(per_seed=raw, **{name: delta_summary([reports[s]['results'][tag][name]['delta_r1_pp']
                for s in c['seeds']]) for name in ('spatial_vs_official', 'channel_vs_official', 'spatial_vs_channel')})
        primary = results[c['primary_dataset']]
        write(output/'summary.json', dict(seeds=c['seeds'], primary_dataset=c['primary_dataset'], results=results,
            primary_screen=dict(spatial_beats_official_all_three=primary['spatial_vs_official']['all_three_positive'],
                                spatial_beats_channel_all_three=primary['spatial_vs_channel']['all_three_positive']),
            limits=['Three seeds and reused development sets are not independent test/generalization proof.',
                'Kernel ablation differs by 512 parameters (0.513%); not exactly equal capacity.',
                'Comparison with channel control addresses spatial interaction beyond extra adaptation capacity, not novelty against all papers.',
                'No CLIP/adversary effect is being tested. No blind test or best-epoch selection.']))
        write(output/'progress.json', dict(phase='complete')); study.finish(output)
        print(read(output/'summary.json'), flush=True)
