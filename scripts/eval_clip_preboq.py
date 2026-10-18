"""Fixed-final RGB-only development evaluation and fresh pre-BoQ probes."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts import clip_preboq_study as study
from scripts import eval_clip_key_study as metrics
from src.models.clip_preboq import ARMS, PreBoQ
from src.models.official_boq import load_official
from src.semantic_region_protocol import owned_run, read, write, sha, state_hash, save_npz

VARIANTS = ('official', *ARMS)


def evaluate(c, args):
    plan, targets, _, _ = study.prerequisites(c)
    datasets = metrics.populations(c, plan)
    initial = None; inventory = {}
    for arm in ARMS:
        run = study.path(c, 'runs')/arm
        if study.verified(run) != study.contract_for(c, arm, 'train') or not all(read(run/'summary.json')['checks'].values()):
            raise ValueError('Training incomplete: ' + arm)
        current = read(run/'initial.json')
        if initial is not None and current != initial: raise ValueError('Unmatched initialization')
        initial = current
        inventory[arm] = dict(checkpoint=sha(run/'last.pt'), contract=sha(run/'contract.json'))
    base = study.identity(c)
    contract = dict(stage='rgb_only_fixed_final_eval', **base, training=inventory, initial=initial,
        datasets={d['tag']: dict(root=str(d['root'].resolve()), references=d['references'], queries=d['queries'],
            image_order_sha256=metrics.order_digest(d['names']), ground_truth_sha256=metrics.gt_digest(d['ground_truth']),
            metadata={p.name: sha(p) for p in sorted(d['root'].glob('*.npy'))}) for d in datasets},
        selection='Fixed last checkpoint, no heldout best-epoch/weight/lambda selection',
        scope='Exploratory single-seed development evaluation. No blind test access. No CLIP at inference.')
    output = study.path(c, 'runs')/'evaluation'
    with owned_run(output, contract, args.resume):
        if (output/'completed.json').exists(): study.verified(output); print('Evaluation complete'); return
        study.old.gpu(c)
        official = load_official(study.path(c, 'official_cache'), study.path(c, 'dino_source'), c['device'])
        frozen = state_hash(official)
        if frozen != initial['official']: raise ValueError('Official model differs')
        models = {}
        for arm in ARMS:
            model = PreBoQ(official, hidden=c['hidden'], gain=c['adapter_gain']).to(c['device'])
            saved = torch.load(study.path(c, 'runs')/arm/'last.pt', map_location='cpu', weights_only=True)
            active = {n: p for n, p in model.named_parameters() if p.requires_grad}
            if (saved['state']['step'] != len(plan['schedule']) or set(active) != set(saved['student'])
                    or saved['state']['contract_sha256'] != inventory[arm]['contract']):
                raise ValueError('Not the registered final checkpoint')
            with torch.no_grad():
                for n, p in active.items():
                    v = saved['student'][n]
                    if p.shape != v.shape or not torch.isfinite(v).all(): raise ValueError('Bad adapter tensor')
                    p.copy_(v.to(p.device))
            if state_hash(model.adapter) != read(study.path(c, 'runs')/arm/'summary.json')['adapter_sha256']:
                raise ValueError('Adapter checksum differs')
            models[arm] = model.requires_grad_(False).eval()
        for d in datasets:
            tag = d['tag']; target = output/(tag+'.json'); ready = output/(tag+'.ready.json')
            is_gsv = tag == 'gsv-adaptation-holdout'
            if ready.exists():
                record = read(ready)
                if sha(target) != record['report_sha256']: raise ValueError('Corrupt evaluation report')
                metrics.validate_completed_dataset(read(target), d, contract)
                if is_gsv and sha(output/'fresh_mean_inputs.npz') != record['probe_sha256']:
                    raise ValueError('Corrupt probe inputs')
                continue
            started = time.monotonic()
            if is_gsv:
                names = [r['path'] for r in plan['images']]
                hashes = [r['image_sha256'] for r in plan['images']]
                destination = {n: i for i, n in enumerate(d['names'])}
                mapping = np.asarray([destination.get(n, -1) for n in names], np.int64)
                means = {a: np.empty((len(names), 768), np.float32) for a in VARIANTS}
            else:
                names, hashes = d['names'], d['hashes']; mapping = np.arange(len(names)); means = None
            dataset = metrics.DualImages(d['root'], names, hashes)
            loader = DataLoader(dataset, batch_size=16, num_workers=c['workers'], shuffle=False)
            descriptors = {a: np.empty((len(d['names']), 12288), np.float32) for a in VARIANTS}
            contents = {}
            with torch.inference_mode():
                for batch, (images, _, indices, digests) in enumerate(tqdm(loader, desc='RGB-only '+tag)):
                    if indices.tolist() != list(range(batch*16, batch*16+len(images))): raise ValueError('Order mismatch')
                    for i, digest in zip(indices.tolist(), digests):
                        if mapping[i] >= 0: contents[int(mapping[i])] = digest
                    for start in range(0, len(images), c['microbatch']):
                        stop = start+c['microbatch']; ids = indices[start:stop].numpy()
                        features = official.backbone(images[start:stop].to(c['device']))
                        values = dict(official=(official.aggregator(features)[0], features.mean((2, 3))),
                                      **{a: m.describe(features) for a, m in models.items()})
                        keep = mapping[ids] >= 0
                        for arm, (value, pooled) in values.items():
                            if not torch.isfinite(value).all() or float((value.norm(dim=1)-1).abs().max()) > 1e-4:
                                raise ValueError('Invalid descriptor')
                            if is_gsv: means[arm][ids] = pooled.cpu().numpy()
                            if keep.any(): descriptors[arm][mapping[ids[keep]]] = value.cpu().numpy()[keep]
                    write(output/'progress.json', dict(phase='extracting', dataset=tag, images=int(indices[-1])+1,
                        total=len(dataset), elapsed_seconds=time.monotonic()-started, clip_inference=False))
                reports = {}
                for arm in VARIANTS:
                    db = torch.from_numpy(descriptors[arm][:d['references']]).to(c['device'])
                    queries = descriptors[arm][d['references']:]
                    rankings = np.concatenate([(torch.from_numpy(queries[s:s+16]).to(c['device'])@db.T)
                        .topk(20, dim=1).indices.cpu().numpy() for s in range(0, len(queries), 16)])
                    reports[arm] = metrics.retrieval_report(rankings, d['ground_truth'], d['references'])
                    del db
            report = dict(tag=tag, dataset=contract['datasets'][tag], variants=reports,
                comparison=metrics.compare_reports(reports), elapsed_seconds=time.monotonic()-started,
                image_content_sha256=metrics.order_digest([n+':'+contents[i] for i, n in enumerate(d['names'])]),
                scope=contract['scope'])
            metrics.validate_completed_dataset(report, d, contract); write(target, report)
            record = dict(report_sha256=sha(target))
            if is_gsv:
                save_npz(output/'fresh_mean_inputs.npz', **means)
                record['probe_sha256'] = sha(output/'fresh_mean_inputs.npz')
            write(ready, record); print(tag, report['comparison'], flush=True)
            del descriptors, means, loader, dataset
        fresh = output/'fresh_probes.json'
        probe_contract = dict(inputs_sha256=sha(output/'fresh_mean_inputs.npz'), seed=c['seed']+9101,
            epochs=c['probe_epochs'], hidden=c['predictor_hidden'], target='True CLIP variation for all four groups')
        if fresh.exists():
            prior = read(fresh)
            if prior['contract'] != probe_contract: raise ValueError('Probe identity differs')
            probes = prior['results']
        else:
            write(output/'progress.json', dict(phase='fresh_cpu_probes'))
            pc = dict(c, hidden=c['predictor_hidden']); probes = {}
            with np.load(output/'fresh_mean_inputs.npz', allow_pickle=False) as inputs:
                for arm in VARIANTS:
                    probes[arm] = study.diagnosis.fit(pc, inputs[arm], targets, plan['train_rows'], c['seed']+9101)[0]
            write(fresh, dict(contract=probe_contract, results=probes,
                scope='Fresh independently initialized probe, not the training adversary; lower R2 alone is not benefit.'))
        checks = dict(official_unchanged=state_hash(official) == frozen,
            checkpoints_unchanged=all(sha(study.path(c, 'runs')/a/'last.pt') == inventory[a]['checkpoint'] for a in ARMS),
            innovation1_unchanged=study.old.source_hashes(study.old.PROTECTED) == base['source']['base']['protected_sources'])
        if not all(checks.values()): raise ValueError('Evaluation integrity failure')
        summary = dict(results={d['tag']: read(output/(d['tag']+'.json'))['comparison'] for d in datasets},
            fresh_probes=probes, checks=checks, blind_test_used=False, clip_inference=False, scope=contract['scope'])
        write(output/'summary.json', summary); write(output/'progress.json', dict(phase='complete'))
        study.finish(output); print(summary, flush=True)
