"""Three fresh matched residual rules; existing sealed conv runs stay read-only."""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_slgd import read, write, sha, code_hash, verify, seal, seed, save, restore, parameter_hash
from scripts.slgd_mid_screen import open_contract, verify_cache_shards, current_code
from scripts.audit_slgd_hard import PlannedPlaces, collate
from scripts.train_hcen_vpr import PlaceBatches, tensor_hash
from scripts.hcen_plain_base import context, load_plain
from scripts.hcen_module_ablation import wait_gpu
from scripts.conv_fusion_controls import references, SOURCES as OLD_SOURCES
from src.models.hcen_followup import build_model as historical_model
from src.hcen_followup_protocol import POLICY as FOLLOWUP_POLICY
from src.hcen_plain_protocol import PLAIN_CHECKPOINT, PLAIN_SHA, canonical_features
from src.hcen_plain_replication import SEEDS
from src.hcen_aux_ablation import gradient_diagnostic
from src.models.conv_residual_ablation import (
    ARMS, PARAMETERS, DISABLED_AUXILIARY_PARAMETERS, build_model,
    module_gradient_status, image_diagnostics,
)
from src.conv_residual_protocol import (
    POLICY, effective_policy, training_schedule, matched_contract, arm_contract,
    diagnostic_summary, summarize, aggregate_reports,
)
from src.slgd_mid_supervision import gps_filtered_retrieval
from src.slgd_hard_protocol import distance_metres
from src.hcen_protocol import benchmark_transform

SOURCES = tuple(dict.fromkeys((
    'scripts/conv_residual_ablation.py', 'src/models/conv_residual_ablation.py',
    'src/conv_residual_protocol.py', 'scripts/run_conv_residual_ablation.sh',
    'tests/test_conv_residual_ablation.py', *OLD_SOURCES,
)))


def forward_batch(model, images, training=False):
    descriptors, references_, diagnostics, deltas = [], [], [], []
    aux = [[], []]
    for chunk in images.cuda().split(4):
        features = canonical_features(model.features(chunk))
        with torch.no_grad():
            reference = model.readout(features[-1])
        value = (checkpoint(lambda a, b, c: model.aggregate(a, b, c, True), *features,
                            use_reentrant=False) if training else model.aggregate(*features, auxiliary=False))
        descriptor, residual = value[0], value[-1]
        if descriptor.ndim != 2 or descriptor.shape[1] != 12288 or float((descriptor.norm(dim=1) - 1).abs().max()) > 1e-4:
            raise ValueError('Invalid/nonunit plain-BoQ descriptor')
        # Collect ONCE, outside the checkpoint. Backward recomputation is not a
        # second observed image and never mutates a statistics accumulator.
        diagnostics.extend(image_diagnostics(features[-1], model.neck.last_raw,
                           residual.detach(), descriptor.detach(), reference, model.residual_arm))
        model.neck.last_raw = None
        descriptors.append(descriptor); references_.append(reference)
        if training:
            aux[0].append(value[1]); aux[1].append(value[2])
        last_norm = features[-1].flatten(1).norm(dim=1)
        deltas.append(residual.detach().flatten(1).norm(dim=1) / last_norm.clamp_min(1e-8))
    return (torch.cat(descriptors), torch.cat(references_),
            tuple(torch.cat(x) for x in aux) if training else (), torch.cat(deltas), diagnostics)


def diagnostics_files(output, dataset, rows, query_indices):
    """Atomic one-row-per-image CSV plus exact per-image distribution summaries."""
    indices = np.asarray(query_indices, dtype=np.int64)
    if len(rows) == 0 or len(indices) == 0 or (indices < 0).any() or (indices >= len(rows)).any():
        raise ValueError('Invalid diagnostic query indices')
    query_set = set(indices.tolist())
    path = output / (dataset + '_diagnostics.csv'); temporary = path.with_suffix('.tmp')
    keys = list(rows[0])
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image_index', 'role', *keys])
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(dict(image_index=index, role='query' if index in query_set else 'reference', **row))
    temporary.replace(path)
    value = dict(all_images=diagnostic_summary(rows),
                 query_images=diagnostic_summary([rows[i] for i in indices]))
    references_ = [row for i, row in enumerate(rows) if i not in query_set]
    if references_:
        value['reference_images'] = diagnostic_summary(references_)
    write(output / (dataset + '_diagnostics.json'), value)


def evaluate_gsv(model, data, heldout, output, workers, baseline):
    from tqdm import tqdm
    plan = heldout['plan']; rows = plan['rows']
    descriptor = np.empty((rows, 12288), np.float32); reference = np.empty_like(descriptor)
    coordinates = np.empty((rows, 2), np.float64); diagnostic_rows = [None] * rows
    loader = DataLoader(PlannedPlaces(data, plan), batch_size=16, num_workers=workers, collate_fn=collate)
    with torch.inference_mode():
        for batch, (images, indices, coords) in enumerate(tqdm(loader, desc='Residual GSV full gallery')):
            write(output / 'progress.json', dict(phase='gsv_evaluation', batch=batch, total=len(loader)))
            d, r, _, _, metrics = forward_batch(model, images)
            ids = indices.numpy(); descriptor[ids] = d.cpu().numpy(); reference[ids] = r.cpu().numpy()
            coordinates[ids] = coords.numpy()
            for index, row in zip(ids, metrics):
                diagnostic_rows[index] = row
    if any(row is None for row in diagnostic_rows):
        raise ValueError('Missing GSV diagnostic image')
    q = np.asarray(plan['query_rows']); g = np.asarray(plan['gallery_rows'])
    positive = np.asarray(plan['positive_gallery_indices']); result = {}
    for name, features in [('baseline', reference), ('student', descriptor)]:
        gallery = torch.from_numpy(features[g]).cuda(); predictions = []
        for start in range(0, len(q), 16):
            rows_ = q[start:start + 16]
            scores = torch.from_numpy(features[rows_]).cuda() @ gallery.T
            allowed = distance_metres(coordinates[rows_], coordinates[g]) >= 25
            allowed[np.arange(len(rows_)), positive[start:start + 16]] = True
            scores.masked_fill_(~torch.from_numpy(allowed).cuda(), -torch.inf)
            predictions.extend(scores.argmax(1).cpu().tolist())
        result[name] = dict(predictions=predictions, outcomes=(np.asarray(predictions) == positive).tolist())
    if result['baseline'] != read(baseline / 'gsv_final.json')['baseline']:
        raise ValueError('GSV baseline differs from sealed direct forward')
    result['descriptor_drift_mean'] = float(np.mean(1 - (descriptor * reference).sum(1)))
    write(output / 'gsv_final.json', result)
    diagnostics_files(output, 'gsv', diagnostic_rows, q)


def evaluate_benchmarks(model, output, workers, baseline):
    from tqdm import tqdm
    from src.dataloaders.valid.mapillary_sls import MapillarySLSDataset
    from src.dataloaders.valid.pittsburgh import PittsburghDataset
    transform = benchmark_transform()
    for dataset in [MapillarySLSDataset(Path('datasets/msls-val'), transform),
                    PittsburghDataset(Path('datasets/pitts30k-val'), transform)]:
        name = dataset.dataset_name; path = output / (name + '_final.json')
        metric_path = output / (name + '_diagnostics.json'); csv_path = output / (name + '_diagnostics.csv')
        # Evaluation resumes only from the same contract/final model. Incomplete
        # diagnostic sets cause a full re-extraction, not a silent partial report.
        if path.exists() and metric_path.exists() and csv_path.exists():
            if read(path)['baseline'] != read(baseline / path.name)['baseline']:
                raise ValueError('Saved benchmark baseline differs')
            continue
        students, references_, metrics = [], [], []
        ordered_metrics = [None] * len(dataset)
        with torch.inference_mode():
            loader = DataLoader(dataset, batch_size=16, num_workers=workers)
            for batch, (images, indices) in enumerate(tqdm(loader, desc=name + ' residual diagnostics')):
                write(output / 'progress.json', dict(phase='benchmark', dataset=name, batch=batch, total=len(loader)))
                d, r, _, _, metrics = forward_batch(model, images)
                students.append(d.cpu()); references_.append(r.cpu())
                # Loader is sequential for descriptor ordering; verify it below.
                expected = list(range(batch * 16, batch * 16 + len(images)))
                if indices.tolist() != expected:
                    raise ValueError('Unexpected benchmark image order')
                for index, row in zip(expected, metrics):
                    ordered_metrics[index] = row
        if any(row is None for row in ordered_metrics):
            raise ValueError('Missing benchmark diagnostic image')
        result = {}
        for key, parts in [('baseline', references_), ('student', students)]:
            all_ = torch.cat(parts).cuda(); db = all_[:dataset.num_references]
            predictions = np.concatenate([(q @ db.T).topk(20, dim=1).indices.cpu().numpy()
                                          for q in all_[dataset.num_references:].split(16)])
            hits = np.asarray([[bool(np.isin(row[:k], gt).any()) for k in (1, 5, 20)]
                               for row, gt in zip(predictions, dataset.ground_truth)])
            result[key] = dict(predictions=predictions.tolist(), outcomes=hits[:, 0].tolist(),
                               correct=int(hits[:, 0].sum()), recall=hits.mean(0).tolist())
            del all_, db
        if result['baseline'] != read(baseline / path.name)['baseline']:
            raise ValueError('Benchmark baseline differs from sealed direct forward')
        result['preprocessing'] = 'Historical tensor bicubic280 FP32 microbatch4; same sealed baseline pixels'
        write(path, result)
        diagnostics_files(output, name, ordered_metrics, np.arange(dataset.num_references, len(dataset)))


def training_contract(args, provenance, identity):
    contract, data, heldout, cache = context(args, 'hierarchy', args.stage)
    schedule = training_schedule(read(args.cache / 'triplets.json')['rows'], args.stage)
    contract.update(policy=effective_policy(args.seed), schedule=schedule, arm=args.arm,
        initialization_seed=args.seed, baseline_completed_sha256=provenance['baseline_completed_sha256'],
        followup_policy=FOLLOWUP_POLICY, reference_evidence=provenance, conv_reference=identity,
        residual_policy=POLICY, code={p: code_hash(ROOT / p) for p in SOURCES})
    original = verify(args.followup_reference / ('seed' + str(args.seed)) / ('conv_' + args.stage))
    if matched_contract(contract) != matched_contract(original):
        raise ValueError('Data/order/optimizer/budget differs from historical conv')
    return contract, data, heldout, cache


def train(args):
    from tqdm import tqdm
    from src.losses.vpr_losses import VPRLossFunction
    baseline, provenance, identity = references(args)
    contract, data, heldout, cache = training_contract(args, provenance, identity)
    verify_cache_shards(args.cache, len(cache['train_places']))
    open_contract(args.output, contract, args.resume)
    if (args.output / 'completed.json').exists():
        verify(args.output); print('Already complete', args.arm, args.seed); return
    visual = load_plain(args); frozen = tensor_hash(visual)
    if frozen != read(baseline / 'summary.json')['frozen_tensor_sha256']:
        raise ValueError('Plain baseline tensor identity differs')
    seed(args.seed); model = build_model(visual, args.arm, args.seed).cuda().eval()
    historical_run = args.followup_reference / ('seed' + str(args.seed)) / ('conv_' + args.stage)
    historical_initial = read(historical_run / 'initial.json')
    if parameter_hash(model) != historical_initial['parameter_sha256']:
        raise ValueError('Exact historical full-neck initialization differs')
    probe = PlannedPlaces(data, heldout['plan'])[0][0]
    with torch.inference_mode():
        d, r, _, _, _ = forward_batch(model, probe); direct = visual(probe.cuda())
    if not torch.equal(d, direct) or not torch.equal(r, direct):
        raise ValueError('Strict zero-start descriptor differs')
    # A nonzero-up probe verifies that the RMS arm matches the old FUNCTION,
    # not merely the trivial all-zero residual at initialization.
    rms_function_check = None
    if args.arm == 'rms_tanh':
        probe_cpu_rng = torch.get_rng_state()
        probe_cuda_rng = torch.cuda.get_rng_state_all()
        old_model = historical_model(visual, 'conv', args.seed).cuda().eval()
        with torch.inference_mode():
            features = canonical_features(model.features(probe.cuda()))
            model.neck.up.weight.fill_(1e-4); old_model.neck.up.weight.fill_(1e-4)
            current = model.neck(features); old = old_model.neck(features)
            if not bool(old[-1].any()) or not torch.equal(current[0], old[0]) or not torch.equal(current[-1], old[-1]):
                raise ValueError('Nonzero RMS neck does not reproduce the sealed formula')
            model.neck.up.weight.zero_()
        rms_function_check = True
        del old_model, features, current, old
        # Do not let this arm-only mechanical probe alter the loader or any
        # later stochastic operation relative to the other arms.
        torch.set_rng_state(probe_cpu_rng)
        torch.cuda.set_rng_state_all(probe_cuda_rng)
    if parameter_hash(model) != historical_initial['parameter_sha256']:
        raise ValueError('Probe changed initial parameters')
    model.neck.last_raw = None
    active = {name: p for name, p in model.named_parameters() if p.requires_grad}
    count = sum(p.numel() for p in active.values())
    if count != PARAMETERS or any(not name.startswith('neck.') for name in active):
        raise ValueError('Unexpected parameter inventory')
    policy = effective_policy(args.seed)
    optimizer = torch.optim.AdamW(active.values(), lr=policy['lr'], weight_decay=0.)
    state = dict(steps=0, rows=[], module_gradients={'0': False, '1': False}, up_gradient_seen=False,
        initial_parameter_sha256=parameter_hash(model), initial_auxiliary_sha256=tensor_hash(model.neck.aux),
        contract_sha256=sha(args.output / 'contract.json'))
    if (args.output / 'last.pt').exists():
        state = restore(args.output / 'last.pt', model, optimizer, state['contract_sha256'])
    if not (args.output / 'initial.json').exists():
        write(args.output / 'initial.json', dict(parameter_sha256=state['initial_parameter_sha256'],
            initialization_seed=args.seed, descriptor_sha256=hashlib.sha256(d.cpu().numpy().tobytes()).hexdigest(),
            frozen_tensor_sha256=frozen, trainable_registered_parameters=count,
            effective_parameters=count - DISABLED_AUXILIARY_PARAMETERS,
            zero_start_error=0., nonzero_rms_function_reproduction=rms_function_check))
        save(args.output / 'last.pt', model, optimizer, state)
    lossfn = VPRLossFunction(); model.train(); schedule = contract['schedule']
    loader = DataLoader(PlaceBatches(data, cache['train_indices'], schedule[state['steps']:]),
                        batch_size=None, num_workers=args.workers)
    for images, labels, coordinates in tqdm(loader, total=len(schedule), initial=state['steps'],
            desc=args.arm + ' ' + str(args.seed) + ' ' + args.stage):
        optimizer.zero_grad(set_to_none=True)
        d, r, aux, delta, metrics = forward_batch(model, images, True)
        labels = labels.cuda(); coords = coordinates.numpy()
        primary = gps_filtered_retrieval(lossfn, d, labels, coords)
        auxiliary = sum(gps_filtered_retrieval(lossfn, x, labels, coords) for x in aux) / len(aux)
        loss = primary + 0. * auxiliary
        with torch.no_grad():
            reference_loss = gps_filtered_retrieval(lossfn, r, labels, coords)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite residual-ablation loss')
        gradients = gradient_diagnostic(primary, 0. * auxiliary, model.named_parameters()) if state['steps'] % 64 == 0 else None
        loss.backward()
        for key, yes in module_gradient_status(model).items():
            state['module_gradients'][key] |= yes
        state['up_gradient_seen'] |= bool(model.neck.up.weight.grad is not None and (model.neck.up.weight.grad != 0).any())
        for parameter in active.values():
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
        norm = torch.nn.utils.clip_grad_norm_(list(active.values()), policy['clip'], error_if_nonfinite=True)
        if any(parameter.grad is not None for parameter in visual.parameters()):
            raise ValueError('Frozen original network received gradient')
        # Only RMS has this relative bound. Other arms are NEVER clipped or
        # rejected simply because their relative residual exceeds 0.1.
        if args.arm == 'rms_tanh' and float(delta.max()) > POLICY['rho'] + 1e-6:
            raise ValueError('RMS image-relative bound exceeded')
        optimizer.step(); state['steps'] += 1
        state['rows'].append(dict(step=state['steps'], primary=float(primary.detach()),
            frozen_baseline_loss=float(reference_loss), auxiliary=float(auxiliary.detach()),
            gradients=gradients, gradient_norm=float(norm),
            relative_residual_mean=float(delta.mean()), relative_residual_max=float(delta.max()),
            places=len(labels.unique()), images=len(labels), forward_diagnostics=diagnostic_summary(metrics)))
        write(args.output / 'progress.json', dict(phase='training', arm=args.arm, seed=args.seed,
            steps=state['steps'], total=len(schedule), last_loss=float(loss.detach())))
        if state['steps'] % 16 == 0:
            save(args.output / 'last.pt', model, optimizer, state)
    if state['steps'] != len(schedule) or tensor_hash(visual) != frozen:
        raise ValueError('Incomplete updates or changed frozen network')
    if tensor_hash(model.neck.aux) != state['initial_auxiliary_sha256']:
        raise ValueError('Disabled auxiliary heads changed')
    if any(int(optimizer.state[p]['step']) != len(schedule) for p in active.values()):
        raise ValueError('Optimizer clocks differ')
    save(args.output / 'last.pt', model, optimizer, state); final = parameter_hash(model)
    restore(args.output / 'last.pt', model, optimizer, state['contract_sha256'])
    if final != parameter_hash(model):
        raise ValueError('Checkpoint roundtrip differs')
    optimizer.zero_grad(set_to_none=True); model.eval()
    if args.stage == 'screen':
        evaluate_gsv(model, data, heldout, args.output, args.workers, baseline)
        evaluate_benchmarks(model, args.output, args.workers, baseline)
    checks = dict(zero_start=True, exact_historical_initialization=True, frozen_plain_unchanged=True,
        checkpoint_roundtrip=True, disabled_auxiliary_unchanged=True,
        nonzero_updates=final != state['initial_parameter_sha256'],
        all_actual_modules_receive_gradient=all(state['module_gradients'].values()),
        up_gradient_seen=state['up_gradient_seen'])
    write(args.output / 'summary.json', dict(arm=args.arm, seed=args.seed, stage=args.stage,
        steps=state['steps'], checks=checks, rows=state['rows'], final_parameter_sha256=final,
        historical_conv_final_parameter_match=final == read(historical_run / 'summary.json')['final_parameter_sha256']
            if args.arm == 'rms_tanh' else None, scope=POLICY['scope']))
    write(args.output / 'progress.json', dict(phase='complete', steps=state['steps']))
    seal(args.output)
    print('RESIDUAL ABLATION COMPLETE', args.seed, args.arm, checks, flush=True)
    if not all(checks.values()):
        raise SystemExit(2)


def verify_arm(args, seed_, arm, stage, provenance, identity):
    path = args.root / ('seed' + str(seed_)) / (arm + '_' + stage)
    contract = verify(path); report = read(path / 'summary.json')
    if contract['arm'] != arm or contract['initialization_seed'] != seed_ or contract['stage'] != stage:
        raise ValueError('Run identity differs')
    if contract['residual_policy'] != POLICY or contract['conv_reference'] != identity or contract['reference_evidence'] != provenance:
        raise ValueError('Residual recipe or evidence differs')
    if contract['policy'] != effective_policy(seed_) or contract['code'] != {p: code_hash(ROOT / p) for p in SOURCES}:
        raise ValueError('Settings or code changed')
    if contract['original_code'] != current_code() or report['steps'] != POLICY[stage + '_steps'] or not all(report['checks'].values()):
        raise ValueError('Run not mechanically complete')
    original = verify(args.followup_reference / ('seed' + str(seed_)) / ('conv_' + stage))
    if matched_contract(contract) != matched_contract(original):
        raise ValueError('Historical schedule/data/optimizer changed')
    return contract


def compare(args):
    baseline, provenance, identity = references(args)
    runs = args.root / ('seed' + str(args.seed))
    contracts = {arm: verify_arm(args, args.seed, arm, args.stage, provenance, identity) for arm in ARMS}
    initial = {arm: read(runs / (arm + '_' + args.stage) / 'initial.json') for arm in ARMS}
    checks = dict(mechanical=True,
        identical_contract_except_arm=all(arm_contract(c) == arm_contract(contracts[ARMS[0]]) for c in contracts.values()),
        identical_full_initial_parameters=len({v['parameter_sha256'] for v in initial.values()}) == 1,
        identical_zero_start_descriptors=len({v['descriptor_sha256'] for v in initial.values()}) == 1,
        nonzero_rms_function_reproduction=initial['rms_tanh']['nonzero_rms_function_reproduction'] is True)
    partitions, standard, diagnostics, historical = {}, {}, {}, {}
    if args.stage == 'screen':
        source = verify(args.source); plan = source['plan']
        if sha(args.source / 'completed.json') != contracts[ARMS[0]]['source_completed_sha256']:
            raise ValueError('Heldout source changed')
        old_run = args.followup_reference / ('seed' + str(args.seed)) / 'conv_screen'
        for dataset in ('gsv_final', 'msls-val_final', 'pitts30k-val_final'):
            b = read(baseline / (dataset + '.json'))['baseline']
            values = {arm: read(runs / (arm + '_screen') / (dataset + '.json')) for arm in ARMS}
            if any(value['baseline'] != b for value in values.values()):
                raise ValueError('Baseline query rankings differ')
            outcomes = {arm: value['student']['outcomes'] for arm, value in values.items()}
            if dataset == 'gsv_final':
                partitions = summarize(outcomes, b['outcomes'], plan['query_place_groups'], plan['calibration_places'])
                metric_name = 'gsv'
            else:
                metric_name = dataset.removesuffix('_final')
                standard[metric_name] = summarize(outcomes, b['outcomes'], np.zeros(len(b['outcomes']), int), 1)['all']
                standard[metric_name]['recall_at_1_5_20'] = dict(baseline=b['recall'], **{arm: v['student']['recall'] for arm, v in values.items()})
            diagnostics[metric_name] = {arm: read(runs / (arm + '_screen') / (metric_name + '_diagnostics.json')) for arm in ARMS}
            historical[metric_name] = values['rms_tanh']['student'] == read(old_run / (dataset + '.json'))['student']
    contract = dict(policy=POLICY, seed=args.seed, stage=args.stage, conv_reference=identity,
        completed_hashes={arm: sha(runs / (arm + '_' + args.stage) / 'completed.json') for arm in ARMS})
    verdict = ('SMOKE_PASS' if args.stage == 'smoke' else 'RESIDUAL_COMPARISON_COMPLETE') if all(checks.values()) else 'MECHANICAL_FAIL'
    open_contract(args.output, contract, args.resume)
    if (args.output / 'completed.json').exists():
        verify(args.output)
    else:
        write(args.output / 'summary.json', dict(verdict=verdict, checks=checks, partitions=partitions,
            standard_benchmark_paired=standard, feature_diagnostics=diagnostics,
            historical_rms_predictions_match=historical, scope=POLICY['scope']))
        seal(args.output)
    print('RESIDUAL COMPARISON', args.seed, verdict, checks, flush=True)
    if not all(checks.values()):
        raise SystemExit(2)


def aggregate(args):
    _, provenance, identity = references(args)
    reports, diagnostic_reports, histories, hashes = {}, {}, {}, {}
    for seed_ in SEEDS:
        root = args.root / ('seed' + str(seed_)); path = root / 'compare_screen'
        c = verify(path); r = read(path / 'summary.json')
        if c['policy'] != POLICY or c['conv_reference'] != identity or r['verdict'] != 'RESIDUAL_COMPARISON_COMPLETE' or not all(r['checks'].values()):
            raise ValueError('Incomplete or changed comparison')
        for arm in ARMS:
            verify_arm(args, seed_, arm, 'screen', provenance, identity)
            if sha(root / (arm + '_screen') / 'completed.json') != c['completed_hashes'][arm]:
                raise ValueError('Comparison/run identity differs')
        reports[seed_] = {'gsv_' + role: r['partitions'][role] for role in ('calibration', 'evaluation', 'all')}
        reports[seed_].update(r['standard_benchmark_paired'])
        diagnostic_reports[seed_] = r['feature_diagnostics']
        histories[seed_] = dict(predictions=r['historical_rms_predictions_match'],
            final_parameter_match=read(root / 'rms_tanh_screen' / 'summary.json')['historical_conv_final_parameter_match'])
        hashes[str(seed_)] = sha(path / 'completed.json')
    contract = dict(policy=POLICY, completed_hashes=hashes, conv_reference=identity,
                     code={p: code_hash(ROOT / p) for p in SOURCES})
    open_contract(args.output, contract, args.resume)
    if (args.output / 'completed.json').exists():
        verify(args.output); return
    write(args.output / 'summary.json', dict(verdict='RESIDUAL_ABLATION_COMPLETE_NOT_PATENTABILITY_PROOF',
        metrics=aggregate_reports(reports), per_seed=reports, feature_diagnostics_by_seed=diagnostic_reports,
        historical_rms_reproduction=histories, scope=POLICY['scope'], fairness=POLICY['fairness']))
    seal(args.output)
    print('ALL THREE RESIDUAL RULES COMPLETE; no automatic success/novelty claim', flush=True)


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['wait_gpu', 'train', 'compare', 'aggregate'])
    parser.add_argument('--arm', choices=ARMS); parser.add_argument('--seed', type=int, choices=SEEDS, default=42)
    parser.add_argument('--stage', choices=['smoke', 'screen'], default='screen')
    parser.add_argument('--checkpoint', type=Path, default=Path(PLAIN_CHECKPOINT))
    parser.add_argument('--reference', type=Path, default=Path('logs/hcen_plain_base_v1'))
    parser.add_argument('--full-reference', type=Path, default=Path('logs/hcen_plain_multiseed_v1'))
    parser.add_argument('--module-reference', type=Path, default=Path('logs/hcen_module_ablation_v1'))
    parser.add_argument('--followup-reference', type=Path, default=Path('logs/hcen_simplified_2048_v1'))
    parser.add_argument('--source', type=Path, default=Path('logs/slgd_hard_v2_audit'))
    parser.add_argument('--cache', type=Path, default=Path('.cache/slgd_mid_v1_pilot'))
    parser.add_argument('--teacher', type=Path, default=Path('logs/slgd_v1/teacher_pilot'))
    parser.add_argument('--dataset-root', type=Path, default=Path('datasets/gsv_cities'))
    parser.add_argument('--root', type=Path, default=Path('logs/conv_residual_ablation_v1'))
    parser.add_argument('--output', type=Path); parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    protected = [args.reference, args.full_reference, args.module_reference,
                 args.followup_reference, args.source, args.teacher, args.cache]
    resolved_root = args.root.resolve(); logs = (ROOT / 'logs').resolve()
    if resolved_root.parent != logs or not resolved_root.name.startswith('conv_residual_ablation_') or any(
            resolved_root.is_relative_to(path.resolve()) or path.resolve().is_relative_to(resolved_root) for path in protected):
        parser.error('Choose logs/conv_residual_ablation_<version>, never a historical reference directory')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        parser.error('Expose physical GPU1 only')
    if args.command == 'wait_gpu':
        wait_gpu(args); return
    if args.output is None or not args.output.resolve().is_relative_to(resolved_root) or args.output.resolve() == resolved_root:
        parser.error('Output must be a child of the isolated experiment root')
    if args.command == 'train' and args.arm is None:
        parser.error('Training requires an arm')
    if args.workers < 0:
        parser.error('workers must be nonnegative')
    if not args.checkpoint.is_file() or sha(args.checkpoint) != PLAIN_SHA:
        parser.error('Pinned original plain-BoQ checkpoint differs')
    if args.command == 'train' and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        parser.error('Need exactly one visible CUDA device')
    import fcntl
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
        if args.command == 'train':
            torch.cuda.set_per_process_memory_fraction(.65, 0)
        try:
            {'train': train, 'compare': compare, 'aggregate': aggregate}[args.command](args)
        except Exception as error:
            if args.output.exists() and (args.output / 'contract.json').is_file() and not (args.output / 'completed.json').exists():
                write(args.output / 'failure.json', dict(status='STOPPED_NOT_COMPLETE',
                    error_type=type(error).__name__, message=str(error)))
            raise


if __name__ == '__main__':
    main()
