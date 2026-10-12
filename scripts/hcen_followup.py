"""Twelve fresh simplified-neck 2048-step runs; preserve all sealed old sources."""
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
from scripts.train_slgd import read, write, sha, code_hash, verify, seal, seed, save, restore, parameter_hash
from scripts.slgd_mid_screen import open_contract, verify_cache_shards, current_code
from scripts.audit_slgd_hard import PlannedPlaces
from scripts.train_hcen_vpr import PlaceBatches, tensor_hash
from scripts.hcen_plain_base import context, load_plain, forward_plain_batch, gsv_evaluate, benchmarks
from scripts.hcen_module_ablation import references as original_references, shared_hash, wait_gpu, SOURCES as OLD_SOURCES
from src.models.hcen_ablation import build_model as build_original
from src.models.hcen_followup import ARMS, COUNTS, build_model, module_gradient_status
from src.hcen_plain_protocol import PLAIN_CHECKPOINT, PLAIN_SHA
from src.hcen_plain_replication import SEEDS
from src.hcen_aux_ablation import gradient_diagnostic
from src.hcen_followup_protocol import POLICY, effective_policy, training_schedule, comparable_contract, summarize, aggregate_reports
from src.slgd_mid_supervision import gps_filtered_retrieval

SOURCES = ('scripts/hcen_followup.py', 'src/models/hcen_followup.py', 'src/hcen_followup_protocol.py', *OLD_SOURCES)


def references(args):
    baseline, roots, provenance = original_references(args)
    aggregate = verify(args.module_reference/'aggregate')
    summary = read(args.module_reference/'aggregate/summary.json')
    if summary['verdict'] != 'MODULE_ABLATION_COMPLETE_NOT_NOVELTY_OR_STABILITY_PROOF':
        raise ValueError('Prior module ablation incomplete')
    if aggregate['full_reference'] != provenance:
        raise ValueError('Prior evidence baseline differs')
    for s in SEEDS:
        run = args.module_reference/('seed'+str(s))
        c = verify(run/'compare_screen')
        if sha(run/'compare_screen/completed.json') != aggregate['completed_hashes'][str(s)]:
            raise ValueError('Prior comparison hash changed')
        for arm in ('linear', 'conv', 'no_sasf'):
            old = verify(run/(arm+'_screen'))
            if old['code'] != {p: code_hash(ROOT/p) for p in OLD_SOURCES}:
                raise ValueError('Sealed module implementation changed')
            if sha(run/(arm+'_screen/completed.json')) != c['completed_hashes'][arm]:
                raise ValueError('Prior run hash changed')
    return baseline, roots, dict(**provenance, module_aggregate_sha256=sha(args.module_reference/'aggregate/completed.json'))


def train(args):
    from tqdm import tqdm
    from src.losses.vpr_losses import VPRLossFunction
    policy = effective_policy(args.seed)
    baseline, roots, provenance = references(args)
    mode = 'last_only' if args.arm == 'last_only' else 'hierarchy'
    contract, data, heldout, cache = context(args, mode, args.stage)
    verify_cache_shards(args.cache, len(cache['train_places']))
    schedule = training_schedule(read(args.cache/'triplets.json')['rows'], args.stage)
    contract.update(policy=policy, schedule=schedule, arm=args.arm, initialization_seed=args.seed,
        baseline_completed_sha256=provenance['baseline_completed_sha256'], followup_policy=POLICY,
        reference_evidence=provenance, code={p: code_hash(ROOT/p) for p in SOURCES})
    old_contract = verify(roots[args.seed]/('hierarchy_'+args.stage))
    for k in ('checkpoint_sha256', 'source_completed_sha256', 'cache_completed_sha256', 'metadata',
              'train_places', 'original_code', 'benchmark_metadata', 'versions'):
        if contract[k] != old_contract[k]:
            raise ValueError('Historical data/base identity differs: '+k)
    if schedule[:8] != old_contract['schedule'][:8]:
        raise ValueError('Original training prefix differs')
    open_contract(args.output, contract, args.resume)
    if (args.output/'completed.json').exists():
        verify(args.output); print('Already complete', args.arm, args.seed); return
    visual = load_plain(args); frozen = tensor_hash(visual)
    if frozen != read(baseline/'summary.json')['frozen_tensor_sha256']:
        raise ValueError('Plain baseline tensors differ')
    seed(args.seed); full = build_original(visual, 'full', args.seed)
    historical = read(roots[args.seed]/('hierarchy_'+args.stage)/'initial.json')
    if parameter_hash(full) != historical['parameter_sha256']:
        raise ValueError('Historical full initialization differs')
    shared = shared_hash(full); del full
    seed(args.seed); model = build_model(visual, args.arm, args.seed).cuda().eval()
    if shared_hash(model) != shared:
        raise ValueError('Common outer initialization differs')
    if args.arm in ('scan', 'last_only', 'linear'):
        old_arm = 'linear' if args.arm == 'linear' else 'no_sasf'
        old_init = read(args.module_reference/('seed'+str(args.seed))/(old_arm+'_screen')/'initial.json')
        if parameter_hash(model) != old_init['parameter_sha256']:
            raise ValueError('Simplified historical initialization differs')
    probe = PlannedPlaces(data, heldout['plan'])[0][0]
    with torch.inference_mode():
        d, r, _, _ = forward_plain_batch(model, probe); direct = visual(probe.cuda())
    if not torch.equal(d, direct) or not torch.equal(r, direct):
        raise ValueError('Strict zero start differs from plain BoQ')
    active = {n: p for n, p in model.named_parameters() if p.requires_grad}
    count = sum(p.numel() for p in active.values())
    if count != COUNTS[args.arm] or any(not n.startswith('neck.') for n in active):
        raise ValueError('Unexpected active parameter inventory')
    optimizer = torch.optim.AdamW(active.values(), lr=policy['lr'], weight_decay=0.)
    state = dict(steps=0, rows=[], module_gradients={'0': False, '1': False}, up_gradient_seen=False,
        initial_parameter_sha256=parameter_hash(model), initial_auxiliary_sha256=tensor_hash(model.neck.aux),
        contract_sha256=sha(args.output/'contract.json'))
    if (args.output/'last.pt').exists():
        state = restore(args.output/'last.pt', model, optimizer, state['contract_sha256'])
    if not (args.output/'initial.json').exists():
        write(args.output/'initial.json', dict(parameter_sha256=state['initial_parameter_sha256'],
            shared_parameter_sha256=shared, historical_full_parameter_sha256=historical['parameter_sha256'],
            initialization_seed=args.seed, descriptor_sha256=hashlib.sha256(d.cpu().numpy().tobytes()).hexdigest(),
            trainable_parameters=count, excluding_zero_weight_auxiliary_parameters=count-65536,
            frozen_tensor_sha256=frozen, zero_start_error=0.))
        save(args.output/'last.pt', model, optimizer, state)
    lossfn = VPRLossFunction(); model.train()
    loader = DataLoader(PlaceBatches(data, cache['train_indices'], schedule[state['steps']:]),
        batch_size=None, num_workers=args.workers)
    for images, labels, coordinates in tqdm(loader, total=len(schedule), initial=state['steps'],
            desc=args.arm+' '+str(args.seed)+' '+args.stage):
        optimizer.zero_grad(set_to_none=True)
        d, r, aux, delta = forward_plain_batch(model, images, True)
        labels = labels.cuda(); coords = coordinates.numpy()
        primary = gps_filtered_retrieval(lossfn, d, labels, coords)
        auxiliary = sum(gps_filtered_retrieval(lossfn, x, labels, coords) for x in aux)/len(aux)
        loss = primary + 0.*auxiliary
        with torch.no_grad():
            reference_loss = gps_filtered_retrieval(lossfn, r, labels, coords)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite follow-up loss')
        gradients = gradient_diagnostic(primary, 0.*auxiliary, model.named_parameters()) if state['steps']%64 == 0 else None
        loss.backward()
        for key, yes in module_gradient_status(model).items():
            state['module_gradients'][key] |= yes
        state['up_gradient_seen'] |= bool(model.neck.up.weight.grad is not None and (model.neck.up.weight.grad != 0).any())
        for p in active.values():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        norm = torch.nn.utils.clip_grad_norm_(list(active.values()), policy['clip'], error_if_nonfinite=True)
        if any(p.grad is not None for p in visual.parameters()):
            raise ValueError('Frozen visual model received gradient')
        if float(delta.max()) > policy['feature_relative_cap']+1e-6:
            raise ValueError('Residual cap exceeded')
        optimizer.step(); state['steps'] += 1
        state['rows'].append(dict(step=state['steps'], primary=float(primary.detach()), frozen_baseline_loss=float(reference_loss),
            auxiliary=float(auxiliary.detach()), gradients=gradients, gradient_norm=float(norm),
            relative_residual_mean=float(delta.mean()), relative_residual_max=float(delta.max()),
            places=len(labels.unique()), images=len(labels)))
        write(args.output/'progress.json', dict(phase='training', arm=args.arm, seed=args.seed,
            steps=state['steps'], total=len(schedule), last_loss=float(loss.detach())))
        if state['steps']%16 == 0:
            save(args.output/'last.pt', model, optimizer, state)
    if state['steps'] != len(schedule) or tensor_hash(visual) != frozen:
        raise ValueError('Incomplete budget or changed frozen model')
    if tensor_hash(model.neck.aux) != state['initial_auxiliary_sha256']:
        raise ValueError('Disabled auxiliary heads changed')
    if any(int(optimizer.state[p]['step']) != len(schedule) for p in active.values()):
        raise ValueError('Optimizer clocks differ')
    save(args.output/'last.pt', model, optimizer, state); final = parameter_hash(model)
    restore(args.output/'last.pt', model, optimizer, state['contract_sha256'])
    if final != parameter_hash(model):
        raise ValueError('Checkpoint roundtrip differs')
    optimizer.zero_grad(set_to_none=True); model.eval()
    if args.stage == 'screen':
        gsv_evaluate(visual, model, data, heldout, args.output, args.workers, baseline)
        benchmarks(visual, model, args.output, args.workers, baseline)
    checks = dict(zero_start=True, shared_outer_initialization=True, frozen_plain_unchanged=True,
        checkpoint_roundtrip=True, disabled_auxiliary_unchanged=True,
        nonzero_updates=final != state['initial_parameter_sha256'],
        both_stages_receive_module_gradient=all(state['module_gradients'].values()), up_gradient_seen=state['up_gradient_seen'])
    write(args.output/'summary.json', dict(arm=args.arm, seed=args.seed, stage=args.stage, steps=state['steps'], checks=checks,
        module_gradients=state['module_gradients'], trainable_parameters=count,
        excluding_zero_weight_auxiliary_parameters=count-65536, rows=state['rows'], final_parameter_sha256=final, scope=POLICY['scope']))
    write(args.output/'progress.json', dict(phase='complete', steps=state['steps'])); seal(args.output)
    print('FOLLOW-UP COMPLETE', args.seed, args.arm, checks, flush=True)
    if not all(checks.values()):
        raise SystemExit(2)


def verify_arm(args, seed_value, arm, stage, provenance):
    run = args.root/('seed'+str(seed_value))/(arm+'_'+stage)
    c = verify(run); summary = read(run/'summary.json')
    if c['arm'] != arm or c['initialization_seed'] != seed_value or c['stage'] != stage:
        raise ValueError('Run identity differs')
    if c['policy'] != effective_policy(seed_value) or c['followup_policy'] != POLICY or c['reference_evidence'] != provenance:
        raise ValueError('Run protocol/reference changed')
    if c['code'] != {p: code_hash(ROOT/p) for p in SOURCES} or c['original_code'] != current_code():
        raise ValueError('Run implementation changed')
    if summary['steps'] != POLICY[stage+'_steps'] or not all(summary['checks'].values()):
        raise ValueError('Incomplete mechanical checks/budget')
    return c


def compare(args):
    baseline, _, provenance = references(args)
    runs = args.root/('seed'+str(args.seed))
    contracts = {a: verify_arm(args, args.seed, a, args.stage, provenance) for a in ARMS}
    common = [comparable_contract(c) for c in contracts.values()]
    if any(c != common[0] for c in common):
        raise ValueError('Unmatched data/order/update protocol')
    initial = {a: read(runs/(a+'_'+args.stage)/'initial.json') for a in ARMS}
    checks = dict(mechanical=True, same_budget=True, same_data_order=True,
        same_shared_outer_initialization=len({x['shared_parameter_sha256'] for x in initial.values()}) == 1,
        identical_scan_last_parameters=initial['scan']['parameter_sha256'] == initial['last_only']['parameter_sha256'],
        same_baseline_zero_start=len({x['descriptor_sha256'] for x in initial.values()}) == 1)
    report = {}; standard = {}
    if args.stage == 'screen':
        source = verify(args.source); plan = source['plan']
        if sha(args.source/'completed.json') != common[0]['source_completed_sha256']:
            raise ValueError('Heldout source differs')
        for dataset in ('gsv_final', 'msls-val_final', 'pitts30k-val_final'):
            b = read(baseline/(dataset+'.json'))['baseline']
            values = {a: read(runs/(a+'_screen')/(dataset+'.json')) for a in ARMS}
            if any(x['baseline'] != b for x in values.values()):
                raise ValueError('Baseline/query ranking differs')
            outcomes = {k: x['student']['outcomes'] for k, x in values.items()}
            if dataset == 'gsv_final':
                report = summarize(outcomes, b['outcomes'], plan['query_place_groups'], plan['calibration_places'])
            else:
                name = dataset.removesuffix('_final')
                standard[name] = summarize(outcomes, b['outcomes'], np.zeros(len(b['outcomes']), int), 1)['all']
                standard[name]['recall_at_1_5_20'] = dict(baseline=b['recall'], **{k: x['student']['recall'] for k, x in values.items()})
    verdict = ('SMOKE_PASS' if args.stage == 'smoke' else 'FOLLOWUP_SCREEN_COMPLETE') if all(checks.values()) else 'MECHANICAL_FAIL'
    contract = dict(policy=POLICY, seed=args.seed, stage=args.stage, reference_evidence=provenance,
        completed_hashes={a: sha(runs/(a+'_'+args.stage)/'completed.json') for a in ARMS})
    open_contract(args.output, contract, args.resume)
    if (args.output/'completed.json').exists():
        verify(args.output)
    else:
        write(args.output/'summary.json', dict(verdict=verdict, checks=checks, partitions=report,
            standard_benchmark_paired=standard, scope=POLICY['scope'])); seal(args.output)
    print('COMPARISON', args.seed, verdict, checks, flush=True)
    if not all(checks.values()):
        raise SystemExit(2)


def aggregate(args):
    _, _, provenance = references(args); reports = {}; hashes = {}
    for s in SEEDS:
        run = args.root/('seed'+str(s)); c = verify(run/'compare_screen'); r = read(run/'compare_screen/summary.json')
        if c['reference_evidence'] != provenance or c['policy'] != POLICY or r['verdict'] != 'FOLLOWUP_SCREEN_COMPLETE' or not all(r['checks'].values()):
            raise ValueError('Comparison incomplete or protocol differs')
        for arm in ARMS:
            verify_arm(args, s, arm, 'screen', provenance)
            if sha(run/(arm+'_screen/completed.json')) != c['completed_hashes'][arm]:
                raise ValueError('Comparison/run hash differs')
        reports[s] = {'gsv_'+role: r['partitions'][role] for role in ('calibration', 'evaluation', 'all')}
        reports[s].update(r['standard_benchmark_paired']); hashes[str(s)] = sha(run/'compare_screen/completed.json')
    contract = dict(policy=POLICY, completed_hashes=hashes, reference_evidence=provenance, code={p: code_hash(ROOT/p) for p in SOURCES})
    open_contract(args.output, contract, args.resume)
    if (args.output/'completed.json').exists():
        verify(args.output); return
    result = aggregate_reports(reports)
    write(args.output/'summary.json', dict(verdict='FOLLOWUP_COMPLETE_NOT_STABLE_GAIN_PROOF', **result,
        per_seed=reports, fairness=POLICY['fairness'], scope=POLICY['scope']))
    seal(args.output); print('ALL FOLLOW-UP RUNS COMPLETE', result['directional_checks'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['wait_gpu', 'train', 'compare', 'aggregate'])
    p.add_argument('--arm', choices=ARMS); p.add_argument('--seed', type=int, choices=SEEDS, default=42)
    p.add_argument('--stage', choices=['smoke', 'screen'], default='screen')
    p.add_argument('--checkpoint', type=Path, default=Path(PLAIN_CHECKPOINT))
    p.add_argument('--reference', type=Path, default=Path('logs/hcen_plain_base_v1'))
    p.add_argument('--full-reference', type=Path, default=Path('logs/hcen_plain_multiseed_v1'))
    p.add_argument('--module-reference', type=Path, default=Path('logs/hcen_module_ablation_v1'))
    p.add_argument('--source', type=Path, default=Path('logs/slgd_hard_v2_audit'))
    p.add_argument('--cache', type=Path, default=Path('.cache/slgd_mid_v1_pilot'))
    p.add_argument('--teacher', type=Path, default=Path('logs/slgd_v1/teacher_pilot'))
    p.add_argument('--dataset-root', type=Path, default=Path('datasets/gsv_cities'))
    p.add_argument('--root', type=Path, default=Path('logs/hcen_simplified_2048_v1'))
    p.add_argument('--output', type=Path); p.add_argument('--workers', type=int, default=4)
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        p.error('Expose physical GPU1 only')
    if args.command == 'wait_gpu':
        wait_gpu(args); return
    if args.output is None or (args.command == 'train' and args.arm is None):
        p.error('Output/arm required')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        p.error('Need one visible CUDA device')
    if not args.checkpoint.is_file() or sha(args.checkpoint) != PLAIN_SHA:
        p.error('Pinned plain checkpoint differs')
    import fcntl
    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root/'worker.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(.65, 0)
    {'train': train, 'compare': compare, 'aggregate': aggregate}[args.command](args)


if __name__ == '__main__':
    main()
