"""Fresh conv-last and SciceVPR-style fusion, vs sealed same-budget local fusion."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
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
from scripts.hcen_module_ablation import wait_gpu
from scripts.hcen_followup import references as prior_references, verify_arm as verify_prior_arm, SOURCES as OLD_SOURCES
from src.hcen_followup_protocol import POLICY as FOLLOWUP_POLICY
from src.hcen_plain_protocol import PLAIN_CHECKPOINT, PLAIN_SHA
from src.hcen_plain_replication import SEEDS
from src.hcen_aux_ablation import gradient_diagnostic
from src.models.conv_fusion_controls import ARMS, COUNTS, build_model, interface_parameters, gradient_inventory, module_gradient_status
from src.conv_fusion_protocol import POLICY, REFERENCE_SHA, training_schedule, effective_policy, matched_contract, summarize, aggregate_reports
from src.slgd_mid_supervision import gps_filtered_retrieval

SOURCES = ('scripts/conv_fusion_controls.py', 'src/models/conv_fusion_controls.py', 'src/conv_fusion_protocol.py', *OLD_SOURCES)


def references(args):
    baseline, _, provenance = prior_references(args)
    root = args.followup_reference
    aggregate = verify(root/'aggregate'); summary = read(root/'aggregate/summary.json')
    if sha(root/'aggregate/completed.json') != REFERENCE_SHA or summary['verdict'] != 'FOLLOWUP_COMPLETE_NOT_STABLE_GAIN_PROOF':
        raise ValueError('Pinned 2048-step reference aggregate differs')
    if aggregate['policy'] != FOLLOWUP_POLICY or aggregate['reference_evidence'] != provenance:
        raise ValueError('Previous scientific reference differs')
    prior_args = SimpleNamespace(**{**vars(args), 'root': root})
    for s in SEEDS:
        run = root/('seed'+str(s)); c = verify(run/'compare_screen')
        if sha(run/'compare_screen/completed.json') != aggregate['completed_hashes'][str(s)]:
            raise ValueError('Prior comparison hash differs')
        if not all(read(run/'compare_screen/summary.json')['checks'].values()):
            raise ValueError('Prior mechanical check failed')
        for stage in ('smoke', 'screen'):
            verify_prior_arm(prior_args, s, 'conv', stage, provenance)
        if sha(run/'conv_screen/completed.json') != c['completed_hashes']['conv']:
            raise ValueError('Sealed conv run identity differs')
    identity = dict(aggregate_sha256=REFERENCE_SHA, completed_hashes={str(s): sha(root/('seed'+str(s))/'conv_screen/completed.json') for s in SEEDS})
    return baseline, provenance, identity


def interface_hash(model):
    h = hashlib.sha256()
    for n, p in interface_parameters(model).items():
        h.update(n.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train(args):
    from tqdm import tqdm
    from src.losses.vpr_losses import VPRLossFunction
    policy = effective_policy(args.seed)
    baseline, provenance, identity = references(args)
    mode = 'last_only' if args.arm == 'conv_last' else 'hierarchy'
    contract, data, heldout, cache = context(args, mode, args.stage)
    verify_cache_shards(args.cache, len(cache['train_places']))
    schedule = training_schedule(read(args.cache/'triplets.json')['rows'], args.stage)
    contract.update(policy=policy, schedule=schedule, arm=args.arm, initialization_seed=args.seed,
        baseline_completed_sha256=provenance['baseline_completed_sha256'], followup_policy=FOLLOWUP_POLICY,
        reference_evidence=provenance, conv_reference=identity, conv_control_policy=POLICY,
        code={p: code_hash(ROOT/p) for p in SOURCES})
    original_run = args.followup_reference/('seed'+str(args.seed))/('conv_'+args.stage)
    original = verify(original_run)
    if matched_contract(contract) != matched_contract(original):
        raise ValueError('Data/images/order/budget/optimizer differ from sealed convolution')
    open_contract(args.output, contract, args.resume)
    if (args.output/'completed.json').exists():
        verify(args.output); print('Already complete', args.arm, args.seed); return
    visual = load_plain(args); frozen = tensor_hash(visual)
    if frozen != read(baseline/'summary.json')['frozen_tensor_sha256']:
        raise ValueError('Plain baseline tensor identity differs')
    seed(args.seed); reference_model = build_model(visual, 'conv', args.seed)
    historical = read(original_run/'initial.json')
    reference_initial = parameter_hash(reference_model)
    if reference_initial != historical['parameter_sha256']:
        raise ValueError('Real historical conv initialization not reproduced')
    shared = interface_hash(reference_model); del reference_model
    seed(args.seed); model = build_model(visual, args.arm, args.seed).cuda().eval()
    if interface_hash(model) != shared:
        raise ValueError('Common LN/up/disabled-head interface initialization differs')
    if args.arm == 'conv_last' and parameter_hash(model) != reference_initial:
        raise ValueError('Depth-only conv ablation changes weights')
    probe = PlannedPlaces(data, heldout['plan'])[0][0]
    with torch.inference_mode():
        d, r, _, _ = forward_plain_batch(model, probe); direct = visual(probe.cuda())
    if not torch.equal(d, direct) or not torch.equal(r, direct):
        raise ValueError('Strict zero-start descriptor differs')
    active = {n: p for n, p in model.named_parameters() if p.requires_grad}
    count = sum(p.numel() for p in active.values())
    if count != COUNTS[args.arm] or any(not n.startswith('neck.') for n in active):
        raise ValueError('Unexpected active parameter inventory')
    optimizer = torch.optim.AdamW(active.values(), lr=policy['lr'], weight_decay=0.)
    state = dict(steps=0, rows=[], module_gradients={k: False for k in gradient_inventory(args.arm)}, up_gradient_seen=False,
        initial_parameter_sha256=parameter_hash(model), initial_auxiliary_sha256=tensor_hash(model.neck.aux),
        contract_sha256=sha(args.output/'contract.json'))
    if (args.output/'last.pt').exists():
        state = restore(args.output/'last.pt', model, optimizer, state['contract_sha256'])
    if not (args.output/'initial.json').exists():
        write(args.output/'initial.json', dict(parameter_sha256=state['initial_parameter_sha256'],
            shared_interface_sha256=shared, historical_conv_parameter_sha256=reference_initial, initialization_seed=args.seed,
            descriptor_sha256=hashlib.sha256(d.cpu().numpy().tobytes()).hexdigest(), trainable_parameters=count,
            excluding_zero_weight_auxiliary_parameters=count-65536, frozen_tensor_sha256=frozen, zero_start_error=0.))
        save(args.output/'last.pt', model, optimizer, state)
    lossfn = VPRLossFunction(); model.train()
    loader = DataLoader(PlaceBatches(data, cache['train_indices'], schedule[state['steps']:]), batch_size=None, num_workers=args.workers)
    for images, labels, coordinates in tqdm(loader, total=len(schedule), initial=state['steps'], desc=args.arm+' '+str(args.seed)+' '+args.stage):
        optimizer.zero_grad(set_to_none=True)
        d, r, aux, delta = forward_plain_batch(model, images, True); labels = labels.cuda(); coords = coordinates.numpy()
        primary = gps_filtered_retrieval(lossfn, d, labels, coords)
        auxiliary = sum(gps_filtered_retrieval(lossfn, x, labels, coords) for x in aux)/len(aux)
        loss = primary+0.*auxiliary
        with torch.no_grad():
            reference_loss = gps_filtered_retrieval(lossfn, r, labels, coords)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite fusion-control loss')
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
            raise ValueError('Feature residual bound exceeded')
        optimizer.step(); state['steps'] += 1
        state['rows'].append(dict(step=state['steps'], primary=float(primary.detach()), frozen_baseline_loss=float(reference_loss),
            auxiliary=float(auxiliary.detach()), gradients=gradients, gradient_norm=float(norm),
            relative_residual_mean=float(delta.mean()), relative_residual_max=float(delta.max()), places=len(labels.unique()), images=len(labels)))
        write(args.output/'progress.json', dict(phase='training', arm=args.arm, seed=args.seed, steps=state['steps'], total=len(schedule), last_loss=float(loss.detach())))
        if state['steps']%16 == 0:
            save(args.output/'last.pt', model, optimizer, state)
    if state['steps'] != len(schedule) or tensor_hash(visual) != frozen:
        raise ValueError('Incomplete updates or changed frozen model')
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
    checks = dict(zero_start=True, common_interface_initialization=True, frozen_plain_unchanged=True,
        checkpoint_roundtrip=True, disabled_auxiliary_unchanged=True, nonzero_updates=final != state['initial_parameter_sha256'],
        all_actual_modules_receive_gradient=all(state['module_gradients'].values()), up_gradient_seen=state['up_gradient_seen'])
    write(args.output/'summary.json', dict(arm=args.arm, seed=args.seed, stage=args.stage, steps=state['steps'], checks=checks,
        module_gradients=state['module_gradients'], trainable_parameters=count, excluding_zero_weight_auxiliary_parameters=count-65536,
        rows=state['rows'], final_parameter_sha256=final, scope=POLICY['scope']))
    write(args.output/'progress.json', dict(phase='complete', steps=state['steps'])); seal(args.output)
    print('FUSION CONTROL COMPLETE', args.seed, args.arm, checks, flush=True)
    if not all(checks.values()):
        raise SystemExit(2)


def verify_arm(args, s, arm, stage, provenance, identity):
    run = args.root/('seed'+str(s))/(arm+'_'+stage); c = verify(run); r = read(run/'summary.json')
    if c['arm'] != arm or c['initialization_seed'] != s or c['stage'] != stage:
        raise ValueError('Arm/seed/stage differs')
    if c['conv_control_policy'] != POLICY or c['conv_reference'] != identity or c['reference_evidence'] != provenance:
        raise ValueError('Policy or evidence changed')
    if c['policy'] != effective_policy(s) or c['code'] != {p: code_hash(ROOT/p) for p in SOURCES}:
        raise ValueError('Scientific settings or implementation changed')
    if c['original_code'] != current_code() or r['steps'] != POLICY[stage+'_steps'] or not all(r['checks'].values()):
        raise ValueError('Incomplete mechanics/budget or original implementation changed')
    original = verify(args.followup_reference/('seed'+str(s))/('conv_'+stage))
    if matched_contract(c) != matched_contract(original):
        raise ValueError('Unmatched sealed convolution control')
    return c


def compare(args):
    baseline, provenance, identity = references(args); runs = args.root/('seed'+str(args.seed))
    contracts = {a: verify_arm(args, args.seed, a, args.stage, provenance, identity) for a in ARMS}
    initial = {a: read(runs/(a+'_'+args.stage)/'initial.json') for a in ARMS}
    original_run = args.followup_reference/('seed'+str(args.seed))/('conv_'+args.stage)
    old = read(original_run/'initial.json')
    checks = dict(mechanical=True, same_budget_data_optimizer=True,
        exact_conv_last_initialization=initial['conv_last']['parameter_sha256'] == old['parameter_sha256'],
        same_common_interface_initialization=len({x['shared_interface_sha256'] for x in initial.values()}) == 1,
        same_baseline_zero_start=all(x['descriptor_sha256'] == old['descriptor_sha256'] for x in initial.values()))
    report = {}; standard = {}
    if args.stage == 'screen':
        source = verify(args.source); plan = source['plan']
        if sha(args.source/'completed.json') != contracts['conv_last']['source_completed_sha256']:
            raise ValueError('Query source changed')
        for dataset in ('gsv_final', 'msls-val_final', 'pitts30k-val_final'):
            b = read(baseline/(dataset+'.json'))['baseline']
            values = {a: read(runs/(a+'_screen')/(dataset+'.json')) for a in ARMS}
            values['conv'] = read(original_run/(dataset+'.json'))
            if any(x['baseline'] != b for x in values.values()):
                raise ValueError('Baseline/query ranking differs')
            outcomes = {k: x['student']['outcomes'] for k, x in values.items()}
            if dataset == 'gsv_final':
                report = summarize(outcomes, b['outcomes'], plan['query_place_groups'], plan['calibration_places'])
            else:
                name = dataset.removesuffix('_final')
                standard[name] = summarize(outcomes, b['outcomes'], np.zeros(len(b['outcomes']), int), 1)['all']
                standard[name]['recall_at_1_5_20'] = dict(baseline=b['recall'], **{k: x['student']['recall'] for k, x in values.items()})
    verdict = ('SMOKE_PASS' if args.stage == 'smoke' else 'FUSION_CONTROLS_COMPLETE') if all(checks.values()) else 'MECHANICAL_FAIL'
    contract = dict(policy=POLICY, seed=args.seed, stage=args.stage, conv_reference=identity,
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
    _, provenance, identity = references(args); reports = {}; hashes = {}
    for s in SEEDS:
        run = args.root/('seed'+str(s)); c = verify(run/'compare_screen'); r = read(run/'compare_screen/summary.json')
        if c['policy'] != POLICY or c['conv_reference'] != identity or r['verdict'] != 'FUSION_CONTROLS_COMPLETE' or not all(r['checks'].values()):
            raise ValueError('Comparison incomplete or changed')
        for arm in ARMS:
            verify_arm(args, s, arm, 'screen', provenance, identity)
            if sha(run/(arm+'_screen/completed.json')) != c['completed_hashes'][arm]:
                raise ValueError('Comparison/run identity differs')
        reports[s] = {'gsv_'+role: r['partitions'][role] for role in ('calibration', 'evaluation', 'all')}
        reports[s].update(r['standard_benchmark_paired']); hashes[str(s)] = sha(run/'compare_screen/completed.json')
    contract = dict(policy=POLICY, completed_hashes=hashes, conv_reference=identity, code={p: code_hash(ROOT/p) for p in SOURCES})
    open_contract(args.output, contract, args.resume)
    if (args.output/'completed.json').exists():
        verify(args.output); return
    result = aggregate_reports(reports)
    write(args.output/'summary.json', dict(verdict='FUSION_ABLATION_COMPLETE_NOT_NOVELTY_PROOF', **result,
        per_seed=reports, parameter_counts=COUNTS, fairness=POLICY['fairness'], scope=POLICY['scope']))
    seal(args.output); print('ALL FUSION CONTROLS COMPLETE', result['directional_checks'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['wait_gpu', 'train', 'compare', 'aggregate']); p.add_argument('--arm', choices=ARMS)
    p.add_argument('--seed', type=int, choices=SEEDS, default=42); p.add_argument('--stage', choices=['smoke', 'screen'], default='screen')
    p.add_argument('--checkpoint', type=Path, default=Path(PLAIN_CHECKPOINT))
    p.add_argument('--reference', type=Path, default=Path('logs/hcen_plain_base_v1'))
    p.add_argument('--full-reference', type=Path, default=Path('logs/hcen_plain_multiseed_v1'))
    p.add_argument('--module-reference', type=Path, default=Path('logs/hcen_module_ablation_v1'))
    p.add_argument('--followup-reference', type=Path, default=Path('logs/hcen_simplified_2048_v1'))
    p.add_argument('--source', type=Path, default=Path('logs/slgd_hard_v2_audit')); p.add_argument('--cache', type=Path, default=Path('.cache/slgd_mid_v1_pilot'))
    p.add_argument('--teacher', type=Path, default=Path('logs/slgd_v1/teacher_pilot')); p.add_argument('--dataset-root', type=Path, default=Path('datasets/gsv_cities'))
    p.add_argument('--root', type=Path, default=Path('logs/conv_fusion_controls_v1')); p.add_argument('--output', type=Path)
    p.add_argument('--workers', type=int, default=4); p.add_argument('--resume', action='store_true'); args = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        p.error('Expose physical GPU1 only')
    if args.command == 'wait_gpu':
        wait_gpu(args); return
    if args.output is None or (args.command == 'train' and args.arm is None):
        p.error('Output/arm required')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        p.error('Need one visible CUDA device')
    if not args.checkpoint.is_file() or sha(args.checkpoint) != PLAIN_SHA:
        p.error('Pinned checkpoint differs')
    import fcntl
    args.root.mkdir(parents=True, exist_ok=True); lock = (args.root/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(.65, 0)
    {'train': train, 'compare': compare, 'aggregate': aggregate}[args.command](args)


if __name__ == '__main__':
    main()
