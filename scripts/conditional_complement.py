"""Three bounded warm-start controls; never overwrite an earlier experiment."""
import argparse
import os
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

from scripts.train_slgd import (
    read, write, sha, code_hash, verify, seal, seed, save, restore, parameter_hash)
from scripts.slgd_mid_screen import open_contract
from scripts.conv_official_layers import base_contract, SOURCES as OLD_SOURCES
from scripts.conv_official_diagnosis import author, make_adapter, load_parameters
from scripts.conv_official_boq import evaluate
from scripts.conv_residual_ablation import forward_batch
from scripts.train_hcen_vpr import PlaceBatches, tensor_hash
from scripts.hcen_module_ablation import wait_gpu
from src.hcen_plain_protocol import canonical_features
from src.official_layer_selection import selected_features
from src.models.conv_residual_ablation import module_gradient_status
from src.models.conditional_complement import (
    AnchoredConditionalNeck, anchor_parameters, conditional_gradient_status)
from src.conditional_complement_protocol import (
    ARMS, POLICY, WARM_PARAMETER_SHA, named_hash, matched_contract,
    compare_outcomes, candidate_verdict)
from src.slgd_mid_supervision import gps_filtered_retrieval

SOURCES = tuple(dict.fromkeys((
    'scripts/conditional_complement.py', 'src/models/conditional_complement.py',
    'src/conditional_complement_protocol.py', 'scripts/run_conditional_complement.sh',
    'tests/test_conditional_complement.py', *OLD_SOURCES)))


def study_contract(args):
    c, data, cache = base_contract(args, 'conditional_' + args.stage)
    warm = verify(args.warm_start)
    summary = read(args.warm_start / 'summary.json')
    if (warm['arm'] != 'last_only' or warm['fusion_layers'] != [12, 12, 12]
            or summary['final_parameter_sha256'] != WARM_PARAMETER_SHA
            or not all(summary['checks'].values())):
        raise ValueError('Not the pinned completed last-only warm start')
    if warm['code'] != {p: code_hash(ROOT / p) for p in warm['code']}:
        raise ValueError('Warm-start sealed source code changed')
    for key in ('metadata', 'train_places', 'benchmark_metadata',
                'checkpoint_sha256', 'versions', 'schedule', 'local_dino_sources'):
        if warm[key] != c[key]:
            raise ValueError('Warm-start population/runtime changed: ' + key)
    if c['schedule'][:512] * 4 != c['schedule']:
        raise ValueError('Expected the original fixed2048 schedule')
    if args.stage == 'smoke':
        c['schedule'] = c['schedule'][:POLICY['smoke_steps']]
    c.update(policy=POLICY, arm=args.arm,
             warm_completed_sha256=sha(args.warm_start / 'completed.json'),
             warm_checkpoint_sha256=sha(args.warm_start / 'last.pt'),
             code={p: code_hash(ROOT / p) for p in SOURCES})
    return c, data, cache


def make_model(args, images):
    """Warm-start all arms identically, probe BEFORE restoring any new run."""
    visual = author(args)
    frozen = tensor_hash(visual)
    model = make_adapter(visual, POLICY['rho'])
    model.fusion_layers = (12, 12, 12)
    model.features = MethodType(selected_features, model)
    old_state = load_parameters(model, args.warm_start / 'last.pt')
    if (old_state['steps'] != POLICY['warm_updates'] or parameter_hash(model) != WARM_PARAMETER_SHA
            or old_state['contract_sha256'] != sha(args.warm_start / 'contract.json')):
        raise ValueError('Warm-start checkpoint parameters/steps changed')
    probes = {280: images[:4], 322: F.interpolate(
        images[:4], (322, 322), mode='bicubic', align_corners=False, antialias=True)}
    warm_descriptors = {}
    with torch.inference_mode():
        for size, pixels in probes.items():
            warm_descriptors[size] = forward_batch(model, pixels)[0].clone()
    seed(POLICY['seed'])
    model.neck = AnchoredConditionalNeck(
        model.neck, args.arm, dim=POLICY['dim'], rank=POLICY['rank']).cuda()
    # Capture identical native depths in ALL arms; routing happens in the neck.
    model.fusion_layers = tuple(POLICY['layers'])
    errors = {}
    with torch.inference_mode():
        for size, pixels in probes.items():
            current = forward_batch(model, pixels)[0]
            errors[str(size)] = float((current - warm_descriptors[size]).abs().max())
    if max(errors.values()) > 1e-6:
        raise ValueError('Conditional branch breaks warm-start descriptor equality')
    active = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not active or any(not n.startswith('neck.') for n in active):
        raise ValueError('Only neck parameters may be trained')
    anchor = anchor_parameters(model)
    supplement = {n: p for n, p in active.items() if n.startswith('neck.complement.')}
    initial = dict(
        zero_addition_errors=errors, warm_parameter_sha256=WARM_PARAMETER_SHA,
        anchor_parameter_sha256=named_hash(anchor),
        complement_parameter_sha256=named_hash(supplement) if supplement else None,
        initial_parameter_sha256=parameter_hash(model),
        trainable_parameters=sum(p.numel() for p in active.values()),
        anchor_parameters=sum(p.numel() for p in anchor.values()),
        complement_parameters=sum(p.numel() for p in supplement.values()),
        frozen_auxiliary_parameters=sum(p.numel() for p in model.neck.aux.parameters()),
        frozen_tensor_sha256=frozen,
        probe='First4 GSV training images;322 is mechanical resize, not benchmark model selection')
    # A constructs no extra weights; B/C do. Reset after construction/probes so
    # every arm starts the training loader and checkpoint RNG at the same seed.
    seed(POLICY['seed'])
    return model, active, initial


def training_forward(model, images):
    """Checkpoint trainable neck/readout; collect each microbatch only once."""
    descriptors, delta, stats = [], [], {}
    for pixels in images.cuda().split(POLICY['microbatch']):
        features = canonical_features(model.features(pixels))
        descriptor, residual = checkpoint(
            lambda a, b, c: model.aggregate(a, b, c, auxiliary=False),
            *features, use_reentrant=False)
        if (descriptor.shape != (len(pixels), 12288)
                or not bool(torch.isfinite(descriptor).all())
                or float((descriptor.detach().norm(dim=1) - 1).abs().max()) > 1e-4):
            raise ValueError('Nonfinite/nonunit descriptor')
        descriptors.append(descriptor)
        delta.append(residual.detach().flatten(1).norm(dim=1)
                     / features[-1].flatten(1).norm(dim=1).clamp_min(1e-8))
        for key, value in model.neck.last_conditional_stats.items():
            stats.setdefault(key, []).append(value)
        model.neck.last_raw = None
        model.neck.last_conditional_stats = None
    return (torch.cat(descriptors), torch.cat(delta),
            {key: float(torch.cat(values).mean()) for key, values in stats.items()})


def train(args):
    from tqdm import tqdm
    from src.losses.vpr_losses import VPRLossFunction
    c, data, cache = study_contract(args)
    if args.stage == 'train':
        smoke = verify(args.root / 'smoke_aggregate')
        smoke_report = read(args.root / 'smoke_aggregate/summary.json')
        if (smoke['policy'] != POLICY or smoke['code'] != c['code']
                or smoke_report['verdict'] != 'SMOKE_PASS_DISCARD_UPDATES'
                or not all(smoke_report['checks'].values())):
            raise ValueError('All three current-code smoke controls must pass before training')
        for arm in ARMS:
            path = args.root / ('smoke_' + arm)
            smoke_arm = verify(path)
            if (sha(path / 'completed.json') != smoke['completed'][arm]
                    or smoke_arm['warm_checkpoint_sha256'] != c['warm_checkpoint_sha256']):
                raise ValueError('Smoke was not performed on the same warm checkpoint')
    open_contract(args.output, c, args.resume)
    if (args.output / 'completed.json').exists():
        verify(args.output)
        print('Already complete', args.output, flush=True)
        return
    images, _, _ = PlaceBatches(data, cache['train_indices'], c['schedule'])[0]
    model, active, initial = make_model(args, images)
    initial_aux = tensor_hash(model.neck.aux)
    groups = [dict(params=list(anchor_parameters(model).values()), lr=POLICY['anchor_lr'])]
    extra = [p for n, p in active.items() if n.startswith('neck.complement.')]
    if extra:
        groups.append(dict(params=extra, lr=POLICY['complement_lr']))
    optimizer = torch.optim.AdamW(groups, weight_decay=POLICY['weight_decay'])
    state = dict(steps=0, rows=[], module_gradients={'0': False, '1': False},
                 conditional_gradients={k: False for k in conditional_gradient_status(model.neck)},
                 anchor_up_gradient=False, contract_sha256=sha(args.output / 'contract.json'))
    if (args.output / 'initial.json').exists():
        if read(args.output / 'initial.json') != initial:
            raise ValueError('Reconstructed initial model differs')
        if not (args.output / 'last.pt').is_file():
            raise ValueError('Initial record exists but checkpoint is missing')
    else:
        write(args.output / 'initial.json', initial)
    if (args.output / 'last.pt').is_file():
        state = restore(args.output / 'last.pt', model, optimizer, state['contract_sha256'])
    else:
        save(args.output / 'last.pt', model, optimizer, state)
    steps = len(c['schedule'])
    if not 0 <= state['steps'] <= steps:
        raise ValueError('Invalid checkpoint step count')
    model.train()
    lossfn = VPRLossFunction()
    loader = DataLoader(PlaceBatches(data, cache['train_indices'], c['schedule'][state['steps']:]),
                        batch_size=None, num_workers=args.workers)
    for images, labels, coords in tqdm(loader, total=steps, initial=state['steps'],
                                      desc=f'Conditional {args.arm} {args.stage}'):
        optimizer.zero_grad(set_to_none=True)
        descriptor, delta, stats = training_forward(model, images)
        loss = gps_filtered_retrieval(lossfn, descriptor, labels.cuda(), coords.numpy())
        if not bool(torch.isfinite(loss)) or float(delta.max()) > POLICY['rho'] + 1e-6:
            raise ValueError('Nonfinite loss or relative residual cap violation')
        loss.backward()
        for key, yes in module_gradient_status(model).items():
            state['module_gradients'][key] |= yes
        for key, yes in conditional_gradient_status(model.neck).items():
            state['conditional_gradients'][key] |= yes
        state['anchor_up_gradient'] |= bool(
            model.neck.up.weight.grad is not None and (model.neck.up.weight.grad != 0).any())
        if any(p.grad is None for p in active.values()):
            raise ValueError('Unused trainable parameter; no dummy zero-gradient padding')
        norm = torch.nn.utils.clip_grad_norm_(list(active.values()), POLICY['clip'], error_if_nonfinite=True)
        if any(p.grad is not None for p in model.visual.parameters()):
            raise ValueError('Frozen official network received gradients')
        optimizer.step()
        state['steps'] += 1
        state['rows'].append(dict(step=state['steps'], loss=float(loss.detach()),
            gradient_norm=float(norm), relative_residual_mean=float(delta.mean()),
            relative_residual_max=float(delta.max()), conditional=stats))
        write(args.output / 'progress.json', dict(phase='training', arm=args.arm,
            steps=state['steps'], total=steps, loss=float(loss.detach())))
        if state['steps'] % 16 == 0:
            save(args.output / 'last.pt', model, optimizer, state)
    save(args.output / 'last.pt', model, optimizer, state)
    final = parameter_hash(model)
    restore(args.output / 'last.pt', model, optimizer, state['contract_sha256'])
    checks = dict(
        zero_addition_reproduces_warm=max(initial['zero_addition_errors'].values()) <= 1e-6,
        complete_steps=state['steps'] == steps,
        frozen_official_unchanged=tensor_hash(model.visual) == initial['frozen_tensor_sha256'],
        frozen_auxiliary_unchanged=tensor_hash(model.neck.aux) == initial_aux,
        anchor_module_gradients=all(state['module_gradients'].values()),
        anchor_up_gradient=state['anchor_up_gradient'],
        conditional_gradients=all(state['conditional_gradients'].values()),
        parameters_updated=final != initial['initial_parameter_sha256'],
        checkpoint_roundtrip=parameter_hash(model) == final,
        optimizer_clocks=all(int(optimizer.state[p]['step']) == steps for p in active.values()))
    if not all(checks.values()):
        raise ValueError('Mechanics failed: ' + str(checks))
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    if args.stage == 'train':
        evaluate(SimpleNamespace(**{**vars(args), 'root': args.reference}), model, args.output, True)
    write(args.output / 'summary.json', dict(checks=checks, arm=args.arm, stage=args.stage,
        additional_updates=steps, inherited_updates=POLICY['warm_updates'],
        final_parameter_sha256=final, rows=state['rows'],
        conditional_gradient_inventory=state['conditional_gradients'], scope=POLICY['scope']))
    write(args.output / 'progress.json', dict(phase='complete', steps=steps))
    seal(args.output)


def aggregate(args):
    paths = {arm: args.root / (args.stage + '_' + arm) for arm in ARMS}
    contracts = {arm: verify(path) for arm, path in paths.items()}
    initials = {arm: read(path / 'initial.json') for arm, path in paths.items()}
    reports = {arm: read(path / 'summary.json') for arm, path in paths.items()}
    first = contracts[ARMS[0]]
    if first['policy'] != POLICY or first['code'] != {p: code_hash(ROOT / p) for p in SOURCES}:
        raise ValueError('Policy/source changed')
    checks = dict(
        matched_contracts=all(matched_contract(c) == matched_contract(first) for c in contracts.values()),
        matched_warm_anchors=len({v['anchor_parameter_sha256'] for v in initials.values()}) == 1,
        matched_bc_init=initials[ARMS[1]]['initial_parameter_sha256'] == initials[ARMS[2]]['initial_parameter_sha256'],
        matched_bc_complement_init=initials[ARMS[1]]['complement_parameter_sha256'] == initials[ARMS[2]]['complement_parameter_sha256'] is not None,
        matched_bc_capacity=initials[ARMS[1]]['trainable_parameters'] == initials[ARMS[2]]['trainable_parameters'],
        mechanical=all(all(r['checks'].values()) for r in reports.values()))
    checks['registered_arm_and_stage'] = all(
        contracts[arm]['arm'] == arm and reports[arm]['arm'] == arm
        and reports[arm]['stage'] == args.stage
        and reports[arm]['additional_updates'] == len(contracts[arm]['schedule'])
        for arm in ARMS)
    if not all(checks.values()):
        raise ValueError('Three-arm fairness failed: ' + str(checks))
    c = dict(policy=POLICY, stage=args.stage, code=first['code'],
             completed={arm: sha(path / 'completed.json') for arm, path in paths.items()})
    open_contract(args.output, c, args.resume)
    if (args.output / 'completed.json').exists():
        verify(args.output)
        return
    metrics = {}
    verdict = dict(verdict='SMOKE_PASS_DISCARD_UPDATES', stable_gain_proven=False)
    if args.stage == 'train':
        verify(args.reference / 'baseline')
        verify(args.warm_start)
        if (sha(args.reference / 'baseline/completed.json') != first['official_baseline_completed_sha256']
                or sha(args.warm_start / 'completed.json') != first['warm_completed_sha256']):
            raise ValueError('Reference result hashes changed')
        for size in POLICY['eval_sizes']:
            for ds in ('msls-val', 'pitts30k-val'):
                tag = f'{ds}_{size}'
                values = {arm: read(path / (tag + '.json')) for arm, path in paths.items()}
                baseline = read(args.reference / 'baseline' / (tag + '.json'))
                warm = read(args.warm_start / (tag + '.json'))
                metrics[tag] = compare_outcomes(
                    {arm: row['outcomes'] for arm, row in values.items()},
                    baseline['outcomes'], warm['outcomes'])
                metrics[tag]['recall_at_1_5_10_20'] = {
                    'official': baseline['recall'], 'warm_start': warm['recall'],
                    **{arm: row['recall'] for arm, row in values.items()}}
        verdict = candidate_verdict(metrics)
    write(args.output / 'summary.json', dict(**verdict, checks=checks, metrics=metrics,
        parameter_inventory={arm: {k: v for k, v in row.items() if k.endswith('parameters')}
                             for arm, row in initials.items()},
        formal_training_runs=3 if args.stage == 'train' else 0,
        no_automatic_followup=True, scope=POLICY['scope']))
    seal(args.output)
    write(args.root / 'progress.json', dict(phase='complete' if args.stage == 'train' else 'smoke_complete',
        verdict=verdict['verdict'], formal_training_runs=3 if args.stage == 'train' else 0))
    print(verdict, flush=True)


def main():
    os.chdir(ROOT)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('wait_gpu', 'train', 'aggregate'))
    p.add_argument('--arm', choices=ARMS)
    p.add_argument('--stage', choices=('smoke', 'train'), default='smoke')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--root', type=Path, default=Path('logs/conditional_complement_v1'))
    p.add_argument('--output', type=Path)
    p.add_argument('--warm-start', type=Path, default=Path('logs/conv_official_layers_v1/last_only'))
    p.add_argument('--reference', type=Path, default=Path('logs/conv_official_boq_v1'))
    p.add_argument('--diagnosis', type=Path, default=Path('logs/conv_official_diagnosis_v1'))
    p.add_argument('--official-cache', type=Path, default=Path('.cache/boq_official_v1'))
    p.add_argument('--dino-source', type=Path, default=Path('/home/wt/.cache/torch/hub/facebookresearch_dinov2_main'))
    p.add_argument('--dataset-root', type=Path, default=Path('datasets/gsv_cities'))
    p.add_argument('--source', type=Path, default=Path('logs/slgd_hard_v2_audit'))
    p.add_argument('--teacher', type=Path, default=Path('logs/slgd_v1/teacher_pilot'))
    p.add_argument('--cache', type=Path, default=Path('.cache/slgd_mid_v1_pilot'))
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        p.error('Expose physical GPU1 only')
    protected = [args.warm_start, args.reference, args.diagnosis, args.source, args.teacher, args.cache]
    root = args.root.resolve()
    if (root.parent != (ROOT / 'logs').resolve() or not root.name.startswith('conditional_complement_')
            or any(root.is_relative_to(x.resolve()) or x.resolve().is_relative_to(root) for x in protected)):
        p.error('Use a separate logs/conditional_complement_<version> root')
    if args.workers < 0:
        p.error('workers must be nonnegative')
    if args.command == 'wait_gpu':
        wait_gpu(args)
        return
    if args.command == 'train' and not args.arm:
        p.error('Choose one registered arm')
    name = args.stage + '_' + (args.arm if args.command == 'train' else 'aggregate')
    if args.output is None or args.output.resolve().parent != root or args.output.name != name:
        p.error('Output must be the exact stage/arm child of the new root: ' + name)
    import fcntl
    args.root.mkdir(parents=True, exist_ok=True)
    # Both direct commands and the shell launcher honor the project-wide lock.
    with (ROOT / 'logs/query_relation_screen_v1.lock').open('a') as shared:
        if os.environ.get('CONDITIONAL_SHARED_LOCK') == 'held':
            inherited = os.fstat(9)
            expected = os.fstat(shared.fileno())
            if (inherited.st_ino, inherited.st_dev) != (expected.st_ino, expected.st_dev):
                raise ValueError('Inherited shared-lock descriptor does not match the project lock')
            fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            fcntl.flock(shared, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (args.root / 'worker.lock').open('a') as worker:
            fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            if args.command == 'train':
                if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                    p.error('Need exactly one visible CUDA device')
                torch.cuda.set_per_process_memory_fraction(.65, 0)
            try:
                {'train': train, 'aggregate': aggregate}[args.command](args)
            except Exception as error:
                failure = dict(phase='failed', stage=args.stage, arm=args.arm,
                               error_type=type(error).__name__, message=str(error))
                write(args.root / 'progress.json', failure)
                if args.output.exists() and not (args.output / 'completed.json').exists():
                    write(args.output / 'failure.json', failure)
                raise


if __name__ == '__main__':
    main()
