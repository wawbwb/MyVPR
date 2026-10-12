"""Run seed43/44 without modifying or invalidating the sealed seed42 code."""
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
from scripts.train_slgd import RU_SHA, read, write, sha, code_hash, verify, seal, seed, save, restore, parameter_hash
from scripts.slgd_mid_screen import dataset, sources, load_visual, open_contract, verify_cache_shards, current_code
from scripts.train_hcen_vpr import PlaceBatches, tensor_hash, forward_batch, gsv_evaluate, benchmarks
from scripts.train_hcen_aux_ablation import previous_provenance, SOURCES as BASE_SOURCES
from scripts.audit_slgd_hard import PlannedPlaces
from src.models.hcen_vpr import HCENVPR
from src.slgd_mid_supervision import gps_filtered_retrieval
from src.hcen_aux_ablation import ARMS, training_schedule, objective, gradient_diagnostic, paired_report
from src.hcen_replication import POLICY as REPLICATION_POLICY, NEW_SEEDS, effective_policy, comparable_contract, aggregate_reports

SOURCES = ('scripts/hcen_multiseed.py', 'src/hcen_replication.py', *BASE_SOURCES)


def reference_provenance(root):
    contract = verify(root/'compare_screen')
    if read(root/'compare_screen/summary.json')['verdict'] != 'FACTORIAL_COMPLETE_NOT_FULL_TRAINING':
        raise ValueError('Seed42 comparison incomplete')
    for arm in ARMS:
        run = root/(arm+'_screen'); c = verify(run)
        if sha(run/'completed.json') != contract['completed_hashes'][arm]:
            raise ValueError('Seed42 completed run differs')
        if c['code'] != {p: code_hash(ROOT/p) for p in BASE_SOURCES} or c['policy'] != effective_policy(42):
            raise ValueError('Sealed seed42 source or policy changed')
    return dict(comparison_sha256=sha(root/'compare_screen/completed.json'))


def train(args):
    from tqdm import tqdm
    from src.losses.vpr_losses import VPRLossFunction
    POLICY = effective_policy(args.seed)
    teacher, heldout, metadata = sources(SimpleNamespace(**{**vars(args), 'stage': 'pilot'}))
    cache = verify(args.cache)
    if cache['code'] != current_code() or cache['stage'] != 'pilot' or cache['metadata'] != metadata:
        raise ValueError('Original full training cache differs')
    if cache['source_completed_sha256'] != sha(args.source/'completed.json'):
        raise ValueError('Heldout source differs')
    if cache['teacher_completed_sha256'] != sha(args.teacher/'completed.json') or cache['train_places'] != teacher['train_places'][:len(cache['train_places'])]:
        raise ValueError('Original train split differs')
    verify_cache_shards(args.cache, len(cache['train_places']))
    schedule = training_schedule(read(args.cache/'triplets.json')['rows'], args.stage)
    provenance = previous_provenance(args.previous)
    reference = reference_provenance(args.reference)
    benchmark_metadata = {str(p): sha(p) for name in ('msls-val', 'pitts30k-val')
                          for p in sorted((Path('datasets')/name).glob('*.npy'))}
    contract = dict(arm=args.arm, **ARMS[args.arm], stage=args.stage, policy=POLICY,
        initialization_seed=args.seed, replication_policy=REPLICATION_POLICY, replication_reference=reference,
        ru_sha256=RU_SHA, cache_completed_sha256=sha(args.cache/'completed.json'),
        source_completed_sha256=sha(args.source/'completed.json'), previous=provenance,
        metadata=metadata, benchmark_metadata=benchmark_metadata, train_places=cache['train_places'],
        schedule=schedule, original_code=current_code(), code={p: code_hash(ROOT/p) for p in SOURCES},
        versions=dict(torch=torch.__version__, numpy=np.__version__))
    original = verify(args.reference/(args.arm+'_'+('screen' if args.stage == 'screen' else 'smoke')))
    if comparable_contract(original) != comparable_contract(contract):
        raise ValueError('Protocol differs from seed42 beyond initialization')
    open_contract(args.output, contract, args.resume)
    if (args.output/'completed.json').exists():
        verify(args.output); print('Already complete', args.arm, args.seed); return
    data = dataset(args)
    if [str(data.places_ids[i]) for i in cache['train_indices']] != cache['train_places']:
        raise ValueError('Original training identities differ')
    plan = heldout['plan']
    if [str(data.places_ids[i]) for i in plan['dataset_indices']] != plan['query_places']+plan['distractor_places']:
        raise ValueError('Original holdout identities differ')
    seed(42); visual = load_visual(args); frozen_hash = tensor_hash(visual)
    seed(args.seed); model = HCENVPR(visual, ARMS[args.arm]['mode']).cuda().eval()
    probe = PlannedPlaces(data, plan)[0][0]
    with torch.inference_mode():
        d, r, _, _ = forward_batch(model, probe)
    error = float((d-r).abs().max())
    if error > 2e-6 or not torch.isfinite(d).all():
        raise ValueError('Zero start differs from original RU')
    active = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not active or any(not n.startswith('neck.') for n in active):
        raise ValueError('Only the original HCEN neck may train')
    optimizer = torch.optim.AdamW(active.values(), lr=POLICY['lr'], weight_decay=0.)
    state = dict(steps=0, rows=[], ssm_gradient_seen=False, up_gradient_seen=False,
        initial_auxiliary_sha256=tensor_hash(model.neck.aux),
        initial_parameter_sha256=parameter_hash(model), contract_sha256=sha(args.output/'contract.json'))
    if (args.output/'last.pt').exists():
        state = restore(args.output/'last.pt', model, optimizer, state['contract_sha256'])
    if not (args.output/'initial.json').exists():
        write(args.output/'initial.json', dict(zero_start_error=error,
            initialization_seed=args.seed, parameter_sha256=state['initial_parameter_sha256'],
            descriptor_sha256=hashlib.sha256(d.cpu().numpy().tobytes()).hexdigest(),
            frozen_tensor_sha256=frozen_hash, trainable_parameters=sum(p.numel() for p in active.values())))
        save(args.output/'last.pt', model, optimizer, state)
    write(args.output/'source_outcomes.json', read(args.source/'outcomes.json'))
    lossfn = VPRLossFunction(); model.train()
    loader = DataLoader(PlaceBatches(data, cache['train_indices'], schedule[state['steps']:]),
                        batch_size=None, num_workers=args.workers)
    weight = ARMS[args.arm]['auxiliary_weight']
    for images, labels, coordinates in tqdm(loader, total=len(schedule), initial=state['steps'],
                                             desc=args.arm+' '+args.stage):
        optimizer.zero_grad(set_to_none=True)
        d, r, aux, delta = forward_batch(model, images, True); labels = labels.cuda(); coords = coordinates.numpy()
        primary = gps_filtered_retrieval(lossfn, d, labels, coords)
        auxiliary = sum(gps_filtered_retrieval(lossfn, x, labels, coords) for x in aux)/len(aux)
        loss = objective(primary, auxiliary, args.arm)
        with torch.no_grad():
            reference_loss = gps_filtered_retrieval(lossfn, r, labels, coords)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite ablation objective')
        diagnostic = gradient_diagnostic(primary, weight*auxiliary, model.named_parameters()) if state['steps']%64 == 0 else None
        loss.backward()
        state['ssm_gradient_seen'] |= any('ssm.select' in n and p.grad is not None and bool((p.grad != 0).any()) for n, p in active.items())
        state['up_gradient_seen'] |= bool(model.neck.up.weight.grad is not None and (model.neck.up.weight.grad != 0).any())
        for p in active.values():
            if p.grad is None: p.grad = torch.zeros_like(p)
        norm = torch.nn.utils.clip_grad_norm_(list(active.values()), POLICY['clip'], error_if_nonfinite=True)
        if any(p.grad is not None for p in visual.parameters()):
            raise ValueError('Frozen RU received gradient')
        if float(delta.max()) > POLICY['feature_relative_cap']+1e-6:
            raise ValueError('Original residual bound exceeded')
        optimizer.step(); state['steps'] += 1
        state['rows'].append(dict(step=state['steps'], loss=float(loss.detach()), primary=float(primary.detach()),
            frozen_ru_loss=float(reference_loss), auxiliary=float(auxiliary.detach()), auxiliary_weight=weight,
            gradients=diagnostic, gradient_norm=float(norm), relative_residual_mean=float(delta.mean()),
            relative_residual_max=float(delta.max()), places=len(labels.unique()), images=len(labels)))
        write(args.output/'progress.json', dict(phase='training', arm=args.arm, steps=state['steps'],
                                               total=len(schedule), last_loss=state['rows'][-1]['loss']))
        if state['steps']%16 == 0: save(args.output/'last.pt', model, optimizer, state)
    if state['steps'] != len(schedule) or tensor_hash(visual) != frozen_hash:
        raise ValueError('Budget incomplete or frozen RU changed')
    save(args.output/'last.pt', model, optimizer, state)
    final_hash = parameter_hash(model)
    restore(args.output/'last.pt', model, optimizer, state['contract_sha256'])
    if parameter_hash(model) != final_hash:
        raise ValueError('Checkpoint roundtrip differs')
    if any(int(optimizer.state[p]['step']) != len(schedule) for p in active.values()):
        raise ValueError('Optimizer clocks differ')
    aux_unchanged = tensor_hash(model.neck.aux) == state['initial_auxiliary_sha256']
    if weight == 0 and not aux_unchanged:
        raise ValueError('Disabled auxiliary head changed')
    optimizer.zero_grad(set_to_none=True); model.eval()
    if args.stage == 'screen':
        gsv_evaluate(model, data, heldout, args.output, args.workers)
        benchmarks(model, data, args.output, args.workers)
    checks = dict(zero_start=error <= 2e-6, frozen_ru_unchanged=True, checkpoint_roundtrip=True,
        nonzero_updates=final_hash != state['initial_parameter_sha256'],
        ssm_gradient_seen=state['ssm_gradient_seen'], up_gradient_seen=state['up_gradient_seen'],
        disabled_auxiliary_unchanged=weight != 0 or aux_unchanged)
    write(args.output/'summary.json', dict(arm=args.arm, stage=args.stage, steps=state['steps'], checks=checks,
        initialization_seed=args.seed, initial_parameter_sha256=state['initial_parameter_sha256'],
        final_parameter_sha256=final_hash, rows=state['rows'], scope=REPLICATION_POLICY['scope']))
    write(args.output/'progress.json', dict(phase='complete', steps=state['steps']))
    seal(args.output); print('COMPLETE', args.arm, args.seed, checks, flush=True)
    if not all(checks.values()): raise SystemExit(2)


def compare(args):
    runs = args.root/('seed'+str(args.seed)); contracts = {}; summaries = {}; initials = {}
    for arm in ARMS:
        run = runs/(arm+'_'+args.stage)
        contracts[arm] = verify(run); summaries[arm] = read(run/'summary.json'); initials[arm] = read(run/'initial.json')
        if contracts[arm]['arm'] != arm or {k: contracts[arm][k] for k in ('mode', 'auxiliary_weight')} != ARMS[arm]:
            raise ValueError('Arm identity differs')
        if contracts[arm]['initialization_seed'] != args.seed or contracts[arm]['policy'] != effective_policy(args.seed):
            raise ValueError('Seed/policy differs')
        if contracts[arm]['code'] != {p: code_hash(ROOT/p) for p in SOURCES} or contracts[arm]['original_code'] != current_code():
            raise ValueError('Training source implementation differs')
        original = verify(args.reference/(arm+'_'+args.stage))
        if comparable_contract(original) != comparable_contract(contracts[arm]):
            raise ValueError('Protocol no longer matches seed42')
    matched = [{k: v for k, v in c.items() if k not in ('arm', 'mode', 'auxiliary_weight')} for c in contracts.values()]
    if any(c != matched[0] for c in matched[1:]): raise ValueError('Unmatched four-arm controls')
    checks = dict(mechanical=all(all(s['checks'].values()) for s in summaries.values()),
        identical_initial_parameters=len({i['parameter_sha256'] for i in initials.values()}) == 1,
        identical_initial_descriptors=len({i['descriptor_sha256'] for i in initials.values()}) == 1,
        identical_budget=all(s['steps'] == effective_policy(args.seed)[args.stage+'_steps'] for s in summaries.values()),
        same_ru_zero_start_as_seed42=all(i['descriptor_sha256'] == read(args.reference/(a+'_'+args.stage)/'initial.json')['descriptor_sha256'] for a,i in initials.items()),
        different_initialization_from_seed42=all(i['parameter_sha256'] != read(args.reference/(a+'_'+args.stage)/'initial.json')['parameter_sha256'] for a,i in initials.items()))
    report = {}; standard = {}; source = verify(args.source)
    if sha(args.source/'completed.json') != matched[0]['source_completed_sha256']:
        raise ValueError('Comparison heldout differs')
    if args.stage == 'screen':
        old = read(args.source/'outcomes.json'); plan = source['plan']
        results = {a: read(runs/(a+'_screen')/'gsv_final.json') for a in ARMS}
        if any(r['ru']['outcomes'] != old['ru'] for r in results.values()): raise ValueError('GSV baseline differs')
        report = paired_report({a: r['student']['outcomes'] for a,r in results.items()}, old['ru'],
                               plan['query_place_groups'], old['hard'], plan['calibration_places'])
        for name in ('msls-val', 'pitts30k-val'):
            results = {a: read(runs/(a+'_screen')/(name+'_final.json')) for a in ARMS}
            ru = results['last_noaux']['ru']
            if any(r['ru'] != ru for r in results.values()): raise ValueError('Benchmark RU rankings differ')
            standard[name] = paired_report({a:r['student']['outcomes'] for a,r in results.items()},
                ru['outcomes'], np.zeros(len(ru['outcomes']),int), ~np.asarray(ru['outcomes'],bool),1)['all']
            standard[name]['recall_at_1_5_20'] = {a:r['student']['recall'] for a,r in results.items()}
    verdict = ('SMOKE_PASS' if args.stage == 'smoke' else 'FACTORIAL_COMPLETE_NOT_FULL_TRAINING') if all(checks.values()) else 'MECHANICAL_FAIL'
    contract = dict(replication_policy=REPLICATION_POLICY, initialization_seed=args.seed, stage=args.stage,
                    completed_hashes={a:sha(runs/(a+'_'+args.stage)/'completed.json') for a in ARMS})
    open_contract(args.output, contract, args.resume)
    if (args.output/'completed.json').exists(): verify(args.output)
    else:
        write(args.output/'summary.json',dict(verdict=verdict, checks=checks, initialization_seed=args.seed,
            partitions=report, standard_benchmark_paired=standard, scope=REPLICATION_POLICY['scope']))
        seal(args.output)
    print('COMPARISON', args.seed, verdict, checks, flush=True)
    if not all(checks.values()): raise SystemExit(2)


def aggregate(args):
    reference_provenance(args.reference)
    roots = {42:args.reference, 43:args.root/'seed43', 44:args.root/'seed44'}
    reports = {}; hashes = {}; initial_parameters = {}; descriptor_hashes = set(); frozen_hashes = set()
    for s, root in roots.items():
        c = verify(root/'compare_screen'); report = read(root/'compare_screen/summary.json')
        if report['verdict'] != 'FACTORIAL_COMPLETE_NOT_FULL_TRAINING' or not all(report['checks'].values()):
            raise ValueError('Incomplete/mechanically invalid seed')
        hashes[str(s)] = sha(root/'compare_screen/completed.json')
        initial_parameters[s] = set()
        for arm in ARMS:
            run = root/(arm+'_screen'); contract = verify(run)
            if sha(run/'completed.json') != c['completed_hashes'][arm]: raise ValueError('Comparison/run hash differs')
            original = verify(args.reference/(arm+'_screen'))
            if comparable_contract(contract) != comparable_contract(original): raise ValueError('Seeds used different scientific protocols')
            expected = BASE_SOURCES if s == 42 else SOURCES
            if contract['code'] != {p:code_hash(ROOT/p) for p in expected}: raise ValueError('Source hash differs')
            if s != 42 and contract['replication_reference'] != reference_provenance(args.reference):
                raise ValueError('Replication reference identity differs')
            initial = read(run/'initial.json'); initial_parameters[s].add(initial['parameter_sha256'])
            descriptor_hashes.add(initial['descriptor_sha256']); frozen_hashes.add(initial['frozen_tensor_sha256'])
            g = read(run/'gsv_final.json'); g0 = read(args.reference/(arm+'_screen')/'gsv_final.json')
            if g['ru'] != g0['ru']: raise ValueError('Different GSV RU/query rankings across seeds')
            for name in ('msls-val','pitts30k-val'):
                v = read(run/(name+'_final.json')); v0 = read(args.reference/(arm+'_screen')/(name+'_final.json'))
                if v['ru'] != v0['ru'] or v['preprocessing'] != v0['preprocessing']:
                    raise ValueError('Different benchmark baseline/protocol across seeds')
        reports[s] = {'gsv_'+role:report['partitions'][role] for role in ('calibration','evaluation','all')}
        reports[s].update(report['standard_benchmark_paired'])
    if any(len(v) != 1 for v in initial_parameters.values()) or len(set.union(*initial_parameters.values())) != 3:
        raise ValueError('Initialization hashes do not identify three matched but distinct seeds')
    if len(descriptor_hashes) != 1 or len(frozen_hashes) != 1: raise ValueError('Zero-start/frozen RU differs across seeds')
    contract = dict(policy=REPLICATION_POLICY, completed_hashes=hashes, code={p:code_hash(ROOT/p) for p in SOURCES})
    open_contract(args.output, contract, args.resume)
    if (args.output/'completed.json').exists(): verify(args.output); return
    write(args.output/'summary.json',dict(verdict='REPLICATION_COMPLETE_NOT_STABLE_GAIN_PROOF',
        checks=dict(matched_scientific_protocol=True, three_distinct_initializations=True, same_frozen_ru=True),
        metrics=aggregate_reports(reports), per_seed=reports, scope=REPLICATION_POLICY['scope'],
        interpretation='Means/std describe initialization variability on fixed queries. Seeds do not triple the number of independent test queries; no pooled significance claim.'))
    seal(args.output); print('ALL THREE SEEDS COMPLETE', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['train','compare','aggregate']); p.add_argument('--arm',choices=tuple(ARMS))
    p.add_argument('--seed',type=int,choices=NEW_SEEDS,default=43); p.add_argument('--stage',choices=['smoke','screen'],default='screen')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path('logs/slgd_hard_v2_audit'))
    p.add_argument('--cache',type=Path,default=Path('.cache/slgd_mid_v1_pilot'))
    p.add_argument('--teacher',type=Path,default=Path('logs/slgd_v1/teacher_pilot'))
    p.add_argument('--previous',type=Path,default=Path('logs/hcen_vpr_v2'))
    p.add_argument('--reference',type=Path,default=Path('logs/hcen_aux_ablation_v1'))
    p.add_argument('--dataset-root',type=Path,default=Path('datasets/gsv_cities'))
    p.add_argument('--root',type=Path,default=Path('logs/hcen_multiseed_v1'))
    p.add_argument('--output',type=Path,required=True); p.add_argument('--workers',type=int,default=4)
    p.add_argument('--resume',action='store_true'); args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        p.error('Expose physical GPU1 only')
    if not args.checkpoint.is_file() or sha(args.checkpoint) != RU_SHA: p.error('Original RU checkpoint differs')
    if args.command == 'train' and args.arm is None: p.error('Training arm required')
    import fcntl
    args.root.mkdir(parents=True,exist_ok=True)
    lock=(args.root/'worker.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(.65,0)
    {'train':train,'compare':compare,'aggregate':aggregate}[args.command](args)


if __name__=='__main__':main()
