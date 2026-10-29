"""Matched official ResNet50 transfer. No training without --allow-training."""
import argparse
import datetime
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

from scripts.train_slgd import read, write, sha, code_hash, verify, seal, seed, save, restore, parameter_hash
from scripts.train_hcen_vpr import PlaceBatches, tensor_hash
from scripts.slgd_mid_screen import open_contract
from src.models.resnet_module_transfer import (
    ARMS, verify_resnet_download, load_official_resnet, ResNetModuleTransfer, gradient_groups)
from src.resnet_transfer_protocol import POLICY, paired, assessment, project_sources

ENTRY_SOURCES = ('scripts/resnet_module_transfer.py', 'scripts/prepare_resnet_boq.py',
    'scripts/run_resnet_module_transfer.sh', 'src/models/resnet_module_transfer.py',
    'src/resnet_transfer_protocol.py', 'tests/test_resnet_module_transfer.py')


def input_transform(size):
    from torchvision import transforms as T
    return T.Compose([T.ToTensor(), T.Resize((size, size),
        interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.Normalize([.485, .456, .406], [.229, .224, .225])])


def benchmark_datasets(args, size):
    from src.dataloaders.valid.mapillary_sls import MapillarySLSDataset
    from src.dataloaders.valid.pittsburgh import PittsburghDataset
    transform = input_transform(size)
    return (MapillarySLSDataset(args.msls_root, transform),
            PittsburghDataset(args.pitts_root, transform))


def training_inputs(args):
    from src.dataloaders.train.gsv_cities import GSVCitiesDataset
    plan = verify(args.training_plan)
    cache = verify(args.training_cache)
    if plan['cache_completed_sha256'] != sha(args.training_cache/'completed.json'):
        raise ValueError('Training index cache differs from the original fixed schedule')
    if plan['train_places'] != cache['train_places'] or len(cache['train_places']) != 8192:
        raise ValueError('Expected the fixed8192 GSV-only train-place pool')
    metadata = {p.name: sha(p) for p in sorted((args.dataset_root/'Dataframes').glob('*.csv'))}
    if not metadata or metadata != cache['metadata'] or metadata != plan['metadata']:
        raise ValueError('GSV metadata identity changed')
    schedule = plan['schedule']
    if (len(schedule) != POLICY['steps'] or schedule != schedule[:512]*4
            or any(not row or len(row)>16 or len(set(row))!=len(row)
                   or any(type(x) is not int or x<0 or x>=8192 for x in row) for row in schedule)):
        raise ValueError('Unexpected immutable training schedule')
    seed(POLICY['seed'])
    data = GSVCitiesDataset(dataset_path=args.dataset_root,
        cities=sorted(p.stem for p in (args.dataset_root/'Dataframes').glob('*.csv')),
        img_per_place=4, transform=input_transform(POLICY['train_size']), return_metadata=True)
    indices = cache['train_indices']
    if (len(indices) != 8192 or len(set(indices)) != 8192
            or [str(data.places_ids[i]) for i in indices] != cache['train_places']):
        raise ValueError('Training dataset index/place identity differs')
    return data, indices, schedule, metadata


def contract(args, stage, data_identity):
    _, indices, schedule, metadata = data_identity
    benchmark_metadata = {}
    for root in (args.msls_root, args.pitts_root):
        files = sorted(root.glob('*.npy'))
        if not files:
            raise ValueError('Missing benchmark metadata: ' + str(root))
        benchmark_metadata.update({str(p.resolve()): sha(p) for p in files})
    source_names = project_sources(ROOT, ENTRY_SOURCES)
    return dict(stage=stage, policy=POLICY, official=verify_resnet_download(args.official_cache),
        training_plan_completed_sha256=sha(args.training_plan/'completed.json'),
        training_cache_completed_sha256=sha(args.training_cache/'completed.json'),
        metadata=metadata, train_indices=indices, schedule=schedule,
        benchmark_metadata=benchmark_metadata,
        roots={k: str(getattr(args, k).resolve()) for k in
               ('official_cache', 'dataset_root', 'msls_root', 'pitts_root')},
        code={p: code_hash(ROOT/p) for p in source_names},
        versions=dict(torch=str(torch.__version__), numpy=str(np.__version__),
            torchvision=str(__import__('torchvision').__version__)))


def prepare_run(args, stage, identity):
    c = contract(args, stage, identity)
    if stage in ('smoke', 'train'):
        c['arm'] = args.arm
        verify(args.root/'baseline')
        c['baseline_completed_sha256'] = sha(args.root/'baseline/completed.json')
        baseline_c = read(args.root/'baseline/contract.json')
        for key in ('policy', 'official', 'training_plan_completed_sha256',
                    'training_cache_completed_sha256', 'metadata', 'schedule',
                    'benchmark_metadata', 'roots', 'code', 'versions'):
            if c[key] != baseline_c[key]:
                raise ValueError('Baseline/training identity differs: ' + key)
    open_contract(args.output, c, args.resume)
    return c


def evaluate(args, model, output):
    from tqdm import tqdm
    model.eval()
    for size in POLICY['eval_sizes']:
        for dataset in benchmark_datasets(args, size):
            tag = f'{dataset.dataset_name}_{size}'
            result_path = output/(tag+'.json')
            ordered_paths = np.concatenate((dataset.dbImages, dataset.qImages))
            paths_sha = __import__('hashlib').sha256(
                '\n'.join(str(x) for x in ordered_paths).encode()).hexdigest()
            if result_path.exists():
                old = read(result_path)
                if old['image_order_sha256'] != paths_sha:
                    raise ValueError('Finished evaluation population changed')
                continue
            features = np.empty((len(dataset), POLICY['descriptor_dim']), np.float32)
            loader = DataLoader(dataset, batch_size=16, num_workers=args.workers, shuffle=False)
            with torch.inference_mode():
                for batch, (images, indices) in enumerate(tqdm(loader, desc=tag)):
                    if indices.tolist() != list(range(batch*16, batch*16+len(images))):
                        raise ValueError('Benchmark extraction order differs')
                    descriptors = torch.cat([model(x.cuda()) for x in images.split(POLICY['microbatch'])])
                    if (not bool(torch.isfinite(descriptors).all())
                            or float((descriptors.norm(dim=1)-1).abs().max())>1e-4):
                        raise ValueError('Nonfinite/nonunit global descriptor')
                    features[indices.numpy()] = descriptors.cpu().numpy()
                    write(output/'progress.json', dict(phase='evaluation', tag=tag,
                        batches=batch+1, total=len(loader)))
                # Only descriptor matrices, no dense patch/token cache.
                database = torch.from_numpy(features[:dataset.num_references]).cuda()
                predictions = np.concatenate([(torch.from_numpy(q).cuda() @ database.T)
                    .topk(20, dim=1).indices.cpu().numpy()
                    for q in np.array_split(features[dataset.num_references:],
                        max(1, (dataset.num_queries+15)//16))])
            hits = np.asarray([[bool(np.isin(row[:k], gt).any()) for k in (1,5,10,20)]
                for row, gt in zip(predictions, dataset.ground_truth)], dtype=bool)
            value = dict(queries=len(hits), correct=int(hits[:,0].sum()),
                outcomes=hits[:,0].tolist(), predictions=predictions.tolist(),
                recall_k=[1,5,10,20], recall=hits.mean(0).tolist(), input_size=size,
                image_order_sha256=paths_sha, descriptor_dim=POLICY['descriptor_dim'],
                scope=POLICY['scope'])
            if output.name != 'baseline':
                base = read(args.root/'baseline'/(tag+'.json'))
                if base['image_order_sha256'] != paths_sha:
                    raise ValueError('Baseline/candidate query identities differ')
                value['paired'] = paired(base['outcomes'], value['outcomes'])
            write(result_path, value)
            print(tag, value['correct'], value['recall'], value.get('paired',{}), flush=True)
            del database, features


def preflight(args):
    """CPU/disk checks only, no update or CUDA context."""
    identity = training_inputs(args)
    c = prepare_run(args, 'preflight', identity)
    if (args.output/'completed.json').exists():
        verify(args.output)
        print('Preflight already complete; training has NOT started', flush=True)
        return
    visual = load_official_resnet(args.official_cache)
    initial = tensor_hash(visual)
    probes = torch.randn(2,3,64,64, generator=torch.Generator().manual_seed(7))
    with torch.inference_mode():
        direct = visual(probes)
        states, counts, errors = {}, {}, {}
        for arm in ARMS:
            model = ResNetModuleTransfer(visual, arm).eval()
            errors[arm] = float((model(probes)-direct).abs().max())
            if errors[arm] > 1e-6:
                raise ValueError('ResNet zero-start mismatch: ' + arm)
            states[arm] = parameter_hash(model)
            counts[arm] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    checks = dict(strict_official_load=True, cropped_backbone=len(visual.backbone.net)==7,
        official_descriptor_dim=direct.shape==(2,16384), frozen_no_parameters=not any(p.requires_grad for p in visual.parameters()),
        frozen_buffers_unchanged=tensor_hash(visual)==initial,
        conv_initialization=states['conv_last']==states['conv_cross'],
        conditional_initialization=states['conditional_last']==states['conditional_cross'],
        conv_capacity=counts['conv_last']==counts['conv_cross'],
        conditional_capacity=counts['conditional_last']==counts['conditional_cross'],
        zero_start=max(errors.values())<=1e-6)
    if not all(checks.values()):
        raise ValueError('CPU preflight failed: ' + str(checks))
    write(args.output/'summary.json', dict(checks=checks, zero_start_errors=errors,
        active_parameters=counts, initial_parameter_sha256=states,
        official_tensor_sha256=initial, verdict='PREFLIGHT_PASS_NO_TRAINING',
        cuda_context_initialized=torch.cuda.is_initialized(), code_files=len(c['code'])))
    seal(args.output)
    print('PREFLIGHT_PASS_NO_TRAINING', counts, flush=True)


def baseline(args):
    prepare_run(args, 'baseline', training_inputs(args))
    if (args.output/'completed.json').exists():
        verify(args.output)
        return
    seed(POLICY['seed'])
    visual = load_official_resnet(args.official_cache, 'cuda')
    frozen = tensor_hash(visual)
    evaluate(args, visual, args.output)
    if tensor_hash(visual) != frozen:
        raise ValueError('Baseline changed frozen buffers')
    write(args.output/'summary.json', dict(official_tensor_sha256=frozen,
        author_reported_msls_r1=.912, measured_msls_r1=read(args.output/'msls-val_384.json')['recall'][0],
        note='Do not tune preprocessing or GT to force README rounded agreement',
        scope=POLICY['scope']))
    seal(args.output)


def batch_forward(model, images, training):
    desc, ratios, raws = [], [], []
    for chunk in images.cuda().split(POLICY['microbatch']):
        maps = model.features(chunk)
        if training:
            d, residual = checkpoint(model.aggregate, *maps, use_reentrant=False)
        else:
            d, residual = model.aggregate(*maps)
        desc.append(d)
        ratios.append(residual.detach().flatten(1).norm(dim=1)
                      / maps[-1].flatten(1).norm(dim=1).clamp_min(1e-12))
        raws.append(model.neck.last_raw.square().mean().sqrt())
    return torch.cat(desc), torch.cat(ratios), torch.stack(raws).mean()


def train(args):
    from tqdm import tqdm
    from src.losses.vpr_losses import VPRLossFunction
    from src.slgd_mid_supervision import gps_filtered_retrieval
    identity = training_inputs(args)
    prepare_run(args, args.stage, identity)
    if (args.output/'completed.json').exists():
        verify(args.output)
        print('Already complete', args.output, flush=True)
        return
    if args.stage == 'train':
        verify(args.root/'smoke_aggregate')
        if read(args.root/'smoke_aggregate/summary.json')['verdict'] != 'SMOKE_PASS_DISCARD_UPDATES':
            raise ValueError('Complete all four smoke checks first')
    seed(POLICY['seed'])
    visual = load_official_resnet(args.official_cache, 'cuda')
    frozen = tensor_hash(visual)
    if frozen != read(args.root/'baseline/summary.json')['official_tensor_sha256']:
        raise ValueError('Official baseline differs')
    model = ResNetModuleTransfer(visual, args.arm).cuda().eval()
    active = {n:p for n,p in model.named_parameters() if p.requires_grad}
    if not active or any(not n.startswith('neck.') or n.startswith('neck.aux.') for n in active):
        raise ValueError('Only real nonauxiliary neck parameters may train')
    zero_errors = {}
    with torch.inference_mode():
        for size in POLICY['eval_sizes']:
            pixels = torch.randn(2,3,size,size, generator=torch.Generator().manual_seed(7))
            zero_errors[str(size)] = float((model(pixels.cuda())-visual(pixels.cuda())).abs().max())
    if max(zero_errors.values())>1e-6:
        raise ValueError('Full-resolution zero-start descriptor mismatch')
    optimizer = torch.optim.AdamW(active.values(), lr=POLICY['lr'], weight_decay=0.)
    state = dict(steps=0, rows=[], gradients={}, contract_sha256=sha(args.output/'contract.json'),
        initial_parameter_sha256=parameter_hash(model))
    if (args.output/'last.pt').exists():
        state = restore(args.output/'last.pt', model, optimizer, state['contract_sha256'])
    if not (args.output/'initial.json').exists():
        write(args.output/'initial.json', dict(initial_parameter_sha256=state['initial_parameter_sha256'],
            active_parameters=sum(p.numel() for p in active.values()), zero_start_errors=zero_errors,
            official_tensor_sha256=frozen))
        save(args.output/'last.pt', model, optimizer, state)
    data, indices, schedule, _ = identity
    schedule = schedule[:POLICY['smoke_steps']] if args.stage=='smoke' else schedule
    loader = DataLoader(PlaceBatches(data, indices, schedule[state['steps']:]),
        batch_size=None, num_workers=args.workers)
    lossfn = VPRLossFunction()
    model.train()
    for images, labels, coordinates in tqdm(loader, initial=state['steps'], total=len(schedule),
            desc=f'ResNet {args.stage} {args.arm}'):
        optimizer.zero_grad(set_to_none=True)
        d, ratio, raw = batch_forward(model, images, True)
        loss = gps_filtered_retrieval(lossfn, d, labels.cuda(), coordinates.numpy())
        if not bool(torch.isfinite(loss)) or float(ratio.max())>POLICY['rho']+1e-6:
            raise ValueError('Invalid loss or relative residual bound')
        loss.backward()
        for name, valid in gradient_groups(model).items():
            state['gradients'][name] = state['gradients'].get(name,False) or valid
        if any(p.grad is not None for p in visual.parameters()):
            raise ValueError('Frozen official model received gradients')
        if any(p.grad is None for p in active.values()):
            raise ValueError('Unused trainable parameter detected')
        norm = torch.nn.utils.clip_grad_norm_(list(active.values()), POLICY['clip'], error_if_nonfinite=True)
        optimizer.step()
        state['steps'] += 1
        row = dict(step=state['steps'], loss=float(loss.detach()), gradient_norm=float(norm),
            relative_residual_mean=float(ratio.mean()), relative_residual_max=float(ratio.max()), raw_rms=float(raw))
        state['rows'].append(row)
        write(args.output/'progress.json', dict(phase='training', arm=args.arm, total=len(schedule), **row))
        if state['steps']%16 == 0:
            save(args.output/'last.pt', model, optimizer, state)
    save(args.output/'last.pt', model, optimizer, state)
    final = parameter_hash(model)
    restored = restore(args.output/'last.pt', model, optimizer, state['contract_sha256'])
    checks = dict(frozen_author_and_bn_unchanged=tensor_hash(visual)==frozen,
        all_steps=state['steps']==len(schedule), all_gradient_groups=all(state['gradients'].values()),
        zero_start=max(zero_errors.values())<=1e-6, checkpoint_roundtrip=parameter_hash(model)==final,
        restored_steps=restored['steps']==state['steps'], nonzero_update=final!=state['initial_parameter_sha256'],
        optimizer_clocks=all(int(optimizer.state[p]['step'])==len(schedule) for p in active.values()))
    if not all(checks.values()):
        raise ValueError('Training mechanics failed: ' + str(checks))
    optimizer.zero_grad(set_to_none=True)
    if args.stage=='train':
        evaluate(args, model, args.output)
    write(args.output/'summary.json', dict(checks=checks, rows=state['rows'], steps=state['steps'],
        final_parameter_sha256=final, arm=args.arm, scope=POLICY['scope']))
    write(args.output/'progress.json', dict(phase='complete', stage=args.stage, arm=args.arm))
    seal(args.output)


def aggregate(args):
    identity = training_inputs(args)
    c = contract(args, args.stage+'_aggregate', identity)
    verify(args.root/'baseline')
    baseline_seal = sha(args.root/'baseline/completed.json')
    contracts, initials, summaries, seals = {}, {}, {}, {}
    for arm in ARMS:
        path = args.root/(args.stage+'_'+arm)
        contracts[arm] = verify(path)
        if contracts[arm]['arm'] != arm or contracts[arm]['stage'] != args.stage:
            raise ValueError('Mismatched aggregate arm/stage')
        if contracts[arm]['baseline_completed_sha256'] != baseline_seal:
            raise ValueError('Aggregate baseline identity differs')
        for key in ('policy','official','metadata','schedule','benchmark_metadata','roots','code','versions',
                    'training_plan_completed_sha256','training_cache_completed_sha256'):
            if contracts[arm][key] != c[key]:
                raise ValueError('Mismatched aggregate contract: ' + key)
        initials[arm] = read(path/'initial.json')
        summaries[arm] = read(path/'summary.json')
        seals[arm] = sha(path/'completed.json')
    mechanical = dict(all_run_checks=all(all(s['checks'].values()) for s in summaries.values()))
    for a,b in (('conv_last','conv_cross'),('conditional_last','conditional_cross')):
        mechanical[a+'_matched_capacity'] = initials[a]['active_parameters']==initials[b]['active_parameters']
        mechanical[a+'_matched_initialization'] = initials[a]['initial_parameter_sha256']==initials[b]['initial_parameter_sha256']
    if not all(mechanical.values()):
        raise ValueError('Matched study mechanics failed: ' + str(mechanical))
    c['completed'] = seals
    open_contract(args.output, c, args.resume)
    if (args.output/'completed.json').exists():
        verify(args.output)
        return
    result = dict(mechanical_checks=mechanical, verdict='SMOKE_PASS_DISCARD_UPDATES')
    if args.stage=='train':
        parts = {}
        for size in POLICY['eval_sizes']:
            for ds in ('msls-val','pitts30k-val'):
                tag = f'{ds}_{size}'
                base = read(args.root/'baseline'/(tag+'.json'))
                values = {arm: read(args.root/('train_'+arm)/(tag+'.json')) for arm in ARMS}
                if any(v['image_order_sha256']!=base['image_order_sha256'] for v in values.values()):
                    raise ValueError('Mismatched query identities in aggregate')
                part = dict(queries=base['queries'], official_correct=base['correct'], official_recall=base['recall'])
                for arm,v in values.items():
                    part[arm] = dict(correct=v['correct'], recall=v['recall'], vs_official=paired(base['outcomes'],v['outcomes']))
                for cross,last in (('conv_cross','conv_last'),('conditional_cross','conditional_last')):
                    part[cross+'_vs_'+last] = paired(values[last]['outcomes'], values[cross]['outcomes'])
                part['conditional_cross_vs_conv_cross'] = paired(values['conv_cross']['outcomes'], values['conditional_cross']['outcomes'])
                parts[tag] = part
        result.update(assessment(parts), results=parts)
    write(args.output/'summary.json', result)
    seal(args.output)
    print(result['verdict'], flush=True)


def main():
    os.chdir(ROOT)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('preflight','baseline','train','aggregate'))
    p.add_argument('--stage', choices=('smoke','train'), default='smoke')
    p.add_argument('--arm', choices=ARMS)
    p.add_argument('--allow-training', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--root', type=Path, default=Path('logs/resnet_module_transfer_v1'))
    p.add_argument('--output', type=Path)
    p.add_argument('--official-cache', type=Path, default=Path('.cache/boq_resnet50_official_v1'))
    p.add_argument('--training-plan', type=Path, default=Path('logs/conv_official_boq_v1/seed42/train'))
    p.add_argument('--training-cache', type=Path, default=Path('.cache/slgd_mid_v1_pilot'))
    p.add_argument('--dataset-root', type=Path, default=Path('datasets/gsv_cities'))
    p.add_argument('--msls-root', type=Path, default=Path('datasets/msls-val'))
    p.add_argument('--pitts-root', type=Path, default=Path('datasets/pitts30k-val'))
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if args.command=='train' and not args.allow_training:
        p.error('Training is opt-in: add --allow-training only after approval')
    if args.command=='train' and not args.arm:
        p.error('--arm is required for training')
    if args.workers<0 or args.root.resolve().parent!=(ROOT/'logs').resolve() or not args.root.name.startswith('resnet_module_transfer_'):
        p.error('Use workers>=0 and an isolated logs/resnet_module_transfer_<version> root')
    name = (f'{args.stage}_{args.arm}' if args.command=='train' else
            args.stage+'_aggregate' if args.command=='aggregate' else args.command)
    args.output = args.output or args.root/name
    if args.output.resolve().parent!=args.root.resolve() or args.output.name!=name:
        p.error('Output must be the matching direct child of the isolated root')
    import fcntl
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    with (args.root/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        shared = None
        if args.command in ('baseline','train'):
            if os.environ.get('CUDA_VISIBLE_DEVICES')!='1' or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
                p.error('Expose physical GPU1 only: CUDA_VISIBLE_DEVICES=1')
            shared = (ROOT/'logs/query_relation_screen_v1.lock').open('a')
            fcntl.flock(shared, fcntl.LOCK_EX|fcntl.LOCK_NB)
            torch.cuda.set_per_process_memory_fraction(.65,0)
        try:
            globals()[args.command](args)
        except Exception as error:
            if (args.output/'contract.json').exists() and not (args.output/'completed.json').exists():
                write(args.output/'failure.json',dict(type=type(error).__name__,message=str(error),
                    time=datetime.datetime.now().astimezone().isoformat()))
            raise
        finally:
            if shared is not None:
                shared.close()


if __name__=='__main__':
    main()
