"""QR-BoQ matched fixed-last screen. No development checkpoint selection."""
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
from scripts.train_depth_query import seed, MatchedPlaces, save, restore, RU_SHA
from scripts.candidate_set_screen import read, write, sha, complete
from scripts.adaptive_pair_budget import verified
from scripts.eval_condition_robustness import load_inference_model_from_ckpt
from src.models.depth_query import DepthQueryVPR
from src.models.query_relation import MODES, install
from src.query_relation_training import score_development
from src.dsa_training import schedules
from src.dataloaders.train.gsv_cities import GSVCitiesDataset
from src.dataloaders.valid.mapillary_sls import MapillarySLSDataset
from src.dataloaders.valid.pittsburgh import PittsburghDataset
from src.losses.vpr_losses import VPRLossFunction

POLICY = dict(seed=42, places=4096, views=4, places_per_batch=16, epochs=3,
              lr=1e-4, weight_decay=0., edge_width=32, precision='fp32', cuda_allocator_fraction=.4, microbatch_images=8,
              batches_per_epoch=320, broad_batches=256, hard_batches=64,
              selection='Fixed epoch3 only; GSV diagnostic only, no best checkpoint',
              scope='Single-seed QR relational structure screen; exposed MSLS/Pitts, not independent final tests')

def forward_chunks(model, images):
    return torch.cat([model(chunk) for chunk in images.split(8)], dim=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=MODES, required=True)
    p.add_argument('--preflight-run', type=Path, default=Path('logs/query_relation_preflight_v1'))
    p.add_argument('--plan', type=Path, default=Path('doc/local_value_plan_v1'))
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, default=Path('datasets'))
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    import fcntl
    from torchvision.transforms import v2 as T
    from tqdm import tqdm
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Expose physical GPU1 only')
    torch.cuda.set_per_process_memory_fraction(.4, 0)
    if not a.checkpoint.is_file() or sha(a.checkpoint) != RU_SHA:
        raise ValueError('Original RU required')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    lock = (a.output.parent/(a.output.name+'.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = [T.ToImage(), T.Resize((280, 280), interpolation=T.InterpolationMode.BICUBIC, antialias=True)]
    end = [T.ToDtype(torch.float32, scale=True), T.Normalize([.485, .456, .406], [.229, .224, .225])]
    clean = T.Compose(base+end)
    augment = T.Compose(base+[T.ColorJitter(.4, .4, .4, .1), T.RandomGrayscale(.2)]+end)
    train = GSVCitiesDataset(dataset_path=a.dataset_root/'gsv_cities', cities='all', img_per_place=4, transform=augment)
    seed(42)
    dev = GSVCitiesDataset(dataset_path=a.dataset_root/'gsv_cities', cities='all', img_per_place=4, transform=clean)
    if list(train.places_ids) != list(dev.places_ids): raise ValueError('Unmatched dataset ordering')
    order = sorted(range(len(train)), key=lambda i: hashlib.sha256(f'dsq-subset42:{train.places_ids[i]}'.encode()).hexdigest())
    if len(order) < 5120: raise ValueError('Insufficient places')
    indices = order[:32 if a.smoke else 4096]
    development = order[4096:4128 if a.smoke else 5120]
    verified(a.plan)
    plan = read(a.plan/'plan.json')
    plan_hash = sha(a.plan/'completed.json')
    if plan['train_places'] != [str(train.places_ids[i]) for i in order[:4096]] or plan['dev_places'] != [str(dev.places_ids[i]) for i in order[4096:5120]]:
        raise ValueError('Hard plan training/development places differ')
    id_to_index = {str(v): i for i, v in enumerate(train.places_ids)}
    if len(plan['epochs']) != 3:
        raise ValueError('Expected three fixed batch schedules')
    for schedule in plan['epochs']:
        if len(schedule['batches']) != 256 or any(len(b) != 16 or len(set(b)) != 16 or not set(b).issubset(set(plan['train_places'])) for b in schedule['batches']):
            raise ValueError('Invalid or leaking hard batch schedule')
    verified(a.preflight_run)
    pc = read(a.preflight_run/'contract.json')
    if pc['plan_sha256'] != plan_hash or pc['ru_sha256'] != RU_SHA:
        raise ValueError('Preflight source differs')
    for name in ('src/models/query_relation.py', 'src/models/depth_query.py', 'src/models/aggregators/boq.py'):
        digest = hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if pc['code'].get(name) != digest: raise ValueError('Model changed since preflight: '+name)
    audit = read(a.preflight_run/'summary.json')
    if set(audit['preflights']) != set(MODES) or any(not v['frozen_unchanged'] for v in audit['preflights'].values()):
        raise ValueError('Incomplete preflight')
    schedule = schedules(plan)
    if a.smoke:
        schedule = [[plan['epochs'][0]['batches'][0], schedule[0][0]]]
        indices = [id_to_index[v] for b in schedule[0] for v in b]
    source = ['scripts/train_query_relation.py', 'src/models/query_relation.py', 'src/query_relation_training.py',
              'src/models/depth_query.py', 'src/dsa_training.py', 'scripts/train_depth_query.py', 'src/models/aggregators/boq.py',
              'src/losses/vpr_losses.py', 'src/dataloaders/train/gsv_cities.py',
              'scripts/eval_condition_robustness.py', 'src/models/backbones/dinov2.py']
    metadata = sorted((a.dataset_root/'gsv_cities/Dataframes').glob('*.csv'))
    if not metadata: raise ValueError('Missing GSV metadata')
    for path in metadata:
        if pc['metadata'].get(str(path)) != sha(path): raise ValueError('Preflight GSV metadata changed')
    for name in ('msls-val', 'pitts30k-val'): metadata += sorted((a.dataset_root/name).glob('*.npy'))
    contract = dict(mode=a.mode, smoke=a.smoke, policy=POLICY, checkpoint_sha256=RU_SHA, plan_sha256=plan_hash,
        preflight_completed_sha256=sha(a.preflight_run/'completed.json'), batch_schedule=schedule,
        train_places=[str(train.places_ids[i]) for i in indices], dev_places=[str(dev.places_ids[i]) for i in development],
        data={str(f): sha(f) for f in metadata}, code={f: hashlib.sha256((ROOT/f).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for f in source})
    if a.output.exists():
        if not a.resume or read(a.output/'contract.json') != contract: raise ValueError('Run contract mismatch')
        if (a.output/'completed.json').exists(): verified(a.output); print('Already complete'); return
    else:
        a.output.mkdir()
        write(a.output/'contract.json', contract)
    visual = load_inference_model_from_ckpt(a.checkpoint, 'cpu').cuda().eval()
    sample = MatchedPlaces(dev, development[:1], 0)[0][0].cuda()
    with torch.inference_mode(): expected = visual(sample)
    model = DepthQueryVPR(visual, 'baseline').cuda().eval().requires_grad_(False)
    seed(42); adapter = install(model.aggregator, a.mode)
    model.cuda().eval(); adapters = [adapter]
    with torch.inference_mode(): error = float((model(sample)-expected).abs().max())
    if not np.isfinite(error) or error > 2e-6: raise ValueError('Zero-start descriptor mismatch')
    active = [p for p in model.parameters() if p.requires_grad]
    expected_active = {id(p) for m in adapters for part in (m.project, m.edge, m.output) for p in part.parameters()}
    if {id(p) for p in active} != expected_active or sum(p.numel() for p in active) != 20736:
        raise ValueError('Unexpected trainable parameters')
    print('Zero-start error', error, 'trainable parameters', sum(p.numel() for p in active), flush=True)
    trainable_names = {k for k,p in model.named_parameters() if p.requires_grad}
    frozen = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k not in trainable_names}
    opt = torch.optim.AdamW(active, lr=POLICY['lr'], weight_decay=0.)
    loss_fn = VPRLossFunction()
    state = dict(epoch=0, cursor=0, loss_sum=0., zero_batches=0, steps=0, history=[], contract_sha256=sha(a.output/'contract.json'))
    if (a.output/'last.pt').exists():
        state = restore(torch.load(a.output/'last.pt', map_location='cpu', weights_only=True), model, opt)
        if state['contract_sha256'] != sha(a.output/'contract.json'): raise ValueError('Resume identity mismatch')

    def evaluate_dev(epoch):
        features = []
        gallery_order = (order[:32] if a.smoke else order[:4096]) + development
        query_start = (len(gallery_order)-len(development))*4
        data = MatchedPlaces(dev, gallery_order, 0)
        write(a.output/'progress.json', dict(phase='gsv_diagnostic', epoch=epoch))
        with torch.inference_mode():
            for images, _ in tqdm(DataLoader(data, batch_size=8, num_workers=a.workers), desc=f'{a.mode} GSV e{epoch}'):
                features.append(forward_chunks(model, images.flatten(0, 1).cuda()).cpu())
        all_features = torch.cat(features)
        query_features = all_features[query_start:]
        reference = a.output/'initial_dev.pt'
        if epoch == 0:
            torch.save(query_features, a.output/'initial_dev.tmp'); (a.output/'initial_dev.tmp').replace(reference)
        drift = float((query_features-torch.load(reference, map_location='cpu', weights_only=True)).norm(dim=1).mean())
        result = score_development(all_features.cuda(), query_start)
        if epoch == 0 and not a.smoke and result['correct'] != audit['expanded_gallery']['correct']:
            raise ValueError('Expanded gallery RU baseline differs')
        result['mean_descriptor_l2_drift'] = drift
        write(a.output/f'gsv_epoch{epoch:02d}.json', result)
        return {k: result[k] for k in ('correct', 'queries', 'margin_loss', 'mean_descriptor_l2_drift')}

    def benchmark(epoch):
        result = {}
        for ds in (MapillarySLSDataset(a.dataset_root/'msls-val', clean), PittsburghDataset(a.dataset_root/'pitts30k-val', clean)):
            file = a.output/f'{ds.dataset_name}_epoch{epoch:02d}.json'
            if file.exists(): result[ds.dataset_name] = read(file)['metrics']; continue
            write(a.output/'progress.json', dict(phase='benchmark', epoch=epoch, dataset=ds.dataset_name))
            descriptors = []
            with torch.inference_mode():
                for images, _ in tqdm(DataLoader(ds, batch_size=32, num_workers=a.workers), desc=f'{a.mode} {ds.dataset_name} e{epoch}'):
                    descriptors.append(forward_chunks(model, images.cuda()).cpu())
            features = torch.cat(descriptors).cuda()
            database = features[:ds.num_references]
            predictions = np.concatenate([(q@database.T).topk(20, dim=1).indices.cpu().numpy() for q in features[ds.num_references:].split(64)])
            hits = np.asarray([[bool(np.isin(row[:k], gt).any()) for k in (1, 5, 20)] for row, gt in zip(predictions, ds.ground_truth)])
            counts = hits.sum(0)
            metrics = dict(queries=ds.num_queries, correct=int(counts[0]), r1=float(hits[:, 0].mean()), r5=float(hits[:, 1].mean()), r20=float(hits[:, 2].mean()))
            if epoch == 0 and int(counts[0]) != {'msls-val': 675, 'pitts30k-val': 7160}[ds.dataset_name]:
                raise ValueError('Frozen RU recall not reproduced')
            write(file, dict(metrics=metrics, predictions=predictions.tolist(), correct=hits[:, 0].tolist()))
            result[ds.dataset_name] = metrics
            del features, database, descriptors
        return result

    if not state['history']:
        initial = evaluate_dev(0)
        state['history'] = [dict(epoch=0, development=initial)]
        save(a.output/'last.pt', model, opt, state)
    # Evaluate the original model even when resuming nonzero adapters.
    if not a.smoke and not (a.output/'pitts30k-val_epoch00.json').exists():
        for m in adapters: m.enabled = False
        try: benchmark(0)
        finally:
            for m in adapters: m.enabled = True
    probe_norm = None
    for epoch in range(state['epoch'], 1 if a.smoke else POLICY['epochs']):
        batches = schedule[epoch]
        if a.smoke: batches = batches[:2]
        if any(len(b) != 16 or len(set(b)) != 16 for b in batches): raise ValueError('Invalid batches')
        data_order = [id_to_index[v] for b in batches for v in b]
        data = MatchedPlaces(train, data_order[state['cursor']:], epoch)
        # Frozen backbone/BoQ remain eval; only the new relation branch trains.
        model.eval()
        for images, labels in tqdm(DataLoader(data, batch_size=16, num_workers=a.workers), desc=f'{a.mode} hard-mix train e{epoch+1}'):
            opt.zero_grad(set_to_none=True)
            descriptors = forward_chunks(model, images.flatten(0, 1).cuda())
            loss, _ = loss_fn(descriptors, labels.flatten().cuda())
            if not torch.isfinite(loss): raise ValueError('Nonfinite VPR loss')
            if a.smoke:
                probe = (descriptors*torch.linspace(-1, 1, descriptors.shape[1], device='cuda')).sum()
                grads = torch.autograd.grad(probe, active, retain_graph=True)
                probe_norm = sum(float(g.abs().sum()) for g in grads)
                if not np.isfinite(probe_norm) or probe_norm == 0: raise ValueError('Dead QR branch')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True)
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad): raise ValueError('Frozen parameter gradient')
            if float(loss.detach()) != 0:
                opt.step()
                state['steps'] += 1
            else: state['zero_batches'] += 1
            state['loss_sum'] += float(loss.detach())*len(images)
            state['cursor'] += len(images)
            write(a.output/'progress.json', dict(phase='training', mode=a.mode, epoch=epoch+1, done=state['cursor'], total=len(data_order)))
            if state['cursor'] % 256 == 0: save(a.output/'last.pt', model, opt, state)
        validation = evaluate_dev(epoch+1)
        state['history'].append(dict(epoch=epoch+1, development=validation, train_loss=state['loss_sum']/len(data_order),
             zero_batches=state['zero_batches'], optimizer_steps_total=state['steps'], adapter_weight_norms=[float(p.detach().norm()) for p in active]))
        state.update(epoch=epoch+1, cursor=0, loss_sum=0., zero_batches=0)
        save(a.output/'last.pt', model, opt, state)
        write(a.output/'history.json', state['history'])
    for k, v in model.state_dict().items():
        if k in frozen and not torch.equal(v.cpu(), frozen[k]): raise ValueError('Historical weights or buffers changed: '+k)
    if a.smoke:
        if state['steps'] < 1 or not any(float(p.detach().norm()) > 0 for p in active):
            raise ValueError('Smoke produced no actual VPR update')
        before = [p.detach().clone() for p in active]
        restore(torch.load(a.output/'last.pt', map_location='cpu', weights_only=True), model, opt)
        if any(not torch.equal(x, y) for x, y in zip(before, active)): raise ValueError('Resume roundtrip mismatch')
        write(a.output/'preflight.json', dict(initial_error=error, probe_gradient=probe_norm, optimizer_steps=state['steps'], frozen_unchanged=True, resume_roundtrip=True))
    else:
        benchmark(state['epoch'])
    write(a.output/'summary.json', dict(mode=a.mode, policy=POLICY, reported_epoch=state['epoch'], history=state['history'],
                                      trainable_parameters=sum(p.numel() for p in active), frozen_unchanged=True))
    write(a.output/'progress.json', dict(phase='complete'))
    complete(a.output)
    print('COMPLETE', a.output, flush=True)


if __name__ == '__main__': main()
