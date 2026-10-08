"""Matched frozen-RU residual screen. All arm choices fixed before evaluation."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_depth_query import seed, MatchedPlaces, save, restore, RU_SHA
from scripts.candidate_set_screen import read, write, sha, complete
from scripts.adaptive_pair_budget import verified
from scripts.eval_condition_robustness import load_inference_model_from_ckpt
from src.models.protected_mamba import MODES, ProtectedMambaVPR, preservation_loss
from src.dataloaders.train.gsv_cities import GSVCitiesDataset
from src.dataloaders.valid.mapillary_sls import MapillarySLSDataset
from src.dataloaders.valid.pittsburgh import PittsburghDataset
from src.losses.vpr_losses import VPRLossFunction
from src.query_relation_training import score_development
from src.scan_mamba_diagnostics import fixed_step, gradient_comparison

POLICY = dict(revision=1, seed=42, size=280, views=4, places_per_batch=16,
    holdout_places=1024, full_epochs=3, smoke_steps=4, pilot_steps=128,
    lr=1e-4, weight_decay=0., clip=1., precision='fp32', microbatch=4,
    cap=.1, temperature=.07, relation_weight=1., drift_weight=.1,
    frozen='Entire RU backbone, gate and BoQ; only spatial mixer is optimized',
    optimizer_clock='Unconditional AdamW on all batches, None gradients become zero',
    selection='Fixed last; mechanical pilot gate only; no benchmark tuning',
    scope='Single-seed historically exposed MSLS/Pitts, not independent final tests')


def code_hash(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=MODES, required=True)
    p.add_argument('--stage', choices=('smoke', 'pilot', 'full'), default='pilot')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    from torchvision.transforms import v2 as T
    from tqdm import tqdm
    import fcntl
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or torch.cuda.device_count() != 1:
        raise ValueError('Expose physical GPU1 only')
    if not a.checkpoint.is_file() or sha(a.checkpoint) != RU_SHA:
        raise ValueError('Original RU required')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    lock = (a.output.parent/(a.output.name+'.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.cuda.set_per_process_memory_fraction(.4, 0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed(42)
    start = [T.ToImage(), T.Resize((280, 280), interpolation=T.InterpolationMode.BICUBIC, antialias=True)]
    end = [T.ToDtype(torch.float32, scale=True), T.Normalize([.485, .456, .406], [.229, .224, .225])]
    augment = T.Compose(start+[T.ColorJitter(.4, .4, .4, .1), T.RandomGrayscale(.2)]+end)
    clean = T.Compose(start+end)
    train = GSVCitiesDataset(dataset_path=Path('datasets/gsv_cities'), cities='all', img_per_place=4, transform=augment)
    seed(42)
    dev = GSVCitiesDataset(dataset_path=Path('datasets/gsv_cities'), cities='all', img_per_place=4, transform=clean)
    if list(train.places_ids) != list(dev.places_ids):
        raise ValueError('Dataset order differs')
    order = sorted(range(len(train)), key=lambda i: hashlib.sha256(f'scan-mamba42:{train.places_ids[i]}'.encode()).hexdigest())
    if len(order) < 6144:
        raise ValueError('Insufficient places')
    holdout, training = order[:1024], order[1024:]
    schedule = []
    for epoch in range(3 if a.stage == 'full' else 1):
        indices = np.random.default_rng(62031+epoch).permutation(training).tolist()
        batches = [indices[i:i+16] for i in range(0, len(indices)-15, 16)]
        schedule.append(batches if a.stage == 'full' else batches[:4 if a.stage == 'smoke' else 128])
    paths = ['scripts/train_protected_mamba.py', 'src/models/protected_mamba.py',
        'src/models/scan_mamba.py', 'src/models/depth_query.py', 'src/models/aggregators/boq.py',
        'src/losses/vpr_losses.py', 'scripts/train_depth_query.py', 'src/dataloaders/train/gsv_cities.py',
        'scripts/eval_condition_robustness.py', 'src/models/backbones/dinov2.py',
        'src/scan_mamba_diagnostics.py', 'src/query_relation_training.py']
    metadata = sorted(Path('datasets/gsv_cities/Dataframes').glob('*.csv'))
    if not metadata:
        raise ValueError('Missing GSV metadata')
    if a.stage == 'full':
        for directory in ('msls-val', 'pitts30k-val'):
            metadata.extend(sorted((Path('datasets')/directory).glob('*.npy')))
    contract = dict(mode=a.mode, stage=a.stage, policy=POLICY, ru_sha256=RU_SHA,
        source={name: code_hash(ROOT/name) for name in paths}, metadata={str(f): sha(f) for f in metadata},
        versions=dict(torch=torch.__version__, numpy=np.__version__),
        train_places=[str(train.places_ids[i]) for i in training], holdout_places=[str(train.places_ids[i]) for i in holdout],
        schedule=[[[str(train.places_ids[i]) for i in b] for b in epoch] for epoch in schedule])
    if a.output.exists():
        if not a.resume or read(a.output/'contract.json') != contract:
            raise ValueError('Run contract differs')
        if (a.output/'completed.json').exists():
            verified(a.output); print('Already complete'); return
    else:
        a.output.mkdir(); write(a.output/'contract.json', contract)
    seed(42)
    visual = load_inference_model_from_ckpt(a.checkpoint, 'cpu').cuda().eval()
    sample = MatchedPlaces(dev, holdout[:1], 0)[0][0].cuda()
    with torch.no_grad():
        expected = visual(sample)
    seed(42)
    model = ProtectedMambaVPR(visual, a.mode, POLICY['cap']).cuda().eval()
    with torch.no_grad():
        err = float((model(sample)-expected).abs().max())
    if not np.isfinite(err) or err > 2e-6:
        raise ValueError('Zero-start RU mismatch')
    active = {k: v for k, v in model.named_parameters() if v.requires_grad}
    if not active or any(not k.startswith('mixer.') for k in active):
        raise ValueError('Only mixer parameters may train')
    frozen = {k: v.detach().cpu().clone() for k, v in model.base.state_dict().items()}
    opt = torch.optim.AdamW(active.values(), lr=POLICY['lr'], weight_decay=0.)
    lossfn = VPRLossFunction()
    state = dict(epoch=0, cursor=0, steps=0, rows=[], ssm_gradient_seen=False,
                 contract_sha256=sha(a.output/'contract.json'))
    if (a.output/'last.pt').exists():
        state = restore(torch.load(a.output/'last.pt', map_location='cpu', weights_only=True), model, opt)
        if state['contract_sha256'] != sha(a.output/'contract.json'):
            raise ValueError('Resume mismatch')
    print('MODE', a.mode, 'STAGE', a.stage, 'zero error', err, 'trainable', sum(v.numel() for v in active.values()), flush=True)

    def descriptors(images, training=False, bypass=False):
        result = []
        for chunk in images.split(4):
            features = model.features(chunk)
            result.append(checkpoint(model.aggregate, features, use_reentrant=False) if training
                          else model.aggregate(features, bypass=bypass))
        return tuple(torch.cat([r[i] for r in result]) for i in range(3))

    @torch.no_grad()
    def probe(tag):
        rows = []
        indices = [i for b in schedule[0][:4 if a.stage == 'smoke' else 8] for i in b]
        for images, labels in DataLoader(MatchedPlaces(train, indices, 0), batch_size=16, num_workers=a.workers):
            d, ru, delta = descriptors(images.flatten(0, 1).cuda())
            target = labels.flatten().cuda()
            vpr, _ = lossfn(d, target); frozen_loss, _ = lossfn(ru, target)
            kl, drift = preservation_loss(d, ru)
            rows.append(dict(vpr=float(vpr), frozen_vpr=float(frozen_loss), relation=float(kl), drift=float(drift),
                             delta_l2=float(delta.norm(dim=-1).mean())))
        write(a.output/f'probe_{tag}.json', rows)
        ds, references, deltas = [], [], []
        for images, _ in DataLoader(MatchedPlaces(dev, holdout[:8 if a.stage == 'smoke' else 128], 0), batch_size=8, num_workers=a.workers):
            pixels = images.flatten(0, 1).cuda()
            d, ru, delta = descriptors(pixels)
            bypass = descriptors(pixels, bypass=True)[0]
            if not torch.equal(ru, bypass):
                raise ValueError('Bypass is not exact frozen RU')
            ds.append(d.cpu()); references.append(ru.cpu()); deltas.append(delta.norm(dim=-1).cpu())
        f, r = torch.cat(ds), torch.cat(references)
        result = score_development(f.cuda(), 0)
        norms = torch.cat(deltas)
        result.update(descriptor_drift=float((f-r).norm(dim=-1).mean()), delta_l2_max=float(norms.max()),
            delta_l2_mean=float(norms.mean()), min_cosine_to_ru=float((f*r).sum(-1).min()), bypass_matches_ru=True)
        if result['delta_l2_max'] > POLICY['cap']+1e-6 or result['min_cosine_to_ru'] < (1+POLICY['cap']**2)**-.5-2e-6:
            raise ValueError('Tangent bound violated')
        write(a.output/f'holdout_{tag}.json', result)

    @torch.no_grad()
    def benchmark(tag):
        for ds in (MapillarySLSDataset(Path('datasets/msls-val'), clean), PittsburghDataset(Path('datasets/pitts30k-val'), clean)):
            path = a.output/f'{ds.dataset_name}_{tag}.json'
            if path.exists():
                continue
            write(a.output/'progress.json', dict(phase='benchmark', mode=a.mode, tag=tag, dataset=ds.dataset_name))
            features = []
            for images, _ in tqdm(DataLoader(ds, batch_size=16, num_workers=a.workers), desc=f'{ds.dataset_name} {tag}'):
                features.append(descriptors(images.cuda(), bypass=(tag == 'initial'))[0].cpu())
            f = torch.cat(features).cuda(); db = f[:ds.num_references]
            predictions = np.concatenate([(q@db.T).topk(20, dim=1).indices.cpu().numpy() for q in f[ds.num_references:].split(32)])
            hits = np.asarray([[np.isin(row[:k], gt).any() for k in (1, 5, 20)] for row, gt in zip(predictions, ds.ground_truth)])
            if tag == 'initial' and int(hits[:, 0].sum()) != {'msls-val': 675, 'pitts30k-val': 7160}[ds.dataset_name]:
                raise ValueError('RU recall mismatch')
            write(path, dict(correct=hits[:, 0].tolist(), predictions=predictions.tolist(), recall=hits.mean(0).tolist(), correct_count=int(hits[:, 0].sum())))

    def gradient_probe(tag):
        indices = [i for b in schedule[0][:2] for i in b]
        rows = []
        for images, labels in DataLoader(MatchedPlaces(train, indices, 0), batch_size=16, num_workers=a.workers):
            opt.zero_grad(set_to_none=True)
            d, ru, _ = descriptors(images.flatten(0, 1).cuda(), True)
            vpr, _ = lossfn(d, labels.flatten().cuda())
            kl, drift = preservation_loss(d, ru)
            grads = gradient_comparison(vpr, kl+.1*drift, active.items(), weight=1.)['mixer']
            rows.append(dict(vpr=float(vpr.detach()), relation=float(kl.detach()), drift=float(drift.detach()),
                vpr_gradient_norm=grads['vpr_norm'], preservation_gradient_norm=grads['weighted_consistency_norm'],
                preservation_to_vpr=grads['consistency_to_vpr'], gradient_cosine=grads['cosine']))
        opt.zero_grad(set_to_none=True)
        write(a.output/f'gradient_{tag}.json', dict(batches=rows,
            scope='Read-only same two training batches; preservation gradient counterfactual for mamba_plain'))

    if not (a.output/'holdout_initial.json').exists():
        if state['steps'] != 0:
            raise ValueError('Missing initial diagnostic')
        probe('initial'); save(a.output/'last.pt', model, opt, state)
    if not (a.output/'gradient_initial.json').exists():
        if state['steps'] != 0:
            raise ValueError('Missing initial gradient diagnostic')
        gradient_probe('initial')
    if a.stage == 'full' and not all((a.output/f'{ds}_initial.json').exists() for ds in ('msls-val', 'pitts30k-val')):
        if state['steps'] != 0:
            raise ValueError('Missing initial benchmark')
        benchmark('initial')
    preserved = a.mode.endswith('preserved')
    for epoch in range(state['epoch'], len(schedule)):
        indices = [i for b in schedule[epoch][state['cursor']:] for i in b]
        loader = DataLoader(MatchedPlaces(train, indices, epoch), batch_size=16, num_workers=a.workers)
        for images, labels in tqdm(loader, desc=f'{a.mode} {a.stage} e{epoch+1}'):
            t = time.monotonic(); opt.zero_grad(set_to_none=True)
            d, ru, delta = descriptors(images.flatten(0, 1).cuda(), True)
            vpr, _ = lossfn(d, labels.flatten().cuda())
            kl, drift = preservation_loss(d, ru, POLICY['temperature'])
            loss = vpr + (POLICY['relation_weight']*kl+POLICY['drift_weight']*drift if preserved else 0)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(active.values(), 1., error_if_nonfinite=True))
            ssmgrad = 0.
            if a.mode.startswith('mamba'):
                value = model.mixer.ssm.select.weight.grad
                ssmgrad = float(value.abs().sum()) if value is not None else 0.
                state['ssm_gradient_seen'] |= ssmgrad > 0
            if any(p.grad is not None for p in model.base.parameters()):
                raise ValueError('Frozen RU accumulated gradients')
            fixed_step(opt, active.values()); state['steps'] += 1; state['cursor'] += 1
            state['rows'].append(dict(epoch=epoch+1, batch=state['cursor'], vpr=float(vpr.detach()),
                relation=float(kl.detach()), drift=float(drift.detach()), delta_l2=float(delta.detach().norm(dim=-1).mean()),
                gradient_norm=grad, ssm_gradient=ssmgrad, seconds=time.monotonic()-t))
            write(a.output/'progress.json', dict(phase='training', mode=a.mode, stage=a.stage, epoch=epoch+1,
                  done=state['cursor'], total=len(schedule[epoch]), steps=state['steps']))
            if state['cursor'] % 16 == 0:
                save(a.output/'last.pt', model, opt, state)
        state.update(epoch=epoch+1, cursor=0); save(a.output/'last.pt', model, opt, state)
    if state['steps'] != sum(map(len, schedule)) or not any(r['vpr'] > 0 for r in state['rows']):
        raise ValueError('Missing updates or informative batches')
    if any(int(opt.state[p]['step']) != state['steps'] for p in active.values()):
        raise ValueError('Adam clocks differ')
    if a.mode.startswith('mamba') and not state['ssm_gradient_seen']:
        raise ValueError('No selective-state gradient')
    probe('final')
    gradient_probe('final')
    if a.stage == 'full':
        benchmark('final')
    if any(not torch.equal(v.cpu(), frozen[k]) for k, v in model.base.state_dict().items()):
        raise ValueError('Frozen RU changed')
    before = {k: v.detach().clone() for k, v in active.items()}
    restore(torch.load(a.output/'last.pt', map_location='cpu', weights_only=True), model, opt)
    if any(not torch.equal(v, before[k]) for k, v in active.items()):
        raise ValueError('Checkpoint roundtrip mismatch')
    write(a.output/'history.json', state['rows'])
    write(a.output/'summary.json', dict(mode=a.mode, stage=a.stage, optimizer_steps=state['steps'],
        zero_start_error=err, frozen_ru_unchanged=True, checkpoint_roundtrip=True, matched_optimizer_clock=True,
        ssm_gradient_seen=state['ssm_gradient_seen'], nonzero_vpr_batches=sum(r['vpr'] > 0 for r in state['rows']),
        trainable_parameters=sum(v.numel() for v in active.values()), max_memory_allocated=torch.cuda.max_memory_allocated(),
        scope=POLICY['scope']))
    write(a.output/'progress.json', dict(phase='complete', mode=a.mode, stage=a.stage))
    complete(a.output); print('COMPLETE', a.output, flush=True)


if __name__ == '__main__':
    main()
