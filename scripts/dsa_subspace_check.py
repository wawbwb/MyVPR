"""Train-only DSA basis fit, place-disjoint validation, and zero-update preflight."""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.candidate_set_screen import read, write, sha, complete
from scripts.adaptive_pair_budget import verified
from scripts.train_depth_query import RU_SHA, seed, MatchedPlaces
from scripts.eval_condition_robustness import load_inference_model_from_ckpt
from src.dataloaders.train.gsv_cities import GSVCitiesDataset
from src.discriminative_subspace import MODES, fit, shuffle_places, ratio, overlap, assess
from src.models.discriminative_subspace import SubspaceVPR


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--source', type=Path, default=Path('logs/second_order_query/mean_outer_screen_v1'))
    p.add_argument('--dataset-root', type=Path, default=Path('datasets/gsv_cities'))
    p.add_argument('--output', type=Path, default=Path('logs/dsa_phase0_v1'))
    p.add_argument('--resume', action='store_true')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    import fcntl
    from torchvision.transforms import v2 as T
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Expose physical GPU1 only with CUDA_VISIBLE_DEVICES=1')
    if not a.checkpoint.is_file() or sha(a.checkpoint) != RU_SHA:
        raise ValueError('Expected original RU checkpoint')
    verified(a.source)
    source = read(a.source/'contract.json')
    train_ids, dev_ids = source['train_places'], source['dev_places']
    if len(set(train_ids)) != 4096 or len(set(dev_ids)) != 1024 or set(train_ids)&set(dev_ids):
        raise ValueError('Expected original disjoint 4096/1024-place split')
    metadata = sorted((a.dataset_root/'Dataframes').glob('*.csv'))
    if not metadata: raise ValueError('Missing GSV metadata')
    for f in metadata:
        if source['data'].get(str(f)) != sha(f): raise ValueError('Changed GSV metadata: '+str(f))
    code_files = ['scripts/dsa_subspace_check.py', 'src/discriminative_subspace.py',
                  'src/models/discriminative_subspace.py', 'scripts/train_depth_query.py',
                  'scripts/eval_condition_robustness.py', 'src/dataloaders/train/gsv_cities.py',
                  'src/models/backbones/dinov2.py', 'src/models/aggregators/boq.py']
    contract = dict(checkpoint_sha256=RU_SHA, source_completed_sha256=sha(a.source/'completed.json'),
        train_places=train_ids, dev_places=dev_ids, metadata={str(f):sha(f) for f in metadata},
        policy=dict(rank=16, views=4, size=280, blocks=[11,12], seed=42, shuffle_seed=42029,
                    shrinkage=.1, ridge=1e-6, ratio_gain=1.05, retained_between=.25, max_overlap=.95,
                    extraction='mean norm1 patch tokens, epoch0 clean views; no CLS', optimizer_steps=0),
        code={f:hashlib.sha256((ROOT/f).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for f in code_files})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    lock=(a.output.parent/(a.output.name+'.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.output.exists():
        if not a.resume or read(a.output/'contract.json') != contract: raise ValueError('Existing run/contract mismatch')
        if (a.output/'completed.json').exists(): verified(a.output); print('Already complete'); return
    else:
        a.output.mkdir(); write(a.output/'contract.json',contract)
    seed(42)
    clean=T.Compose([T.ToImage(),T.Resize((280,280),interpolation=T.InterpolationMode.BICUBIC,antialias=True),
                     T.ToDtype(torch.float32,scale=True),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    dataset=GSVCitiesDataset(dataset_path=a.dataset_root,cities='all',img_per_place=4,transform=clean)
    lookup={str(pid):i for i,pid in enumerate(dataset.places_ids)}
    indices=[lookup[pid] for pid in train_ids+dev_ids]
    visual=load_inference_model_from_ckpt(a.checkpoint,'cpu').cuda().eval().requires_grad_(False)
    dino=visual.backbone.dino
    shards=a.output/'vectors'; shards.mkdir(exist_ok=True)
    arrays=[]
    loader=DataLoader(MatchedPlaces(dataset,indices,0),batch_size=8,num_workers=a.workers,shuffle=False)
    for bi,(images,_) in enumerate(tqdm(loader,desc='DSA image-level vectors')):
        path=shards/f'{bi:04d}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as z: values=z['vectors'].copy()
        else:
            with torch.inference_mode():
                x=dino.prepare_tokens_with_masks(images.flatten(0,1).cuda()); collected=[]
                for j,block in enumerate(dino.blocks):
                    if j in (10,11): collected.append(block.norm1(x)[:,1:].mean(1).cpu())
                    x=block(x)
                values=torch.stack(collected,1).reshape(len(images),4,2,768).numpy()
            with path.with_suffix('.tmp').open('wb') as f: np.savez(f,vectors=values)
            path.with_suffix('.tmp').replace(path)
        if values.shape != (len(images),4,2,768) or not np.isfinite(values).all():
            raise ValueError('Invalid vector shard: '+str(path))
        arrays.append(values)
        write(a.output/'progress.json',dict(phase='extract',places=min((bi+1)*8,5120),total=5120))
    vectors=torch.from_numpy(np.concatenate(arrays)); bases={m:[] for m in MODES}; report={}
    write(a.output/'progress.json',dict(phase='fit_train_only'))
    for layer in range(2):
        train,dev=vectors[:4096,:,layer],vectors[4096:,:,layer]
        layer_report={}; metrics={}
        for mode in MODES:
            fitting=shuffle_places(train) if mode=='shuffled_fisher' else train
            u,details=fit(fitting,fisher=mode!='pca')
            bases[mode].append(u.float()); metrics[mode]=ratio(dev,u)
            halves=[fit(fitting[i::2],fisher=mode!='pca')[0] for i in (0,1)]
            layer_report[mode]=dict(fit=details,train=ratio(train,u),development=metrics[mode],
                                   train_half_overlap=overlap(*halves))
        between=overlap(bases['place_fisher'][-1],bases['shuffled_fisher'][-1])
        report[str(layer+11)]=dict(modes=layer_report,true_shuffled_overlap=between,gate=assess(metrics,between))
    torch.save(bases,a.output/'bases.tmp'); (a.output/'bases.tmp').replace(a.output/'bases.pt')
    write(a.output/'progress.json',dict(phase='zero_update_gradient_preflight'))
    sample=MatchedPlaces(dataset,indices[:1],0)[0][0][:2].cuda()
    with torch.inference_mode(): expected=visual(sample).clone()
    model=SubspaceVPR(visual,bases['place_fisher']).cuda().eval()
    active={k:p for k,p in model.named_parameters() if p.requires_grad}
    if sum(p.numel() for p in active.values()) != 49152: raise ValueError('Trainable count mismatch')
    probes={}
    for mode in MODES:
        with torch.no_grad():
            for adapter,u in zip(model.adapters,bases[mode]): adapter.basis.copy_(u.cuda())
        model.zero_grad(set_to_none=True)
        actual=model(sample); error=float((actual.detach()-expected).abs().max())
        if error > 2e-6 or not torch.isfinite(actual).all(): raise ValueError('Zero-start mismatch')
        (actual*torch.linspace(-1,1,actual.shape[1],device=actual.device)).sum().backward()
        grads={k:float(v.grad.abs().sum()) if v.grad is not None else 0. for k,v in active.items()}
        if any(not np.isfinite(v) or v<=0 for v in grads.values()): raise ValueError('Dead/nonfinite adapter gradient')
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad): raise ValueError('Frozen gradient')
        if any(torch.count_nonzero(p).item() for p in active.values()): raise ValueError('Unexpected parameter update')
        probes[mode]=dict(zero_start_error=error,probe_grad_l1=grads)
        del actual
    summary=dict(layers=report,preflight=probes,trainable_parameters=49152,optimizer_steps=0,
        verdict='PASS_PHASE0' if all(r['gate']['pass_gate'] for r in report.values()) else 'STOP_SUBSPACE_GATE',
        scope='Exploratory place-disjoint GSV subspace diagnostic; no VPR training or recall improvement claim')
    write(a.output/'summary.json',summary)
    write(a.output/'progress.json',dict(phase='complete',verdict=summary['verdict']))
    complete(a.output); print(summary,flush=True)


if __name__=='__main__': main()
