"""Author BoQ + existing conv/RMS innovation; separate immutable transfer runs."""
import argparse
import datetime
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.train_slgd import read, write, sha, code_hash, verify, seal, seed, save, restore, parameter_hash
from scripts.slgd_mid_screen import open_contract, verify_cache_shards
from scripts.hcen_plain_base import context
from scripts.train_hcen_vpr import PlaceBatches, tensor_hash
from scripts.hcen_module_ablation import wait_gpu
from scripts.conv_residual_ablation import forward_batch, diagnostics_files, SOURCES as OLD_SOURCES
from src.hcen_followup_protocol import training_schedule
from src.models.conv_residual_ablation import PARAMETERS, module_gradient_status
from src.models.official_boq import COMMIT, WEIGHT_SHA, load_official, verify_download, build_adapter
from src.slgd_mid_supervision import gps_filtered_retrieval

SOURCES = tuple(dict.fromkeys(('scripts/conv_official_boq.py', 'src/models/official_boq.py',
    'scripts/run_conv_official_boq.sh', 'tests/test_official_boq.py', *OLD_SOURCES)))
POLICY = dict(revision=1, seeds=[42, 43, 44], layers=[8, 10, 12], train_size=280,
    eval_sizes=[280, 322], primary_eval_size=322, smoke_steps=8, steps=2048,
    lr=1e-4, weight_decay=0., clip=1., microbatch=4, precision='fp32', rho=.1,
    trainable_parameters=PARAMETERS, effective_parameters=PARAMETERS-65536,
    auxiliary_weight=0., rank_weight=0.,
    selection='Fresh official initialization per seed; smoke is discarded; fixed last2048; no best seed/epoch/size selection',
    data='Reuse verified GSV-only fixed four-view training schedule and historical RU-mined negative indices; no RU model/features/teacher supervision',
    training='Freeze the complete author encoder and BoQ; update existing conv/RMS neck only',
    scope='Official-weight transfer on exposed MSLS validation and Pitts30k validation. Not full GSV retraining, independent blind test, or Pitts30k-test comparison')


def contract(args, stage, initialization_seed=None):
    download = verify_download(args.official_cache)
    base, data, heldout, cache = context(args, 'hierarchy', 'screen')
    verify_cache_shards(args.cache, len(cache['train_places']))
    dino_hashes = {str(p.relative_to(args.dino_source)): code_hash(p)
        for p in sorted(args.dino_source.rglob('*.py'))}
    if not dino_hashes:
        raise ValueError('Missing local DINO source')
    base.update(stage=stage, policy=POLICY, checkpoint_sha256=WEIGHT_SHA,
        author_download=download, local_dino_sources=dino_hashes,
        original_local_baseline_not_used=True,
        code={p: code_hash(ROOT / p) for p in SOURCES},
        versions=dict(torch=str(torch.__version__), numpy=str(np.__version__),
                      torchvision=__import__('torchvision').__version__),
        initialization_seed=initialization_seed,
        schedule=training_schedule(read(args.cache/'triplets.json')['rows'],
            'smoke' if stage == 'smoke' else 'screen') if stage in ('smoke','train') else None)
    return base, data, cache


def benchmark_datasets(size):
    from torchvision import transforms as T
    from src.dataloaders.valid.mapillary_sls import MapillarySLSDataset
    from src.dataloaders.valid.pittsburgh import PittsburghDataset
    transform = T.Compose([T.ToTensor(), T.Resize((size, size),
        interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.Normalize([.485, .456, .406], [.229, .224, .225])])
    return (MapillarySLSDataset(Path('datasets/msls-val'), transform),
            PittsburghDataset(Path('datasets/pitts30k-val'), transform))


def retrieve(features, dataset):
    features = torch.as_tensor(features, device='cuda')
    db = features[:dataset.num_references]
    predictions = np.concatenate([(q @ db.T).topk(20, dim=1).indices.cpu().numpy()
        for q in features[dataset.num_references:].split(16)])
    hits = np.asarray([[bool(np.isin(row[:k], gt).any()) for k in (1,5,10,20)]
        for row, gt in zip(predictions, dataset.ground_truth)], dtype=bool)
    return dict(predictions=predictions.tolist(), outcomes=hits[:,0].tolist(),
        correct=int(hits[:,0].sum()), queries=len(hits),
        recall_k=[1,5,10,20], recall=hits.mean(0).tolist())


def evaluate(args, model, output, adapter=False):
    from tqdm import tqdm
    model.eval()
    for size in POLICY['eval_sizes']:
        for dataset in benchmark_datasets(size):
            tag = f'{dataset.dataset_name}_{size}'
            report_path = output / (tag + '.json')
            if report_path.is_file():
                print('Reuse finished evaluation', tag, flush=True)
                continue
            descriptors = np.empty((len(dataset),12288),np.float32)
            rows = []
            loader = DataLoader(dataset,batch_size=16,num_workers=args.workers)
            with torch.inference_mode():
                for batch,(images,indices) in enumerate(tqdm(loader,desc=tag)):
                    expected = list(range(batch*16,batch*16+len(images)))
                    if indices.tolist() != expected:
                        raise ValueError('Unexpected benchmark order')
                    if adapter:
                        d, _, _, _, metrics = forward_batch(model, images)
                        rows.extend(metrics)
                    else:
                        d = torch.cat([model(chunk.cuda()) for chunk in images.split(4)])
                    if not bool(torch.isfinite(d).all()) or float((d.norm(dim=1)-1).abs().max()) > 1e-4:
                        raise ValueError('Invalid author/adapted descriptor')
                    descriptors[indices.numpy()] = d.cpu().numpy()
                    write(output/'progress.json',dict(phase='evaluation',dataset=tag,
                        batch=batch+1,total=len(loader)))
            result = retrieve(descriptors,dataset)
            result.update(input_size=size, author_weight_sha256=WEIGHT_SHA,
                preprocessing='Tensor bicubic antialias then ImageNet normalization',
                dataset_note=('740-query MSLS-val, not MSLS challenge' if dataset.dataset_name=='msls-val'
                              else '7608-query Pitts30k-val, NOT 6816-query Pitts30k-test'))
            if adapter:
                reference = read(args.root/'baseline'/(tag+'.json'))
                base, current = np.asarray(reference['outcomes'],bool), np.asarray(result['outcomes'],bool)
                if len(base) != len(current):
                    raise ValueError('Baseline/student population mismatch')
                result['paired'] = dict(baseline_correct=reference['correct'],
                    corrections=np.flatnonzero(~base & current).tolist(),
                    regressions=np.flatnonzero(base & ~current).tolist(),
                    net=int(current.sum()-base.sum()),
                    delta_r1_pp=100*float(current.mean()-base.mean()))
                diagnostics_files(output,tag,rows,np.arange(dataset.num_references,len(dataset)))
            write(report_path,result)
            print('EVAL',tag,'correct',result['correct'],'R1',result['recall'][0],
                  'paired',result.get('paired',{}),flush=True)


def baseline(args):
    c, _, _ = contract(args,'baseline')
    open_contract(args.output,c,args.resume)
    if (args.output/'completed.json').is_file():
        verify(args.output); print('Author baseline already complete'); return
    seed(42)
    visual=load_official(args.official_cache,args.dino_source,'cuda')
    frozen=tensor_hash(visual)
    evaluate(args,visual,args.output)
    msls=read(args.output/'msls-val_322.json')
    write(args.output/'summary.json',dict(phase='baseline_complete',
        frozen_tensor_sha256=frozen,official_reported_msls_r1=.938,
        measured_msls_r1=msls['recall'][0],
        matches_official_one_decimal=round(msls['recall'][0]*100,1)==93.8,
        note='Do not modify preprocessing/GT or select a resolution to force paper agreement',scope=POLICY['scope']))
    write(args.output/'progress.json',dict(phase='complete'))
    seal(args.output)


def train(args):
    from tqdm import tqdm
    from src.losses.vpr_losses import VPRLossFunction
    c,data,cache=contract(args,args.stage,args.seed)
    verify(args.root/'baseline')
    c['baseline_completed_sha256']=sha(args.root/'baseline/completed.json')
    open_contract(args.output,c,args.resume)
    if (args.output/'completed.json').is_file():
        verify(args.output); print('Already complete',args.stage,args.seed); return
    seed(42)
    visual=load_official(args.official_cache,args.dino_source,'cuda')
    frozen=tensor_hash(visual)
    if frozen != read(args.root/'baseline/summary.json')['frozen_tensor_sha256']:
        raise ValueError('Author baseline parameters changed')
    seed(args.seed); model=build_adapter(visual,args.seed).cuda().eval()
    # Cover both native training and official testing grids before any update.
    zero_errors={}
    with torch.inference_mode():
        for size in (280,322):
            dataset_=benchmark_datasets(size)[0]
            for batch_size in (1,4):
                probe=torch.stack([dataset_[i][0] for i in range(batch_size)])
                d,r,_,_,_=forward_batch(model,probe)
                direct=visual(probe.cuda())
                error=float(torch.maximum((d-direct).abs().max(),(r-direct).abs().max()))
                if error > 1e-6:
                    raise ValueError(f'Author zero-start mismatch at {size}/{batch_size}: {error}')
                zero_errors[f'{size}_batch{batch_size}']=error
    active={n:p for n,p in model.named_parameters() if p.requires_grad}
    if sum(p.numel() for p in active.values())!=PARAMETERS or any(not n.startswith('neck.') for n in active):
        raise ValueError('Unexpected trainable parameter inventory')
    optimizer=torch.optim.AdamW(active.values(),lr=POLICY['lr'],weight_decay=0.)
    state=dict(steps=0,rows=[],module_gradients={'0':False,'1':False},
        up_gradient_seen=False,contract_sha256=sha(args.output/'contract.json'),
        initial_parameter_sha256=parameter_hash(model),
        initial_auxiliary_sha256=tensor_hash(model.neck.aux))
    if (args.output/'last.pt').is_file():
        state=restore(args.output/'last.pt',model,optimizer,state['contract_sha256'])
    if not (args.output/'initial.json').exists():
        write(args.output/'initial.json',dict(zero_start_errors=zero_errors,
            trainable_parameters=PARAMETERS,effective_parameters=PARAMETERS-65536,
            initial_parameter_sha256=state['initial_parameter_sha256'],
            frozen_tensor_sha256=frozen))
        save(args.output/'last.pt',model,optimizer,state)
    schedule=c['schedule']; lossfn=VPRLossFunction();model.train()
    loader=DataLoader(PlaceBatches(data,cache['train_indices'],schedule[state['steps']:]),
        batch_size=None,num_workers=args.workers)
    for images,labels,coordinates in tqdm(loader,total=len(schedule),initial=state['steps'],
            desc=f'Official BoQ conv {args.seed} {args.stage}'):
        optimizer.zero_grad(set_to_none=True)
        d,r,aux,delta,_=forward_batch(model,images,True)
        loss=gps_filtered_retrieval(lossfn,d,labels.cuda(),coordinates.numpy())
        auxiliary=sum(gps_filtered_retrieval(lossfn,x,labels.cuda(),coordinates.numpy()) for x in aux)/len(aux)
        total=loss+0.*auxiliary
        if not bool(torch.isfinite(total)) or float(delta.max())>.1+1e-6:
            raise ValueError('Nonfinite loss or invalid relative residual bound')
        total.backward()
        for key,value in module_gradient_status(model).items():state['module_gradients'][key]|=value
        state['up_gradient_seen']|=bool(model.neck.up.weight.grad is not None and (model.neck.up.weight.grad!=0).any())
        for p in active.values():
            if p.grad is None:p.grad=torch.zeros_like(p)
        norm=torch.nn.utils.clip_grad_norm_(list(active.values()),1.,error_if_nonfinite=True)
        if any(p.grad is not None for p in visual.parameters()):
            raise ValueError('Official frozen model unexpectedly received gradients')
        optimizer.step();state['steps']+=1
        state['rows'].append(dict(step=state['steps'],loss=float(loss.detach()),gradient_norm=float(norm),
            relative_residual_mean=float(delta.mean()),relative_residual_max=float(delta.max())))
        write(args.output/'progress.json',dict(phase='training',seed=args.seed,
            steps=state['steps'],total=len(schedule),loss=float(loss.detach())))
        if state['steps']%16==0:save(args.output/'last.pt',model,optimizer,state)
    save(args.output/'last.pt',model,optimizer,state)
    final=parameter_hash(model)
    restore(args.output/'last.pt',model,optimizer,state['contract_sha256'])
    checks=dict(zero_start=all(e<=1e-6 for e in zero_errors.values()),
        frozen_author_unchanged=tensor_hash(visual)==frozen,
        module_gradients=all(state['module_gradients'].values()),up_gradient=state['up_gradient_seen'],
        complete_steps=state['steps']==len(schedule),checkpoint_roundtrip=parameter_hash(model)==final,
        nonzero_updates=final!=state['initial_parameter_sha256'],
        auxiliary_unchanged=tensor_hash(model.neck.aux)==state['initial_auxiliary_sha256'],
        optimizer_clocks=all(int(optimizer.state[p]['step'])==len(schedule) for p in active.values()))
    if not all(checks.values()):raise ValueError('Mechanics failed: '+str(checks))
    optimizer.zero_grad(set_to_none=True);model.eval()
    if args.stage=='train':evaluate(args,model,args.output,True)
    write(args.output/'summary.json',dict(seed=args.seed,stage=args.stage,steps=state['steps'],
        checks=checks,rows=state['rows'],final_parameter_sha256=final,scope=POLICY['scope']))
    write(args.output/'progress.json',dict(phase='complete',steps=state['steps']))
    seal(args.output);print('OFFICIAL TRANSFER COMPLETE',args.stage,args.seed,checks,flush=True)


def aggregate(args):
    baseline_c=verify(args.root/'baseline');parts={};completed={}
    for s in POLICY['seeds']:
        p=args.root/f'seed{s}'/'train';c=verify(p);summary=read(p/'summary.json')
        if c['initialization_seed']!=s or c['policy']!=POLICY or summary['steps']!=2048 or not all(summary['checks'].values()):
            raise ValueError('Incomplete or changed official transfer')
        completed[str(s)]=sha(p/'completed.json')
        for size in POLICY['eval_sizes']:
            for name in ('msls-val','pitts30k-val'):
                tag=f'{name}_{size}';base=read(args.root/'baseline'/(tag+'.json'))
                value=read(p/(tag+'.json'))
                parts.setdefault(tag,dict(queries=base['queries'],baseline_correct=base['correct'],
                    baseline_recall=base['recall'],per_seed={}))['per_seed'][str(s)]={
                        k:value[k] for k in ('correct','recall','paired')}
    for part in parts.values():
        values=list(part['per_seed'].values());n=part['queries']
        part['mean_r1']=statistics.mean(v['correct'] for v in values)/n
        part['mean_delta_pp']=100*(statistics.mean(v['correct'] for v in values)-part['baseline_correct'])/n
        part['sample_std_correct']=statistics.stdev(v['correct'] for v in values)
    c=dict(policy=POLICY,official_baseline=sha(args.root/'baseline/completed.json'),completed=completed,
        code={p:code_hash(ROOT/p) for p in SOURCES})
    open_contract(args.output,c,args.resume)
    if (args.output/'completed.json').is_file():verify(args.output);return
    write(args.output/'summary.json',dict(verdict='OFFICIAL_WEIGHT_TRANSFER_COMPLETE',
        results=parts,primary_size=322,scope=POLICY['scope']))
    seal(args.output)
    print('OFFICIAL THREE-SEED TRANSFER FINISHED',flush=True)


def main():
    os.chdir(ROOT)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['wait_gpu','baseline','train','aggregate'])
    p.add_argument('--stage',choices=['smoke','train'],default='train')
    p.add_argument('--seed',type=int,choices=[42,43,44],default=42)
    p.add_argument('--root',type=Path,default=Path('logs/conv_official_boq_v1'))
    p.add_argument('--output',type=Path);p.add_argument('--resume',action='store_true')
    p.add_argument('--official-cache',type=Path,default=Path('.cache/boq_official_v1'))
    p.add_argument('--dino-source',type=Path,default=Path('/home/wt/.cache/torch/hub/facebookresearch_dinov2_main'))
    p.add_argument('--dataset-root',type=Path,default=Path('datasets/gsv_cities'))
    p.add_argument('--source',type=Path,default=Path('logs/slgd_hard_v2_audit'))
    p.add_argument('--teacher',type=Path,default=Path('logs/slgd_v1/teacher_pilot'))
    p.add_argument('--cache',type=Path,default=Path('.cache/slgd_mid_v1_pilot'))
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':p.error('Expose physical GPU1 only')
    if args.root.resolve().parent != (ROOT/'logs').resolve() or not args.root.name.startswith('conv_official_boq_'):
        p.error('Use an isolated logs/conv_official_boq_<version> directory')
    if args.command=='wait_gpu':wait_gpu(args);return
    if args.output is None or not args.output.resolve().is_relative_to(args.root.resolve()) or args.output.resolve()==args.root.resolve():
        p.error('Output must be a child of the new experiment root')
    if args.workers<0:p.error('Workers must be nonnegative')
    import fcntl
    args.root.mkdir(parents=True,exist_ok=True)
    with (args.root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        if args.command!='aggregate':torch.cuda.set_per_process_memory_fraction(.65,0)
        try:{'baseline':baseline,'train':train,'aggregate':aggregate}[args.command](args)
        except Exception as e:
            if (args.output/'contract.json').is_file() and not (args.output/'completed.json').is_file():
                write(args.output/'failure.json',dict(error_type=type(e).__name__,message=str(e),
                    time=datetime.datetime.now().astimezone().isoformat()))
            raise


if __name__=='__main__':main()
