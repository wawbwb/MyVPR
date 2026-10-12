"""Three controls on the actual plain-BoQ checkpoint, with baseline sealing first."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_slgd import read,write,sha,code_hash,verify,seal,seed,save,restore,parameter_hash
from scripts.slgd_mid_screen import dataset,sources,current_code,open_contract,verify_cache_shards
from scripts.audit_slgd_hard import PlannedPlaces,collate
from scripts.train_hcen_vpr import PlaceBatches,tensor_hash
from scripts.train_hcen_aux_ablation import SOURCES as BASE_SOURCES
from scripts.eval_condition_robustness import load_inference_model_from_ckpt,strip_compiled_model_prefix
from src.hcen_protocol import benchmark_transform
from src.hcen_aux_ablation import training_schedule,gradient_diagnostic
from src.hcen_plain_protocol import POLICY,PLAIN_SHA,PLAIN_CHECKPOINT,MODES,validate_checkpoint,paired_report,PlainHCENVPR,canonical_features
from src.slgd_mid_supervision import gps_filtered_retrieval
from src.slgd_hard_protocol import distance_metres

SOURCES=('scripts/hcen_plain_base.py','src/hcen_plain_protocol.py',*BASE_SOURCES)


def forward_plain_batch(model,images,training=False):
    descriptors=[];references=[];aux=[[],[]];deltas=[]
    for chunk in images.cuda().split(4):
        features=canonical_features(model.features(chunk))
        with torch.no_grad():references.append(model.readout(features[-1]))
        value=checkpoint(lambda a,b,c:model.aggregate(a,b,c,True),*features,use_reentrant=False) if training else model.aggregate(*features,auxiliary=False)
        descriptors.append(value[0])
        if training:
            aux[0].append(value[1]);aux[1].append(value[2])
        last=features[-1]
        deltas.append(value[-1].detach().flatten(1).norm(dim=1)/last.flatten(1).norm(dim=1).clamp_min(1e-8))
    return torch.cat(descriptors),torch.cat(references),tuple(torch.cat(x) for x in aux) if training else (),torch.cat(deltas)


def load_plain(args):
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    state=strip_compiled_model_prefix(ck['state_dict']);validate_checkpoint(ck['hyper_parameters'],state)
    seed(42);visual=load_inference_model_from_ckpt(args.checkpoint,'cpu').cuda().eval().requires_grad_(False)
    if visual.semantic_region_gate is not None or visual.spatial_attn_head is not None:
        raise ValueError('Plain visual model unexpectedly has a custom gate')
    # The generic historical loader validates each module; here also require all keys.
    if set(visual.state_dict())!=set(state):raise ValueError('Checkpoint tensors silently omitted')
    if any(not torch.equal(p.detach().cpu(),state[n]) for n,p in visual.state_dict().items()):
        raise ValueError('Loaded plain checkpoint tensor differs')
    del ck,state
    return visual


def context(args,mode,stage):
    teacher,heldout,metadata=sources(SimpleNamespace(**{**vars(args),'stage':'pilot'}))
    cache=verify(args.cache)
    if cache['code']!=current_code() or cache['metadata']!=metadata or cache['stage']!='pilot':
        raise ValueError('Historical fixed training population differs')
    if cache['source_completed_sha256']!=sha(args.source/'completed.json') or cache['teacher_completed_sha256']!=sha(args.teacher/'completed.json'):
        raise ValueError('Historical source identities differ')
    if cache['train_places']!=teacher['train_places'][:len(cache['train_places'])]:raise ValueError('Historical training split differs')
    contract=dict(mode=mode,stage=stage,policy=POLICY,checkpoint_sha256=PLAIN_SHA,
        source_completed_sha256=sha(args.source/'completed.json'),cache_completed_sha256=sha(args.cache/'completed.json'),
        metadata=metadata,train_places=cache['train_places'],original_code=current_code(),
        benchmark_metadata={str(p):sha(p) for name in ('msls-val','pitts30k-val') for p in sorted((Path('datasets')/name).glob('*.npy'))},
        code={p:code_hash(ROOT/p) for p in SOURCES},versions=dict(torch=torch.__version__,numpy=np.__version__))
    data=dataset(args);plan=heldout['plan']
    if [str(data.places_ids[i]) for i in cache['train_indices']]!=cache['train_places']:
        raise ValueError('Training place indices differ')
    if [str(data.places_ids[i]) for i in plan['dataset_indices']]!=plan['query_places']+plan['distractor_places']:
        raise ValueError('Heldout place indices differ')
    return contract,data,heldout,cache


def extract(visual,model,images):
    if model is not None:
        d,r,_,_=forward_plain_batch(model,images)
    else:
        d=torch.cat([visual(x.cuda()) for x in images.split(4)]);r=d
    if d.ndim!=2 or d.shape[1]!=12288 or not torch.isfinite(d).all() or not torch.isfinite(r).all():
        raise ValueError('Invalid plain-BoQ descriptor')
    if float((d.norm(dim=1)-1).abs().max())>1e-4:raise ValueError('Nonunit descriptor')
    return d,r


def gsv_evaluate(visual,model,data,heldout,output,workers,baseline=None):
    from tqdm import tqdm
    plan=heldout['plan'];desc=np.empty((plan['rows'],12288),np.float32)
    ref=np.empty_like(desc) if model is not None else desc
    coordinates=np.empty((plan['rows'],2),np.float64)
    loader=DataLoader(PlannedPlaces(data,plan),batch_size=16,num_workers=workers,collate_fn=collate)
    with torch.inference_mode():
        for batch,(images,indices,coords) in enumerate(tqdm(loader,desc='Plain-BoQ GSV full gallery')):
            write(output/'progress.json',dict(phase='gsv_evaluation',batch=batch,total=len(loader)))
            d,r=extract(visual,model,images);desc[indices.numpy()]=d.cpu().numpy();ref[indices.numpy()]=r.cpu().numpy()
            coordinates[indices.numpy()]=coords.numpy()
    q=np.asarray(plan['query_rows']);g=np.asarray(plan['gallery_rows']);positive=np.asarray(plan['positive_gallery_indices'])
    result={}
    for name,features in [('baseline',ref),('student',desc)]:
        gallery=torch.from_numpy(features[g]).cuda();pred=[]
        for start in range(0,len(q),16):
            rows=q[start:start+16];scores=torch.from_numpy(features[rows]).cuda()@gallery.T
            allowed=distance_metres(coordinates[rows],coordinates[g])>=25
            allowed[np.arange(len(rows)),positive[start:start+16]]=True
            scores.masked_fill_(~torch.from_numpy(allowed).cuda(),-torch.inf);pred.extend(scores.argmax(1).cpu().tolist())
        result[name]=dict(predictions=pred,outcomes=(np.asarray(pred)==positive).tolist())
    if baseline is not None and result['baseline']!=read(baseline/'gsv_final.json')['baseline']:
        raise ValueError('GSV plain baseline differs from independently sealed direct forward')
    result['descriptor_drift_mean']=float(np.mean(1-(desc*ref).sum(1)))
    write(output/'gsv_final.json',result)


def benchmarks(visual,model,output,workers,baseline=None):
    from tqdm import tqdm
    from src.dataloaders.valid.mapillary_sls import MapillarySLSDataset
    from src.dataloaders.valid.pittsburgh import PittsburghDataset
    transform=benchmark_transform()
    for ds in [MapillarySLSDataset(Path('datasets/msls-val'),transform),PittsburghDataset(Path('datasets/pitts30k-val'),transform)]:
        path=output/(ds.dataset_name+'_final.json')
        if path.exists():
            saved=read(path)
            if baseline is not None and saved['baseline']!=read(baseline/path.name)['baseline']:raise ValueError('Saved baseline differs')
            continue
        student=[];reference=[]
        with torch.inference_mode():
            loader=DataLoader(ds,batch_size=16,num_workers=workers)
            for batch,(images,_) in enumerate(tqdm(loader,desc=ds.dataset_name+' plain-BoQ')):
                write(output/'progress.json',dict(phase='benchmark',dataset=ds.dataset_name,batch=batch,total=len(loader)))
                d,r=extract(visual,model,images);student.append(d.cpu());reference.append(r.cpu())
        result={}
        for name,parts in [('baseline',reference),('student',student)]:
            all_=torch.cat(parts).cuda();db=all_[:ds.num_references]
            pred=np.concatenate([(q@db.T).topk(20,dim=1).indices.cpu().numpy() for q in all_[ds.num_references:].split(16)])
            hits=np.asarray([[bool(np.isin(row[:k],gt).any()) for k in (1,5,20)] for row,gt in zip(pred,ds.ground_truth)])
            result[name]=dict(predictions=pred.tolist(),outcomes=hits[:,0].tolist(),correct=int(hits[:,0].sum()),recall=hits.mean(0).tolist())
            del all_,db
        if baseline is not None and result['baseline']!=read(baseline/path.name)['baseline']:
            raise ValueError('Plain benchmark baseline differs from independently sealed direct forward')
        if model is None and ds.dataset_name=='msls-val' and result['baseline']['correct']!=POLICY['historical_msls_correct']:
            write(output/'baseline_mismatch.json',dict(dataset=ds.dataset_name,expected=POLICY['historical_msls_correct'],observed=result['baseline']['correct']))
            raise ValueError('Plain MSLS historical baseline mismatch; stop before any training')
        result['preprocessing']='Historical tensor bicubic280 FP32 microbatch4; same as sealed RU HCEN benchmark pixels'
        write(path,result)


def baseline(args):
    contract,data,heldout,_=context(args,'plain_baseline','baseline')
    open_contract(args.output,contract,args.resume)
    if (args.output/'completed.json').exists():verify(args.output);print('Plain baseline already complete');return
    visual=load_plain(args);before=tensor_hash(visual)
    probe=PlannedPlaces(data,heldout['plan'])[0][0]
    with torch.inference_mode():
        direct=visual(probe.cuda())
        for mode in MODES:
            seed(42);neck=PlainHCENVPR(visual,mode).cuda().eval();d,r,_,_=forward_plain_batch(neck,probe)
            if not torch.equal(d,direct) or not torch.equal(r,direct):raise ValueError('Plain direct forward/zero-start differs')
        del neck
    gsv_evaluate(visual,None,data,heldout,args.output,args.workers)
    benchmarks(visual,None,args.output,args.workers)
    if tensor_hash(visual)!=before:raise ValueError('Plain baseline tensor changed')
    write(args.output/'summary.json',dict(verdict='PLAIN_BASELINE_VERIFIED',checkpoint_sha256=PLAIN_SHA,
        frozen_tensor_sha256=before,zero_start_matches_direct_forward=True,
        checks=dict(no_ru_or_extra_modules=True,strict_full_state_loaded=True,zero_start=True,frozen=True),scope=POLICY['scope']))
    write(args.output/'progress.json',dict(phase='complete'));seal(args.output)
    print('PLAIN BASELINE VERIFIED',flush=True)


def baseline_identity(args):
    path=args.root/'baseline';c=verify(path);s=read(path/'summary.json')
    if c['checkpoint_sha256']!=PLAIN_SHA or c['code']!={p:code_hash(ROOT/p) for p in SOURCES} or s['verdict']!='PLAIN_BASELINE_VERIFIED':
        raise ValueError('Plain baseline provenance differs')
    return path,sha(path/'completed.json')


def train(args):
    from tqdm import tqdm
    from src.losses.vpr_losses import VPRLossFunction
    reference,reference_sha=baseline_identity(args)
    contract,data,heldout,cache=context(args,args.mode,args.stage)
    verify_cache_shards(args.cache,len(cache['train_places']))
    schedule=training_schedule(read(args.cache/'triplets.json')['rows'],args.stage)
    contract.update(schedule=schedule,baseline_completed_sha256=reference_sha)
    open_contract(args.output,contract,args.resume)
    if (args.output/'completed.json').exists():verify(args.output);print('Already complete',args.mode);return
    visual=load_plain(args);frozen=tensor_hash(visual)
    if frozen!=read(reference/'summary.json')['frozen_tensor_sha256']:raise ValueError('Baseline model tensors differ')
    seed(42);model=PlainHCENVPR(visual,args.mode).cuda().eval()
    probe=PlannedPlaces(data,heldout['plan'])[0][0]
    with torch.inference_mode():
        d,r,_,_=forward_plain_batch(model,probe);direct=visual(probe.cuda())
    if not torch.equal(d,direct) or not torch.equal(r,direct):raise ValueError('Plain zero start differs')
    active={n:p for n,p in model.named_parameters() if p.requires_grad}
    if any(not n.startswith('neck.') for n in active):raise ValueError('Only HCEN neck may train')
    optimizer=torch.optim.AdamW(active.values(),lr=POLICY['lr'],weight_decay=0.)
    state=dict(steps=0,rows=[],ssm_gradient_seen=False,up_gradient_seen=False,
        initial_parameter_sha256=parameter_hash(model),initial_auxiliary_sha256=tensor_hash(model.neck.aux),
        contract_sha256=sha(args.output/'contract.json'))
    if (args.output/'last.pt').exists():state=restore(args.output/'last.pt',model,optimizer,state['contract_sha256'])
    if not (args.output/'initial.json').exists():
        write(args.output/'initial.json',dict(parameter_sha256=state['initial_parameter_sha256'],
            descriptor_sha256=hashlib.sha256(d.cpu().numpy().tobytes()).hexdigest(),trainable_parameters=sum(p.numel() for p in active.values()),
            frozen_tensor_sha256=frozen,zero_start_error=0.))
        save(args.output/'last.pt',model,optimizer,state)
    lossfn=VPRLossFunction();model.train()
    loader=DataLoader(PlaceBatches(data,cache['train_indices'],schedule[state['steps']:]),batch_size=None,num_workers=args.workers)
    for images,labels,coordinates in tqdm(loader,total=len(schedule),initial=state['steps'],desc='Plain '+args.mode+' '+args.stage):
        optimizer.zero_grad(set_to_none=True)
        d,r,aux,delta=forward_plain_batch(model,images,True);labels=labels.cuda();coords=coordinates.numpy()
        primary=gps_filtered_retrieval(lossfn,d,labels,coords)
        auxiliary=sum(gps_filtered_retrieval(lossfn,x,labels,coords) for x in aux)/len(aux)
        loss=primary+0.*auxiliary
        with torch.no_grad():reference_loss=gps_filtered_retrieval(lossfn,r,labels,coords)
        if not torch.isfinite(loss):raise ValueError('Nonfinite plain-base HCEN loss')
        gradients=gradient_diagnostic(primary,0.*auxiliary,model.named_parameters()) if state['steps']%64==0 else None
        loss.backward()
        state['ssm_gradient_seen']|=any('ssm.select' in n and p.grad is not None and bool((p.grad!=0).any()) for n,p in active.items())
        state['up_gradient_seen']|=bool(model.neck.up.weight.grad is not None and (model.neck.up.weight.grad!=0).any())
        for p in active.values():
            if p.grad is None:p.grad=torch.zeros_like(p)
        norm=torch.nn.utils.clip_grad_norm_(list(active.values()),POLICY['clip'],error_if_nonfinite=True)
        if any(p.grad is not None for p in visual.parameters()):raise ValueError('Frozen plain model received gradient')
        if float(delta.max())>POLICY['feature_relative_cap']+1e-6:raise ValueError('Residual cap exceeded')
        optimizer.step();state['steps']+=1
        state['rows'].append(dict(step=state['steps'],primary=float(primary.detach()),frozen_baseline_loss=float(reference_loss),
            auxiliary=float(auxiliary.detach()),gradient_norm=float(norm),gradients=gradients,
            relative_residual_mean=float(delta.mean()),relative_residual_max=float(delta.max()),places=len(labels.unique()),images=len(labels)))
        write(args.output/'progress.json',dict(phase='training',mode=args.mode,steps=state['steps'],total=len(schedule),last_loss=float(loss.detach())))
        if state['steps']%16==0:save(args.output/'last.pt',model,optimizer,state)
    if state['steps']!=len(schedule) or tensor_hash(visual)!=frozen:raise ValueError('Budget incomplete or plain base mutated')
    if tensor_hash(model.neck.aux)!=state['initial_auxiliary_sha256']:raise ValueError('Zero-weight auxiliary head changed')
    if any(int(optimizer.state[p]['step'])!=len(schedule) for p in active.values()):raise ValueError('Optimizer clocks differ')
    save(args.output/'last.pt',model,optimizer,state);final=parameter_hash(model)
    restore(args.output/'last.pt',model,optimizer,state['contract_sha256'])
    if final!=parameter_hash(model):raise ValueError('Checkpoint roundtrip differs')
    optimizer.zero_grad(set_to_none=True);model.eval()
    if args.stage=='screen':
        gsv_evaluate(visual,model,data,heldout,args.output,args.workers,reference)
        benchmarks(visual,model,args.output,args.workers,reference)
    checks=dict(zero_start=True,frozen_plain_unchanged=True,checkpoint_roundtrip=True,disabled_auxiliary_unchanged=True,
        nonzero_updates=final!=state['initial_parameter_sha256'],ssm_gradient_seen=state['ssm_gradient_seen'],up_gradient_seen=state['up_gradient_seen'])
    write(args.output/'summary.json',dict(mode=args.mode,stage=args.stage,steps=state['steps'],checks=checks,rows=state['rows'],
        initial_parameter_sha256=state['initial_parameter_sha256'],final_parameter_sha256=final,scope=POLICY['scope']))
    write(args.output/'progress.json',dict(phase='complete',steps=state['steps']));seal(args.output)
    print('PLAIN HCEN COMPLETE',args.mode,checks,flush=True)
    if not all(checks.values()):raise SystemExit(2)


def compare(args):
    reference,reference_sha=baseline_identity(args)
    contracts={m:verify(args.root/(m+'_'+args.stage)) for m in MODES}
    common=[{k:v for k,v in c.items() if k!='mode'} for c in contracts.values()]
    if common[0]!=common[1] or common[0]['baseline_completed_sha256']!=reference_sha:raise ValueError('Unmatched plain controls')
    if common[0]['policy']!=POLICY or common[0]['code']!={p:code_hash(ROOT/p) for p in SOURCES}:raise ValueError('Plain implementation changed')
    initial={m:read(args.root/(m+'_'+args.stage)/'initial.json') for m in MODES}
    summaries={m:read(args.root/(m+'_'+args.stage)/'summary.json') for m in MODES}
    checks=dict(mechanical=all(all(s['checks'].values()) for s in summaries.values()),
        identical_parameters=initial['last_only']['parameter_sha256']==initial['hierarchy']['parameter_sha256'],
        identical_zero_start=initial['last_only']['descriptor_sha256']==initial['hierarchy']['descriptor_sha256'],
        identical_budget=all(s['steps']==POLICY[args.stage+'_steps'] for s in summaries.values()))
    report={};standard={};source=verify(args.source)
    if sha(args.source/'completed.json')!=common[0]['source_completed_sha256']:raise ValueError('Heldout source changed')
    if args.stage=='screen':
        plan=source['plan'];b=read(reference/'gsv_final.json')['baseline']
        values={m:read(args.root/(m+'_screen')/'gsv_final.json') for m in MODES}
        if any(v['baseline']!=b for v in values.values()):raise ValueError('GSV baseline/query ranking mismatch')
        report=paired_report({m:v['student']['outcomes'] for m,v in values.items()},b['outcomes'],plan['query_place_groups'],plan['calibration_places'])
        for name in ('msls-val','pitts30k-val'):
            b=read(reference/(name+'_final.json'))['baseline'];values={m:read(args.root/(m+'_screen')/(name+'_final.json')) for m in MODES}
            if any(v['baseline']!=b for v in values.values()):raise ValueError('Benchmark baseline/query ranking mismatch')
            standard[name]=paired_report({m:v['student']['outcomes'] for m,v in values.items()},b['outcomes'],np.zeros(len(b['outcomes']),int),1)['all']
            standard[name]['recall_at_1_5_20']=dict(baseline=b['recall'],**{m:v['student']['recall'] for m,v in values.items()})
    verdict=('SMOKE_PASS' if args.stage=='smoke' else 'PLAIN_BASE_SCREEN_COMPLETE_NOT_STABLE_GAIN_PROOF') if all(checks.values()) else 'MECHANICAL_FAIL'
    contract=dict(policy=POLICY,stage=args.stage,baseline_completed_sha256=reference_sha,
        completed_hashes={m:sha(args.root/(m+'_'+args.stage)/'completed.json') for m in MODES})
    open_contract(args.output,contract,args.resume)
    if (args.output/'completed.json').exists():verify(args.output)
    else:
        write(args.output/'summary.json',dict(verdict=verdict,checks=checks,partitions=report,standard_benchmark_paired=standard,scope=POLICY['scope']))
        seal(args.output)
    print('COMPARISON',verdict,checks,flush=True)
    if not all(checks.values()):raise SystemExit(2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['baseline','train','compare']);p.add_argument('--mode',choices=MODES)
    p.add_argument('--stage',choices=['smoke','screen'],default='screen')
    p.add_argument('--checkpoint',type=Path,default=Path(PLAIN_CHECKPOINT))
    p.add_argument('--source',type=Path,default=Path('logs/slgd_hard_v2_audit'))
    p.add_argument('--cache',type=Path,default=Path('.cache/slgd_mid_v1_pilot'))
    p.add_argument('--teacher',type=Path,default=Path('logs/slgd_v1/teacher_pilot'))
    p.add_argument('--dataset-root',type=Path,default=Path('datasets/gsv_cities'))
    p.add_argument('--root',type=Path,default=Path('logs/hcen_plain_base_v1'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1' or not torch.cuda.is_available() or torch.cuda.device_count()!=1:p.error('Expose physical GPU1 only')
    if not args.checkpoint.is_file() or sha(args.checkpoint)!=PLAIN_SHA:p.error('Pinned plain-BoQ checkpoint differs')
    if args.command=='train' and args.mode is None:p.error('Training mode required')
    import fcntl
    args.root.mkdir(parents=True,exist_ok=True);lock=(args.root/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.65,0)
    {'baseline':baseline,'train':train,'compare':compare}[args.command](args)


if __name__=='__main__':main()
