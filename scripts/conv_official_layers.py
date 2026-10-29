"""Official-weight native-layer screen; lock calibration choice before evaluation."""
import argparse
from collections import OrderedDict
import os
from pathlib import Path
import shutil
import sys
from types import MethodType,SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.train_slgd import read,write,sha,code_hash,verify,seal,save,restore,parameter_hash
from scripts.slgd_mid_screen import open_contract
from scripts.audit_slgd_hard import PlannedPlaces,collate,save_npz
from scripts.audit_slgd_layers import FinePairs
from scripts.conv_official_boq import contract as original_contract,evaluate
from scripts.conv_official_diagnosis import author,make_adapter,SOURCES as OLD_SOURCES
from scripts.conv_residual_ablation import forward_batch
from scripts.train_hcen_vpr import PlaceBatches,tensor_hash
from scripts.hcen_module_ablation import wait_gpu
from src.models.conv_residual_ablation import PARAMETERS,module_gradient_status
from src.slgd_layer_protocol import LayerCapture,quantized_tokens
from src.models.slgd import mutual_score
from src.slgd_hard_protocol import distance_metres
from src.slgd_mid_supervision import gps_filtered_retrieval
from src.official_transfer_diagnosis import paired_summary
from src.official_layer_selection import LAYERS,POLICY,summarize,choose_layers,selected_features

SOURCES=tuple(dict.fromkeys(('scripts/conv_official_layers.py','src/official_layer_selection.py',
    'scripts/run_conv_official_layers.sh','tests/test_official_layer_selection.py',
    'scripts/audit_slgd_layers.py','src/slgd_layer_protocol.py',*OLD_SOURCES)))


def checked_npz(path):
    if sha(path)!=read(path.with_suffix('.json'))['sha256']:raise ValueError('Cache checksum differs: '+str(path))
    with np.load(path,allow_pickle=False) as value:return {k:value[k] for k in value.files}


def base_contract(args,stage):
    c,data,cache=original_contract(args,stage,42)
    baseline=verify(args.reference/'baseline')
    control=verify(args.diagnosis/'e3_low_rho')
    for old in (baseline,control):
        if old['code']!={p:code_hash(ROOT/p) for p in old['code']}:raise ValueError('Old sealed code changed')
    schedule=control['schedule']
    if len(schedule)!=2048 or control['experiment_settings']!=dict(kind='train',rho=.025,lr=1e-4,negatives='historical'):
        raise ValueError('Not the matched E3 control')
    for key in ('metadata','train_places','benchmark_metadata','checkpoint_sha256','versions'):
        if c[key]!=control[key]:raise ValueError('Control population/precision differs: '+key)
    plan=read(args.source/'contract.json')['plan']
    if set(c['train_places']) & set(plan['query_places']+plan['distractor_places']):raise ValueError('Train/selection overlap')
    c.update(policy=POLICY,stage=stage,schedule=schedule,plan=plan,
        official_baseline_completed_sha256=sha(args.reference/'baseline/completed.json'),
        control_completed_sha256=sha(args.diagnosis/'e3_low_rho/completed.json'),
        code={p:code_hash(ROOT/p) for p in SOURCES})
    return c,data,cache


def globals_and_candidates(args,data,plan,visual):
    from tqdm import tqdm
    path=args.output/'global_shards';path.mkdir(exist_ok=True)
    n=plan['rows'];descriptors=np.empty((n,12288),np.float32);coordinates=np.empty((n,2),np.float64)
    if shutil.disk_usage(args.output).free<n*12288*4*1.2:raise ValueError('Need about0.8GiB free for compact global cache')
    loader=DataLoader(PlannedPlaces(data,plan),batch_size=16,num_workers=args.workers,collate_fn=collate)
    hashes={}
    with torch.inference_mode():
        for batch,(images,indices,coords) in enumerate(tqdm(loader,desc='Official GSV global descriptors')):
            p=path/f'{batch:05d}.npz'
            if p.exists():stored=checked_npz(p)
            else:
                d=torch.cat([visual(x.cuda()) for x in images.split(4)]).cpu().numpy()
                save_npz(p,dict(indices=indices.numpy(),descriptors=d,coordinates=coords.numpy()))
                stored=checked_npz(p)
            ids=indices.numpy();d=stored['descriptors'];gps=stored['coordinates']
            if not np.array_equal(stored['indices'],ids) or d.shape!=(len(ids),12288) or d.dtype!=np.float32 or not np.isfinite(d).all():raise ValueError('Invalid global cache')
            if gps.shape!=(len(ids),2) or not np.array_equal(gps,coords.numpy()) or np.max(np.abs(np.linalg.norm(d,axis=1)-1))>1e-4:raise ValueError('Invalid GPS or descriptor norm')
            descriptors[ids]=d;coordinates[ids]=gps;hashes[p.name]=sha(p)
            write(args.output/'progress.json',dict(phase='global_cache',batch=batch+1,total=len(loader)))
    write(args.output/'global_manifest.json',dict(files=hashes,patch_tokens_stored_on_disk=False))
    candidates=[];scores=[];q=np.asarray(plan['query_rows']);g=np.asarray(plan['gallery_rows'])
    gallery=torch.from_numpy(descriptors[g]).cuda()
    with torch.inference_mode():
        for start in range(0,len(q),16):
            ids=q[start:start+16]
            s=torch.from_numpy(descriptors[ids]).cuda()@gallery.T
            allowed=distance_metres(coordinates[ids],coordinates[g])>=25
            allowed[np.arange(len(ids)),np.asarray(plan['positive_gallery_indices'])[start:start+16]]=True
            s.masked_fill_(~torch.from_numpy(allowed).cuda(),-torch.inf)
            v,i=s.topk(POLICY['candidate_k'],dim=1)
            if not bool(torch.isfinite(v).all()):raise ValueError('Insufficient GPS-safe candidates')
            candidates.extend(i.cpu().numpy());scores.extend(v.cpu().numpy())
    del descriptors,gallery
    candidates=np.asarray(candidates,np.int64)
    save_npz(args.output/'candidate_pack.npz',dict(candidates=candidates,global_scores=np.asarray(scores,np.float32)))
    return candidates


def screen(args):
    from tqdm import tqdm
    c,data,_=base_contract(args,'official_layer_screen')
    open_contract(args.output,c,args.resume)
    if (args.output/'completed.json').exists():verify(args.output);return
    visual=author(args);frozen=tensor_hash(visual);plan=c['plan']
    candidates=globals_and_candidates(args,data,plan,visual)
    positive=np.asarray(plan['positive_gallery_indices']);groups=np.asarray(plan['query_place_groups'])
    all_scores=np.empty((len(groups),len(LAYERS),20),np.float32);all_valid=np.empty(all_scores.shape,bool)
    image_rows=FinePairs(data,plan,candidates,[])
    qrows=np.asarray(plan['query_rows']);grows=np.asarray(plan['gallery_rows'])
    scores_path=args.output/'pair_scores';scores_path.mkdir(exist_ok=True)
    tokens=OrderedDict();capture=LayerCapture(visual.backbone,LAYERS)
    file_hashes={}
    try:
        for role,ids in [('calibration',np.flatnonzero(groups<plan['calibration_places'])),
                         ('evaluation',np.flatnonzero(groups>=plan['calibration_places']))]:
            for number,q in enumerate(tqdm(ids,desc='Official native layer '+role)):
                p=scores_path/f'{q:05d}.npz'
                if p.exists():stored=checked_npz(p)
                else:
                    needed=[int(qrows[q]),*map(int,grows[candidates[q]])]
                    for row in needed:
                        if row in tokens:tokens.move_to_end(row)
                    missing=[row for row in needed if row not in tokens]
                    with torch.inference_mode():
                        for start in range(0,len(missing),4):
                            rows=missing[start:start+4]
                            images=torch.stack([image_rows.row_image(row) for row in rows]).cuda()
                            features,_=capture(images)
                            packed=np.stack([quantized_tokens(features[layer],20) for layer in LAYERS],axis=1)
                            for row,value in zip(rows,packed):tokens[row]=value
                    # No eviction before the current21 images have been scored.
                    values=np.empty((len(LAYERS),20),np.float32);valid=np.empty(values.shape,bool)
                    with torch.inference_mode():
                        for i in range(len(LAYERS)):
                            left=torch.nn.functional.normalize(torch.from_numpy(tokens[needed[0]][i].astype(np.float32)).cuda(),dim=-1)
                            for start in range(0,20,4):
                                right=np.stack([tokens[row][i] for row in needed[1+start:1+start+4]])
                                right=torch.nn.functional.normalize(torch.from_numpy(right.astype(np.float32)).cuda(),dim=-1)
                                v,count=mutual_score(left[None].expand(len(right),-1,-1),right)
                                values[i,start:start+len(right)]=v.cpu().numpy()
                                valid[i,start:start+len(right)]=(count>=POLICY['min_matches']).cpu().numpy()
                    while len(tokens)>POLICY['token_lru_images']:tokens.popitem(last=False)
                    save_npz(p,dict(query=np.asarray([q],np.int64),scores=values,valid=valid))
                    stored=checked_npz(p)
                if stored['query'].tolist()!=[int(q)] or stored['scores'].shape!=(len(LAYERS),20) or stored['valid'].dtype!=bool or not np.isfinite(stored['scores']).all():raise ValueError('Invalid layer scores')
                all_scores[q]=stored['scores'];all_valid[q]=stored['valid'];file_hashes[p.name]=sha(p)
                if number%16==0:write(args.output/'progress.json',dict(phase='layer_matching',partition=role,queries=number+1,total=len(ids)))
            result=summarize(candidates[ids],positive[ids],groups[ids],all_scores[ids],all_valid[ids],ids)
            write(args.output/(role+'.json'),result)
            if role=='calibration':
                selection=dict(**choose_layers(result),calibration_sha256=sha(args.output/'calibration.json'),
                    candidate_pack_sha256=sha(args.output/'candidate_pack.npz'))
                if (args.output/'selection.json').exists() and read(args.output/'selection.json')!=selection:raise ValueError('Locked selection changed')
                write(args.output/'selection.json',selection)
                print('LOCKED BEFORE EVALUATION',selection,flush=True)
    finally:capture.close()
    if tensor_hash(visual)!=frozen:raise ValueError('Official weights changed during screening')
    write(args.output/'pair_manifest.json',dict(files=file_hashes))
    selection=read(args.output/'selection.json')
    write(args.output/'summary.json',dict(selection=selection,calibration=read(args.output/'calibration.json'),
        evaluation=read(args.output/'evaluation.json'),native_tokens=400,
        patch_tokens_stored_on_disk=False,global_cache_estimated_gib=plan['rows']*12288*4/1024**3,
        frozen_unchanged=True,scope=POLICY['scope']))
    write(args.output/'progress.json',dict(phase='complete',selection=selection));seal(args.output)


def training(args):
    from tqdm import tqdm
    from src.losses.vpr_losses import VPRLossFunction
    screen_c=verify(args.root/'screen');selection=read(args.root/'screen/selection.json')
    if not selection['gate']:raise ValueError('No calibration-supported alternative; do not force training')
    c,data,cache=base_contract(args,'official_selected_layer_training')
    layers=selection['layers'] if args.arm=='selected' else [12,12,12]
    c.update(arm=args.arm,fusion_layers=layers,screen_completed_sha256=sha(args.root/'screen/completed.json'))
    if screen_c['policy']!=POLICY:raise ValueError('Changed layer screen')
    open_contract(args.output,c,args.resume)
    if (args.output/'completed.json').exists():verify(args.output);return
    visual=author(args);frozen=tensor_hash(visual);model=make_adapter(visual,POLICY['rho'])
    model.fusion_layers=tuple(layers);model.features=MethodType(selected_features,model)
    initial=parameter_hash(model)
    if initial!=read(args.diagnosis/'e3_low_rho/initial.json')['initial_parameter_sha256']:raise ValueError('Matched initialization differs')
    batches=PlaceBatches(data,cache['train_indices'],c['schedule'])
    images,_,_=batches[0]
    with torch.inference_mode():
        d,r,_,_,_=forward_batch(model,images[:4]);direct=visual(images[:4].cuda())
        zero=max(float((d-direct).abs().max()),float((r-direct).abs().max()))
    if zero>1e-6:raise ValueError('Layer routing breaks official zero-start')
    active={n:p for n,p in model.named_parameters() if p.requires_grad}
    if sum(p.numel() for p in active.values())!=PARAMETERS or any(not n.startswith('neck.') for n in active):raise ValueError('Changed model capacity')
    optimizer=torch.optim.AdamW(active.values(),lr=POLICY['lr'],weight_decay=0.)
    state=dict(steps=0,rows=[],module_gradients={'0':False,'1':False},up_gradient_seen=False,
        initial_auxiliary_sha256=tensor_hash(model.neck.aux),contract_sha256=sha(args.output/'contract.json'))
    if (args.output/'last.pt').exists():state=restore(args.output/'last.pt',model,optimizer,state['contract_sha256'])
    if not (args.output/'initial.json').exists():
        write(args.output/'initial.json',dict(initial_parameter_sha256=initial,zero_start_error=zero,fusion_layers=layers))
        save(args.output/'last.pt',model,optimizer,state)
    model.train();lossfn=VPRLossFunction()
    loader=DataLoader(PlaceBatches(data,cache['train_indices'],c['schedule'][state['steps']:]),batch_size=None,num_workers=args.workers)
    for images,labels,coords in tqdm(loader,total=2048,initial=state['steps'],desc='Official layers '+args.arm):
        optimizer.zero_grad(set_to_none=True)
        d,r,aux,delta,_=forward_batch(model,images,True);labels=labels.cuda();gps=coords.numpy()
        primary=gps_filtered_retrieval(lossfn,d,labels,gps)
        auxiliary=sum(gps_filtered_retrieval(lossfn,x,labels,gps) for x in aux)/len(aux)
        loss=primary+0.*auxiliary
        if not bool(torch.isfinite(loss)) or float(delta.max())>POLICY['rho']+1e-6:raise ValueError('Nonfinite loss/cap violation')
        loss.backward()
        for k,v in module_gradient_status(model).items():state['module_gradients'][k]|=v
        state['up_gradient_seen']|=bool(model.neck.up.weight.grad is not None and (model.neck.up.weight.grad!=0).any())
        for p in active.values():
            if p.grad is None:p.grad=torch.zeros_like(p)
        norm=torch.nn.utils.clip_grad_norm_(list(active.values()),1.,error_if_nonfinite=True)
        if any(p.grad is not None for p in visual.parameters()):raise ValueError('Frozen official gradients')
        optimizer.step();state['steps']+=1
        state['rows'].append(dict(step=state['steps'],loss=float(primary.detach()),gradient_norm=float(norm),relative_residual_mean=float(delta.mean())))
        write(args.output/'progress.json',dict(phase='training',steps=state['steps'],total=2048,layers=layers))
        if state['steps']%16==0:save(args.output/'last.pt',model,optimizer,state)
    save(args.output/'last.pt',model,optimizer,state);final=parameter_hash(model)
    restore(args.output/'last.pt',model,optimizer,state['contract_sha256'])
    checks=dict(zero_start=zero<=1e-6,frozen_unchanged=tensor_hash(visual)==frozen,
        module_gradients=all(state['module_gradients'].values()),up_gradient=state['up_gradient_seen'],
        complete_steps=state['steps']==2048,checkpoint_roundtrip=parameter_hash(model)==final,
        auxiliary_unchanged=tensor_hash(model.neck.aux)==state['initial_auxiliary_sha256'],
        optimizer_clocks=all(int(optimizer.state[p]['step'])==2048 for p in active.values()))
    if not all(checks.values()):raise ValueError('Layer control mechanics failed: '+str(checks))
    optimizer.zero_grad(set_to_none=True);model.eval()
    evaluate(SimpleNamespace(**{**vars(args),'root':args.reference}),model,args.output,True)
    write(args.output/'summary.json',dict(checks=checks,initial_parameter_sha256=initial,final_parameter_sha256=final,
        fusion_layers=layers,rows=state['rows'],scope=POLICY['scope']))
    write(args.output/'progress.json',dict(phase='complete',steps=2048));seal(args.output)


def aggregate(args):
    c=verify(args.root/'screen');selection=read(args.root/'screen/selection.json')
    if c['policy']!=POLICY or c['code']!={p:code_hash(ROOT/p) for p in SOURCES}:raise ValueError('Changed screen implementation')
    result={};completed={}
    if selection['gate']:
        for arm in ('selected','last_only'):
            path=args.root/arm;contract=verify(path);summary=read(path/'summary.json')
            if not all(summary['checks'].values()) or contract['code']!=c['code'] or contract['policy']!=POLICY:raise ValueError('Unverified training')
            completed[arm]=sha(path/'completed.json');result[arm]={}
            for size in (280,322):
                for ds in ('msls-val','pitts30k-val'):
                    tag=f'{ds}_{size}';row=read(path/(tag+'.json'));control=read(args.diagnosis/'e3_low_rho'/(tag+'.json'))
                    result[arm][tag]=dict(correct=row['correct'],queries=row['queries'],recall=row['recall'],
                        vs_official=row['paired'],vs_81012=paired_summary(control['outcomes'],row['outcomes']))
        for size in (280,322):
            for ds in ('msls-val','pitts30k-val'):
                tag=f'{ds}_{size}'
                result['selected'][tag]['vs_last_only']=paired_summary(read(args.root/'last_only'/(tag+'.json'))['outcomes'],read(args.root/'selected'/(tag+'.json'))['outcomes'])
    contract=dict(policy=POLICY,screen_completed_sha256=sha(args.root/'screen/completed.json'),training_completed=completed,code=c['code'])
    open_contract(args.output,contract,args.resume)
    if (args.output/'completed.json').exists():verify(args.output);return
    write(args.output/'summary.json',dict(verdict='LAYER_SCREEN_AND_MATCHED_CONTROLS_COMPLETE' if selection['gate'] else 'SCREEN_COMPLETE_NO_SUPPORTED_ALTERNATIVE',
        selection=selection,results=result,new_training_runs=len(completed),additional_runs_authorized=0,
        limitations='MNN candidate-rank screening is not the learned conv network; no guarantee selected layers improve; GSV partitions and benchmarks previously exposed; two new runs seed42 only',scope=POLICY['scope']))
    seal(args.output);write(args.root/'progress.json',dict(phase='complete',training_runs=len(completed)))


def main():
    os.chdir(ROOT);p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('wait_gpu','screen','gate','train','aggregate'))
    p.add_argument('--arm',choices=('selected','last_only'));p.add_argument('--resume',action='store_true')
    p.add_argument('--root',type=Path,default=Path('logs/conv_official_layers_v1'));p.add_argument('--output',type=Path)
    p.add_argument('--reference',type=Path,default=Path('logs/conv_official_boq_v1'))
    p.add_argument('--diagnosis',type=Path,default=Path('logs/conv_official_diagnosis_v1'))
    p.add_argument('--official-cache',type=Path,default=Path('.cache/boq_official_v1'))
    p.add_argument('--dino-source',type=Path,default=Path('/home/wt/.cache/torch/hub/facebookresearch_dinov2_main'))
    p.add_argument('--dataset-root',type=Path,default=Path('datasets/gsv_cities'))
    p.add_argument('--source',type=Path,default=Path('logs/slgd_hard_v2_audit'))
    p.add_argument('--teacher',type=Path,default=Path('logs/slgd_v1/teacher_pilot'))
    p.add_argument('--cache',type=Path,default=Path('.cache/slgd_mid_v1_pilot'));p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':p.error('Expose physical GPU1 only')
    if args.root.resolve().parent!=(ROOT/'logs').resolve() or not args.root.name.startswith('conv_official_layers_'):p.error('Use isolated layer root')
    if args.workers<0:p.error('Workers must be nonnegative')
    if args.command=='wait_gpu':wait_gpu(args);return
    if args.command=='gate':
        verify(args.root/'screen');selection=read(args.root/'screen/selection.json');print(selection,flush=True)
        raise SystemExit(0 if selection['gate'] else 2)
    expected=args.arm if args.command=='train' else args.command
    if not expected or args.output is None or args.output.resolve().parent!=args.root.resolve() or args.output.name!=expected:p.error('Output must match stage directly under new root')
    import fcntl
    args.root.mkdir(exist_ok=True)
    with (args.root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        if args.command!='aggregate':torch.cuda.set_per_process_memory_fraction(.65,0)
        write(args.root/'progress.json',dict(phase='running',stage=args.command,arm=args.arm))
        try:{'screen':screen,'train':training,'aggregate':aggregate}[args.command](args)
        except Exception as error:
            failure=dict(phase='failed',stage=args.command,type=type(error).__name__,message=str(error))
            write(args.root/'progress.json',failure)
            if args.output.exists() and not (args.output/'completed.json').exists():write(args.output/'failure.json',failure)
            raise


if __name__=='__main__':main()
