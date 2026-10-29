"""One closed five-item diagnosis of local-positive/author-negative transfer."""
import argparse
import copy
import datetime
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from scripts.train_slgd import read,write,sha,code_hash,verify,seal,seed,save,restore,parameter_hash,MatchedPlaces
from scripts.slgd_mid_screen import open_contract
from scripts.train_hcen_vpr import PlaceBatches,tensor_hash
from scripts.conv_official_boq import contract as original_contract,benchmark_datasets,evaluate,SOURCES as OLD_SOURCES
from scripts.conv_residual_ablation import forward_batch
from scripts.hcen_module_ablation import wait_gpu
from src.models.official_boq import load_official,build_adapter,OfficialBoQ,native_feature_layout
from src.models.conv_residual_ablation import PARAMETERS,module_gradient_status
from src.hcen_plain_protocol import PLAIN_SHA,PLAIN_CHECKPOINT,validate_checkpoint
from src.slgd_layer_protocol import LayerCapture
from src.slgd_mid_supervision import gps_filtered_retrieval
from src.official_transfer_diagnosis import POLICY,EXPERIMENTS,ScaledRMSNeck,choose_negative_batch,paired_summary,gradient_comparison

SOURCES=tuple(dict.fromkeys(('scripts/conv_official_diagnosis.py','src/official_transfer_diagnosis.py',
    'scripts/run_conv_official_diagnosis.sh','tests/test_official_transfer_diagnosis.py',*OLD_SOURCES)))


def reference(args):
    result={}
    for name,path in [('official_baseline',args.reference/'baseline'),
                      ('official_seed42',args.reference/'seed42/train'),
                      ('local_seed42',args.local_reference/'seed42/rms_tanh_screen')]:
        verify(path); result[name]=sha(path/'completed.json')
    return result


def make_contract(args, item):
    c,data,cache=original_contract(args,'diagnosis',42)
    c.update(policy=POLICY,experiment=item,experiment_settings=EXPERIMENTS.get(item),
        prior_results=reference(args),code={p:code_hash(ROOT/p) for p in SOURCES})
    c['schedule']=read(args.reference/'seed42/train/contract.json')['schedule']
    if len(c['schedule'])!=2048 or c['schedule'][:512]*4!=c['schedule']:
        raise ValueError('Unexpected original training schedule')
    return c,data,cache


def author(args):
    seed(42)
    visual=load_official(args.official_cache,args.dino_source,'cuda')
    if tensor_hash(visual)!=read(args.reference/'baseline/summary.json')['frozen_tensor_sha256']:
        raise ValueError('Official baseline changed')
    return visual


def make_adapter(visual,rho):
    seed(42);model=build_adapter(visual,42).cuda().eval()
    model.neck=ScaledRMSNeck(model.neck,rho)
    return model


def load_parameters(model,path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    active={n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(active)!=set(state['parameters']):raise ValueError('Adapter parameter identity differs')
    with torch.no_grad():
        for name,p in active.items():p.copy_(state['parameters'][name].to(p.device))
    return state['state']


def make_local_wrapper(official):
    from src.models.backbones.dinov2 import DinoV2
    from src.models.aggregators.boq import BoQ
    with patch('torch.hub.load',return_value=copy.deepcopy(official.backbone.dino)):
        backbone=DinoV2(num_unfrozen_blocks=2)
    value=OfficialBoQ(backbone,BoQ(768,384,64,2,32)).cuda()
    value.load_state_dict(official.state_dict(),strict=True)
    return value.requires_grad_(False).eval()


def gradient_probe(aggregator,features,native):
    x=features.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        y=aggregator(native_feature_layout(x) if native else x.contiguous())[0]
        objective=(y[:,:16]*torch.arange(1,17,device=y.device)).sum()
        gradient=torch.autograd.grad(objective,x)[0]
    return y.detach(),gradient.detach()


def weights_comparison(official_state,local_state):
    if set(official_state)!=set(local_state):raise ValueError('Base checkpoint keys differ')
    groups={}
    for name,a in official_state.items():
        b=local_state[name]
        if a.shape!=b.shape:raise ValueError('Base shape differs: '+name)
        group='aggregator' if name.startswith('aggregator.') else 'backbone_other'
        if '.blocks.' in name:
            index=int(name.split('.blocks.')[1].split('.')[0])
            group='backbone_first10' if index<10 else 'backbone_last2'
        row=groups.setdefault(group,dict(tensors=0,changed_tensors=0,elements=0,square_difference=0.,
            official_square_norm=0.,local_square_norm=0.,dot_product=0.))
        a,b=a.detach().cpu().double(),b.detach().cpu().double()
        row['tensors']+=1;row['changed_tensors']+=int(not torch.equal(a,b));row['elements']+=a.numel()
        row['square_difference']+=float((a-b).square().sum())
        row['official_square_norm']+=float(a.square().sum());row['local_square_norm']+=float(b.square().sum())
        row['dot_product']+=float((a*b).sum())
    for row in groups.values():
        row['relative_l2_vs_official']=(row['square_difference']/row['official_square_norm'])**.5 if row['official_square_norm'] else None
        denominator=(row['official_square_norm']*row['local_square_norm'])**.5
        row['parameter_cosine']=row['dot_product']/denominator if denominator else None
    return groups


def audit(args):
    c,data,cache=make_contract(args,'e1_implementation')
    open_contract(args.output,c,args.resume)
    if (args.output/'completed.json').exists():verify(args.output);return
    o=author(args);local=make_local_wrapper(o)
    probe_results=[];probe_images={}
    # Preserve the original cross-layout FP32 observations, but distinguish
    # implementation differences from layout-dependent CUDA roundoff. No
    # training implementation or original experiment checkpoint is changed.
    double_official=copy.deepcopy(o.aggregator).double()
    double_local=copy.deepcopy(local.aggregator).double()
    for size in (280,322):
        ds=benchmark_datasets(size)[0]
        ids=[int(i*ds.num_references//8) for i in range(8)]+[
            ds.num_references+int(i*ds.num_queries//8) for i in range(8)]
        pixels=torch.stack([ds[i][0] for i in ids]);probe_images[size]=pixels
        for start in range(0,len(pixels),4):
            images=pixels[start:start+4].cuda()
            with torch.inference_mode():
                a,b=o(images),local(images)
                raw_o,raw_l=o.backbone(images),local.backbone(images)
            # Perturb the same tensor; test the nonzero residual input path too.
            feature=raw_o.detach().clone()+.025*raw_o.square().mean(1,keepdim=True).sqrt()*torch.sin(raw_o)
            y1,g1=gradient_probe(o.aggregator,feature,True)
            y2,g2=gradient_probe(local.aggregator,feature,False)
            matched_y,matched_g=gradient_probe(local.aggregator,feature,True)
            double_y1,double_g1=gradient_probe(double_official,feature.double(),True)
            double_y2,double_g2=gradient_probe(double_local,feature.double(),False)
            relative=float((g1-g2).norm()/g1.norm().clamp_min(1e-12))
            row=dict(size=size,indices=ids[start:start+4],raw_max_error=float((raw_o-raw_l).abs().max()),
                direct_descriptor_max_error=float((a-b).abs().max()),
                nonzero_input_descriptor_max_error=float((y1-y2).abs().max()),
                gradient_relative_l2=relative,official_gradient_norm=float(g1.norm()),
                local_gradient_norm=float(g2.norm()),
                matched_layout_descriptor_max_error=float((y1-matched_y).abs().max()),
                matched_layout_fp32_gradient=gradient_comparison(g1,matched_g,1e-4),
                cross_layout_fp64_descriptor_max_error=float((double_y1-double_y2).abs().max()),
                cross_layout_fp64_gradient=gradient_comparison(double_g1,double_g2,1e-10))
            probe_results.append(row)
    del double_official,double_local
    checks=dict(same_weights_raw=all(r['raw_max_error']==0 for r in probe_results),
        same_weights_forward=all(r['direct_descriptor_max_error']<=2e-6 and r['nonzero_input_descriptor_max_error']<=2e-6 for r in probe_results),
        same_weights_matched_layout_gradient=all(r['matched_layout_descriptor_max_error']<=2e-6 and r['matched_layout_fp32_gradient']['passed'] for r in probe_results),
        same_weights_cross_layout_fp64_gradient=all(r['cross_layout_fp64_descriptor_max_error']<=1e-10 and r['cross_layout_fp64_gradient']['passed'] for r in probe_results))
    ck=torch.load(args.local_checkpoint,map_location='cpu',weights_only=False)
    from scripts.eval_condition_robustness import strip_compiled_model_prefix
    state=strip_compiled_model_prefix(ck['state_dict']);validate_checkpoint(ck['hyper_parameters'],state)
    if sha(args.local_checkpoint)!=PLAIN_SHA:raise ValueError('Local reference checkpoint differs')
    weight_stats=weights_comparison(o.state_dict(),state)
    local.load_state_dict(state,strict=True);del ck,state
    layer_results=[]
    for size,pixels in probe_images.items():
        for label,visual in [('official',o),('local',local)]:
            capture=LayerCapture(visual.backbone,(8,10,12));values={k:[] for k in (8,10,12)}
            try:
                with torch.inference_mode():
                    for chunk in pixels.cuda().split(4):
                        features,_=capture(chunk)
                        for k,f in features.items():values[k].extend(f.square().mean(1).sqrt().flatten(1).mean(1).cpu().tolist())
            finally:capture.close()
            layer_results.append(dict(model=label,size=size,images=len(pixels),
                rms_by_layer={str(k):float(np.mean(v)) for k,v in values.items()}))
    original_batches=PlaceBatches(data,cache['train_indices'],c['schedule'])
    from src.losses.vpr_losses import VPRLossFunction
    lossfn=VPRLossFunction();loss_results=[]
    for index in (0,64,128,192,256,320,384,448):
        images,labels,coords=original_batches[index]
        for label,visual in [('official',o),('local',local)]:
            with torch.inference_mode():
                d=torch.cat([visual(x.cuda()) for x in images.split(4)])
                loss=gps_filtered_retrieval(lossfn,d,labels.cuda(),coords.numpy())
            loss_results.append(dict(batch=index,model=label,images=len(images),loss=float(loss)))
    write(args.output/'summary.json',dict(checks=checks,same_weights_probes=probe_results,
        weight_difference=weight_stats,feature_scale=layer_results,matched_training_losses=loss_results,
        numerical_localization=dict(original_cross_layout_fp32_within_1e4=all(r['gradient_relative_l2']<=1e-4 for r in probe_results),
            note='FP32 cross-layout differences remain recorded, not rounded away. Identical layout FP32 and cross-layout FP64 must separately pass; training keeps the existing author-native layout.'),
        note='Identical architecture is checked with identical weights; parameter/feature differences do not alone identify the cause of recall change'))
    print('E1 AUDIT',checks,flush=True)
    if not all(checks.values()):raise ValueError('Implementation audit failed; do not attribute this to model quality')
    write(args.output/'progress.json',dict(phase='complete'))
    seal(args.output)


def evaluate_new(args,model):
    # Read ONLY the old official baseline; write results in the new output.
    evaluation_args=SimpleNamespace(**{**vars(args),'root':args.reference})
    evaluate(evaluation_args,model,args.output,True)


def attenuate(args):
    c,_,_=make_contract(args,'e2_attenuate');open_contract(args.output,c,args.resume)
    if (args.output/'completed.json').exists():verify(args.output);return
    model=make_adapter(author(args),.025)
    load_parameters(model,args.reference/'seed42/train/last.pt')
    expected=read(args.reference/'seed42/train/summary.json')['final_parameter_sha256']
    if parameter_hash(model)!=expected:raise ValueError('Attenuation changed trained weights')
    evaluate_new(args,model)
    write(args.output/'summary.json',dict(experiment='e2_attenuate',inference_scale=.25,
        trained_parameters_unchanged=True,new_training_updates=0,scope=POLICY['scope']))
    write(args.output/'progress.json',dict(phase='complete'));seal(args.output)


def official_mining(args,c,data,cache):
    p=args.root/'official_training_mining'
    mine_c=copy.deepcopy(c);mine_c.update(experiment='official_training_mining',experiment_settings=None)
    open_contract(p,mine_c,p.exists())
    if (p/'completed.json').exists():verify(p);return read(p/'schedule.json')['rows']
    from tqdm import tqdm
    visual=author(args);frozen=tensor_hash(visual)
    count=len(cache['train_indices']);descriptors=np.empty((count,12288),np.float32);coords=np.empty((count,2),np.float64)
    loader=DataLoader(MatchedPlaces(data,cache['train_indices'],99),batch_size=8,num_workers=args.workers)
    with torch.inference_mode():
        for batch,(images,_,metadata) in enumerate(tqdm(loader,desc='E5 official train-only negative descriptors')):
            b,k=images.shape[:2]
            d=torch.cat([visual(x.cuda()) for x in images.flatten(0,1).split(4)]).reshape(b,k,-1)
            mean=F.normalize(d.mean(1),dim=1)
            start=batch*8;descriptors[start:start+b]=mean.cpu().numpy()
            coords[start:start+b]=metadata['coordinates'][:,0].numpy()
            write(p/'progress.json',dict(phase='mining_official_train_only',batch=batch+1,total=len(loader)))
    if not np.isfinite(descriptors).all() or not np.isfinite(coords).all() or tensor_hash(visual)!=frozen:
        raise ValueError('Invalid official mining extraction')
    all_=torch.from_numpy(descriptors).cuda();schedule=[];records=[]
    with torch.inference_mode():
        for index,batch in enumerate(tqdm(c['schedule'][:512],desc='E5 official negatives same anchors')):
            anchors=batch[:8]
            scores=(all_[anchors]@all_.T).cpu().numpy()
            replacement,selections=choose_negative_batch(anchors,scores,coords,len(batch))
            schedule.append(replacement);records.append(dict(batch=index,original=batch,replacement=replacement,selections=selections))
    schedule=schedule*4
    if len(schedule)!=2048 or any(a[:8]!=b[:8] or len(a)!=len(b) for a,b in zip(schedule,c['schedule'])):
        raise ValueError('Official mining altered anchors or training image budget')
    temporary=p/'place_descriptors.tmp'
    with temporary.open('wb') as stream:np.save(stream,descriptors,allow_pickle=False)
    temporary.replace(p/'place_descriptors.npy')
    write(p/'schedule.json',dict(rows=schedule))
    write(p/'summary.json',dict(train_places=count,train_images=count*4,
        original_schedule_sha256=sha(args.reference/'seed42/train/contract.json'),
        changed_batches=sum(a!=b for a,b in zip(schedule,c['schedule'])),
        same_anchors=True,same_batch_sizes=True,benchmark_images_used=0,records=records))
    write(p/'progress.json',dict(phase='complete'));seal(p)
    del all_,visual
    return schedule


def training(args):
    from tqdm import tqdm
    from src.losses.vpr_losses import VPRLossFunction
    settings=EXPERIMENTS[args.experiment]
    c,data,cache=make_contract(args,args.experiment)
    if settings['negatives']=='official':
        c['schedule']=official_mining(args,c,data,cache)
        c['official_mining_completed_sha256']=sha(args.root/'official_training_mining/completed.json')
    open_contract(args.output,c,args.resume)
    if (args.output/'completed.json').exists():verify(args.output);return
    visual=author(args);frozen=tensor_hash(visual);model=make_adapter(visual,settings['rho'])
    initial=parameter_hash(model)
    if initial!=read(args.reference/'seed42/train/initial.json')['initial_parameter_sha256']:
        raise ValueError('New train initialization differs from original seed42')
    zero_errors={}
    with torch.inference_mode():
        for size in (280,322):
            ds=benchmark_datasets(size)[0];images=torch.stack([ds[i][0] for i in range(4)])
            d,ref,_,_,_=forward_batch(model,images);direct=visual(images.cuda())
            error=max(float((d-direct).abs().max()),float((ref-direct).abs().max()))
            if error>1e-6:raise ValueError('New control zero-start mismatch')
            zero_errors[str(size)]=error
    active={n:p for n,p in model.named_parameters() if p.requires_grad}
    if sum(p.numel() for p in active.values())!=PARAMETERS or any(not n.startswith('neck.') for n in active):
        raise ValueError('Unexpected trainable parameters')
    optimizer=torch.optim.AdamW(active.values(),lr=settings['lr'],weight_decay=0.)
    state=dict(steps=0,rows=[],module_gradients={'0':False,'1':False},up_gradient_seen=False,
        initial_parameter_sha256=initial,initial_auxiliary_sha256=tensor_hash(model.neck.aux),
        contract_sha256=sha(args.output/'contract.json'))
    if (args.output/'last.pt').exists():state=restore(args.output/'last.pt',model,optimizer,state['contract_sha256'])
    if not (args.output/'initial.json').exists():
        write(args.output/'initial.json',dict(initial_parameter_sha256=initial,zero_errors=zero_errors,
            frozen_tensor_sha256=frozen,settings=settings));save(args.output/'last.pt',model,optimizer,state)
    lossfn=VPRLossFunction();model.train()
    loader=DataLoader(PlaceBatches(data,cache['train_indices'],c['schedule'][state['steps']:]),
        batch_size=None,num_workers=args.workers)
    for images,labels,coords in tqdm(loader,total=2048,initial=state['steps'],desc=args.experiment):
        optimizer.zero_grad(set_to_none=True)
        d,ref,aux,delta,_=forward_batch(model,images,True)
        labels=labels.cuda();coordinates=coords.numpy()
        primary=gps_filtered_retrieval(lossfn,d,labels,coordinates)
        auxiliary=sum(gps_filtered_retrieval(lossfn,x,labels,coordinates) for x in aux)/len(aux)
        loss=primary+0.*auxiliary
        with torch.no_grad():frozen_loss=gps_filtered_retrieval(lossfn,ref,labels,coordinates)
        if not bool(torch.isfinite(loss)) or float(delta.max())>settings['rho']+1e-6:
            raise ValueError('Nonfinite loss or control amplitude violation')
        loss.backward()
        for k,v in module_gradient_status(model).items():state['module_gradients'][k]|=v
        state['up_gradient_seen']|=bool(model.neck.up.weight.grad is not None and (model.neck.up.weight.grad!=0).any())
        for p in active.values():
            if p.grad is None:p.grad=torch.zeros_like(p)
        norm=torch.nn.utils.clip_grad_norm_(list(active.values()),1.,error_if_nonfinite=True)
        if any(p.grad is not None for p in visual.parameters()):raise ValueError('Frozen official gradients')
        optimizer.step();state['steps']+=1
        state['rows'].append(dict(step=state['steps'],loss=float(primary.detach()),frozen_loss=float(frozen_loss),
            gradient_norm=float(norm),relative_residual_mean=float(delta.mean()),relative_residual_max=float(delta.max())))
        write(args.output/'progress.json',dict(phase='training',experiment=args.experiment,steps=state['steps'],total=2048,loss=float(primary.detach())))
        if state['steps']%16==0:save(args.output/'last.pt',model,optimizer,state)
    save(args.output/'last.pt',model,optimizer,state);final=parameter_hash(model)
    restore(args.output/'last.pt',model,optimizer,state['contract_sha256'])
    checks=dict(zero_start=all(x<=1e-6 for x in zero_errors.values()),frozen_unchanged=tensor_hash(visual)==frozen,
        module_gradients=all(state['module_gradients'].values()),up_gradient=state['up_gradient_seen'],
        full_budget=state['steps']==2048,checkpoint_roundtrip=parameter_hash(model)==final,
        auxiliary_unchanged=tensor_hash(model.neck.aux)==state['initial_auxiliary_sha256'],
        optimizer_clocks=all(int(optimizer.state[p]['step'])==2048 for p in active.values()))
    if not all(checks.values()):raise ValueError('Control mechanics failed: '+str(checks))
    optimizer.zero_grad(set_to_none=True);model.eval();evaluate_new(args,model)
    write(args.output/'summary.json',dict(experiment=args.experiment,checks=checks,settings=settings,
        steps=2048,rows=state['rows'],final_parameter_sha256=final,scope=POLICY['scope']))
    write(args.output/'progress.json',dict(phase='complete',steps=2048));seal(args.output)


def aggregate(args):
    reports={};hashes={}
    for name in EXPERIMENTS:
        path=args.root/name;c=verify(path)
        if c['policy']!=POLICY or c['experiment']!=name or c['code']!={p:code_hash(ROOT/p) for p in SOURCES}:
            raise ValueError('Wrong or changed diagnosis contract')
        hashes[name]=sha(path/'completed.json');s=read(path/'summary.json')
        if name=='e1_implementation':
            if not all(s['checks'].values()):raise ValueError('Implementation audit failed')
            reports[name]=s;continue
        if name in ('e3_low_rho','e4_low_lr','e5_native_negatives') and not all(s['checks'].values()):
            raise ValueError('Unverified new training')
        reports[name]={}
        for size in (280,322):
            for dataset in ('msls-val','pitts30k-val'):
                tag=f'{dataset}_{size}';row=read(path/(tag+'.json'))
                previous=read(args.reference/'seed42/train'/(tag+'.json'))
                reports[name][tag]=dict(correct=row['correct'],queries=row['queries'],recall=row['recall'],
                    paired_vs_official=row['paired'],paired_vs_original_adapter=paired_summary(previous['outcomes'],row['outcomes']))
    c=dict(policy=POLICY,results=hashes,code={p:code_hash(ROOT/p) for p in SOURCES})
    open_contract(args.output,c,args.resume)
    if (args.output/'completed.json').exists():verify(args.output);return
    direction={name:{tag:row['paired_vs_official']['net'] for tag,row in report.items() if tag.endswith('_322')}
        for name,report in reports.items() if name!='e1_implementation'}
    write(args.output/'summary.json',dict(verdict='FIVE_ITEM_DIAGNOSIS_COMPLETE',reports=reports,
        primary_net_vs_official=direction,additional_experiments_authorized=0,scope=POLICY['scope'],
        interpretations=dict(e2='Attenuation relief tests excessive learned intervention, not a new trained method',
            e3='Compare e3 vs original adapter and vs e2 to separate training cap from post-hoc scaling',
            e4='A lower rate tests optimization-path sensitivity; fixed-budget undertraining remains possible',
            e5='Only training negative identities changed; effects can reflect hardness and batch-negative diversity',
            limit='Three new controls each seed42 only; mixed or tiny outcomes cannot conclusively attribute all causes')))
    seal(args.output);write(args.root/'progress.json',dict(phase='complete',experiment_count=5))
    print('ALL FIVE ITEMS COMPLETE; no automatic additional experiment',direction,flush=True)


def main():
    os.chdir(ROOT);p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['wait_gpu','run','aggregate'])
    p.add_argument('--experiment',choices=list(EXPERIMENTS));p.add_argument('--resume',action='store_true')
    p.add_argument('--root',type=Path,default=Path('logs/conv_official_diagnosis_v1'));p.add_argument('--output',type=Path)
    p.add_argument('--reference',type=Path,default=Path('logs/conv_official_boq_v1'))
    p.add_argument('--local-reference',type=Path,default=Path('logs/conv_residual_ablation_v1'))
    p.add_argument('--local-checkpoint',type=Path,default=Path(PLAIN_CHECKPOINT))
    p.add_argument('--official-cache',type=Path,default=Path('.cache/boq_official_v1'))
    p.add_argument('--dino-source',type=Path,default=Path('/home/wt/.cache/torch/hub/facebookresearch_dinov2_main'))
    p.add_argument('--dataset-root',type=Path,default=Path('datasets/gsv_cities'))
    p.add_argument('--source',type=Path,default=Path('logs/slgd_hard_v2_audit'))
    p.add_argument('--teacher',type=Path,default=Path('logs/slgd_v1/teacher_pilot'))
    p.add_argument('--cache',type=Path,default=Path('.cache/slgd_mid_v1_pilot'));p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':p.error('Expose physical GPU1 only')
    if args.root.resolve().parent!=(ROOT/'logs').resolve() or not args.root.name.startswith('conv_official_diagnosis_'):
        p.error('Use isolated logs/conv_official_diagnosis_<version>')
    if args.command=='wait_gpu':wait_gpu(args);return
    if args.output is None or not args.output.resolve().is_relative_to(args.root.resolve()) or args.output.resolve()==args.root.resolve():
        p.error('Choose a child output of the diagnosis root')
    if args.command=='run' and args.experiment is None:p.error('Need a registered experiment')
    if args.workers<0:p.error('Workers must be nonnegative')
    import fcntl
    args.root.mkdir(exist_ok=True)
    with (args.root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        if args.command!='aggregate':torch.cuda.set_per_process_memory_fraction(.65,0)
        write(args.root/'progress.json',dict(phase='running',experiment=args.experiment if args.command=='run' else 'aggregate',experiment_limit=5))
        try:
            if args.command=='aggregate':aggregate(args)
            elif args.experiment=='e1_implementation':audit(args)
            elif args.experiment=='e2_attenuate':attenuate(args)
            else:training(args)
        except Exception as e:
            write(args.root/'progress.json',dict(phase='failed',experiment=args.experiment,
                experiment_limit=5,type=type(e).__name__,message=str(e)))
            if args.output.exists() and not (args.output/'completed.json').exists():
                write(args.output/'failure.json',dict(type=type(e).__name__,message=str(e)))
            raise


if __name__=='__main__':main()
