"""Train-only hard mining + same-architecture native L12/L10 guidance screen."""
import argparse
import functools
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_slgd import (RU_SHA,read,write,sha,code_hash,verify,seal,seed,
                                MatchedPlaces,save,restore,parameter_hash)
from scripts.audit_slgd_hard import PlannedPlaces,collate
from src.slgd_layer_protocol import LayerCapture
from src.slgd_mid_supervision import (MID_POLICY,native_teacher_margin,guidance_loss,
    gps_filtered_retrieval,paired_report)
from src.slgd_hard_protocol import distance_metres
from src.models.slgd import SLGDStudent,LocalProjector

SOURCES=('scripts/slgd_mid_screen.py','src/slgd_mid_supervision.py',
    'src/slgd_layer_protocol.py','src/models/slgd.py','scripts/train_slgd.py',
    'scripts/audit_slgd_hard.py','src/slgd_hard_protocol.py',
    'src/models/backbones/dinov2.py','src/models/aggregators/boq.py',
    'src/models/aggregators/salad.py','src/models/semantic_region_gate.py',
    'src/dataloaders/train/gsv_cities.py','src/losses/vpr_losses.py',
    'scripts/eval_condition_robustness.py')


def dataset(args):
    from torchvision import transforms as T
    from src.dataloaders.train.gsv_cities import GSVCitiesDataset
    seed(42)
    transform=T.Compose([T.Resize((280,280),interpolation=T.InterpolationMode.BICUBIC),T.ToTensor(),
                         T.Normalize([.485,.456,.406],[.229,.224,.225])])
    cities=sorted(p.stem for p in (args.dataset_root/'Dataframes').glob('*.csv'))
    return GSVCitiesDataset(dataset_path=args.dataset_root,cities=cities,img_per_place=4,
                           transform=transform,return_metadata=True)


def current_code():
    return {name:code_hash(ROOT/name) for name in SOURCES}


def sources(args):
    teacher=verify(args.teacher)
    heldout=verify(args.source)
    if teacher['checkpoint_sha256']!=RU_SHA or heldout['ru_sha256']!=RU_SHA or heldout['smoke']!=(args.stage=='smoke'):
        raise ValueError('RU/stage/source identity differs')
    for source in (teacher,heldout):
        for name,digest in source['code'].items():
            if code_hash(ROOT/name)!=digest:
                raise ValueError('Historical source changed: '+name)
    metadata={p.name:sha(p) for p in sorted((args.dataset_root/'Dataframes').glob('*.csv'))}
    if metadata!=teacher['metadata'] or metadata!=heldout['original_metadata']:
        raise ValueError('GSV metadata changed')
    if set(teacher['train_places']) & set(heldout['plan']['query_places']+heldout['plan']['distractor_places']):
        raise ValueError('Train/holdout gallery overlap')
    return teacher,heldout,metadata


def open_contract(path,contract,resume):
    if path.exists():
        if not resume or read(path/'contract.json')!=contract:
            raise ValueError('Existing output or immutable contract mismatch: '+str(path))
    else:
        path.mkdir(parents=True)
        write(path/'contract.json',contract)


def load_visual(args):
    from scripts.eval_condition_robustness import load_inference_model_from_ckpt
    return load_inference_model_from_ckpt(args.checkpoint,'cpu').cuda().eval().requires_grad_(False)


@torch.no_grad()
def frozen_descriptor(visual,features):
    gated=visual.semantic_region_gate(features)[0] if visual.semantic_region_gate is not None else features
    result=visual.aggregator(gated)
    return result[0] if isinstance(result,tuple) else result


class RowTriplets(Dataset):
    def __init__(self,data,indices,triplets,start=0):
        self.data=MatchedPlaces(data,indices,99)
        self.triplets=triplets[start:]

    @functools.lru_cache(maxsize=16)
    def place_images(self,place):
        return self.data[place][0]

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self,index):
        rows=self.triplets[index]
        return torch.stack([self.place_images(int(row)//4)[int(row)%4] for row in rows]),torch.as_tensor(rows)


def prepare(args):
    from tqdm import tqdm
    teacher,heldout,metadata=sources(args)
    count=MID_POLICY[args.stage+'_places']
    ids=teacher['train_places'][:count]
    data=dataset(args)
    lookup={str(x):i for i,x in enumerate(data.places_ids)}
    indices=[lookup[x] for x in ids]
    contract=dict(stage=args.stage,policy=MID_POLICY,checkpoint_sha256=RU_SHA,train_places=ids,
        train_indices=indices,source_completed_sha256=sha(args.source/'completed.json'),
        teacher_completed_sha256=sha(args.teacher/'completed.json'),metadata=metadata,code=current_code(),
        versions=dict(torch=torch.__version__,numpy=np.__version__))
    open_contract(args.cache,contract,args.resume)
    if (args.cache/'completed.json').exists():
        verify(args.cache)
        verify_cache_shards(args.cache,count)
        print('Hard mining cache already complete',flush=True)
        return
    shards=args.cache/'shards'
    shards.mkdir(exist_ok=True)
    global_rows=np.empty((count*4,12288),np.float32)
    coords=np.empty((count*4,2),np.float64)
    visual=None
    hashes={}
    loader=DataLoader(MatchedPlaces(data,indices,99),batch_size=16,num_workers=args.workers)
    for batch,(images,labels,metadata_batch) in enumerate(tqdm(loader,desc='Train-only frozen RU gallery')):
        path=shards/f'{batch:05d}.npz'
        rows=np.arange(batch*64,min((batch+1)*64,count*4),dtype=np.int64)
        write(args.cache/'progress.json',dict(phase='encode_training_gallery',batch=batch,total=(count+15)//16))
        reused=False
        if path.exists() and path.with_suffix('.json').exists() and sha(path)==read(path.with_suffix('.json'))['sha256']:
            with np.load(path,allow_pickle=False) as z:
                g,c=z['descriptors'],z['coordinates']
                if not np.array_equal(z['rows'],rows):
                    raise ValueError('Training shard identities differ')
            reused=True
        if not reused:
            if visual is None:
                visual=load_visual(args)
            parts=[]
            with torch.inference_mode():
                for chunk in images.flatten(0,1).cuda().split(4):
                    features=visual.backbone(chunk)
                    if isinstance(features,tuple):features=features[0]
                    parts.append(frozen_descriptor(visual,features).cpu().numpy())
            g=np.concatenate(parts).astype(np.float32)
            c=metadata_batch['coordinates'].flatten(0,1).numpy().astype(np.float64)
            temporary=path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream,rows=rows,descriptors=g,coordinates=c)
            temporary.replace(path)
            write(path.with_suffix('.json'),dict(sha256=sha(path)))
        if g.shape!=(len(rows),12288) or g.dtype!=np.float32 or c.shape!=(len(rows),2) or not np.isfinite(g).all() or not np.isfinite(c).all():
            raise ValueError('Invalid train gallery shard')
        if np.max(np.abs(np.linalg.norm(g,axis=-1)-1))>1e-4:
            raise ValueError('Non-unit RU training descriptors')
        global_rows[rows],coords[rows]=g,c
        hashes[path.name]=sha(path)
    del visual,loader
    torch.cuda.empty_cache()
    gallery=torch.from_numpy(global_rows).cuda()
    anchors=np.arange(0,count*4,4)
    positives=anchors[:,None]+np.arange(1,4)[None]
    triples=[]
    gaps=[]
    for start in tqdm(range(0,count,16),desc='Mine weakest positive / full train-gallery hard negative'):
        rows=anchors[start:start+16]
        ps=positives[start:start+16]
        sim=gallery[torch.as_tensor(rows,device='cuda')] @ gallery.T
        pidx=torch.as_tensor(ps,device='cuda')
        pos=pidx.gather(1,sim.gather(1,pidx).argmin(1)[:,None])[:,0]
        allowed=distance_metres(coords[rows],coords)>=25
        allowed &= rows[:,None]//4!=np.arange(count*4)[None]//4
        if not allowed.any(1).all():raise ValueError('No GPS-safe training negative')
        neg=sim.masked_fill(~torch.from_numpy(allowed).cuda(),-torch.inf).argmax(1)
        batch_ids=torch.arange(len(rows),device='cuda')
        gaps.extend((sim[batch_ids,pos]-sim[batch_ids,neg]).cpu().tolist())
        triples.extend(np.stack((rows,pos.cpu().numpy(),neg.cpu().numpy()),axis=1).tolist())
    del gallery
    write(args.cache/'triplets.json',dict(rows=triples,ru_margins=gaps))
    write(args.cache/'manifest.json',dict(complete=True,shards=hashes))
    write(args.cache/'summary.json',dict(train_places=count,images=count*4,triplets=len(triples),
        nonpositive_ru_margins=sum(x<=0 for x in gaps),margin_below_005=sum(x<.05 for x in gaps),
        mean_ru_margin=float(np.mean(gaps)),descriptor_cache_gib=global_rows.nbytes/1024**3,
        note='Weakest positive may have limited overlap; native teacher abstains on invalid/nonpositive margins'))
    write(args.cache/'progress.json',dict(phase='complete'))
    seal(args.cache)
    print('Prepared train-only fixed triplets:',read(args.cache/'summary.json'),flush=True)


def verify_cache_shards(path,count):
    hashes=read(path/'manifest.json')['shards']
    if len(hashes)!=(count+15)//16:
        raise ValueError('Incomplete train gallery shards')
    for name,digest in hashes.items():
        target=(path/'shards'/name).resolve()
        if not target.is_relative_to((path/'shards').resolve()) or sha(target)!=digest:
            raise ValueError('Train cache shard hash differs')


def train(args):
    from tqdm import tqdm
    from src.losses.vpr_losses import VPRLossFunction
    cache=verify(args.cache)
    if cache['policy']!=MID_POLICY or cache['code']!=current_code() or cache['stage']!=args.stage:
        raise ValueError('Train cache/code/policy differs')
    teacher,heldout,metadata=sources(args)
    if cache['source_completed_sha256']!=sha(args.source/'completed.json') or cache['teacher_completed_sha256']!=sha(args.teacher/'completed.json'):
        raise ValueError('Train cache source differs')
    verify_cache_shards(args.cache,len(cache['train_places']))
    contract=dict(**cache,mode=args.mode,cache_completed_sha256=sha(args.cache/'completed.json'))
    open_contract(args.output,contract,args.resume)
    if (args.output/'completed.json').exists():
        verify(args.output)
        print('Training already complete:',args.mode,flush=True)
        return
    data=dataset(args)
    if [str(data.places_ids[i]) for i in cache['train_indices']]!=cache['train_places']:
        raise ValueError('Train dataset identities changed')
    seed(42)
    visual=load_visual(args)
    init=LocalProjector().cuda().eval().requires_grad_(False)
    init.load_state_dict(torch.load(args.teacher/'last.pt',map_location='cpu',weights_only=True)['parameters'],strict=True)
    seed(42)
    model=SLGDStudent(visual,init,slots=16,beta=.2).cuda().eval()
    globals_=list(visual.aggregator.parameters())
    gids={id(x) for x in globals_}
    local=[x for x in model.parameters() if x.requires_grad and id(x) not in gids]
    active=globals_+local
    optimizer=torch.optim.AdamW([dict(params=globals_,lr=MID_POLICY['boq_lr']),
                                dict(params=local,lr=MID_POLICY['local_lr'])],weight_decay=0)
    state=dict(steps=0,rows=[],contract_sha256=sha(args.output/'contract.json'),
               initial_parameter_sha256=parameter_hash(model))
    if (args.output/'last.pt').exists():
        state=restore(args.output/'last.pt',model,optimizer,state['contract_sha256'])
    plan=heldout['plan']
    if not (args.output/'initial.json').exists():
        samples=PlannedPlaces(data,plan)
        images=samples[0][0]
        with torch.inference_mode():
            probe=model(images.cuda()).cpu().numpy()
        import hashlib
        write(args.output/'initial.json',dict(descriptor_sha256=hashlib.sha256(probe.tobytes()).hexdigest(),
            descriptor_dimension=probe.shape[1],norm_error=float(np.max(np.abs(np.linalg.norm(probe,axis=1)-1))),
            note='Same initial hybrid vector for every arm; initial hybrid is NOT identical to RU'))
        save(args.output/'last.pt',model,optimizer,state)
    capture=LayerCapture(visual.backbone,[10,12])
    triplets=np.asarray(read(args.cache/'triplets.json')['rows'],np.int64)
    total=MID_POLICY[args.stage+'_steps']
    if len(triplets)!=total*16:
        raise ValueError('Fixed triplet schedule/budget differs')
    # Original coordinate array is small; no feature cache is loaded into training RAM.
    coordinate_parts=[]
    for name in sorted(read(args.cache/'manifest.json')['shards']):
        with np.load(args.cache/'shards'/name,allow_pickle=False) as stored:
            coordinate_parts.append(stored['coordinates'])
    coords=np.concatenate(coordinate_parts)
    del coordinate_parts
    loss_function=VPRLossFunction()
    loader=DataLoader(RowTriplets(data,cache['train_indices'],triplets,state['steps']*16),
                      batch_size=16,num_workers=args.workers,drop_last=True)
    model.train()
    features={}
    bar=tqdm(loader,total=total,initial=state['steps'],desc=args.mode+' '+args.stage)
    for images,rows in bar:
        optimizer.zero_grad(set_to_none=True)
        features={10:[],12:[]}
        with torch.no_grad():
            for chunk in images.flatten(0,1).cuda().split(4):
                values,last=capture(chunk)
                for layer in features:features[layer].append(values[layer])
        features={layer:torch.cat(value) for layer,value in features.items()}
        descriptor=model.aggregate(features[12],return_training_tokens=False)[0]
        labels=rows.flatten().cuda()//4
        retrieval=gps_filtered_retrieval(loss_function,descriptor,labels,coords[rows.flatten().numpy()])
        a=torch.arange(0,len(descriptor),3,device='cuda')
        p,n=a+1,a+2
        diagnostic={}
        auxiliary=descriptor.sum()*0
        reliable=torch.zeros(len(a),dtype=bool,device='cuda')
        activity=reliable
        for layer in (10,12):
            margin,flags=native_teacher_margin(features[layer],a,p,n)
            diagnostic[f'L{layer}']=dict(reliable_fraction=float(flags.float().mean()),
                                        mean_margin=float(margin.mean()))
            if (args.mode=='mid10' and layer==10) or (args.mode=='last12' and layer==12):
                auxiliary,activity=guidance_loss(descriptor,a,p,n,margin,flags)
                reliable=flags
        loss=retrieval+MID_POLICY['auxiliary_weight']*auxiliary
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        probe=None
        if args.mode!='retrieval' and state['steps']%64==0:
            gv=torch.autograd.grad(retrieval,active,retain_graph=True,allow_unused=True)
            ga=torch.autograd.grad(MID_POLICY['auxiliary_weight']*auxiliary,active,retain_graph=True,allow_unused=True)
            vn=sum(float(x.detach().double().square().sum()) for x in gv if x is not None)
            an=sum(float(x.detach().double().square().sum()) for x in ga if x is not None)
            dot=sum(float((x.detach().double()*y.detach().double()).sum()) for x,y in zip(gv,ga) if x is not None and y is not None)
            probe=dict(retrieval_norm=vn**.5,auxiliary_norm=an**.5,
                       cosine=dot/(vn*an)**.5 if vn*an else None)
        loss.backward()
        for parameter in active:
            if parameter.grad is None:parameter.grad=torch.zeros_like(parameter)
        norm=torch.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True)
        if any(x.grad is not None for x in visual.backbone.parameters()):
            raise ValueError('Frozen DINO received gradient')
        optimizer.step()
        state['steps']+=1
        row=dict(step=state['steps'],loss=float(loss.detach()),retrieval=float(retrieval.detach()),
                 auxiliary=float(auxiliary.detach()),gradient_norm=float(norm),
                 reliable_fraction=float(reliable.float().mean()),active_fraction=float(activity.float().mean()),
                 teacher_diagnostic=diagnostic,unique_train_images=len(set(rows.flatten().tolist())))
        if probe is not None:row['gradient_probe']=probe
        state['rows'].append(row)
        bar.set_postfix(loss=row['loss'],aux=row['auxiliary'],active=row['active_fraction'])
        write(args.output/'progress.json',dict(phase='training',mode=args.mode,steps=state['steps'],total=total))
        if state['steps']%16==0:save(args.output/'last.pt',model,optimizer,state)
    capture.close()
    del loader,features
    if state['steps']!=total:raise ValueError('Training stopped before fixed final step')
    save(args.output/'last.pt',model,optimizer,state)
    before=parameter_hash(model)
    restore(args.output/'last.pt',model,optimizer,state['contract_sha256'])
    if parameter_hash(model)!=before:raise ValueError('Checkpoint roundtrip differs')
    if any(int(optimizer.state[x]['step'])!=total for x in active):raise ValueError('Optimizer clocks differ')
    # Native teacher hooks/MNN removed: evaluation calls only single-image student.forward.
    model.eval()
    if any(block._forward_hooks for block in visual.backbone.dino.blocks):
        raise ValueError('Teacher instrumentation still attached during inference')
    descriptors=np.empty((plan['rows'],14336),np.float32)
    coordinates=np.empty((plan['rows'],2),np.float64)
    eval_loader=DataLoader(PlannedPlaces(data,plan),batch_size=16,num_workers=args.workers,collate_fn=collate)
    with torch.inference_mode():
        for batch,(images,indices,coord) in enumerate(tqdm(eval_loader,desc=args.mode+' one-stage GSV gallery')):
            write(args.output/'progress.json',dict(phase='one_stage_evaluation',batch=batch,
                                                   total=(len(plan['dataset_indices'])+15)//16,steps=total))
            parts=[model(chunk).cpu().numpy() for chunk in images.cuda().split(4)]
            descriptors[indices.numpy()]=np.concatenate(parts)
            coordinates[indices.numpy()]=coord.numpy()
    qrows=np.asarray(plan['query_rows']); grows=np.asarray(plan['gallery_rows'])
    positive=np.asarray(plan['positive_gallery_indices'])
    gallery=torch.from_numpy(descriptors[grows]).cuda()
    predictions=[]
    for start in tqdm(range(0,len(qrows),16),desc=args.mode+' single-dot-product retrieval'):
        rows=qrows[start:start+16]
        score=torch.from_numpy(descriptors[rows]).cuda() @ gallery.T
        allowed=distance_metres(coordinates[rows],coordinates[grows])>=25
        allowed[np.arange(len(rows)),positive[start:start+16]]=True
        score.masked_fill_(~torch.from_numpy(allowed).cuda(),-torch.inf)
        predictions.extend(score.argmax(1).cpu().tolist())
    outcome=(np.asarray(predictions)==positive).tolist()
    final=dict(queries=len(qrows),references=len(grows),correct=sum(outcome),outcomes=outcome,
               predictions=predictions,descriptor_dimension=14336,
               norm_error=float(np.max(np.abs(np.linalg.norm(descriptors,axis=1)-1))),
               inference='Single-image fixed vector / full-gallery dot product; NO teacher, MNN, or candidate reranking')
    write(args.output/'final.json',final)
    write(args.output/'summary.json',dict(mode=args.mode,stage=args.stage,steps=total,
        initial_parameter_sha256=state['initial_parameter_sha256'],final_parameter_sha256=before,
        changed=before!=state['initial_parameter_sha256'],checkpoint_roundtrip=True,
        nonzero_gradient_steps=sum(x['gradient_norm']>0 for x in state['rows']),
        auxiliary_active_steps=sum(x['active_fraction']>0 for x in state['rows']),
        mean_reliable_fraction=float(np.mean([x['reliable_fraction'] for x in state['rows']])),
        mean_active_fraction=float(np.mean([x['active_fraction'] for x in state['rows']])),
        trainable_parameters=sum(x.numel() for x in active),rows=state['rows'],scope=MID_POLICY['scope']))
    write(args.output/'progress.json',dict(phase='complete',steps=total))
    seal(args.output)
    print('COMPLETE',args.mode,'correct',sum(outcome),'/',len(outcome),flush=True)


def compare(args):
    contracts={}; finals={}; summaries={}; initials={}
    for mode in MID_POLICY['modes']:
        path=args.root/f'{mode}_{args.stage}'
        contracts[mode]=verify(path)
        finals[mode]=read(path/'final.json')
        summaries[mode]=read(path/'summary.json')
        initials[mode]=read(path/'initial.json')
        if contracts[mode]['mode']!=mode or contracts[mode]['stage']!=args.stage or contracts[mode]['code']!=current_code():
            raise ValueError('Unmatched run mode/stage/code')
    for mode in ('last12','mid10'):
        reference=dict(contracts['retrieval']); actual=dict(contracts[mode])
        reference.pop('mode'); actual.pop('mode')
        if reference!=actual:raise ValueError('Unmatched experiment contracts')
    initial_hashes={x['initial_parameter_sha256'] for x in summaries.values()}
    checks=dict(identical_initial_parameters=len(initial_hashes)==1,
        identical_initial_descriptors=all(x==initials['retrieval'] for x in initials.values()),
        completed_fixed_budget=all(x['steps']==MID_POLICY[args.stage+'_steps'] for x in summaries.values()),
        finite_nonzero_updates=all(x['changed'] and x['nonzero_gradient_steps']>0 and x['checkpoint_roundtrip'] for x in summaries.values()),
        last_auxiliary_active=summaries['last12']['auxiliary_active_steps']>0,
        mid_auxiliary_active=summaries['mid10']['auxiliary_active_steps']>0)
    heldout=verify(args.source)
    if sha(args.source/'completed.json')!=contracts['retrieval']['source_completed_sha256']:
        raise ValueError('Comparison source differs from training source')
    if any(len(x['outcomes'])!=len(heldout['plan']['query_rows']) for x in finals.values()):
        raise ValueError('Final query count differs')
    o=read(args.source/'outcomes.json')
    report=paired_report({mode:x['outcomes'] for mode,x in finals.items()},heldout['plan']['query_place_groups'],
                        o['ru'],o['hard'],heldout['plan']['calibration_places'])
    if args.stage=='smoke':
        # The small smoke gallery may already satisfy every teacher margin.
        # Unit tests exercise synthetic violations; do not relabel inactivity as a code error.
        mechanical={k:v for k,v in checks.items() if not k.endswith('auxiliary_active')}
        verdict='SMOKE_PASS' if all(mechanical.values()) else 'MECHANICAL_FAIL'
    elif not all(checks.values()):
        verdict='INACTIVE_OR_UNMATCHED_SUPERVISION'
    else:
        gains=all(x['comparisons']['mid10_vs_last12']['net']>=MID_POLICY['min_net_vs_last'] and
                  x['comparisons']['mid10_vs_retrieval']['net']>=MID_POLICY['min_net_vs_retrieval']
                  for x in report.values())
        verdict='EXPLORATORY_MID_LAYER_GAIN' if gains else 'NO_FIXED_LAST_MID_LAYER_GAIN'
    comparison_contract=dict(stage=args.stage,policy=MID_POLICY,code=current_code(),
        completed_hashes={mode:sha(args.root/f'{mode}_{args.stage}'/'completed.json') for mode in MID_POLICY['modes']})
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'completed.json').exists():
        if verify(args.output)!=comparison_contract:
            raise ValueError('Existing comparison belongs to different runs')
        print('Comparison already complete:',read(args.output/'summary.json')['verdict'],flush=True)
        if args.stage=='smoke' and read(args.output/'summary.json')['verdict']!='SMOKE_PASS':
            raise SystemExit(2)
        return
    write(args.output/'contract.json',comparison_contract)
    write(args.output/'summary.json',dict(verdict=verdict,checks=checks,partitions=report,
        activity={mode:dict(active_steps=summaries[mode]['auxiliary_active_steps'],
                           mean_coverage=summaries[mode]['mean_reliable_fraction']) for mode in MID_POLICY['modes']},
        scope=MID_POLICY['scope'],note='Neither pilot nor smoke demonstrates semantic/novelty/stable benefit'))
    seal(args.output)
    print('COMPARISON:',verdict,checks,flush=True)
    if args.stage=='smoke' and verdict!='SMOKE_PASS':raise SystemExit(2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','train','compare'])
    p.add_argument('--stage',choices=['smoke','pilot'],required=True)
    p.add_argument('--mode',choices=MID_POLICY['modes'])
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--teacher',type=Path,default=Path('logs/slgd_v1/teacher_pilot'))
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--root',type=Path,default=Path('logs/slgd_mid_v1'))
    p.add_argument('--dataset-root',type=Path,default=Path('datasets/gsv_cities'))
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1' or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        p.error('Expose physical GPU1 only')
    if not args.checkpoint.is_file() or sha(args.checkpoint)!=RU_SHA:
        p.error('Original RU missing or SHA differs')
    if args.command in ('train','compare') and args.output is None:p.error('Output required')
    if args.command=='train' and args.mode is None:p.error('Train mode required')
    import fcntl
    args.root.mkdir(parents=True,exist_ok=True)
    lock=(args.root/'worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(.65,0)
    {'prepare':prepare,'train':train,'compare':compare}[args.command](args)


if __name__=='__main__':main()
