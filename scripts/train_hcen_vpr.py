"""Fixed-budget ReID-inspired HCEN neck; existing RU stays entirely frozen."""
import argparse
import hashlib
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader,Dataset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_slgd import RU_SHA,read,write,sha,code_hash,verify,seal,seed,MatchedPlaces,save,restore,parameter_hash
from scripts.slgd_mid_screen import dataset,sources,load_visual,frozen_descriptor,open_contract,verify_cache_shards,current_code
from scripts.audit_slgd_hard import PlannedPlaces,collate
from src.models.hcen_vpr import HCENVPR,MODES
from src.hcen_protocol import POLICY,training_schedule,ranking_regularizer,paired_partitions
from src.slgd_mid_supervision import gps_filtered_retrieval
from src.slgd_hard_protocol import distance_metres

SOURCES=('scripts/train_hcen_vpr.py','src/models/hcen_vpr.py','src/hcen_protocol.py',
         'src/models/scan_mamba.py','src/slgd_mid_supervision.py',
         'src/dataloaders/valid/mapillary_sls.py','src/dataloaders/valid/pittsburgh.py')


class PlaceBatches(Dataset):
    def __init__(self,data,indices,schedule):
        self.places=MatchedPlaces(data,indices,99);self.schedule=schedule

    def __len__(self):return len(self.schedule)

    def __getitem__(self,index):
        values=[self.places[p] for p in self.schedule[index]]
        return (torch.cat([v[0] for v in values]),
                torch.tensor(self.schedule[index],dtype=torch.long).repeat_interleave(4),
                torch.cat([v[2]['coordinates'] for v in values]))


def tensor_hash(module):
    h=hashlib.sha256()
    for name,value in module.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def forward_batch(model,images,training=False):
    descriptors=[];references=[];aux=[[],[]];deltas=[]
    for chunk in images.cuda().split(4):
        features=model.features(chunk)
        with torch.no_grad():references.append(frozen_descriptor(model.visual,features[-1]))
        value=checkpoint(lambda a,b,c:model.aggregate(a,b,c,True),*features,use_reentrant=False) if training else model.aggregate(*features,auxiliary=False)
        descriptors.append(value[0])
        if training:
            aux[0].append(value[1]);aux[1].append(value[2])
        last=features[-1]
        deltas.append((value[-1].detach().flatten(1).norm(dim=1)/last.flatten(1).norm(dim=1).clamp_min(1e-8)))
    return (torch.cat(descriptors),torch.cat(references),
            tuple(torch.cat(x) for x in aux) if training else (),torch.cat(deltas))


def gsv_evaluate(model,data,heldout,output,workers):
    from tqdm import tqdm
    plan=heldout['plan'];desc=np.empty((plan['rows'],12288),np.float32);ref=np.empty_like(desc)
    coordinates=np.empty((plan['rows'],2),np.float64)
    loader=DataLoader(PlannedPlaces(data,plan),batch_size=16,num_workers=workers,collate_fn=collate)
    with torch.inference_mode():
        for batch,(images,indices,coords) in enumerate(tqdm(loader,desc='HCEN full GSV gallery')):
            write(output/'progress.json',dict(phase='gsv_evaluation',batch=batch,total=len(loader)))
            d,r,_,_=forward_batch(model,images)
            desc[indices.numpy()]=d.cpu().numpy();ref[indices.numpy()]=r.cpu().numpy()
            coordinates[indices.numpy()]=coords.numpy()
    result={}
    q=np.asarray(plan['query_rows']);g=np.asarray(plan['gallery_rows']);positive=np.asarray(plan['positive_gallery_indices'])
    for name,features in [('ru',ref),('student',desc)]:
        gallery=torch.from_numpy(features[g]).cuda();pred=[]
        for start in range(0,len(q),16):
            rows=q[start:start+16]
            scores=torch.from_numpy(features[rows]).cuda()@gallery.T
            allowed=distance_metres(coordinates[rows],coordinates[g])>=25
            allowed[np.arange(len(rows)),positive[start:start+16]]=True
            scores.masked_fill_(~torch.from_numpy(allowed).cuda(),-torch.inf)
            pred.extend(scores.argmax(1).cpu().tolist())
        result[name]=dict(predictions=pred,outcomes=(np.asarray(pred)==positive).tolist())
    if result['ru']['outcomes']!=read(output/'source_outcomes.json')['ru']:
        raise ValueError('Full GSV frozen RU outcomes differ from original')
    result['descriptor_drift_mean']=float(np.mean(1-(desc*ref).sum(1)))
    write(output/'gsv_final.json',result)


def benchmarks(model,data,output,workers):
    from tqdm import tqdm
    from src.dataloaders.valid.mapillary_sls import MapillarySLSDataset
    from src.dataloaders.valid.pittsburgh import PittsburghDataset
    for ds in [MapillarySLSDataset(Path('datasets/msls-val'),data.transform),
               PittsburghDataset(Path('datasets/pitts30k-val'),data.transform)]:
        path=output/(ds.dataset_name+'_final.json')
        if path.exists():continue
        descriptor=[];reference=[]
        loader=DataLoader(ds,batch_size=16,num_workers=workers)
        with torch.inference_mode():
            for batch,(images,_) in enumerate(tqdm(loader,desc=ds.dataset_name+' HCEN / frozen RU')):
                write(output/'progress.json',dict(phase='benchmark',dataset=ds.dataset_name,batch=batch,total=len(loader)))
                d,r,_,_=forward_batch(model,images)
                descriptor.append(d.cpu());reference.append(r.cpu())
        result={}
        for name,parts in [('ru',reference),('student',descriptor)]:
            all_=torch.cat(parts).cuda();db=all_[:ds.num_references]
            pred=np.concatenate([(q@db.T).topk(20,dim=1).indices.cpu().numpy()
                                 for q in all_[ds.num_references:].split(16)])
            hits=np.asarray([[bool(np.isin(row[:k],gt).any()) for k in (1,5,20)]
                             for row,gt in zip(pred,ds.ground_truth)])
            result[name]=dict(predictions=pred.tolist(),outcomes=hits[:,0].tolist(),
                              correct=int(hits[:,0].sum()),recall=hits.mean(0).tolist())
            del all_,db
        if result['ru']['correct']!={'msls-val':675,'pitts30k-val':7160}[ds.dataset_name]:
            raise ValueError('Standard benchmark frozen RU mismatch: '+ds.dataset_name)
        ru=np.asarray(result['ru']['outcomes']);st=np.asarray(result['student']['outcomes'])
        result['paired']=dict(corrections=int((~ru&st).sum()),regressions=int((ru&~st).sum()),net=int(st.sum())-int(ru.sum()))
        write(path,result)


def train(args):
    from tqdm import tqdm
    from src.losses.vpr_losses import VPRLossFunction
    teacher,heldout,metadata=sources(args)
    cache=verify(args.cache)
    if cache['code']!=current_code() or cache['stage']!=args.stage or cache['metadata']!=metadata:
        raise ValueError('Original training cache/source differs')
    if cache['source_completed_sha256']!=sha(args.source/'completed.json'):
        raise ValueError('Heldout source identity differs')
    if cache['teacher_completed_sha256']!=sha(args.teacher/'completed.json') or cache['train_places']!=teacher['train_places'][:len(cache['train_places'])]:
        raise ValueError('Original train split identity differs')
    verify_cache_shards(args.cache,len(cache['train_places']))
    schedule=training_schedule(read(args.cache/'triplets.json')['rows'],args.stage)
    benchmark_meta={str(p):sha(p) for name in ['msls-val','pitts30k-val']
                    for p in sorted((Path('datasets')/name).glob('*.npy'))} if args.stage=='pilot' else {}
    contract=dict(mode=args.mode,stage=args.stage,policy=POLICY,ru_sha256=RU_SHA,
        cache_completed_sha256=sha(args.cache/'completed.json'),source_completed_sha256=sha(args.source/'completed.json'),
        metadata=metadata,benchmark_metadata=benchmark_meta,train_places=cache['train_places'],schedule=schedule,
        original_code=current_code(),code={p:code_hash(ROOT/p) for p in SOURCES},paper_sha256=sha(args.reference_paper),
        versions=dict(torch=torch.__version__,numpy=np.__version__))
    open_contract(args.output,contract,args.resume)
    if (args.output/'completed.json').exists():verify(args.output);print('Already complete',args.mode);return
    data=dataset(args)
    if [str(data.places_ids[i]) for i in cache['train_indices']]!=cache['train_places']:
        raise ValueError('Original training identities differ')
    if [str(data.places_ids[i]) for i in heldout['plan']['dataset_indices']]!=heldout['plan']['query_places']+heldout['plan']['distractor_places']:
        raise ValueError('Original heldout identities differ')
    seed(42);visual=load_visual(args)
    frozen_hash=tensor_hash(visual)
    seed(42);model=HCENVPR(visual,args.mode).cuda().eval()
    probe=PlannedPlaces(data,heldout['plan'])[0][0]
    with torch.inference_mode():
        d,r,_,delta=forward_batch(model,probe)
    error=float((d-r).abs().max())
    if error>2e-6 or not torch.isfinite(d).all():raise ValueError('Zero start differs from RU')
    active={name:p for name,p in model.named_parameters() if p.requires_grad}
    if not active or any(not name.startswith('neck.') for name in active):raise ValueError('Only neck may train')
    optimizer=torch.optim.AdamW(active.values(),lr=POLICY['lr'],weight_decay=0.)
    state=dict(steps=0,rows=[],ssm_gradient_seen=False,rank_gradient_seen=False,
               contract_sha256=sha(args.output/'contract.json'),initial_parameter_sha256=parameter_hash(model))
    if (args.output/'last.pt').exists():state=restore(args.output/'last.pt',model,optimizer,state['contract_sha256'])
    if not (args.output/'initial.json').exists():
        write(args.output/'initial.json',dict(zero_start_error=error,parameter_sha256=state['initial_parameter_sha256'],
            descriptor_sha256=hashlib.sha256(d.cpu().numpy().tobytes()).hexdigest(),
            frozen_tensor_sha256=frozen_hash,trainable_parameters=sum(p.numel() for p in active.values())))
        save(args.output/'last.pt',model,optimizer,state)
    write(args.output/'source_outcomes.json',read(args.source/'outcomes.json'))
    lossfn=VPRLossFunction();model.train()
    loader=DataLoader(PlaceBatches(data,cache['train_indices'],schedule[state['steps']:]),
                      batch_size=None,num_workers=args.workers)
    for images,labels,coordinates in tqdm(loader,total=len(schedule),initial=state['steps'],desc=args.mode+' '+args.stage):
        optimizer.zero_grad(set_to_none=True)
        d,r,aux,delta=forward_batch(model,images,True);labels=labels.cuda()
        coords=coordinates.numpy()
        vpr=gps_filtered_retrieval(lossfn,d,labels,coords)
        auxiliary=sum(gps_filtered_retrieval(lossfn,x,labels,coords) for x in aux)/len(aux)
        rank,stats=ranking_regularizer(d,r,labels,coords)
        weight=POLICY['rank_weight'] if args.mode=='hierarchy_rank' else 0.
        loss=vpr+POLICY['auxiliary_weight']*auxiliary+weight*rank
        if not torch.isfinite(loss):raise ValueError('Nonfinite HCEN loss')
        rank_grad=None
        if weight and state['steps']%32==0:
            gradient=torch.autograd.grad(weight*rank,model.neck.up.weight,retain_graph=True,allow_unused=True)[0]
            rank_grad=float(gradient.norm()) if gradient is not None else 0.
            state['rank_gradient_seen'] |= rank_grad>0
        loss.backward()
        state['ssm_gradient_seen'] |= any('ssm.select' in name and p.grad is not None and bool((p.grad!=0).any()) for name,p in active.items())
        for p in active.values():
            if p.grad is None:p.grad=torch.zeros_like(p)
        norm=torch.nn.utils.clip_grad_norm_(list(active.values()),1.,error_if_nonfinite=True)
        if any(p.grad is not None for p in visual.parameters()):raise ValueError('Frozen RU received gradient')
        if float(delta.max())>.100001:raise ValueError('Feature residual bound exceeded')
        optimizer.step();state['steps']+=1
        state['rows'].append(dict(step=state['steps'],loss=float(loss.detach()),vpr=float(vpr.detach()),
            auxiliary=float(auxiliary.detach()),rank=float(rank.detach()),rank_stats=stats,rank_up_gradient=rank_grad,
            gradient_norm=float(norm),relative_feature_residual_mean=float(delta.mean()),relative_feature_residual_max=float(delta.max()),
            places=len(labels.unique()),images=len(labels)))
        write(args.output/'progress.json',dict(phase='training',mode=args.mode,steps=state['steps'],total=len(schedule),
                                              last_loss=state['rows'][-1]['loss']))
        if state['steps']%16==0:save(args.output/'last.pt',model,optimizer,state)
    if state['steps']!=len(schedule):raise ValueError('Incomplete fixed budget')
    if tensor_hash(visual)!=frozen_hash:raise ValueError('Frozen RU tensor mutation')
    save(args.output/'last.pt',model,optimizer,state)
    final_hash=parameter_hash(model);restore(args.output/'last.pt',model,optimizer,state['contract_sha256'])
    if parameter_hash(model)!=final_hash:raise ValueError('Checkpoint roundtrip differs')
    if any(int(optimizer.state[p]['step'])!=len(schedule) for p in active.values()):raise ValueError('Optimizer clocks differ')
    optimizer.zero_grad(set_to_none=True);model.eval()
    gsv_evaluate(model,data,heldout,args.output,args.workers)
    if args.stage=='pilot':benchmarks(model,data,args.output,args.workers)
    checks=dict(zero_start=error<=2e-6,frozen_ru_unchanged=True,checkpoint_roundtrip=True,
                nonzero_updates=final_hash!=state['initial_parameter_sha256'],ssm_gradient_seen=state['ssm_gradient_seen'])
    write(args.output/'summary.json',dict(mode=args.mode,stage=args.stage,steps=state['steps'],checks=checks,
        rank_gradient_seen=state['rank_gradient_seen'],initial_parameter_sha256=state['initial_parameter_sha256'],
        final_parameter_sha256=final_hash,rows=state['rows'],scope=POLICY['scope']))
    write(args.output/'progress.json',dict(phase='complete',steps=state['steps']))
    seal(args.output)
    print('COMPLETE',args.mode,checks,flush=True)


def compare(args):
    contracts={};summaries={};initials={};outcomes={}
    for mode in MODES:
        run=args.root/(mode+'_'+args.stage)
        contracts[mode]=verify(run);summaries[mode]=read(run/'summary.json');initials[mode]=read(run/'initial.json')
        outcomes[mode]=read(run/'gsv_final.json')['student']['outcomes']
    for mode in MODES[1:]:
        left=dict(contracts[MODES[0]]);right=dict(contracts[mode]);left.pop('mode');right.pop('mode')
        if left!=right:raise ValueError('Unmatched HCEN controls')
    if contracts['last_only']['policy']!=POLICY or contracts['last_only']['original_code']!=current_code() or contracts['last_only']['code']!={p:code_hash(ROOT/p) for p in SOURCES}:
        raise ValueError('Comparison code/policy differs from training')
    checks=dict(mechanical=all(all(s['checks'].values()) for s in summaries.values()),
        identical_initial_parameters=len({i['parameter_sha256'] for i in initials.values()})==1,
        identical_initial_descriptors=len({i['descriptor_sha256'] for i in initials.values()})==1,
        identical_budget=all(s['steps']==POLICY[args.stage+'_steps'] for s in summaries.values()))
    source=verify(args.source)
    if sha(args.source/'completed.json')!=contracts['last_only']['source_completed_sha256']:raise ValueError('Comparison source differs')
    old=read(args.source/'outcomes.json');plan=source['plan']
    report=paired_partitions(outcomes,old['ru'],plan['query_place_groups'],old['hard'],plan['calibration_places'])
    standard={}
    if args.stage=='pilot':
        for name in ['msls-val','pitts30k-val']:
            results={mode:read(args.root/(mode+'_'+args.stage)/(name+'_final.json')) for mode in MODES}
            ru=results['last_only']['ru']['outcomes']
            if any(r['ru']['outcomes']!=ru for r in results.values()):raise ValueError('Benchmark RU outcomes differ across arms')
            standard[name]=paired_partitions({m:r['student']['outcomes'] for m,r in results.items()},ru,
                np.zeros(len(ru),int),~np.asarray(ru,bool),1)['all']
    verdict='SMOKE_PASS' if all(checks.values()) else 'MECHANICAL_FAIL'
    if args.stage=='pilot' and all(checks.values()):verdict='PILOT_COMPLETE_NOT_FULL_TRAINING'
    contract=dict(policy=POLICY,stage=args.stage,completed_hashes={m:sha(args.root/(m+'_'+args.stage)/'completed.json') for m in MODES})
    open_contract(args.output,contract,args.resume)
    if (args.output/'completed.json').exists():
        verify(args.output)
        if read(args.output/'summary.json')['verdict']=='MECHANICAL_FAIL':raise SystemExit(2)
        return
    write(args.output/'summary.json',dict(verdict=verdict,checks=checks,partitions=report,standard_benchmark_paired=standard,
        rank_gradient_seen=summaries['hierarchy_rank']['rank_gradient_seen'],scope=POLICY['scope']))
    seal(args.output);print('COMPARISON',verdict,checks,flush=True)
    if not all(checks.values()):raise SystemExit(2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['train','compare'])
    p.add_argument('--mode',choices=MODES)
    p.add_argument('--stage',choices=['smoke','pilot'],required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--teacher',type=Path,default=Path('logs/slgd_v1/teacher_pilot'))
    p.add_argument('--dataset-root',type=Path,default=Path('datasets/gsv_cities'))
    p.add_argument('--root',type=Path,default=Path('logs/hcen_vpr_v1'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-paper',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1' or not torch.cuda.is_available() or torch.cuda.device_count()!=1:p.error('Expose physical GPU1 only')
    if not args.checkpoint.is_file() or sha(args.checkpoint)!=RU_SHA:p.error('Original RU checkpoint differs')
    if args.command=='train' and args.mode is None:p.error('Training mode required')
    import fcntl
    args.root.mkdir(parents=True,exist_ok=True);lock=(args.root/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.65,0)
    {'train':train,'compare':compare}[args.command](args)


if __name__=='__main__':main()
