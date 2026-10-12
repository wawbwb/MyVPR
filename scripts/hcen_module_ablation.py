"""Nine fresh module ablations vs sealed three-seed full HCEN; GPU-safe queue."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_slgd import read,write,sha,code_hash,verify,seal,seed,save,restore,parameter_hash
from scripts.slgd_mid_screen import open_contract,verify_cache_shards,current_code
from scripts.audit_slgd_hard import PlannedPlaces
from scripts.train_hcen_vpr import PlaceBatches,tensor_hash
from scripts.hcen_plain_base import context,load_plain,forward_plain_batch,gsv_evaluate,benchmarks,SOURCES as BASE_SOURCES
from scripts.hcen_plain_multiseed import reference_identity,SOURCES as MULTI_SOURCES
from src.hcen_aux_ablation import training_schedule,gradient_diagnostic
from src.hcen_plain_protocol import PLAIN_CHECKPOINT,PLAIN_SHA
from src.hcen_plain_replication import SEEDS,effective_policy
from src.models.hcen_ablation import VARIANTS,COUNTS,build_model,common_parameters,module_gradient_status
from src.hcen_module_protocol import POLICY as ABLATION_POLICY,matched_reference_contract,summarize,aggregate_reports
from src.slgd_mid_supervision import gps_filtered_retrieval

SOURCES=('scripts/hcen_module_ablation.py','src/models/hcen_ablation.py','src/hcen_module_protocol.py',*MULTI_SOURCES)


def references(args):
    baseline,seed42=reference_identity(args)
    aggregate=verify(args.full_reference/'aggregate')
    summary=read(args.full_reference/'aggregate/summary.json')
    if summary['verdict']!='PLAIN_REPLICATION_COMPLETE_NOT_STABLE_GAIN_PROOF' or not all(summary['checks'].values()):
        raise ValueError('Full-HCEN three-seed reference incomplete')
    roots={42:args.reference,43:args.full_reference/'seed43',44:args.full_reference/'seed44'}
    for s,root in roots.items():
        comparison=verify(root/'compare_screen')
        if sha(root/'compare_screen/completed.json')!=aggregate['completed_hashes'][str(s)]:raise ValueError('Reference aggregate identity differs')
        expected=BASE_SOURCES if s==42 else MULTI_SOURCES
        for mode in ('last_only','hierarchy'):
            run=root/(mode+'_screen');c=verify(run)
            if c['code']!={p:code_hash(ROOT/p) for p in expected}:raise ValueError('Sealed full-reference implementation changed')
            if sha(run/'completed.json')!=comparison['completed_hashes'][mode]:raise ValueError('Sealed reference run differs')
    return baseline,roots,dict(**seed42,full_aggregate_sha256=sha(args.full_reference/'aggregate/completed.json'))


def shared_hash(model):
    h=hashlib.sha256()
    for n,p in common_parameters(model).items():h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def wait_gpu(args):
    args.root.mkdir(parents=True,exist_ok=True);stable=0
    while True:
        text=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        used,util=[int(x.strip()) for x in text.strip().split(',')]
        stable=stable+1 if used<=1500 and util<=5 else 0
        write(args.root/'gpu_wait.json',dict(phase='gpu_available' if stable>=3 else 'waiting_for_gpu1',
            gpu1_memory_used_mib=used,gpu1_utilization_percent=util,consecutive_idle_checks=stable,
            policy='Need memory<=1500MiB and utilization<=5% for three 15s checks; never stop other processes'))
        if stable>=3:print('GPU1 available; proceed with mechanics and fixed protocol',flush=True);return
        print('WAIT_GPU1',used,'MiB',util,'%; do not interrupt other tasks',flush=True);time.sleep(15)


def train(args):
    from tqdm import tqdm
    from src.losses.vpr_losses import VPRLossFunction
    POLICY=effective_policy(args.seed)
    baseline,roots,provenance=references(args)
    contract,data,heldout,cache=context(args,'hierarchy',args.stage)
    verify_cache_shards(args.cache,len(cache['train_places']))
    schedule=training_schedule(read(args.cache/'triplets.json')['rows'],args.stage)
    contract.update(policy=POLICY,schedule=schedule,variant=args.variant,initialization_seed=args.seed,
        baseline_completed_sha256=provenance['baseline_completed_sha256'],module_policy=ABLATION_POLICY,
        full_reference=provenance,code={p:code_hash(ROOT/p) for p in SOURCES})
    original=verify(roots[args.seed]/('hierarchy_'+args.stage))
    if matched_reference_contract(contract)!=matched_reference_contract(original):raise ValueError('Fixed scientific protocol differs from full HCEN')
    open_contract(args.output,contract,args.resume)
    if (args.output/'completed.json').exists():verify(args.output);print('Already complete',args.variant,args.seed);return
    visual=load_plain(args);frozen=tensor_hash(visual)
    if frozen!=read(baseline/'summary.json')['frozen_tensor_sha256']:raise ValueError('Plain baseline tensors differ')
    seed(args.seed);full=build_model(visual,'full',args.seed)
    old_initial=read(roots[args.seed]/('hierarchy_'+args.stage)/'initial.json')
    if parameter_hash(full)!=old_initial['parameter_sha256']:raise ValueError('Historical full neck initialization not reproduced')
    shared=shared_hash(full);del full
    seed(args.seed);model=build_model(visual,args.variant,args.seed).cuda().eval()
    if shared_hash(model)!=shared:raise ValueError('Common outer-neck initialization differs')
    probe=PlannedPlaces(data,heldout['plan'])[0][0]
    with torch.inference_mode():
        d,r,_,_=forward_plain_batch(model,probe);direct=visual(probe.cuda())
    if not torch.equal(d,direct) or not torch.equal(r,direct):raise ValueError('Strict zero start differs from direct plain BoQ')
    active={n:p for n,p in model.named_parameters() if p.requires_grad}
    count=sum(p.numel() for p in active.values())
    if count!=COUNTS[args.variant] or any(not n.startswith('neck.') for n in active):raise ValueError('Unexpected ablation parameter inventory')
    optimizer=torch.optim.AdamW(active.values(),lr=POLICY['lr'],weight_decay=0.)
    state=dict(steps=0,rows=[],module_gradients={'0':False,'1':False},up_gradient_seen=False,
        initial_parameter_sha256=parameter_hash(model),initial_auxiliary_sha256=tensor_hash(model.neck.aux),contract_sha256=sha(args.output/'contract.json'))
    if (args.output/'last.pt').exists():state=restore(args.output/'last.pt',model,optimizer,state['contract_sha256'])
    if not (args.output/'initial.json').exists():
        write(args.output/'initial.json',dict(parameter_sha256=state['initial_parameter_sha256'],shared_parameter_sha256=shared,
            historical_full_parameter_sha256=old_initial['parameter_sha256'],initialization_seed=args.seed,
            descriptor_sha256=hashlib.sha256(d.cpu().numpy().tobytes()).hexdigest(),trainable_parameters=count,
            excluding_zero_weight_auxiliary_parameters=count-65536,frozen_tensor_sha256=frozen,zero_start_error=0.))
        save(args.output/'last.pt',model,optimizer,state)
    lossfn=VPRLossFunction();model.train()
    loader=DataLoader(PlaceBatches(data,cache['train_indices'],schedule[state['steps']:]),batch_size=None,num_workers=args.workers)
    for images,labels,coordinates in tqdm(loader,total=len(schedule),initial=state['steps'],desc=args.variant+' '+str(args.seed)+' '+args.stage):
        optimizer.zero_grad(set_to_none=True)
        d,r,aux,delta=forward_plain_batch(model,images,True);labels=labels.cuda();coords=coordinates.numpy()
        primary=gps_filtered_retrieval(lossfn,d,labels,coords)
        auxiliary=sum(gps_filtered_retrieval(lossfn,x,labels,coords) for x in aux)/len(aux)
        loss=primary+0.*auxiliary
        with torch.no_grad():reference_loss=gps_filtered_retrieval(lossfn,r,labels,coords)
        if not torch.isfinite(loss):raise ValueError('Nonfinite module-ablation loss')
        gradients=gradient_diagnostic(primary,0.*auxiliary,model.named_parameters()) if state['steps']%64==0 else None
        loss.backward()
        status=module_gradient_status(model)
        for key,value in status.items():state['module_gradients'][key]|=value
        state['up_gradient_seen']|=bool(model.neck.up.weight.grad is not None and (model.neck.up.weight.grad!=0).any())
        for p in active.values():
            if p.grad is None:p.grad=torch.zeros_like(p)
        norm=torch.nn.utils.clip_grad_norm_(list(active.values()),POLICY['clip'],error_if_nonfinite=True)
        if any(p.grad is not None for p in visual.parameters()):raise ValueError('Frozen plain model received gradient')
        if float(delta.max())>POLICY['feature_relative_cap']+1e-6:raise ValueError('Residual bound exceeded')
        optimizer.step();state['steps']+=1
        state['rows'].append(dict(step=state['steps'],primary=float(primary.detach()),frozen_baseline_loss=float(reference_loss),
            auxiliary=float(auxiliary.detach()),gradients=gradients,gradient_norm=float(norm),
            relative_residual_mean=float(delta.mean()),relative_residual_max=float(delta.max()),places=len(labels.unique()),images=len(labels)))
        write(args.output/'progress.json',dict(phase='training',variant=args.variant,seed=args.seed,steps=state['steps'],total=len(schedule),last_loss=float(loss.detach())))
        if state['steps']%16==0:save(args.output/'last.pt',model,optimizer,state)
    if state['steps']!=len(schedule) or tensor_hash(visual)!=frozen:raise ValueError('Budget incomplete or frozen base changed')
    if tensor_hash(model.neck.aux)!=state['initial_auxiliary_sha256']:raise ValueError('Zero-weight auxiliary head changed')
    if any(int(optimizer.state[p]['step'])!=len(schedule) for p in active.values()):raise ValueError('Optimizer clocks differ')
    save(args.output/'last.pt',model,optimizer,state);final=parameter_hash(model)
    restore(args.output/'last.pt',model,optimizer,state['contract_sha256'])
    if final!=parameter_hash(model):raise ValueError('Checkpoint roundtrip differs')
    optimizer.zero_grad(set_to_none=True);model.eval()
    if args.stage=='screen':
        gsv_evaluate(visual,model,data,heldout,args.output,args.workers,baseline)
        benchmarks(visual,model,args.output,args.workers,baseline)
    checks=dict(zero_start=True,shared_outer_initialization=True,frozen_plain_unchanged=True,checkpoint_roundtrip=True,
        disabled_auxiliary_unchanged=True,nonzero_updates=final!=state['initial_parameter_sha256'],
        both_stages_receive_module_gradient=all(state['module_gradients'].values()),up_gradient_seen=state['up_gradient_seen'])
    write(args.output/'summary.json',dict(variant=args.variant,seed=args.seed,stage=args.stage,steps=state['steps'],checks=checks,
        module_gradients=state['module_gradients'],trainable_parameters=count,excluding_zero_weight_auxiliary_parameters=count-65536,
        rows=state['rows'],final_parameter_sha256=final,scope=ABLATION_POLICY['scope']))
    write(args.output/'progress.json',dict(phase='complete',steps=state['steps']));seal(args.output)
    print('MODULE ABLATION COMPLETE',args.seed,args.variant,checks,flush=True)
    if not all(checks.values()):raise SystemExit(2)


def compare(args):
    baseline,roots,provenance=references(args);runs=args.root/('seed'+str(args.seed))
    contracts={v:verify(runs/(v+'_'+args.stage)) for v in VARIANTS}
    original=verify(roots[args.seed]/('hierarchy_'+args.stage))
    for v,c in contracts.items():
        if c['variant']!=v or c['initialization_seed']!=args.seed or c['module_policy']!=ABLATION_POLICY:raise ValueError('Wrong ablation identity')
        if c['code']!={p:code_hash(ROOT/p) for p in SOURCES} or c['original_code']!=current_code():raise ValueError('Implementation differs')
        if c['full_reference']!=provenance or matched_reference_contract(c)!=matched_reference_contract(original):raise ValueError('Unmatched full-HCEN protocol')
    initial={v:read(runs/(v+'_'+args.stage)/'initial.json') for v in VARIANTS}
    summary={v:read(runs/(v+'_'+args.stage)/'summary.json') for v in VARIANTS}
    old=read(roots[args.seed]/('hierarchy_'+args.stage)/'initial.json')
    checks=dict(mechanical=all(all(s['checks'].values()) for s in summary.values()),
        same_shared_outer_initialization=len({x['shared_parameter_sha256'] for x in initial.values()})==1,
        exact_historical_full_initialization=all(x['historical_full_parameter_sha256']==old['parameter_sha256'] for x in initial.values()),
        same_baseline_zero_start=all(x['descriptor_sha256']==old['descriptor_sha256'] for x in initial.values()),
        same_budget=all(s['steps']==effective_policy(args.seed)[args.stage+'_steps'] for s in summary.values()))
    report={};standard={}
    if args.stage=='screen':
        source=verify(args.source);plan=source['plan']
        if sha(args.source/'completed.json')!=original['source_completed_sha256']:raise ValueError('Query source differs')
        for dataset in ('gsv_final','msls-val_final','pitts30k-val_final'):
            b=read(baseline/(dataset+'.json'))['baseline'];values={v:read(runs/(v+'_screen')/(dataset+'.json')) for v in VARIANTS}
            values['full']=read(roots[args.seed]/'hierarchy_screen'/(dataset+'.json'))
            values['last_only']=read(roots[args.seed]/'last_only_screen'/(dataset+'.json'))
            if any(x['baseline']!=b for x in values.values()):raise ValueError('Baseline/query ranking differs')
            outcomes={k:x['student']['outcomes'] for k,x in values.items()}
            if dataset=='gsv_final':report=summarize(outcomes,b['outcomes'],plan['query_place_groups'],plan['calibration_places'])
            else:
                name=dataset.removesuffix('_final');standard[name]=summarize(outcomes,b['outcomes'],np.zeros(len(b['outcomes']),int),1)['all']
                standard[name]['recall_at_1_5_20']=dict(baseline=b['recall'],**{k:x['student']['recall'] for k,x in values.items()})
    verdict=('SMOKE_PASS' if args.stage=='smoke' else 'MODULE_SCREEN_COMPLETE_NOT_NECESSITY_PROOF') if all(checks.values()) else 'MECHANICAL_FAIL'
    contract=dict(policy=ABLATION_POLICY,seed=args.seed,stage=args.stage,full_reference=provenance,
        completed_hashes={v:sha(runs/(v+'_'+args.stage)/'completed.json') for v in VARIANTS})
    open_contract(args.output,contract,args.resume)
    if (args.output/'completed.json').exists():verify(args.output)
    else:
        write(args.output/'summary.json',dict(verdict=verdict,checks=checks,partitions=report,standard_benchmark_paired=standard,scope=ABLATION_POLICY['scope']))
        seal(args.output)
    print('COMPARISON',args.seed,verdict,checks,flush=True)
    if not all(checks.values()):raise SystemExit(2)


def aggregate(args):
    baseline,roots,provenance=references(args);reports={};hashes={}
    for s in SEEDS:
        run=args.root/('seed'+str(s));c=verify(run/'compare_screen');r=read(run/'compare_screen/summary.json')
        if r['verdict']!='MODULE_SCREEN_COMPLETE_NOT_NECESSITY_PROOF' or not all(r['checks'].values()):raise ValueError('Incomplete module comparison')
        if c['full_reference']!=provenance:raise ValueError('Full reference changed')
        for v in VARIANTS:
            verify(run/(v+'_screen'))
            if sha(run/(v+'_screen/completed.json'))!=c['completed_hashes'][v]:raise ValueError('Comparison/run hash differs')
        reports[s]={'gsv_'+role:r['partitions'][role] for role in ('calibration','evaluation','all')}
        reports[s].update(r['standard_benchmark_paired']);hashes[str(s)]=sha(run/'compare_screen/completed.json')
    contract=dict(policy=ABLATION_POLICY,completed_hashes=hashes,full_reference=provenance,code={p:code_hash(ROOT/p) for p in SOURCES})
    open_contract(args.output,contract,args.resume)
    if (args.output/'completed.json').exists():verify(args.output);return
    write(args.output/'summary.json',dict(verdict='MODULE_ABLATION_COMPLETE_NOT_NOVELTY_OR_STABILITY_PROOF',
        metrics=aggregate_reports(reports),per_seed=reports,parameter_counts=COUNTS,fairness=ABLATION_POLICY['fairness'],scope=ABLATION_POLICY['scope'],
        interpretation='Full vs conv is a nearly parameter-matched mixer replacement; linear/no-SASF also change capacity. Fixed repeated queries are not multiplied independent samples.'))
    seal(args.output);print('ALL MODULE ABLATIONS COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['wait_gpu','train','compare','aggregate']);p.add_argument('--variant',choices=VARIANTS)
    p.add_argument('--seed',type=int,choices=SEEDS,default=42);p.add_argument('--stage',choices=['smoke','screen'],default='screen')
    p.add_argument('--checkpoint',type=Path,default=Path(PLAIN_CHECKPOINT));p.add_argument('--reference',type=Path,default=Path('logs/hcen_plain_base_v1'))
    p.add_argument('--full-reference',type=Path,default=Path('logs/hcen_plain_multiseed_v1'))
    p.add_argument('--source',type=Path,default=Path('logs/slgd_hard_v2_audit'));p.add_argument('--cache',type=Path,default=Path('.cache/slgd_mid_v1_pilot'))
    p.add_argument('--teacher',type=Path,default=Path('logs/slgd_v1/teacher_pilot'));p.add_argument('--dataset-root',type=Path,default=Path('datasets/gsv_cities'))
    p.add_argument('--root',type=Path,default=Path('logs/hcen_module_ablation_v1'));p.add_argument('--output',type=Path)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--resume',action='store_true');args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':p.error('Expose physical GPU1 only')
    if args.command=='wait_gpu':wait_gpu(args);return  # No CUDA initialization/reservation while another task owns GPU1.
    if args.output is None:p.error('Output required')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:p.error('Need one visible CUDA device')
    if not args.checkpoint.is_file() or sha(args.checkpoint)!=PLAIN_SHA:p.error('Pinned plain checkpoint differs')
    if args.command=='train' and args.variant is None:p.error('Ablation variant required')
    import fcntl
    args.root.mkdir(parents=True,exist_ok=True);lock=(args.root/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.65,0)
    {'train':train,'compare':compare,'aggregate':aggregate}[args.command](args)


if __name__=='__main__':main()
