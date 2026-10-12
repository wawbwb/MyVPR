"""Fresh seed43/44 necks on actual plain-BoQ, same sealed baseline and loop."""
import argparse
import hashlib
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_slgd import read,write,sha,code_hash,verify,seal,seed,save,restore,parameter_hash
from scripts.slgd_mid_screen import open_contract,verify_cache_shards,current_code
from scripts.audit_slgd_hard import PlannedPlaces
from scripts.train_hcen_vpr import PlaceBatches,tensor_hash
from scripts.hcen_plain_base import (context,load_plain,forward_plain_batch,gsv_evaluate,benchmarks,
    SOURCES as BASE_SOURCES)
from src.hcen_aux_ablation import training_schedule,gradient_diagnostic
from src.hcen_plain_protocol import MODES,PLAIN_CHECKPOINT,PLAIN_SHA,PlainHCENVPR,paired_report
from src.hcen_plain_replication import (POLICY as REPLICATION_POLICY,NEW_SEEDS,effective_policy,
    comparable_contract,aggregate_reports)
from src.slgd_mid_supervision import gps_filtered_retrieval

SOURCES=('scripts/hcen_plain_multiseed.py','src/hcen_plain_replication.py',*BASE_SOURCES)


def reference_identity(args):
    root=args.reference;c=verify(root/'compare_screen');b=verify(root/'baseline')
    if read(root/'compare_screen/summary.json')['verdict']!='PLAIN_BASE_SCREEN_COMPLETE_NOT_STABLE_GAIN_PROOF':
        raise ValueError('Seed42 plain comparison incomplete')
    if b['checkpoint_sha256']!=PLAIN_SHA or sha(root/'baseline/completed.json')!=c['baseline_completed_sha256']:
        raise ValueError('Independent plain baseline differs')
    if b['code']!={p:code_hash(ROOT/p) for p in BASE_SOURCES}:raise ValueError('Sealed baseline source changed')
    for mode in MODES:
        run=root/(mode+'_screen');old=verify(run)
        if old['policy']!=effective_policy(42) or old['code']!={p:code_hash(ROOT/p) for p in BASE_SOURCES}:
            raise ValueError('Sealed seed42 protocol/source changed')
        if sha(run/'completed.json')!=c['completed_hashes'][mode]:raise ValueError('Seed42 comparison/run identity differs')
    return root/'baseline',dict(comparison_sha256=sha(root/'compare_screen/completed.json'),
        baseline_completed_sha256=sha(root/'baseline/completed.json'))


def train(args):
    from tqdm import tqdm
    from src.losses.vpr_losses import VPRLossFunction
    POLICY=effective_policy(args.seed)
    reference,provenance=reference_identity(args)
    contract,data,heldout,cache=context(args,args.mode,args.stage)
    verify_cache_shards(args.cache,len(cache['train_places']))
    schedule=training_schedule(read(args.cache/'triplets.json')['rows'],args.stage)
    contract.update(policy=POLICY,schedule=schedule,initialization_seed=args.seed,
        baseline_completed_sha256=provenance['baseline_completed_sha256'],replication_reference=provenance,
        replication_policy=REPLICATION_POLICY,code={p:code_hash(ROOT/p) for p in SOURCES})
    original=verify(args.reference/(args.mode+'_'+args.stage))
    if comparable_contract(contract)!=comparable_contract(original):raise ValueError('Scientific protocol differs beyond seed')
    open_contract(args.output,contract,args.resume)
    if (args.output/'completed.json').exists():verify(args.output);print('Already complete',args.mode,args.seed);return
    visual=load_plain(args);frozen=tensor_hash(visual)
    if frozen!=read(reference/'summary.json')['frozen_tensor_sha256']:raise ValueError('Baseline model tensors differ')
    seed(args.seed);model=PlainHCENVPR(visual,args.mode).cuda().eval()
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
        write(args.output/'initial.json',dict(parameter_sha256=state['initial_parameter_sha256'],initialization_seed=args.seed,
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
    write(args.output/'summary.json',dict(mode=args.mode,stage=args.stage,initialization_seed=args.seed,steps=state['steps'],checks=checks,rows=state['rows'],
        initial_parameter_sha256=state['initial_parameter_sha256'],final_parameter_sha256=final,scope=REPLICATION_POLICY['scope']))
    write(args.output/'progress.json',dict(phase='complete',steps=state['steps']));seal(args.output)
    print('PLAIN HCEN COMPLETE',args.seed,args.mode,checks,flush=True)
    if not all(checks.values()):raise SystemExit(2)


def compare(args):
    reference,provenance=reference_identity(args);runs=args.root/('seed'+str(args.seed))
    contracts={m:verify(runs/(m+'_'+args.stage)) for m in MODES}
    for mode,c in contracts.items():
        if c['mode']!=mode or c['policy']!=effective_policy(args.seed) or c['initialization_seed']!=args.seed:raise ValueError('Seed/mode identity differs')
        if c['code']!={p:code_hash(ROOT/p) for p in SOURCES} or c['original_code']!=current_code():raise ValueError('Implementation differs')
        if c['replication_reference']!=provenance:raise ValueError('Seed42 reference changed')
        if comparable_contract(c)!=comparable_contract(verify(args.reference/(mode+'_'+args.stage))):raise ValueError('Unmatched seed42 protocol')
    common=[{k:v for k,v in c.items() if k!='mode'} for c in contracts.values()]
    if common[0]!=common[1]:raise ValueError('Unmatched two neck controls')
    initial={m:read(runs/(m+'_'+args.stage)/'initial.json') for m in MODES}
    summaries={m:read(runs/(m+'_'+args.stage)/'summary.json') for m in MODES}
    checks=dict(mechanical=all(all(s['checks'].values()) for s in summaries.values()),
        identical_parameters=initial['last_only']['parameter_sha256']==initial['hierarchy']['parameter_sha256'],
        identical_zero_start=initial['last_only']['descriptor_sha256']==initial['hierarchy']['descriptor_sha256'],
        identical_budget=all(s['steps']==effective_policy(args.seed)[args.stage+'_steps'] for s in summaries.values()),
        same_zero_start_as_seed42=all(v['descriptor_sha256']==read(args.reference/(m+'_'+args.stage)/'initial.json')['descriptor_sha256'] for m,v in initial.items()),
        different_initialization_from_seed42=all(v['parameter_sha256']!=read(args.reference/(m+'_'+args.stage)/'initial.json')['parameter_sha256'] for m,v in initial.items()))
    report={};standard={};source=verify(args.source)
    if sha(args.source/'completed.json')!=common[0]['source_completed_sha256']:raise ValueError('Heldout source changed')
    if args.stage=='screen':
        plan=source['plan'];b=read(reference/'gsv_final.json')['baseline']
        values={m:read(runs/(m+'_screen')/'gsv_final.json') for m in MODES}
        if any(v['baseline']!=b for v in values.values()):raise ValueError('GSV baseline/query mismatch')
        report=paired_report({m:v['student']['outcomes'] for m,v in values.items()},b['outcomes'],plan['query_place_groups'],plan['calibration_places'])
        for name in ('msls-val','pitts30k-val'):
            b=read(reference/(name+'_final.json'))['baseline'];values={m:read(runs/(m+'_screen')/(name+'_final.json')) for m in MODES}
            if any(v['baseline']!=b for v in values.values()):raise ValueError('Benchmark baseline/query mismatch')
            standard[name]=paired_report({m:v['student']['outcomes'] for m,v in values.items()},b['outcomes'],np.zeros(len(b['outcomes']),int),1)['all']
            standard[name]['recall_at_1_5_20']=dict(baseline=b['recall'],**{m:v['student']['recall'] for m,v in values.items()})
    verdict=('SMOKE_PASS' if args.stage=='smoke' else 'PLAIN_BASE_SCREEN_COMPLETE_NOT_STABLE_GAIN_PROOF') if all(checks.values()) else 'MECHANICAL_FAIL'
    contract=dict(replication_policy=REPLICATION_POLICY,initialization_seed=args.seed,stage=args.stage,replication_reference=provenance,
        completed_hashes={m:sha(runs/(m+'_'+args.stage)/'completed.json') for m in MODES})
    open_contract(args.output,contract,args.resume)
    if (args.output/'completed.json').exists():verify(args.output)
    else:
        write(args.output/'summary.json',dict(verdict=verdict,checks=checks,initialization_seed=args.seed,
            partitions=report,standard_benchmark_paired=standard,scope=REPLICATION_POLICY['scope']))
        seal(args.output)
    print('COMPARISON',args.seed,verdict,checks,flush=True)
    if not all(checks.values()):raise SystemExit(2)


def aggregate(args):
    reference,provenance=reference_identity(args)
    roots={42:args.reference,43:args.root/'seed43',44:args.root/'seed44'}
    reports={};hashes={};initials={};descriptors=set();frozen=set()
    for s,root in roots.items():
        c=verify(root/'compare_screen');r=read(root/'compare_screen/summary.json')
        if r['verdict']!='PLAIN_BASE_SCREEN_COMPLETE_NOT_STABLE_GAIN_PROOF' or not all(r['checks'].values()):raise ValueError('Incomplete/mechanically invalid seed')
        hashes[str(s)]=sha(root/'compare_screen/completed.json');initials[s]=set()
        for mode in MODES:
            run=root/(mode+'_screen');contract=verify(run)
            if sha(run/'completed.json')!=c['completed_hashes'][mode]:raise ValueError('Run/comparison hash differs')
            if comparable_contract(contract)!=comparable_contract(verify(args.reference/(mode+'_screen'))):raise ValueError('Scientific protocol differs across seeds')
            expected=BASE_SOURCES if s==42 else SOURCES
            if contract['code']!={p:code_hash(ROOT/p) for p in expected}:raise ValueError('Source hash differs')
            if s!=42 and contract['replication_reference']!=provenance:raise ValueError('Replication reference differs')
            v=read(run/'initial.json');initials[s].add(v['parameter_sha256']);descriptors.add(v['descriptor_sha256']);frozen.add(v['frozen_tensor_sha256'])
            for name in ('gsv_final','msls-val_final','pitts30k-val_final'):
                result=read(run/(name+'.json'));original=read(reference/(name+'.json'))
                if result['baseline']!=original['baseline'] or result.get('preprocessing')!=original.get('preprocessing'):
                    raise ValueError('Plain direct baseline predictions/preprocessing differ across seeds')
        reports[s]={'gsv_'+role:r['partitions'][role] for role in ('calibration','evaluation','all')}
        reports[s].update(r['standard_benchmark_paired'])
    if any(len(v)!=1 for v in initials.values()) or len(set.union(*initials.values()))!=3:raise ValueError('Expected three distinct matched initializations')
    if len(descriptors)!=1 or len(frozen)!=1:raise ValueError('Plain zero-start/frozen baseline differs')
    contract=dict(policy=REPLICATION_POLICY,replication_reference=provenance,completed_hashes=hashes,code={p:code_hash(ROOT/p) for p in SOURCES})
    open_contract(args.output,contract,args.resume)
    if (args.output/'completed.json').exists():verify(args.output);return
    write(args.output/'summary.json',dict(verdict='PLAIN_REPLICATION_COMPLETE_NOT_STABLE_GAIN_PROOF',
        checks=dict(matched_protocol=True,three_distinct_initializations=True,same_plain_frozen_baseline=True),
        metrics=aggregate_reports(reports),per_seed=reports,scope=REPLICATION_POLICY['scope'],
        interpretation='Mean/std describe neck initialization variability. Fixed test queries are not tripled independent samples; no pooled significance claim.'))
    seal(args.output);print('ALL THREE PLAIN-BASE SEEDS COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['train','compare','aggregate']);p.add_argument('--mode',choices=MODES)
    p.add_argument('--seed',type=int,choices=NEW_SEEDS,default=43);p.add_argument('--stage',choices=['smoke','screen'],default='screen')
    p.add_argument('--checkpoint',type=Path,default=Path(PLAIN_CHECKPOINT))
    p.add_argument('--reference',type=Path,default=Path('logs/hcen_plain_base_v1'))
    p.add_argument('--source',type=Path,default=Path('logs/slgd_hard_v2_audit'));p.add_argument('--cache',type=Path,default=Path('.cache/slgd_mid_v1_pilot'))
    p.add_argument('--teacher',type=Path,default=Path('logs/slgd_v1/teacher_pilot'));p.add_argument('--dataset-root',type=Path,default=Path('datasets/gsv_cities'))
    p.add_argument('--root',type=Path,default=Path('logs/hcen_plain_multiseed_v1'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--resume',action='store_true');args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1' or not torch.cuda.is_available() or torch.cuda.device_count()!=1:p.error('Expose physical GPU1 only')
    if not args.checkpoint.is_file() or sha(args.checkpoint)!=PLAIN_SHA:p.error('Pinned actual plain-BoQ checkpoint differs')
    if args.command=='train' and args.mode is None:p.error('Training mode required')
    import fcntl
    args.root.mkdir(parents=True,exist_ok=True);lock=(args.root/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.cuda.set_per_process_memory_fraction(.65,0)
    {'train':train,'compare':compare,'aggregate':aggregate}[args.command](args)


if __name__=='__main__':main()
