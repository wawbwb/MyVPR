"""Three matched warm-start controls; train-only official relation preservation."""
import argparse
import datetime
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.train_slgd import read,write,sha,code_hash,verify,seal,save,restore,parameter_hash
from scripts.slgd_mid_screen import open_contract
from scripts.train_hcen_vpr import PlaceBatches,tensor_hash
from scripts.conv_official_boq import contract as original_contract,evaluate
from scripts.conv_official_diagnosis import author,make_adapter,load_parameters,SOURCES as OLD_SOURCES
from scripts.conv_residual_ablation import forward_batch
from scripts.hcen_module_ablation import wait_gpu
from src.models.conv_residual_ablation import PARAMETERS,module_gradient_status
from src.official_adaptation import ARMS,POLICY,preservation_losses,exact_paired
from src.slgd_mid_supervision import gps_filtered_retrieval

SOURCES=tuple(dict.fromkeys(('scripts/conv_official_adapt.py','src/official_adaptation.py',
    'scripts/run_conv_official_adapt.sh','tests/test_official_adaptation.py',*OLD_SOURCES)))


def references(args):
    identities={}
    for name,path in [('official_baseline',args.reference/'baseline'),
                      ('warm_checkpoint',args.reference/'seed42/train'),
                      ('warm_evaluation',args.diagnosis/'e2_attenuate'),
                      ('diagnosis',args.diagnosis/'aggregate')]:
        contract=verify(path)
        if contract['code']!={p:code_hash(ROOT/p) for p in contract['code']}:
            raise ValueError('Sealed source changed: '+name)
        identities[name]=sha(path/'completed.json')
    if read(args.diagnosis/'aggregate/summary.json')['verdict']!='FIVE_ITEM_DIAGNOSIS_COMPLETE':
        raise ValueError('Finish the closed diagnosis first')
    return identities


def make_contract(args):
    c,data,cache=original_contract(args,'adaptation',42)
    schedule=read(args.reference/'seed42/train/contract.json')['schedule']
    if len(schedule)!=2048 or schedule[:512]*4!=schedule:
        raise ValueError('Original matched schedule changed')
    c.update(policy=POLICY,arm=args.arm,arm_settings=ARMS[args.arm],stage='official_adaptation',
        schedule=schedule[:POLICY['steps']],initialization_seed=42,
        prior_results=references(args),warm_checkpoint_sha256=sha(args.reference/'seed42/train/last.pt'),
        code={p:code_hash(ROOT/p) for p in SOURCES})
    return c,data,cache


def train(args):
    from tqdm import tqdm
    from src.losses.vpr_losses import VPRLossFunction
    c,data,cache=make_contract(args)
    open_contract(args.output,c,args.resume)
    if (args.output/'completed.json').exists():
        verify(args.output);print('Already complete',args.arm);return
    visual=author(args);frozen=tensor_hash(visual)
    model=make_adapter(visual,POLICY['rho'])
    load_parameters(model,args.reference/'seed42/train/last.pt')
    initial=parameter_hash(model)
    expected=read(args.reference/'seed42/train/summary.json')['final_parameter_sha256']
    if initial!=expected or model.neck.output_scale!=.25:
        raise ValueError('Not the exact unchanged E2 warm start')
    active={n:p for n,p in model.named_parameters() if p.requires_grad}
    if sum(p.numel() for p in active.values())!=PARAMETERS or any(not n.startswith('neck.') for n in active):
        raise ValueError('Unexpected trainable parameter inventory')
    optimizer=torch.optim.AdamW(active.values(),lr=POLICY['lr'],weight_decay=0.)
    state=dict(steps=0,rows=[],module_gradients={'0':False,'1':False},up_gradient_seen=False,
        initial_parameter_sha256=initial,initial_auxiliary_sha256=tensor_hash(model.neck.aux),
        contract_sha256=sha(args.output/'contract.json'))
    if (args.output/'last.pt').exists():
        state=restore(args.output/'last.pt',model,optimizer,state['contract_sha256'])
    if not (args.output/'initial.json').exists():
        write(args.output/'initial.json',dict(initial_parameter_sha256=initial,
            frozen_tensor_sha256=frozen,arm_settings=ARMS[args.arm],new_optimizer=True,
            exact_e2_warmstart=True,source_steps=2048,residual_scale=.25))
        save(args.output/'last.pt',model,optimizer,state)
    lossfn=VPRLossFunction();model.train()
    settings=ARMS[args.arm]
    loader=DataLoader(PlaceBatches(data,cache['train_indices'],c['schedule'][state['steps']:]),
        batch_size=None,num_workers=args.workers)
    for images,labels,coords in tqdm(loader,total=POLICY['steps'],initial=state['steps'],desc=args.arm):
        optimizer.zero_grad(set_to_none=True)
        d,teacher,_,delta,_=forward_batch(model,images,True)
        labels=labels.cuda();coordinates=coords.numpy()
        primary=gps_filtered_retrieval(lossfn,d,labels,coordinates)
        rank,relation,stats=preservation_losses(d,teacher,labels,coordinates)
        loss=primary+settings['rank_weight']*rank+settings['relation_weight']*relation
        if teacher.requires_grad or not bool(torch.isfinite(loss)) or float(delta.max())>POLICY['rho']+1e-6:
            raise ValueError('Teacher leak/nonfinite loss/residual cap violation')
        loss.backward()
        for name,seen in module_gradient_status(model).items():state['module_gradients'][name]|=seen
        state['up_gradient_seen']|=bool(model.neck.up.weight.grad is not None and (model.neck.up.weight.grad!=0).any())
        for p in active.values():
            if p.grad is None:p.grad=torch.zeros_like(p)
        norm=torch.nn.utils.clip_grad_norm_(list(active.values()),POLICY['clip'],error_if_nonfinite=True)
        if any(p.grad is not None for p in visual.parameters()):raise ValueError('Frozen author received gradients')
        optimizer.step();state['steps']+=1
        state['rows'].append(dict(step=state['steps'],loss=float(loss.detach()),retrieval_loss=float(primary.detach()),
            rank_loss=float(rank.detach()),relation_loss=float(relation.detach()),gradient_norm=float(norm),
            relative_residual_mean=float(delta.mean()),relative_residual_max=float(delta.max()),**stats))
        write(args.output/'progress.json',dict(phase='training',arm=args.arm,steps=state['steps'],
            total=POLICY['steps'],loss=float(loss.detach()),**stats))
        if state['steps']%16==0:save(args.output/'last.pt',model,optimizer,state)
    save(args.output/'last.pt',model,optimizer,state)
    final=parameter_hash(model)
    restore(args.output/'last.pt',model,optimizer,state['contract_sha256'])
    checks=dict(exact_e2_warmstart=initial==expected,frozen_unchanged=tensor_hash(visual)==frozen,
        module_gradients=all(state['module_gradients'].values()),up_gradient=state['up_gradient_seen'],
        full_budget=state['steps']==POLICY['steps'],checkpoint_roundtrip=parameter_hash(model)==final,
        nonzero_parameter_updates=final!=initial,
        auxiliary_unchanged=tensor_hash(model.neck.aux)==state['initial_auxiliary_sha256'],
        optimizer_clocks=all(int(optimizer.state[p]['step'])==POLICY['steps'] for p in active.values()))
    if not all(checks.values()):raise ValueError('Adaptation mechanics failed: '+str(checks))
    optimizer.zero_grad(set_to_none=True);model.eval()
    evaluation_args=SimpleNamespace(**{**vars(args),'root':args.reference})
    evaluate(evaluation_args,model,args.output,True)
    write(args.output/'summary.json',dict(arm=args.arm,checks=checks,settings=settings,steps=state['steps'],
        final_parameter_sha256=final,initial_parameter_sha256=initial,rows=state['rows'],scope=POLICY['scope']))
    write(args.output/'progress.json',dict(phase='complete',steps=state['steps']))
    seal(args.output)
    print('ADAPTATION ARM COMPLETE',args.arm,flush=True)


def aggregate(args):
    references(args);hashes={};reports={};initials=set()
    for arm in ARMS:
        path=args.root/arm;c=verify(path);s=read(path/'summary.json')
        if c['arm']!=arm or c['policy']!=POLICY or c['code']!={p:code_hash(ROOT/p) for p in SOURCES}:
            raise ValueError('Wrong adaptation contract')
        if s['steps']!=POLICY['steps'] or not all(s['checks'].values()):raise ValueError('Incomplete mechanics')
        initials.add(s['initial_parameter_sha256']);hashes[arm]=sha(path/'completed.json')
        reports[arm]={}
        for size in POLICY['eval_sizes']:
            for dataset in ('msls-val','pitts30k-val'):
                tag=f'{dataset}_{size}'
                row=read(path/(tag+'.json'))
                baseline=read(args.reference/'baseline'/(tag+'.json'))
                warm=read(args.diagnosis/'e2_attenuate'/(tag+'.json'))
                control=read(args.root/'metric_control'/(tag+'.json'))
                reports[arm][tag]=dict(queries=row['queries'],correct=row['correct'],recall=row['recall'],
                    vs_official=exact_paired(baseline['outcomes'],row['outcomes']),
                    vs_e2_warmstart=exact_paired(warm['outcomes'],row['outcomes']),
                    vs_metric_control=exact_paired(control['outcomes'],row['outcomes']))
    if len(initials)!=1:raise ValueError('Arms started from different weights')
    c=dict(policy=POLICY,results=hashes,prior_results=references(args),code={p:code_hash(ROOT/p) for p in SOURCES})
    open_contract(args.output,c,args.resume)
    if (args.output/'completed.json').exists():verify(args.output);return
    candidates={arm:all(reports[arm][f'{dataset}_322'][key]['net']>0
        for dataset in ('msls-val','pitts30k-val') for key in ('vs_official','vs_metric_control'))
        for arm in ('rank_preserve','rank_and_relation')}
    write(args.output/'summary.json',dict(verdict='MATCHED_ADAPTATION_COMPLETE',reports=reports,
        primary_single_seed_candidates=candidates,additional_training_authorized=0,scope=POLICY['scope'],
        limitations='Soft in-batch constraints cannot guarantee unseen full-gallery retention; KL can preserve teacher errors; three groups seed42 only; E2 chosen after prior exposed results; fixed1024 is a short continuation, not full retraining'))
    seal(args.output)
    write(args.root/'progress.json',dict(phase='complete',training_runs=3))
    print('THREE ARMS COMPLETE; no automatic extra training',candidates,flush=True)


def main():
    os.chdir(ROOT);p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('wait_gpu','train','aggregate'))
    p.add_argument('--arm',choices=list(ARMS));p.add_argument('--resume',action='store_true')
    p.add_argument('--root',type=Path,default=Path('logs/conv_official_adapt_v1'));p.add_argument('--output',type=Path)
    p.add_argument('--reference',type=Path,default=Path('logs/conv_official_boq_v1'))
    p.add_argument('--diagnosis',type=Path,default=Path('logs/conv_official_diagnosis_v1'))
    p.add_argument('--official-cache',type=Path,default=Path('.cache/boq_official_v1'))
    p.add_argument('--dino-source',type=Path,default=Path('/home/wt/.cache/torch/hub/facebookresearch_dinov2_main'))
    p.add_argument('--dataset-root',type=Path,default=Path('datasets/gsv_cities'))
    p.add_argument('--source',type=Path,default=Path('logs/slgd_hard_v2_audit'))
    p.add_argument('--teacher',type=Path,default=Path('logs/slgd_v1/teacher_pilot'))
    p.add_argument('--cache',type=Path,default=Path('.cache/slgd_mid_v1_pilot'))
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':p.error('Expose physical GPU1 only')
    if args.root.resolve().parent!=(ROOT/'logs').resolve() or not args.root.name.startswith('conv_official_adapt_'):
        p.error('Use isolated logs/conv_official_adapt_<version>')
    if args.workers<0:p.error('Workers must be nonnegative')
    if args.command=='wait_gpu':wait_gpu(args);return
    if args.command=='train' and args.arm is None:p.error('Need an adaptation arm')
    if args.output is None or args.output.resolve().parent!=args.root.resolve():p.error('Choose a direct child output')
    expected=args.arm if args.command=='train' else 'aggregate'
    if args.output.name!=expected:p.error('Output name must match the registered arm')
    import fcntl
    args.root.mkdir(exist_ok=True)
    with (args.root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        if args.command=='train':torch.cuda.set_per_process_memory_fraction(.65,0)
        write(args.root/'progress.json',dict(phase='running',arm=expected,max_training_runs=3))
        try:
            if args.command=='train':train(args)
            else:aggregate(args)
        except Exception as error:
            failure=dict(phase='failed',arm=expected,type=type(error).__name__,message=str(error),
                time=datetime.datetime.now().astimezone().isoformat())
            write(args.root/'progress.json',failure)
            if args.output.exists() and not (args.output/'completed.json').exists():write(args.output/'failure.json',failure)
            raise


if __name__=='__main__':main()
