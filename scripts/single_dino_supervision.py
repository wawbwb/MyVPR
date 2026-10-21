"""Prepared V2 only. No launcher/queue; train AND smoke need explicit consent.

Fixed V1 inference structure, post-exchange auxiliary readout, and per-block
auxiliary gradient cap. Old experiments/innovation1 are read-only inputs.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scripts import single_dino_interaction as old
from src.models.single_dino_interaction import ARMS
from src.models.single_dino_supervision import PostInteractionVPR
from src.models.hetero_interleave import replay_backward
from src.single_dino_grad_balance import combine
from src.semantic_region_protocol import (read,write,sha,state_hash,owned_run,seed_all,
    save_torch,rng_state,restore_rng,gps_retrieval)

SCHEMA='single_dino_supervision_v2'
SOURCES=('src/models/single_dino_supervision.py','src/single_dino_grad_balance.py',
    'scripts/single_dino_supervision.py','config/single_dino_supervision.json',
    'tests/test_single_dino_supervision.py')


def configuration():
    c=read(ROOT/'config/single_dino_supervision.json')
    previous=old.configuration()
    expected=dict(previous,schema=SCHEMA,runs='logs/single_dino_supervision_v2',aux_gradient_max_ratio=.5)
    if c!=expected:raise ValueError('Registered V2 budget/supervision protocol changed')
    return c


def build(c):
    anchor=old.make_anchor(c);models={}
    for arm in ARMS:
        seed_all(c['seed'])
        models[arm]=PostInteractionVPR(anchor,arm,dim=c['dim'],rank=c['rank'],rho=c['rho']).to(c['device'])
    if len({state_hash(m.fusion) for m in models.values()})!=1:raise ValueError('Different fusion initialization')
    ids=[set(map(id,m.fusion.parameters())) for m in models.values()]
    if any(ids[i]&ids[j] for i in range(3) for j in range(i)):raise ValueError('Trainable heads shared')
    return models


def identity(c):
    original=old.configuration();previous=old.key.path(original,'runs')
    engine=old.bound_engine()
    for stage in ('train','eval'):
        engine.verified(previous/stage)
        if not all(read(previous/stage/'summary.json')['checks'].values()):raise ValueError('Invalid V1 baseline')
    value=old.identity(c)
    value.update(schema=SCHEMA,new_code=old.key.source_hashes(SOURCES),
        predecessor={stage:sha(previous/stage/'completed.json') for stage in ('train','eval')},
        inference_path='V1 exact same tensors/rho; auxiliary endpoints moved AFTER exchange',
        optimization='Per-block weighted auxiliary L2 <= 0.5 * main L2; keep unchanged; global clip AFTER combine',
        start='Fresh original seed42 initialization from frozen innovation1, NOT resumed V1 fusion')
    return value


def contract_for(c,stage):
    return dict(stage=stage,**identity(c),scope='Same1024-step single-seed development pilot. V1 vs V2 jointly changes supervision endpoint and gradient cap, not isolated attribution of each fix.')


def verified(output):
    result=read(output/'completed.json')
    if result.get('complete') is not True or result.get('schema')!=SCHEMA:raise ValueError('Incomplete/wrong V2 schema')
    for name,h in result['files'].items():
        p=(output/name).resolve()
        if not p.is_relative_to(output.resolve()) or sha(p)!=h:raise ValueError('Changed V2 artifact: '+name)
    return read(output/'contract.json')


def finish(output):
    files={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*'))
        if p.is_file() and p.name not in ('.lock','completed.json') and not p.name.endswith('.tmp')}
    write(output/'completed.json',dict(complete=True,schema=SCHEMA,files=files))


def component_gradients(model,features,leaves,losses,microbatch):
    """Full-batch descriptor losses; exact micro replay, no batch-local mining."""
    result={}
    for component,loss in losses.items():
        gradients=torch.autograd.grad(loss,leaves,allow_unused=True,retain_graph=True)
        gradients=[torch.zeros_like(z) if g is None else g for z,g in zip(leaves,gradients)]
        model.fusion.zero_grad(set_to_none=True)
        replay_backward(model,features,gradients,microbatch)
        result[component]={n:(torch.zeros_like(p) if p.grad is None else p.grad).detach().clone()
            for n,p in model.fusion.named_parameters()}
    return result


def train(c,a):
    if not a.allow_training:raise ValueError('Explicit --allow-training required for smoke/train')
    from scripts.semantic_region_study import PlaceSteps
    from src.losses.vpr_losses import VPRLossFunction
    plan=old.key.open_plan(c);contract=contract_for(c,a.command);runs=old.key.path(c,'runs')
    if a.command=='train':
        smoke=verified(runs/'smoke')
        if ({k:v for k,v in smoke.items() if k!='stage'}!={k:v for k,v in contract.items() if k!='stage'}
                or not all(read(runs/'smoke/summary.json')['checks'].values())):raise ValueError('Same-version smoke required')
    output=runs/a.command
    with owned_run(output,contract,a.resume):
        if (output/'completed.json').exists():verified(output);print('Already complete');return
        old.key.gpu(c);models=build(c);first=models['aligned']
        initial=dict(frozen=state_hash(first.anchor),fusion={k:state_hash(m.fusion) for k,m in models.items()},
            parameters={k:sum(p.numel() for p in m.fusion.parameters()) for k,m in models.items()})
        if (output/'initial.json').exists() and read(output/'initial.json')!=initial:raise ValueError('Initialization differs')
        write(output/'initial.json',initial)
        with torch.no_grad():
            pixels=PlaceSteps(c,plan,plan['schedule'][:1])[0][0][:2].to(c['device'])
            f=first.encode(pixels);z=first.anchor.readout(f[2])
            errors={arm:float((m.describe(f)[0]-z).abs().max()) for arm,m in models.items()}
        if max(errors.values())>1e-6:raise ValueError('Zero start differs')
        del pixels,f,z
        optimizers={k:torch.optim.AdamW(m.fusion.parameters(),lr=c['learning_rate'],weight_decay=0.) for k,m in models.items()}
        state=dict(step=0,history=[],contract_sha256=sha(output/'contract.json'),
            gradients={k:{n:False for n,p in m.fusion.named_parameters()} for k,m in models.items()},
            caps_satisfied=True,post_exchange_auxiliary_gradient_seen={k:False for k in ARMS})
        def save():
            save_torch(output/'last.pt',dict(fusion={k:m.fusion.state_dict() for k,m in models.items()},
                optimizers={k:o.state_dict() for k,o in optimizers.items()},state=state,rng=rng_state()))
        if (output/'last.pt').exists():
            payload=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
            if payload['state']['contract_sha256']!=state['contract_sha256']:raise ValueError('Resume differs')
            for k,m in models.items():
                m.fusion.load_state_dict(payload['fusion'][k],strict=True)
                optimizers[k].load_state_dict(payload['optimizers'][k])
            state=payload['state'];restore_rng(payload['rng'])
        else:seed_all(c['seed']);save()
        total=c['smoke_steps'] if a.command=='smoke' else c['pilot_steps']
        if not 0<=state['step']<=total:raise ValueError('Invalid resume step')
        loader=DataLoader(PlaceSteps(c,plan,plan['schedule'][state['step']:total]),batch_size=None,
            num_workers=c['workers'],generator=torch.Generator().manual_seed(c['seed']))
        lossfn=VPRLossFunction()
        for m in models.values():m.train()
        for images,labels,coords,_ in tqdm(loader,total=total,initial=state['step'],desc='post-exchange balanced '+a.command):
            started=time.monotonic();labels=labels.to(c['device'])
            with torch.no_grad():
                chunks=[first.encode(x.to(c['device'])) for x in images.split(c['microbatch'])]
                features=tuple(torch.cat([f[j] for f in chunks]) for j in range(6))
                reference=torch.cat([first.anchor.readout(f[2]) for f in chunks])
            row={}
            for arm,m in models.items():
                with torch.no_grad():
                    temporary=[m.describe(f) for f in chunks]
                    leaves=[torch.cat([v[j] for v in temporary]).detach().requires_grad_() for j in range(3)]
                    ratio=torch.cat([v[3]['relative_residual'] for v in temporary])
                del temporary
                if not torch.isfinite(ratio).all() or float(ratio.max())>c['rho']+1e-6:raise ValueError('Residual over cap')
                losses=dict(main=gps_retrieval(lossfn,leaves[0],labels,coords),
                    auxiliary=c['aux_weight']*sum(gps_retrieval(lossfn,z,labels,coords) for z in leaves[1:]),
                    keep=c['keep_weight']*(1-(leaves[0]*reference).sum(1)).clamp_min(0).mean())
                if not all(bool(torch.isfinite(v)) for v in losses.values()):raise ValueError('Nonfinite loss')
                parts=component_gradients(m,features,leaves,losses,c['microbatch'])
                gradients,balance=combine(parts['main'],parts['auxiliary'],parts['keep'],c['aux_gradient_max_ratio'])
                cap_ok=all(v['cap_satisfied'] for v in balance.values())
                state['caps_satisfied']&=cap_ok
                if not cap_ok:raise ValueError('Auxiliary gradient cap violated')
                state['post_exchange_auxiliary_gradient_seen'][arm]|=balance['exchange_scan']['capped_auxiliary_l2']>0
                optimizers[arm].zero_grad(set_to_none=True)
                for n,p in m.fusion.named_parameters():
                    p.grad=gradients[n]
                    state['gradients'][arm][n]|=bool((p.grad!=0).any())
                norm=torch.nn.utils.clip_grad_norm_(m.fusion.parameters(),c['gradient_clip'],error_if_nonfinite=True)
                optimizers[arm].step()
                row[arm]=dict(losses={n:float(v.detach()) for n,v in losses.items()},balance=balance,
                    gradient_norm=float(norm),relative_residual_max=float(ratio.max()))
                del leaves,losses,parts,gradients
            torch.cuda.synchronize();state['step']+=1
            row.update(step=state['step'],seconds=time.monotonic()-started,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
            state['history'].append(row);write(output/'progress.json',dict(phase='training',total=total,**row))
            if state['step']%c['checkpoint_every']==0:save()
            del chunks,features,reference
        save()
        checks=dict(complete_steps=state['step']==total,zero_start=max(errors.values())<=1e-6,
            frozen_anchor_unchanged=state_hash(first.anchor)==initial['frozen'],
            matched_capacity=len(set(initial['parameters'].values()))==1,
            all_fusions_updated=all(state_hash(m.fusion)!=initial['fusion'][k] for k,m in models.items()),
            all_parameters_gradient_seen=all(all(v.values()) for v in state['gradients'].values()),
            capped_post_exchange_auxiliary_gradient_seen=all(state['post_exchange_auxiliary_gradient_seen'].values()),
            auxiliary_caps_satisfied=state['caps_satisfied'],
            innovation1_sources_unchanged=old.key.source_hashes(old.key.PROTECTED)==contract['protected_sources'])
        write(output/'summary.json',dict(checks=checks,steps=state['step'],zero_start_errors=errors,
            final_fusion_sha256={k:state_hash(m.fusion) for k,m in models.items()},
            parameter_inventory=initial['parameters'],history=state['history'],encoder_count=1,
            verdict='SMOKE_DISCARD' if a.command=='smoke' else 'TRAINED_NOT_EVALUATED'))
        if not all(checks.values()):raise ValueError('Mechanical check failed: '+str(checks))
        finish(output)


def evaluator():
    # Private module only. Original V1 globals and on-disk files are untouched.
    spec=importlib.util.spec_from_file_location('_single_dino_v2_evaluation',ROOT/'scripts/single_dino_interaction.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.identity=identity;module.contract_for=contract_for;module.build=build
    module.bound_engine=lambda:SimpleNamespace(verified=verified,finish=finish)
    return module.evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('plan','smoke','train','eval'))
    p.add_argument('--allow-training',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();c=configuration()
    # Guard before identity verification, locks, directories or CUDA work.
    if a.command in ('smoke','train') and not a.allow_training:p.error('Explicit --allow-training required')
    if a.command=='plan':
        plan=old.key.open_plan(c);info=identity(c)
        print(dict(status='PREPARED_NOT_STARTED',training_started=False,steps=c['pilot_steps'],arms=ARMS,
            train_places=len(plan['places'][:8192]),auxiliary_location='post_exchange',auxiliary_gradient_max_ratio=.5,
            frozen_anchor=info['anchor_checkpoint_sha256'],output=str(old.key.path(c,'runs'))));return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':p.error('Physical GPU1 only')
    import fcntl
    with (ROOT/'logs/query_relation_screen_v1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old.require_ready(ROOT,c)
        import subprocess
        used=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=20)
        if int(used.strip())>4096:raise RuntimeError('GPU1 busy; no other tasks killed')
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        if a.command=='eval':evaluator()(c,a)
        else:train(c,a)


if __name__=='__main__':main()
