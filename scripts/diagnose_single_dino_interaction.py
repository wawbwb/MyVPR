"""Read-only trained-model diagnosis. No optimizer, update, or new training.

Outputs only new diagnostic artifacts. Existing checkpoints/source/seals remain
immutable. Same-weight ablations are sensitivity tests, not retrained controls.
"""
import argparse
from contextlib import contextmanager
import copy
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts import single_dino_interaction as study
from scripts import eval_clip_key_study as metrics
from scripts.eval_spatial_adapter_repeat import RGBImages
from scripts.semantic_region_study import PlaceSteps
from src.models.single_dino_interaction import SingleDinoInteraction
from src.single_dino_diagnosis import stats, selected_queries, candidate_union, margin_report
from src.semantic_region_protocol import read, write, sha, state_hash, owned_run, gps_retrieval

SOURCES = ('scripts/diagnose_single_dino_interaction.py', 'src/single_dino_diagnosis.py',
           'tests/test_single_dino_diagnosis.py')
VARIANTS = ('innovation1', 'separate', 'aligned', 'permuted', 'same_weight_separate',
            'same_weight_permuted', 'no_exchange', 'no_context', 'no_scan', 'spatial_mean', 'centered')


def relative(a, b):
    return a.flatten(1).norm(dim=1) / b.flatten(1).norm(dim=1).clamp_min(1e-12)


@contextmanager
def intervention(fusion, route=None, disable=()):
    old = fusion.arm
    hooks = []
    try:
        if route is not None:
            fusion.arm = route
        for name in disable:
            hooks.append(getattr(fusion, name).register_forward_hook(lambda m, i, out: torch.zeros_like(out)))
        yield
    finally:
        fusion.arm = old
        for hook in hooks:
            hook.remove()


def observe(scan, store):
    def hook(module, inputs, output):
        tokens = inputs[0]
        x, z = module.in_proj(tokens).chunk(2, -1)
        x = F.silu(module.conv(x.transpose(1, 2))[..., :tokens.shape[1]].transpose(1, 2))
        skip_only = module.out((x * module.skip) * F.silu(z))
        store['scan_output_over_normalized_input'].extend(relative(output, tokens).cpu().tolist())
        store['recurrence_over_scan_output'].extend(relative(output-skip_only, output).cpu().tolist())
    return scan.register_forward_hook(hook)


def branch_group(name):
    return name.split('.')[0]


def gradient_audit(model, features, labels, coords, initial_state, c):
    """Independent full-batch loss gradients replayed in microbatches, no step."""
    from src.losses.vpr_losses import VPRLossFunction
    from src.models.hetero_interleave import replay_backward
    model.fusion.requires_grad_(True)
    lossfn = VPRLossFunction()
    with torch.no_grad():
        z0 = torch.cat([model.anchor.readout(x) for x in features[2].split(2)])
        outputs = [model.describe(tuple(f[i:i+2] for f in features))[:3] for i in range(0, len(labels), 2)]
        leaves = [torch.cat([o[j] for o in outputs]).detach().requires_grad_() for j in range(3)]
    losses = dict(main=gps_retrieval(lossfn, leaves[0], labels, coords),
        auxiliary=c['aux_weight']*sum(gps_retrieval(lossfn, z, labels, coords) for z in leaves[1:]),
        keep=c['keep_weight']*(1-(leaves[0]*z0).sum(1)).clamp_min(0).mean())
    gradients, norms = {}, {}
    for name, loss in losses.items():
        leaf_grad = torch.autograd.grad(loss, leaves, allow_unused=True, retain_graph=True)
        leaf_grad = [torch.zeros_like(z) if g is None else g for z, g in zip(leaves, leaf_grad)]
        model.fusion.zero_grad(set_to_none=True)
        replay_backward(model, features, leaf_grad, 2)
        gradients[name] = {n: (torch.zeros_like(p) if p.grad is None else p.grad).detach().cpu().clone()
                           for n, p in model.fusion.named_parameters()}
        groups = {}
        for n, g in gradients[name].items():
            group = branch_group(n)
            groups[group] = groups.get(group, 0.) + float(g.double().square().sum())
        norms[name] = dict(loss=float(loss.detach()), gradient_l2_by_group={n: v**.5 for n, v in groups.items()})
    cosines = {}
    for other in ('auxiliary', 'keep'):
        dot = sum(float((g.double()*gradients[other][n].double()).sum()) for n, g in gradients['main'].items())
        aa = sum(float(g.double().square().sum()) for g in gradients['main'].values())
        bb = sum(float(g.double().square().sum()) for g in gradients[other].values())
        cosines[other] = dot/(aa*bb)**.5 if aa*bb > 0 else None
    fresh = SingleDinoInteraction('aligned', dim=c['dim'], rank=c['rank'], rho=c['rho']).to(c['device'])
    fresh.load_state_dict(initial_state, strict=True)
    with torch.no_grad():
        initial_aux = [fresh(features[2][i:i+2])[1] for i in range(0, len(labels), 2)]
        aux0 = c['aux_weight']*sum(gps_retrieval(lossfn, torch.cat([a[j] for a in initial_aux]), labels, coords) for j in range(2))
        base_main = gps_retrieval(lossfn, z0, labels, coords)
    model.fusion.zero_grad(set_to_none=True)
    model.fusion.requires_grad_(False)
    return dict(weighted_components=norms, main_gradient_cosine_with=cosines,
        initial_same_batch_main=float(base_main), initial_same_batch_weighted_auxiliary=float(aux0),
        batch='Max saved aligned main loss among final128 training rows, P16/K4, all64; no optimizer/update; label-informed explanatory selection',
        auxiliary_targets='Context/local GAP descriptors BEFORE exchange_scan; no direct auxiliary gradient to exchange_scan')


def run(c, a):
    engine = study.bound_engine()
    run = study.key.path(c, 'runs')
    engine.verified(run/'train'); engine.verified(run/'eval')
    if engine.verified(run/'train') != study.contract_for(c, 'train'):
        raise ValueError('Sealed training identity changed')
    from src.dataloaders.valid.mapillary_sls import MapillarySLSDataset
    d = MapillarySLSDataset(ROOT/'datasets/msls-val')
    report = read(run/'eval/msls-val.json')
    names = np.concatenate((d.dbImages, d.qImages)).tolist()
    if report['dataset']['order'] != metrics.order_digest(names) or report['dataset']['gt'] != metrics.gt_digest(d.ground_truth):
        raise ValueError('Dataset order/GT changed')
    queries = selected_queries(report)
    refs = candidate_union(report, queries, d.ground_truth)
    contract = dict(schema='single_dino_diagnosis_v1', identity=study.identity(c),
        train_seal=sha(run/'train/completed.json'), eval_seal=sha(run/'eval/completed.json'),
        checkpoint=sha(run/'train/last.pt'), code=study.key.source_hashes(SOURCES),
        queries=queries, references=refs, variants=VARIANTS,
        scope='Label-informed fixed64 explanatory sample; all original errors included. Candidate-union counterfactuals NOT full-DB recall, no training, no test access')
    with owned_run(a.output, contract, a.resume):
        if (a.output/'completed.json').exists():
            seal=read(a.output/'completed.json')
            if not seal['complete'] or any(sha(a.output/f)!=h for f,h in seal['files'].items()): raise ValueError('Changed diagnostic output')
            print('Already complete'); return
        study.key.gpu(c)
        models = study.build(c)
        initial_state = copy.deepcopy(models['aligned'].fusion.state_dict())
        payload = torch.load(run/'train/last.pt', map_location='cpu', weights_only=True)
        frozen = state_hash(models['aligned'].anchor)
        if payload['state']['step'] != c['pilot_steps'] or payload['state']['contract_sha256'] != sha(run/'train/contract.json'):
            raise ValueError('Wrong final checkpoint')
        for arm, model in models.items():
            model.fusion.load_state_dict(payload['fusion'][arm], strict=True)
            if state_hash(model.fusion) != read(run/'train/summary.json')['final_fusion_sha256'][arm]:
                raise ValueError('Wrong arm tensors')
            model.requires_grad_(False).eval()
        before={arm:state_hash(m.fusion) for arm,m in models.items()}
        model=models['aligned']; fusion=model.fusion
        # Reuse frozen encoder outputs; no weights or old reports are written.
        image_ids=refs+[d.num_references+q for q in queries]
        loader=DataLoader(RGBImages(ROOT/'datasets/msls-val',[names[i] for i in image_ids],None),
            batch_size=2,shuffle=False,num_workers=2)
        arrays={v:np.empty((len(image_ids),12288),np.float32) for v in VARIANTS}
        sampled={v:[] for v in VARIANTS if v!='innovation1'}
        mechanisms={n:[] for n in ('residual_relative','spatial_mean_energy_fraction','no_exchange_residual_change_relative',
            'no_scan_residual_change_relative','route_residual_change_relative','tanh_saturation_fraction')}
        scan_stats={name:dict(scan_output_over_normalized_input=[],recurrence_over_scan_output=[])
            for name in ('context_scan','exchange_scan')}
        started=time.monotonic()
        with torch.inference_mode():
            for batch,(pixels,ids,_) in enumerate(tqdm(loader,desc='Fixed checkpoint candidate-union diagnosis')):
                features=model.encode(pixels.to(c['device'])); anchor=features[2]
                base=model.anchor.readout(anchor)
                maps={}
                for arm, m in models.items(): maps[arm]=m.fusion(anchor)[0]
                for label,route,disable in [('same_weight_separate','separate',()),('same_weight_permuted','permuted',()),
                        ('no_exchange',None,('exchange_scan',)),('no_context',None,('context_scan',)),
                        ('no_scan',None,('context_scan','exchange_scan'))]:
                    with intervention(fusion,route,disable): maps[label]=fusion(anchor)[0]
                residual=maps['aligned']-anchor
                maps['spatial_mean']=anchor+residual.mean((2,3),keepdim=True)
                maps['centered']=maps['aligned']-residual.mean((2,3),keepdim=True)
                values={'innovation1':base,**{v:model.anchor.readout(m) for v,m in maps.items()}}
                for v,z in values.items():
                    if not torch.isfinite(z).all(): raise ValueError('Nonfinite descriptor')
                    arrays[v][ids.numpy()]=z.cpu().numpy()
                if batch<32:  # first64 sorted reference images, declared non-random mechanics sample
                    for v,m in maps.items(): sampled[v].extend(relative(m-anchor,anchor).cpu().tolist())
                    mechanisms['residual_relative'].extend(relative(residual,anchor).cpu().tolist())
                    energy=residual.square().sum((1,2,3)).clamp_min(1e-20)
                    mechanisms['spatial_mean_energy_fraction'].extend((residual.mean((2,3),keepdim=True).square().sum((1,2,3))*anchor.shape[-1]*anchor.shape[-2]/energy).cpu().tolist())
                    for name,v in [('no_exchange_residual_change_relative','no_exchange'),('no_scan_residual_change_relative','no_scan'),('route_residual_change_relative','same_weight_permuted')]:
                        mechanisms[name].extend(relative(maps[v]-maps['aligned'],residual).cpu().tolist())
                    scale=c['rho']*anchor.square().mean(1,keepdim=True).sqrt().clamp_min(1e-20)
                    mechanisms['tanh_saturation_fraction'].extend(((residual/scale).abs()>.99).float().mean((1,2,3)).cpu().tolist())
                    hooks=[observe(getattr(fusion,n),s) for n,s in scan_stats.items()]
                    try: fusion(anchor)
                    finally:
                        for h in hooks:h.remove()
                write(a.output/'progress.json',dict(phase='extraction',images=int(ids[-1])+1,total=len(image_ids),elapsed_seconds=time.monotonic()-started))
        reports={}
        ref_top1=report['variants']['innovation1']['predictions']
        baseline_top1=[row[0] for row in ref_top1]
        for v,x in arrays.items():
            reports[v]=margin_report(x[len(refs):]@x[:len(refs)].T, refs,queries,d.ground_truth,baseline_top1)
        checks={}
        for arm in ('innovation1','separate','aligned','permuted'):
            checks[arm+'_original_top1_reproduced']=reports[arm]['top1']==[report['variants'][arm]['predictions'][q][0] for q in queries]
        base=arrays['innovation1']; aligned=arrays['aligned']
        descriptor_changes={}
        for v,x in arrays.items():
            if v=='innovation1':continue
            drift=np.linalg.norm(aligned-base,axis=1)
            descriptor_changes[v]=dict(l2_vs_anchor=stats(np.linalg.norm(x-base,axis=1)),
                l2_vs_aligned=stats(np.linalg.norm(x-aligned,axis=1)),
                relative_to_aligned_descriptor_drift=stats(np.linalg.norm(x-aligned,axis=1)/np.maximum(drift,1e-12)))
        # A saved hard TRAINING batch, not validation-data gradient fitting.
        # Avoid the known zero-main-loss final row; disclose selection explicitly.
        plan=study.key.open_plan(c)
        history=read(run/'train/summary.json')['history']
        selected=max(history[-128:],key=lambda row:row['aligned']['main'])
        row_index=selected['step']-1
        images,labels,coords,_=PlaceSteps(c,plan,plan['schedule'][row_index:row_index+1])[0]
        with torch.no_grad():
            chunks=[model.encode(x.to(c['device'])) for x in images.split(2)]
            features=tuple(torch.cat([x[j] for x in chunks]) for j in range(6))
        gradients=gradient_audit(model,features,labels.to(c['device']),coords,initial_state,c)
        gradients['selected_training_step']=selected['step']
        gradients['saved_step_main']=selected['aligned']['main']
        checks.update(frozen_anchor_unchanged=state_hash(model.anchor)==frozen,
            fusion_tensors_unchanged=before=={arm:state_hash(m.fusion) for arm,m in models.items()},
            checkpoint_unchanged=sha(run/'train/last.pt')==contract['checkpoint'],
            source_unchanged=study.key.source_hashes(study.NEW_SOURCES)=={n:contract['identity']['code'][n] for n in study.NEW_SOURCES},
            protected_sources_unchanged=study.key.source_hashes(study.key.PROTECTED)==contract['identity']['protected_sources'])
        summaries={}
        for v,r in reports.items():
            old=reports['innovation1']; delta=np.asarray(r['positive_negative_margin'])-np.asarray(old['positive_negative_margin'])
            errors=~np.asarray(old['correct'])
            summaries[v]=dict(conditional_correct=sum(r['correct']), margin=stats(r['positive_negative_margin']),
                margin_change_vs_anchor=stats(delta), error_margin_change_vs_anchor=stats(delta[errors]),
                error_margin_improved=int((delta[errors]>1e-6).sum()),error_margin_worsened=int((delta[errors]<-1e-6).sum()),
                error_margin_to_aligned=stats(np.asarray(r['positive_negative_margin'])[errors]-np.asarray(reports['aligned']['positive_negative_margin'])[errors]))
        write(a.output/'per_query.json',dict(query_ids=queries,references=refs,variants=reports))
        write(a.output/'summary.json',dict(checks=checks,queries=len(queries),baseline_error_queries=sum(not t for t in reports['innovation1']['correct']),
            reference_union=len(refs),images=len(image_ids),mechanisms={k:stats(v) for k,v in mechanisms.items()},
            scan={n:{k:stats(v) for k,v in s.items()} for n,s in scan_stats.items()},
            feature_change_vs_anchor={v:stats(x) for v,x in sampled.items()},descriptors=descriptor_changes,
            candidate_union=summaries,gradients=gradients,scope=contract['scope'],training_started=False,
            mechanics_sample='First64 sorted candidate-reference images',elapsed_seconds=time.monotonic()-started))
        if not all(checks.values()):raise ValueError('Diagnosis integrity/reproduction failed: '+str(checks))
        write(a.output/'completed.json',dict(complete=True,schema='single_dino_diagnosis_v1',
            files={p.relative_to(a.output).as_posix():sha(p) for p in sorted(a.output.rglob('*'))
                if p.is_file() and p.name not in ('.lock','completed.json') and not p.name.endswith('.tmp')}))
        print('DIAGNOSIS COMPLETE',a.output,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'logs/single_dino_diagnosis_v1')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();c=study.configuration()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':p.error('PhysicalGPU1 only')
    import fcntl
    with (ROOT/'logs/query_relation_screen_v1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import subprocess
        used=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=20)
        if int(used.strip())>4096:raise RuntimeError('GPU1 busy; no processes killed')
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        run(c,a)


if __name__=='__main__':main()
