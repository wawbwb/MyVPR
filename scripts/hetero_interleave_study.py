"""Matched three-arm ReID-inspired pilot on a frozen, sealed innovation-1.

No CLIP/SAM, no candidate reranking, no change to the old innovation source.
Default plan is CPU/file-only. Training and evaluation require explicit action.
"""
import argparse
from pathlib import Path
import sys
import time
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts import clip_key_study as key
from scripts.semantic_region_study import PlaceSteps
from src.models.hetero_interleave import ARMS, HeterogeneousVPR, replay_backward
from src.models.resnet_module_transfer import load_official_resnet, verify_resnet_download
from src.models.official_boq import verify_download
from src.semantic_region_protocol import (read, write, sha, state_hash, owned_run,
    seed_all, save_torch, rng_state, restore_rng, gps_retrieval)

SCHEMA = 'hetero_interleave_v1'
SOURCES = ('src/models/hetero_interleave.py', 'scripts/hetero_interleave_study.py',
    'scripts/launch_hetero_interleave.py', 'tests/test_hetero_interleave.py',
    'config/hetero_interleave.json', 'src/models/scan_mamba.py',
    'src/models/resnet_module_transfer.py', 'src/semantic_region_protocol.py',
    'scripts/semantic_region_study.py', 'scripts/conv_official_diagnosis.py',
    'src/official_transfer_diagnosis.py', 'src/official_layer_selection.py',
    'src/hcen_plain_protocol.py', 'src/models/hcen_followup.py',
    'scripts/train_slgd.py', 'scripts/clip_key_study.py', 'src/losses/vpr_losses.py',
    'scripts/eval_clip_key_study.py', 'scripts/eval_spatial_adapter_repeat.py',
    *key.PROTECTED)
path = key.path


def configuration():
    c = read(ROOT/'config/hetero_interleave.json')
    expected = dict(schema=SCHEMA, seed=42, train_places=8192, places_per_batch=16,
        views=4, epochs=5, pilot_steps=1024, student_size=322, dim=64, rank=16,
        rho=.025, learning_rate=.0001, keep_weight=.1, aux_weight=.1,
        microbatch=2, smoke_steps=8, checkpoint_every=16, precision='fp32',
        device='cuda:0', cuda_memory_fraction=.2)
    if any(c.get(k) != v for k, v in expected.items()):
        raise ValueError('Registered pilot policy changed')
    return c


def finish(output):
    files = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*'))
        if p.is_file() and p.name not in ('.lock', 'completed.json') and not p.name.endswith('.tmp')}
    write(output/'completed.json', dict(complete=True, schema=SCHEMA, files=files))


def verified(output):
    d = read(output/'completed.json')
    if d.get('complete') is not True or d.get('schema') != SCHEMA:
        raise ValueError('Incomplete/wrong-schema output: '+str(output))
    for n, digest in d['files'].items():
        p = (output/n).resolve()
        if not p.is_relative_to(output.resolve()) or sha(p) != digest:
            raise ValueError('Changed artifact: '+str(p))
    return read(output/'contract.json')


def identity(c):
    from scripts.train_slgd import verify as old_verify
    base = path(c, 'anchor_run')
    old_verify(base)
    summary = read(base/'summary.json')
    if (summary['arm'] != 'c_cross_condition' or not all(summary['checks'].values())
            or summary['final_parameter_sha256'] != c['anchor_parameter_sha256']):
        raise ValueError('Expected final archived innovation-1, not another baseline')
    resnet = verify_resnet_download(path(c, 'resnet_cache'))
    if resnet['weight_sha256'] != c['resnet_sha256']:
        raise ValueError('Pinned official ResNet checkpoint differs')
    return dict(schema=SCHEMA, config=c, code=key.source_hashes(SOURCES),
        protected_sources=key.source_hashes(key.PROTECTED),
        anchor_checkpoint_sha256=sha(base/'last.pt'), anchor_completed_sha256=sha(base/'completed.json'),
        official=verify_download(path(c, 'official_cache')), resnet=resnet,
        plan_sha256=sha(path(c, 'plan')/'plan.json'), plan_completed_sha256=sha(path(c, 'plan')/'completed.json'),
        dino_sources={p.relative_to(path(c, 'dino_source')).as_posix(): sha(p)
            for p in sorted(path(c, 'dino_source').rglob('*.py'))},
        versions={n: __import__('importlib.metadata', fromlist=['version']).version(n)
            for n in ('torch', 'torchvision', 'numpy', 'pillow', 'pytorch-metric-learning')},
        private_reference_sha256='98034e48211800766f4aa0230560315660182636ae50858187c2933ceebac63c')


def make_anchor(c):
    from scripts.conv_official_diagnosis import author, make_adapter, load_parameters
    from scripts.train_slgd import parameter_hash
    from src.models.conditional_complement import AnchoredConditionalNeck
    from src.official_layer_selection import selected_features
    args = SimpleNamespace(official_cache=path(c, 'official_cache'),
        dino_source=path(c, 'dino_source'), reference=path(c, 'reference'))
    visual = author(args)
    base = make_adapter(visual, c['rho'])
    base.neck = AnchoredConditionalNeck(base.neck, 'c_cross_condition').to(c['device'])
    base.fusion_layers = (8, 10, 12)
    base.features = MethodType(selected_features, base)
    state = load_parameters(base, path(c, 'anchor_run')/'last.pt')
    if parameter_hash(base) != c['anchor_parameter_sha256'] or state['steps'] != 2048:
        raise ValueError('Archived innovation-1 tensors were not reconstructed')
    return base.requires_grad_(False).eval()


def build(c):
    anchor = make_anchor(c)
    cnn = load_official_resnet(path(c, 'resnet_cache'), c['device']).backbone
    models = {}
    for arm in ARMS:
        seed_all(c['seed'])
        models[arm] = HeterogeneousVPR(anchor, cnn, arm, dim=c['dim'], rank=c['rank'], rho=c['rho']).to(c['device'])
    hashes = {state_hash(m.fusion) for m in models.values()}
    if len(hashes) != 1:
        raise ValueError('Arms have different new-parameter initialization')
    parameter_ids = [set(map(id, m.fusion.parameters())) for m in models.values()]
    if any(parameter_ids[i] & parameter_ids[j] for i in range(3) for j in range(i)):
        raise ValueError('Fusion parameters are accidentally shared')
    return models


def frozen_hashes(model):
    return dict(anchor=state_hash(model.anchor), resnet=state_hash(model.resnet))


def contract_for(c, stage):
    return dict(stage=stage, **identity(c),
        schedule='First1024 rows of sealed seed42 GSV five-epoch plan; same four views; no holdout training',
        scope='Single-seed fixed-budget three-arm architecture pilot on exposed development data; not independent test/novelty/stable-gain proof')


def train(c, a):
    if not a.allow_training:
        raise ValueError('Explicit --allow-training required')
    plan = key.open_plan(c)
    contract = contract_for(c, a.command)
    runs = path(c, 'runs')
    if a.command == 'train':
        smoke = runs/'smoke'
        old = verified(smoke)
        if ({k:v for k,v in old.items() if k != 'stage'} != {k:v for k,v in contract.items() if k != 'stage'}
                or not all(read(smoke/'summary.json')['checks'].values())):
            raise ValueError('Current-code same-input smoke must pass; smoke updates are discarded')
    output = runs/a.command
    with owned_run(output, contract, a.resume):
        if (output/'completed.json').exists():
            verified(output); print('Already complete', a.command); return
        key.gpu(c)
        models = build(c)
        first = models[ARMS[0]]
        initial = dict(frozen=frozen_hashes(first), fusion={arm: state_hash(m.fusion) for arm,m in models.items()},
            parameters={arm: sum(p.numel() for p in m.fusion.parameters()) for arm,m in models.items()})
        if (output/'initial.json').exists() and read(output/'initial.json') != initial:
            raise ValueError('Reconstructed initial tensors differ')
        write(output/'initial.json', initial)
        pixels = PlaceSteps(c, plan, plan['schedule'][:1])[0][0][:2].to(c['device'])
        with torch.no_grad():
            features = first.encode(pixels)
            anchor_z = first.anchor.readout(features[2])
            errors = {arm: float((m.describe(features)[0]-anchor_z).abs().max()) for arm,m in models.items()}
        if max(errors.values()) > 1e-6:
            raise ValueError('Zero-start differs from innovation-1')
        del pixels, features, anchor_z
        optimizers = {arm: torch.optim.AdamW(m.fusion.parameters(), lr=c['learning_rate'], weight_decay=0.)
            for arm,m in models.items()}
        state = dict(step=0, history=[], contract_sha256=sha(output/'contract.json'),
            gradients={arm: {name: False for name,p in m.fusion.named_parameters()} for arm,m in models.items()})
        def save():
            save_torch(output/'last.pt', dict(fusion={arm:m.fusion.state_dict() for arm,m in models.items()},
                optimizers={arm:opt.state_dict() for arm,opt in optimizers.items()}, state=state, rng=rng_state()))
        if (output/'last.pt').exists():
            payload = torch.load(output/'last.pt', map_location='cpu', weights_only=True)
            if payload['state']['contract_sha256'] != state['contract_sha256']:
                raise ValueError('Resume contract mismatch')
            for arm,m in models.items():
                m.fusion.load_state_dict(payload['fusion'][arm], strict=True)
                optimizers[arm].load_state_dict(payload['optimizers'][arm])
            state = payload['state']; restore_rng(payload['rng'])
        else:
            seed_all(c['seed']); save()
        total = c['smoke_steps'] if a.command == 'smoke' else c['pilot_steps']
        if not 0 <= state['step'] <= total:
            raise ValueError('Invalid checkpoint step')
        loader = DataLoader(PlaceSteps(c, plan, plan['schedule'][state['step']:total]), batch_size=None,
            num_workers=c['workers'], generator=torch.Generator().manual_seed(c['seed']))
        from src.losses.vpr_losses import VPRLossFunction
        lossfn = VPRLossFunction()
        for m in models.values(): m.train()
        for images, labels, coords, _ in tqdm(loader, total=total, initial=state['step'], desc='heterogeneous '+a.command):
            started = time.monotonic(); labels = labels.to(c['device'])
            with torch.no_grad():
                chunks = [first.encode(x.to(c['device'])) for x in images.split(c['microbatch'])]
                features = tuple(torch.cat([chunk[j] for chunk in chunks]) for j in range(6))
                reference = torch.cat([first.anchor.readout(chunk[2]) for chunk in chunks])
            row = {}
            for arm,m in models.items():
                opt = optimizers[arm]; opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    temporary = [m.describe(chunk) for chunk in chunks]
                    leaves = [torch.cat([item[j] for item in temporary]).detach().requires_grad_() for j in range(3)]
                    ratio = torch.cat([item[3]['relative_residual'] for item in temporary])
                if not torch.isfinite(ratio).all() or float(ratio.max()) > c['rho']+1e-6:
                    raise ValueError('Nonfinite/over-cap residual')
                del temporary
                main = gps_retrieval(lossfn, leaves[0], labels, coords)
                auxiliary = sum(gps_retrieval(lossfn, z, labels, coords) for z in leaves[1:])
                keep = (1-(leaves[0]*reference).sum(1)).clamp_min(0).mean()
                loss = main+c['aux_weight']*auxiliary+c['keep_weight']*keep
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                replay_backward(m, features, [z.grad for z in leaves], c['microbatch'])
                for name,p in m.fusion.named_parameters():
                    if p.grad is None or not torch.isfinite(p.grad).all():
                        raise ValueError('Unused/nonfinite active parameter: '+name)
                    state['gradients'][arm][name] |= bool((p.grad != 0).any())
                norm = torch.nn.utils.clip_grad_norm_(m.fusion.parameters(), c['gradient_clip'], error_if_nonfinite=True)
                opt.step()
                row[arm] = dict(main=float(main.detach()), auxiliary=float(auxiliary.detach()), keep=float(keep.detach()),
                    gradient_norm=float(norm), relative_residual_max=float(ratio.max()))
                del leaves, main, auxiliary, keep, loss
            torch.cuda.synchronize(); state['step'] += 1
            row.update(step=state['step'], seconds=time.monotonic()-started, peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
            state['history'].append(row)
            write(output/'progress.json', dict(phase='training', total=total, **row))
            if state['step'] % c['checkpoint_every'] == 0: save()
            del chunks, features, reference
        save()
        checks = dict(complete_steps=state['step']==total, zero_start=max(errors.values()) <= 1e-6,
            frozen_anchor_and_cnn_unchanged=frozen_hashes(first)==initial['frozen'],
            all_new_parameters_have_gradients=all(all(v.values()) for v in state['gradients'].values()),
            all_fusions_updated=all(state_hash(m.fusion)!=initial['fusion'][arm] for arm,m in models.items()),
            matched_capacity=len(set(initial['parameters'].values()))==1,
            innovation1_sources_unchanged=key.source_hashes(key.PROTECTED)==contract['protected_sources'])
        write(output/'summary.json', dict(checks=checks, steps=state['step'], zero_start_errors=errors,
            final_fusion_sha256={arm:state_hash(m.fusion) for arm,m in models.items()},
            parameter_inventory=initial['parameters'], history=state['history'],
            verdict='SMOKE_DISCARD' if a.command=='smoke' else 'TRAINED_NOT_EVALUATED'))
        if not all(checks.values()):
            raise ValueError('Mechanical checks failed: '+str(checks))
        finish(output)


def evaluate(c, a):
    from scripts import eval_clip_key_study as metrics
    from scripts.eval_spatial_adapter_repeat import RGBImages
    run = path(c, 'runs')/'train'
    if verified(run) != contract_for(c, 'train') or not all(read(run/'summary.json')['checks'].values()):
        raise ValueError('Current registered training must be complete')
    plan = key.open_plan(c); datasets = metrics.populations(c, plan)
    contract = dict(stage='eval', **identity(c), checkpoint_sha256=sha(run/'last.pt'),
        datasets={d['tag']:dict(references=d['references'], queries=d['queries'],
            order=metrics.order_digest(d['names']), gt=metrics.gt_digest(d['ground_truth'])) for d in datasets})
    output = path(c, 'runs')/'eval'
    with owned_run(output, contract, a.resume):
        if (output/'completed.json').exists(): verified(output); print('Already evaluated'); return
        key.gpu(c); models = build(c); first = models[ARMS[0]]
        initial_frozen = frozen_hashes(first)
        payload = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
        if payload['state']['step'] != c['pilot_steps'] or payload['state']['contract_sha256'] != sha(run/'contract.json'):
            raise ValueError('Not the fixed-final registered checkpoint')
        for arm,m in models.items():
            m.fusion.load_state_dict(payload['fusion'][arm], strict=True)
            if state_hash(m.fusion) != read(run/'summary.json')['final_fusion_sha256'][arm]:
                raise ValueError('Trained tensor mismatch')
            m.requires_grad_(False).eval()
        variants = ('official', 'innovation1', *ARMS)
        results = {}
        for d in datasets:
            tag = d['tag']; report_path=output/(tag+'.json'); ready=output/(tag+'.ready.json')
            if ready.exists():
                if sha(report_path) != read(ready)['sha256']: raise ValueError('Changed report')
                results[tag] = read(report_path)['comparison']; continue
            started=time.monotonic()
            loader=DataLoader(RGBImages(d['root'], d['names'], d['hashes']), batch_size=16,
                num_workers=c['workers'], shuffle=False)
            arrays={arm:np.empty((len(d['names']),12288),np.float32) for arm in variants}
            contents=[]
            with torch.inference_mode():
                for batch,(images,ids,hashes) in enumerate(tqdm(loader,desc='heterogeneous '+tag)):
                    if ids.tolist()!=list(range(batch*16,batch*16+len(images))): raise ValueError('Image order changed')
                    contents.extend(hashes)
                    for start in range(0,len(images),c['microbatch']):
                        pixels=images[start:start+c['microbatch']].to(c['device'])
                        features=first.encode(pixels)
                        # Official descriptor is a separate reference; no special
                        # no-grad approximation of the innovation-1 baseline.
                        values=dict(official=first.anchor.visual(pixels), innovation1=first.anchor.readout(features[2]),
                            **{arm:m.describe(features)[0] for arm,m in models.items()})
                        for arm,z in values.items():
                            if not torch.isfinite(z).all() or float((z.norm(dim=1)-1).abs().max())>1e-4:
                                raise ValueError('Invalid global descriptor')
                            arrays[arm][ids[start:start+c['microbatch']].numpy()]=z.cpu().numpy()
                    write(output/'progress.json',dict(phase='evaluation',dataset=tag,images=int(ids[-1])+1,
                        total=len(d['names']),elapsed_seconds=time.monotonic()-started))
                reports={}
                for arm in variants:
                    db=torch.tensor(arrays[arm][:d['references']],device=c['device'])
                    queries=arrays[arm][d['references']:]
                    ranks=np.concatenate([(torch.tensor(q,device=c['device'])@db.T).topk(20,dim=1).indices.cpu().numpy()
                        for q in np.array_split(queries,max(1,(len(queries)+15)//16))])
                    reports[arm]=metrics.retrieval_report(ranks,d['ground_truth'],d['references']);del db
            # Archived 322px baseline is reproduced, not treated as a movable target.
            if tag in ('msls-val','pitts30k-val'):
                archived=read(path(c,'anchor_run')/(tag+'_322.json'))
                if reports['innovation1']['outcomes'] != archived['outcomes']:
                    raise ValueError('Innovation-1 archived per-query baseline not reproduced')
            comparison=dict(correct={arm:v['correct'] for arm,v in reports.items()},
                vs_innovation1={arm:metrics.paired_report(reports['innovation1'],reports[arm]) for arm in ARMS},
                hetero_vs_proxy=metrics.paired_report(reports['dino_proxy'],reports['hetero_interleaved']),
                interleaved_vs_separate=metrics.paired_report(reports['separate_scan'],reports['hetero_interleaved']))
            write(report_path,dict(dataset=contract['datasets'][tag],variants=reports,comparison=comparison,
                image_content_digest=metrics.order_digest([n+':'+h for n,h in zip(d['names'],contents)]),
                elapsed_seconds=time.monotonic()-started))
            write(ready,dict(sha256=sha(report_path)));results[tag]=comparison
            print(tag,comparison['correct'],flush=True);del arrays,loader
        checks=dict(frozen_unchanged=frozen_hashes(first)==initial_frozen,
            innovation1_sources_unchanged=key.source_hashes(key.PROTECTED)==contract['protected_sources'],
            training_checkpoint_unchanged=sha(run/'last.pt')==contract['checkpoint_sha256'])
        if not all(checks.values()): raise ValueError('Evaluation integrity failure')
        candidate=all(results[tag]['vs_innovation1']['hetero_interleaved']['net']>0
            and results[tag]['hetero_vs_proxy']['net']>0 and results[tag]['interleaved_vs_separate']['net']>0
            for tag in ('msls-val','pitts30k-val'))
        write(output/'summary.json',dict(checks=checks,results=results,
            verdict='EXPLORATORY_HETERO_INTERACTION_SIGNAL' if candidate else 'NO_MATCHED_HETERO_INTERACTION_GAIN',
            stable_gain_proven=False,blind_test_used=False,scope='Fixed1024 pilot; development datasets; no best checkpoint or threshold search'))
        finish(output)


def main():
    import os
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('plan','smoke','train','eval'))
    p.add_argument('--allow-training',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();c=configuration()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    if a.command=='plan':
        plan=key.open_plan(c);info=identity(c)
        print(dict(status='FILES_READY_NOT_TRAINED',train_places=len(plan['places'][:8192]),
            steps=c['pilot_steps'],anchor_sha256=info['anchor_checkpoint_sha256'],arms=ARMS));return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':p.error('Physical GPU1 only')
    import fcntl
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/query_relation_screen_v1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'smoke':train,'train':train,'eval':evaluate}[a.command](c,a)


if __name__=='__main__':main()
