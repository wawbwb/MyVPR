"""Finite, matched three-arm direct-descriptor pilot; one frozen DINO only."""
import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts import single_dino_interaction as old
from scripts.single_dino_supervision import check_gpu_capacity
from src.models.complementary_covariance import ARMS, ComplementaryVPR, image_seeds
from src.semantic_region_protocol import (read, write, sha, state_hash, owned_run,
    seed_all, save_torch, rng_state, restore_rng, gps_retrieval)

SCHEMA = 'complementary_covariance_v1'
SOURCES = ('src/models/complementary_covariance.py', 'scripts/complementary_covariance.py',
    'scripts/launch_complementary_covariance.py', 'config/complementary_covariance.json',
    'tests/test_complementary_covariance.py', 'scripts/single_dino_supervision.py',
    'scripts/eval_clip_key_study.py', 'scripts/eval_spatial_adapter_repeat.py')


def configuration():
    override = read(ROOT/'config/complementary_covariance.json')
    expected = dict(schema=SCHEMA, runs='logs/complementary_covariance_v1', dim=64, rank=32,
        descriptor_dim=512, complement_weight=.25, warmup_steps=128, branch_loss_weight=1.,
        learning_rate=.0001, pilot_steps=1024, smoke_steps=8)
    if override != expected:
        raise ValueError('Registered covariance pilot differs')
    return dict(old.configuration(), **override)


def identity(c):
    result = old.identity(c)
    result.update(schema=SCHEMA, new_sources=old.key.source_hashes(SOURCES),
        private_references=['6bb336eac52a89823ba67f3f3ebf047e4d19e573382ea686c1653dcb0502aa0e',
            '98034e48211800766f4aa0230560315660182636ae50858187c2933ceebac63c'],
        design='Frozen F1+d0; train local/context cross-statistic descriptor h; concatenate [d0,0.25h]/sqrt(1.0625)',
        scope='Single seed, fixed1024 steps, previously inspected development sets; no stability/novelty claim')
    return result


def contract_for(c, stage):
    return dict(stage=stage, **identity(c))


def finish(output):
    files = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*'))
        if p.is_file() and p.name not in ('.lock', 'completed.json') and not p.name.endswith('.tmp')}
    write(output/'completed.json', dict(complete=True, schema=SCHEMA, files=files))


def verified(output):
    d = read(output/'completed.json')
    if d.get('complete') is not True or d.get('schema') != SCHEMA:
        raise ValueError('Incomplete/wrong schema: '+str(output))
    for name, digest in d['files'].items():
        p = (output/name).resolve()
        if not p.is_relative_to(output.resolve()) or sha(p) != digest:
            raise ValueError('Artifact changed: '+name)
    return read(output/'contract.json')


def build(c):
    anchor = old.make_anchor(c)
    models = {}
    for arm in ARMS:
        seed_all(c['seed'])
        models[arm] = ComplementaryVPR(anchor, arm, dim=c['dim'], rank=c['rank'],
            descriptor_dim=c['descriptor_dim']).to(c['device'])
    if len({state_hash(m.head) for m in models.values()}) != 1:
        raise ValueError('Different initialization')
    ids = [set(map(id, m.head.parameters())) for m in models.values()]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i)):
        raise ValueError('Heads accidentally share parameters')
    return models


def replay(model, chunks, seeds, descriptor_gradients, weight):
    offset = 0
    for features in chunks:
        n = len(features[0])
        outputs = model.describe(features, seeds[offset:offset+n], weight)
        torch.autograd.backward(outputs, tuple(g[offset:offset+n] for g in descriptor_gradients))
        offset += n
    if offset != len(seeds):
        raise ValueError('Replay coverage differs')


def train(c, args):
    from scripts.semantic_region_study import PlaceSteps
    from src.losses.vpr_losses import VPRLossFunction
    plan = old.key.open_plan(c)
    contract = contract_for(c, args.command)
    runs = old.key.path(c, 'runs')
    if args.command == 'train':
        smoke = verified(runs/'smoke')
        if ({k:v for k,v in smoke.items() if k != 'stage'} != {k:v for k,v in contract.items() if k != 'stage'}
                or not all(read(runs/'smoke/summary.json')['checks'].values())):
            raise ValueError('Same-version smoke required')
    output = runs/args.command
    with owned_run(output, contract, args.resume):
        if (output/'completed.json').exists():
            verified(output); print('Already complete'); return
        old.key.gpu(c)
        models = build(c); first = models[ARMS[0]]
        initial = dict(frozen=state_hash(first.anchor), heads={k:state_hash(m.head) for k,m in models.items()},
            parameters={k:sum(p.numel() for p in m.head.parameters()) for k,m in models.items()})
        if (output/'initial.json').exists() and read(output/'initial.json') != initial:
            raise ValueError('Initialization changed')
        write(output/'initial.json', initial)
        sample = PlaceSteps(c, plan, plan['schedule'][:1])[0]
        seeds = image_seeds([plan['images'][int(i)]['image_sha256'] for i in sample[3][:2]])
        with torch.no_grad():
            features = first.encode(sample[0][:2].to(c['device']))
            zero_errors = {a:float((m.describe(features,seeds,0)[0][:,:12288]-features[1]).abs().max())
                for a,m in models.items()}
        del sample, features
        optimizers = {a:torch.optim.AdamW(m.head.parameters(), lr=c['learning_rate'], weight_decay=0.) for a,m in models.items()}
        state = dict(step=0, history=[], contract_sha256=sha(output/'contract.json'),
            gradients={a:{n:False for n,p in m.head.named_parameters()} for a,m in models.items()})
        def save():
            save_torch(output/'last.pt', dict(heads={a:m.head.state_dict() for a,m in models.items()},
                optimizers={a:o.state_dict() for a,o in optimizers.items()}, state=state, rng=rng_state()))
        if (output/'last.pt').exists():
            payload = torch.load(output/'last.pt', map_location='cpu', weights_only=True)
            if payload['state']['contract_sha256'] != state['contract_sha256']:
                raise ValueError('Resume identity differs')
            for arm,m in models.items():
                m.head.load_state_dict(payload['heads'][arm], strict=True)
                optimizers[arm].load_state_dict(payload['optimizers'][arm])
            state = payload['state']; restore_rng(payload['rng'])
        else:
            seed_all(c['seed']); save()
        total = c['smoke_steps'] if args.command == 'smoke' else c['pilot_steps']
        if not 0 <= state['step'] <= total:
            raise ValueError('Invalid cursor')
        loader = DataLoader(PlaceSteps(c,plan,plan['schedule'][state['step']:total]), batch_size=None,
            num_workers=c['workers'], generator=torch.Generator().manual_seed(c['seed']))
        lossfn = VPRLossFunction()
        for m in models.values(): m.train()
        for images, labels, coords, row_ids in tqdm(loader, total=total, initial=state['step'], desc='covariance '+args.command):
            start = time.monotonic(); labels = labels.to(c['device'])
            seeds = image_seeds([plan['images'][int(i)]['image_sha256'] for i in row_ids])
            with torch.no_grad():
                chunks = [first.encode(x.to(c['device'])) for x in images.split(c['microbatch'])]
            weight = c['complement_weight']*min(1., (state['step']+1)/c['warmup_steps'])
            row = {}
            for arm,m in models.items():
                with torch.no_grad():
                    temporary = []; offset = 0
                    for features in chunks:
                        n = len(features[0]); temporary.append(m.describe(features,seeds[offset:offset+n],weight)); offset += n
                    leaves = [torch.cat([v[j] for v in temporary]).detach().requires_grad_() for j in range(2)]
                del temporary
                losses = dict(final=gps_retrieval(lossfn,leaves[0],labels,coords),
                    branch=c['branch_loss_weight']*gps_retrieval(lossfn,leaves[1],labels,coords))
                if not all(bool(torch.isfinite(x)) for x in losses.values()): raise ValueError('Nonfinite loss')
                gradients = torch.autograd.grad(sum(losses.values()), leaves)
                optimizers[arm].zero_grad(set_to_none=True)
                replay(m,chunks,seeds,gradients,weight)
                for name,p in m.head.named_parameters():
                    state['gradients'][arm][name] |= p.grad is not None and bool((p.grad != 0).any())
                norm = torch.nn.utils.clip_grad_norm_(m.head.parameters(),c['gradient_clip'],error_if_nonfinite=True)
                optimizers[arm].step()
                row[arm] = dict(losses={n:float(x.detach()) for n,x in losses.items()},gradient_norm=float(norm))
                del leaves, losses, gradients
            torch.cuda.synchronize(); state['step'] += 1
            row.update(step=state['step'],seconds=time.monotonic()-start,weight=weight,
                peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
            state['history'].append(row)
            write(output/'progress.json',dict(phase='training',total=total,**row))
            if state['step'] % c['checkpoint_every'] == 0: save()
            del chunks
        save()
        checks = dict(complete_steps=state['step']==total,zero_start=max(zero_errors.values())==0,
            frozen_anchor_unchanged=state_hash(first.anchor)==initial['frozen'],
            matched_capacity=len(set(initial['parameters'].values()))==1,
            all_heads_updated=all(state_hash(m.head)!=initial['heads'][a] for a,m in models.items()),
            all_parameters_gradient_seen=all(all(v.values()) for v in state['gradients'].values()),
            innovation1_sources_unchanged=old.key.source_hashes(old.key.PROTECTED)==contract['protected_sources'])
        write(output/'summary.json',dict(checks=checks,steps=state['step'],zero_errors=zero_errors,
            final_head_sha256={a:state_hash(m.head) for a,m in models.items()},parameters=initial['parameters'],
            encoder_count=1,descriptor_dimensions=12800,verdict='SMOKE_DISCARD' if args.command=='smoke' else 'TRAINED_NOT_EVALUATED'))
        if not all(checks.values()): raise ValueError('Mechanical checks failed: '+str(checks))
        finish(output)


def evaluate(c,args):
    from scripts import eval_clip_key_study as metrics
    from scripts.eval_spatial_adapter_repeat import RGBImages
    run = old.key.path(c,'runs')/'train'
    if verified(run) != contract_for(c,'train') or not all(read(run/'summary.json')['checks'].values()):
        raise ValueError('Training incomplete or changed')
    datasets = metrics.populations(c,old.key.open_plan(c))
    contract = dict(stage='eval',**identity(c),checkpoint_sha256=sha(run/'last.pt'),
        datasets={d['tag']:dict(references=d['references'],queries=d['queries'],
            order=metrics.order_digest(d['names']),gt=metrics.gt_digest(d['ground_truth'])) for d in datasets})
    output = old.key.path(c,'runs')/'eval'
    with owned_run(output,contract,args.resume):
        if (output/'completed.json').exists(): verified(output); print('Already evaluated'); return
        old.key.gpu(c); models = build(c); first = models[ARMS[0]]; frozen = state_hash(first.anchor)
        payload = torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        if payload['state']['step'] != c['pilot_steps'] or payload['state']['contract_sha256'] != sha(run/'contract.json'):
            raise ValueError('Not fixed-final checkpoint')
        for arm,m in models.items():
            m.head.load_state_dict(payload['heads'][arm],strict=True)
            if state_hash(m.head) != read(run/'summary.json')['final_head_sha256'][arm]: raise ValueError('Head changed')
            m.requires_grad_(False).eval()
        results = {}; variants = ('innovation1',*ARMS)
        for d in datasets:
            tag = d['tag']; target = output/(tag+'.json'); ready = output/(tag+'.ready.json')
            if ready.exists():
                if sha(target) != read(ready)['sha256']: raise ValueError('Report changed')
                results[tag] = read(target)['comparison']; continue
            start = time.monotonic(); contents = []
            arrays = {a:np.empty((len(d['names']),12288 if a=='innovation1' else 12800),np.float32) for a in variants}
            loader = DataLoader(RGBImages(d['root'],d['names'],d['hashes']),batch_size=16,num_workers=c['workers'],shuffle=False)
            with torch.inference_mode():
                for batch,(images,ids,hashes) in enumerate(tqdm(loader,desc='covariance '+tag)):
                    if ids.tolist() != list(range(batch*16,batch*16+len(images))): raise ValueError('Order changed')
                    contents.extend(hashes)
                    for i in range(0,len(images),c['microbatch']):
                        features = first.encode(images[i:i+c['microbatch']].to(c['device']))
                        seeds = image_seeds(hashes[i:i+c['microbatch']])
                        values = dict(innovation1=features[1],**{a:m.describe(features,seeds,c['complement_weight'])[0] for a,m in models.items()})
                        for arm,z in values.items():
                            if not torch.isfinite(z).all() or float((z.norm(dim=1)-1).abs().max()) > 1e-4: raise ValueError('Invalid descriptor')
                            arrays[arm][ids[i:i+c['microbatch']].numpy()] = z.cpu().numpy()
                    write(output/'progress.json',dict(phase='evaluation',dataset=tag,images=int(ids[-1])+1,total=len(d['names']),elapsed_seconds=time.monotonic()-start))
                reports = {}
                for arm in variants:
                    db = torch.tensor(arrays[arm][:d['references']],device=c['device'])
                    queries = arrays[arm][d['references']:]
                    ranks = np.concatenate([(torch.tensor(q,device=c['device'])@db.T).topk(20,dim=1).indices.cpu().numpy()
                        for q in np.array_split(queries,max(1,(len(queries)+15)//16))])
                    reports[arm] = metrics.retrieval_report(ranks,d['ground_truth'],d['references']); del db
            if tag in ('msls-val','pitts30k-val') and reports['innovation1']['outcomes'] != read(old.key.path(c,'anchor_run')/(tag+'_322.json'))['outcomes']:
                raise ValueError('Innovation1 baseline not reproduced')
            comparison = dict(correct={a:r['correct'] for a,r in reports.items()},
                vs_innovation1={a:metrics.paired_report(reports['innovation1'],reports[a]) for a in ARMS},
                aligned_vs_mean=metrics.paired_report(reports['mean_outer'],reports['aligned_cov']),
                aligned_vs_permuted=metrics.paired_report(reports['permuted_cov'],reports['aligned_cov']))
            write(target,dict(dataset=contract['datasets'][tag],variants=reports,comparison=comparison,
                image_content_digest=metrics.order_digest([n+':'+h for n,h in zip(d['names'],contents)]),elapsed_seconds=time.monotonic()-start))
            write(ready,dict(sha256=sha(target))); results[tag] = comparison
            print(tag,comparison['correct'],flush=True); del arrays,loader
        checks = dict(frozen_anchor_unchanged=state_hash(first.anchor)==frozen,
            innovation1_sources_unchanged=old.key.source_hashes(old.key.PROTECTED)==contract['protected_sources'],
            checkpoint_unchanged=sha(run/'last.pt')==contract['checkpoint_sha256'])
        signal = all(results[t]['vs_innovation1']['aligned_cov']['net']>0 and results[t]['aligned_vs_mean']['net']>0
            and results[t]['aligned_vs_permuted']['net']>0 for t in ('msls-val','pitts30k-val'))
        write(output/'summary.json',dict(checks=checks,results=results,stable_gain_proven=False,blind_test_used=False,
            verdict='EXPLORATORY_COVARIANCE_COMPLEMENT_SIGNAL' if signal else 'NO_MATCHED_TWO_DATASET_COVARIANCE_GAIN'))
        if not all(checks.values()): raise ValueError('Integrity checks failed')
        finish(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('plan','smoke','train','eval'))
    p.add_argument('--allow-training',action='store_true'); p.add_argument('--resume',action='store_true')
    p.add_argument('--allow-shared-gpu',action='store_true')
    args = p.parse_args(); c = configuration()
    if args.command in ('smoke','train') and not args.allow_training: p.error('Explicit training consent required')
    if args.command == 'plan':
        old.key.open_plan(c); print(dict(identity=identity(c),training_started=False)); return
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1': p.error('Physical GPU1 only')
    import fcntl
    import subprocess
    with (ROOT/'logs/query_relation_screen_v1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        memory = subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,timeout=20)
        check_gpu_capacity(*map(int,memory.strip().split(',')),args.allow_shared_gpu,c['cuda_memory_fraction'])
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
        if args.command == 'eval': evaluate(c,args)
        else: train(c,args)


if __name__ == '__main__': main()
