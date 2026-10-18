"""RGB-only final-model evaluation + fresh heldout residual-predictability probes.

Invoked only by explicit `clip_nuisance_study.py eval --evaluate`.
Never modifies old experiments, model parameters, test sets, or training files.
"""
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts import clip_nuisance_study as study
from src.models.clip_nuisance import ARMS,InvariantBoQ,projection
from src.models.official_boq import load_official
from src.semantic_region_protocol import owned_run,read,write,sha,state_hash,save_npz


def evaluate(c,args):
    plan=study.open_plan(c)
    _,targets,_,_=study.prepared(c)
    datasets=study.populations(c,plan)
    base=study.identity(c)
    inventory={};initial=None
    for arm in ARMS:
        run=study.path(c,'runs')/arm
        if study.verified(run)!=study.training_contract(c,arm,'train') or not all(read(run/'summary.json')['checks'].values()):
            raise ValueError('Training incomplete/provenance mismatch: '+arm)
        current=read(run/'initial.json')
        if initial is not None and current!=initial:raise ValueError('Unmatched initialization')
        initial=current
        inventory[arm]=dict(checkpoint=sha(run/'last.pt'),contract=sha(run/'contract.json'),completed=sha(run/'completed.json'))
    contract=dict(stage='rgb_only_eval_and_fresh_probes',**base,training=inventory,initial=initial,
        datasets={d['tag']:dict(root=str(d['root'].resolve()),references=d['references'],queries=d['queries'],
            image_order_sha256=study.order_digest(d['names']),ground_truth_sha256=study.gt_digest(d['ground_truth']),
            metadata={p.name:sha(p) for p in sorted(d['root'].glob('*.npy'))}) for d in datasets},
        inference='RGB-only; original official weights plus visual-only router. No CLIP/predictor.',
        fresh_probe='Discard original adversary, fit common fresh CPU probe per final model with own TRAIN-only normalization',
        scope='Fixed final single-seed exploratory development evaluation; blind test never used')
    output=study.path(c,'runs')/'evaluation'
    with owned_run(output,contract,args.resume):
        if (output/'completed.json').exists():study.verified(output);print('Evaluation already complete',flush=True);return
        study.gpu(c)
        official=load_official(study.path(c,'official_cache'),study.path(c,'dino_source'),c['device'])
        frozen=state_hash(official)
        if frozen!=initial['official']:raise ValueError('Official visual state differs')
        models={}
        for arm in ARMS:
            model=InvariantBoQ(official,c['hidden'],c['bias_cap'],c['key_temperature'],c['seed']).to(c['device'])
            saved=torch.load(study.path(c,'runs')/arm/'last.pt',map_location='cpu',weights_only=True)
            params={n:p for n,p in model.named_parameters() if p.requires_grad}
            if (set(params)!=set(saved['student']) or saved['state']['contract_sha256']!=inventory[arm]['contract']
                    or saved['state']['step']!=2560):raise ValueError('Wrong final checkpoint')
            with torch.no_grad():
                for n,p in params.items():
                    value=saved['student'][n]
                    if p.shape!=value.shape or not torch.isfinite(value).all():raise ValueError('Invalid final parameter')
                    p.copy_(value.to(p.device))
            if state_hash(model.routers)!=read(study.path(c,'runs')/arm/'summary.json')['final_router_sha256']:
                raise ValueError('Final student tensors differ')
            models[arm]=model.requires_grad_(False).eval()
            del saved
        fixed_projection=projection(12288,c['projection_dim'],c['seed']).to(c['device'])
        for d in datasets:
            tag=d['tag'];target=output/(tag+'.json');ready=output/(tag+'.ready.json')
            if ready.exists():
                if sha(target)!=read(ready)['sha256']:raise ValueError('Damaged completed evaluation')
                study.validate_completed_dataset(read(target),d,contract)
                if tag=='gsv-adaptation-holdout':
                    if sha(output/'fresh_probe_inputs.npz')!=read(ready)['probe_inputs_sha256']:
                        raise ValueError('Damaged final probe inputs')
                print('Reuse completed',tag,flush=True);continue
            started=time.monotonic()
            is_gsv=tag=='gsv-adaptation-holdout'
            if is_gsv:
                rows=plan['images'];names=[r['path'] for r in rows]
                dataset=study.DualImages(study.path(c,'dataset_root'),names,[r['image_sha256'] for r in rows])
                destination={n:i for i,n in enumerate(d['names'])}
                mapping=np.asarray([destination.get(n,-1) for n in names],np.int64)
                projected={a:np.empty((len(rows),c['projection_dim']),np.float32) for a in study.VARIANTS}
            else:
                dataset=study.DualImages(d['root'],d['names'],d['hashes'])
                mapping=np.arange(len(dataset))
                projected=None
            descriptors={a:np.empty((len(d['names']),12288),np.float32) for a in study.VARIANTS}
            content_hashes={}
            loader=DataLoader(dataset,batch_size=16,num_workers=c['workers'],shuffle=False)
            with torch.inference_mode():
                for batch,(images,_,indices,digests) in enumerate(tqdm(loader,desc='RGB-only '+tag)):
                    if indices.tolist()!=list(range(batch*16,batch*16+len(images))):raise ValueError('Extraction order differs')
                    for i,digest in zip(indices.tolist(),digests):
                        if mapping[i]>=0:content_hashes[int(mapping[i])]=digest
                    for start in range(0,len(images),c['microbatch']):
                        stop=min(start+c['microbatch'],len(images))
                        ids=indices[start:stop].numpy()
                        features=official.backbone(images[start:stop].to(c['device']))
                        values=dict(official=official.aggregator(features)[0],
                            **{a:m.describe(features) for a,m in models.items()})
                        keep=mapping[ids]>=0
                        for arm,value in values.items():
                            if not torch.isfinite(value).all() or float((value.norm(dim=1)-1).abs().max())>1e-4:
                                raise ValueError('Invalid descriptor')
                            if is_gsv:projected[arm][ids]=(value@fixed_projection).cpu().numpy()
                            if keep.any():descriptors[arm][mapping[ids[keep]]]=value.cpu().numpy()[keep]
                    study.write(output/'progress.json',dict(phase='extracting',dataset=tag,images=int(indices[-1])+1,
                        total_images=len(dataset),training_student=False,clip_inference=False,elapsed_seconds=time.monotonic()-started))
                reports={}
                for arm in study.VARIANTS:
                    database=torch.from_numpy(descriptors[arm][:d['references']]).to(c['device'])
                    queries=descriptors[arm][d['references']:]
                    rankings=np.concatenate([(torch.from_numpy(queries[s:s+16]).to(c['device'])@database.T)
                        .topk(20,dim=1).indices.cpu().numpy() for s in range(0,len(queries),16)])
                    reports[arm]=study.retrieval_report(rankings,d['ground_truth'],d['references'])
                    del database
            report=dict(tag=tag,dataset=contract['datasets'][tag],variants=reports,comparison=study.compare_reports(reports),
                image_content_sha256=study.order_digest([n+':'+content_hashes[i] for i,n in enumerate(d['names'])]),
                elapsed_seconds=time.monotonic()-started,scope=contract['scope'])
            study.validate_completed_dataset(report,d,contract);write(target,report)
            ready_record=dict(sha256=sha(target))
            if is_gsv:
                save_npz(output/'fresh_probe_inputs.npz',**projected)
                ready_record['probe_inputs_sha256']=sha(output/'fresh_probe_inputs.npz')
            write(ready,ready_record)
            print(tag,report['comparison'],flush=True)
            del descriptors,projected,loader,dataset
        fresh=output/'fresh_probes.json'
        probe_contract=dict(inputs_sha256=sha(output/'fresh_probe_inputs.npz'),seed=c['seed']+9101,
                            epochs=c['probe_epochs'],target='True within-place CLIP residual for ALL four groups')
        if fresh.exists():
            old=read(fresh)
            if old['contract']!=probe_contract:raise ValueError('Fresh probe contract differs')
            probes=old['results']
        else:
            write(output/'progress.json',dict(phase='fresh_cpu_probes',training_student=False,clip_inference=False))
            with np.load(output/'fresh_probe_inputs.npz',allow_pickle=False) as inputs:
                probes={a:study.fit_probe(c,inputs[a],targets,plan['train_rows'],c['seed']+9101) for a in study.VARIANTS}
            write(fresh,dict(contract=probe_contract,results=probes,
                scope='Fresh limited-capacity predictor on place-heldout data; lower R2 is not proof of complete disentanglement'))
        checks=dict(official_unchanged=state_hash(official)==frozen,
            innovation1_unchanged=study.source_hashes(study.PROTECTED)==base['base']['protected_sources'],
            checkpoints_unchanged=all(sha(study.path(c,'runs')/a/'last.pt')==inventory[a]['checkpoint'] for a in ARMS))
        if not all(checks.values()):raise ValueError('Evaluation integrity failure')
        summary=dict(results={d['tag']:read(output/(d['tag']+'.json'))['comparison'] for d in datasets},
            fresh_probes=probes,checks=checks,clip_needed_at_inference=False,blind_test_used=False,
            scope='Single-seed exploratory screen. Judge retrieval AND independent probe, not adversary loss alone.')
        write(output/'summary.json',summary);write(output/'progress.json',dict(phase='complete'))
        study.finish(output);print(summary,flush=True)
