"""Frozen DINO block/grid diagnostic on exactly the completed v2 candidate set."""
import argparse
import functools
import os
from pathlib import Path
import shutil
import sys
import uuid
from zipfile import BadZipFile

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_slgd import read, write, sha, code_hash, verify, seal, seed, MatchedPlaces, RU_SHA
from scripts.audit_slgd_hard import PlannedPlaces, collate, shard_rows, validate_shard, save_npz
from src.models.slgd import mutual_score
from src.slgd_layer_protocol import (LAYER_POLICY, LayerCapture, quantized_tokens,
                                     fine_query_ids, summarize_layers, choose_layer)

SOURCES = ('scripts/audit_slgd_layers.py', 'src/slgd_layer_protocol.py',
           'scripts/audit_slgd_hard.py', 'src/slgd_hard_protocol.py',
           'scripts/train_slgd.py', 'src/models/slgd.py',
           'src/dataloaders/train/gsv_cities.py', 'src/models/backbones/dinov2.py',
           'scripts/eval_condition_robustness.py', 'src/models/aggregators/boq.py',
           'src/models/semantic_region_gate.py')
NEW_LAYERS = [x for x in LAYER_POLICY['layers'] if x != 12]


def validate_layer_shard(path, indices):
    if sha(path) != read(path.with_suffix('.json'))['sha256']:
        raise ValueError('Layer shard checksum differs')
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {'indices','tokens'}:
            raise ValueError('Unknown shard fields')
        indices_stored, tokens = z['indices'], z['tokens']
    if indices_stored.dtype != np.int64 or not np.array_equal(indices, indices_stored):
        raise ValueError('Layer shard identities differ')
    if tokens.shape != (len(indices),len(NEW_LAYERS),100,768) or tokens.dtype != np.float16 or not np.isfinite(tokens).all():
        raise ValueError('Layer shard shape/dtype/finite check failed')
    if np.max(np.abs(np.linalg.norm(tokens.astype(np.float32),axis=-1)-1)) > .002:
        raise ValueError('Layer shard has non-unit tokens')
    return tokens


def pair_scores(tokens_left, tokens_right):
    left = torch.nn.functional.normalize(torch.from_numpy(tokens_left.astype(np.float32)).cuda(),dim=-1)
    right = torch.nn.functional.normalize(torch.from_numpy(tokens_right.astype(np.float32)).cuda(),dim=-1)
    with torch.inference_mode():
        score, count = mutual_score(left, right)
    return score.cpu().numpy(), (count >= LAYER_POLICY['min_mutual_matches']).cpu().numpy()


class FinePairs(Dataset):
    """Reconstruct identical views without modifying the historical dataset."""
    def __init__(self, dataset, plan, candidates, query_ids):
        self.data = MatchedPlaces(dataset, plan['dataset_indices'], 99)
        self.plan, self.candidates, self.query_ids = plan, candidates, query_ids
        self.nq = len(plan['query_places'])

    @functools.lru_cache(maxsize=16)
    def place_images(self, place):
        return self.data[place][0]

    def row_image(self, row):
        if row < self.nq*4:
            return self.place_images(row//4)[row%4]
        return self.place_images(self.nq+row-self.nq*4)[0]

    def __len__(self):
        return len(self.query_ids)

    def __getitem__(self, i):
        q = self.query_ids[i]
        rows = [self.plan['query_rows'][q]]+[self.plan['gallery_rows'][int(k)] for k in self.candidates[q]]
        return torch.stack([self.row_image(row) for row in rows]), q


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('logs/slgd_hard_v2_audit'))
    p.add_argument('--source-cache',type=Path,default=Path('.cache/slgd_hard_v2_audit'))
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,default=Path('datasets/gsv_cities'))
    p.add_argument('--output',type=Path,default=Path('logs/slgd_layers_v1'))
    p.add_argument('--cache',type=Path,default=Path('.cache/slgd_layers_v1'))
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--workers',type=int,default=4)
    args = p.parse_args()
    import fcntl
    from torchvision import transforms as T
    from tqdm import tqdm
    from src.dataloaders.train.gsv_cities import GSVCitiesDataset
    from scripts.eval_condition_robustness import load_inference_model_from_ckpt
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        p.error('Expose physical GPU1 only')
    if not args.checkpoint.is_file() or sha(args.checkpoint) != RU_SHA:
        p.error('Original RU missing or SHA256 differs')
    source = verify(args.source)
    if source['smoke'] != args.smoke or source['ru_sha256'] != RU_SHA:
        p.error('Source smoke/full/RU identity differs')
    for name,digest in source['code'].items():
        if code_hash(ROOT/name)!=digest:
            p.error('Frozen source code changed: '+name)
    if read(args.source/'summary.json')['cache_manifest_sha256'] != sha(args.source_cache/'manifest.json'):
        p.error('Source cache manifest hash differs')
    source_manifest = read(args.source_cache/'manifest.json')
    if not source_manifest['complete'] or read(args.source_cache/'contract.json') != source:
        p.error('Source cache contract/complete flag differs')
    if source_manifest['contract_sha256'] != sha(args.source_cache/'contract.json'):
        p.error('Source cache contract checksum differs')
    metadata={x.name:sha(x) for x in sorted((args.dataset_root/'Dataframes').glob('*.csv'))}
    if metadata != source['original_metadata']:
        p.error('GSV metadata differs')
    with np.load(args.source/'candidate_scores.npz',allow_pickle=False) as z:
        candidates, positive, groups, gap = (z[k] for k in ('candidates','positive_gallery','groups','ru_gap'))
        old_scores, old_valid = z['raw_scores'],z['raw_valid']
    plan=source['plan']
    if not np.array_equal(groups,plan['query_place_groups']) or not np.array_equal(positive,plan['positive_gallery_indices']):
        p.error('Source candidate identities differ')
    ru = candidates[:,0] == positive
    hard = ~ru | (gap<=.02)
    fine_ids=fine_query_ids(groups,hard,plan['calibration_places'],24 if args.smoke else LAYER_POLICY['fine_queries_per_partition'])
    contract=dict(policy=LAYER_POLICY,smoke=args.smoke,source_completed_sha256=sha(args.source/'completed.json'),
                  source_cache_manifest_sha256=sha(args.source_cache/'manifest.json'),ru_sha256=RU_SHA,
                  plan=plan,fine_query_ids=fine_ids,metadata=metadata,
                  code={name:code_hash(ROOT/name) for name in SOURCES},
                  versions=dict(torch=torch.__version__,numpy=np.__version__))
    args.cache.parent.mkdir(parents=True,exist_ok=True)
    lock=(args.cache.parent/(args.cache.name+'.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for directory in (args.output,args.cache):
        if directory.exists():
            if not args.resume or read(directory/'contract.json')!=contract:
                p.error('Existing output/cache or immutable contract differs: '+str(directory))
        else:
            directory.mkdir(parents=True)
            write(directory/'contract.json',contract)
    if (args.output/'completed.json').exists():
        verify(args.output)
        if read(args.output/'summary.json')['layer_cache_manifest_sha256']!=sha(args.cache/'manifest.json'):
            p.error('Completed layer manifest changed')
        print('Already complete',flush=True)
        return
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(.65,0)
    estimated=plan['rows']*len(NEW_LAYERS)*100*768*2
    if shutil.disk_usage(args.cache).free < estimated*1.2:
        p.error('Insufficient disk for layer cache')
    ram=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    if ram < plan['rows']*100*768*2*1.5+2*1024**3:
        p.error('Insufficient RAM to score one layer at a time')
    seed(42)
    transform=T.Compose([T.Resize((280,280),interpolation=T.InterpolationMode.BICUBIC),T.ToTensor(),
                         T.Normalize([.485,.456,.406],[.229,.224,.225])])
    cities=sorted(x.stem for x in (args.dataset_root/'Dataframes').glob('*.csv'))
    dataset=GSVCitiesDataset(dataset_path=args.dataset_root,cities=cities,img_per_place=4,
                            transform=transform,return_metadata=True)
    if [str(dataset.places_ids[i]) for i in plan['dataset_indices']] != plan['query_places']+plan['distractor_places']:
        p.error('Dataset ordering/identity changed')
    total=(len(plan['dataset_indices'])+15)//16
    shards=args.cache/'shards'
    shards.mkdir(exist_ok=True)
    hashes={}
    existing=set()
    # Verify old source shards before doing any new extraction.
    for batch in tqdm(range(total),desc='Verify original immutable cache'):
        old_path=args.source_cache/'shards'/f'{batch:05d}.npz'
        if source_manifest['files'].get(old_path.name)!=sha(old_path):
            raise ValueError('Source shard hash differs')
        validate_shard(old_path,shard_rows(plan,batch))
        path=shards/old_path.name
        if path.exists():
            try:
                validate_layer_shard(path,shard_rows(plan,batch))
            except (OSError,ValueError,EOFError,KeyError,BadZipFile) as error:
                quarantine=args.cache/'quarantine'
                quarantine.mkdir(exist_ok=True)
                for old in (path,path.with_suffix('.json')):
                    if old.exists():
                        if not old.resolve().is_relative_to(args.cache.resolve()):
                            raise ValueError('Unsafe quarantine target')
                        old.replace(quarantine/(old.name+'.'+uuid.uuid4().hex+'.bad'))
                print('Quarantined damaged new shard:',batch,str(error),flush=True)
            else:
                existing.add(batch)
                hashes[path.name]=sha(path)
    print(f'Additional pooled cache: {estimated/1024**3:.2f} GiB; full-resolution tokens transient only',flush=True)
    visual=load_inference_model_from_ckpt(args.checkpoint,'cpu').cuda().eval().requires_grad_(False)
    capture=LayerCapture(visual.backbone,LAYER_POLICY['layers'])
    loader=DataLoader(PlannedPlaces(dataset,plan),batch_size=16,num_workers=args.workers,
                      collate_fn=collate,persistent_workers=args.workers>0)
    parity_max=read(args.cache/'parity.json')['legacy_raw_max_abs_error'] if (args.cache/'parity.json').exists() else 0.
    for batch,(images,indices,coordinates) in enumerate(tqdm(loader,desc='Capture L4/6/8/10; reproduce legacy L12')):
        write(args.output/'progress.json',dict(phase='layer_cache',batch=batch,total=total,reused=len(existing)))
        old=validate_shard(args.source_cache/'shards'/f'{batch:05d}.npz',indices.numpy())
        if batch in existing:
            continue
        parts=[]
        pooled12=[]
        ru_parts=[]
        with torch.inference_mode():
            for images_chunk in images.cuda().split(4):
                features,last=capture(images_chunk)
                parts.append(np.stack([quantized_tokens(features[x],10) for x in NEW_LAYERS],axis=1))
                pooled12.append(quantized_tokens(last,10))
                gated=visual.semantic_region_gate(last)[0] if visual.semantic_region_gate is not None else last
                desc=visual.aggregator(gated)
                ru_parts.append((desc[0] if isinstance(desc,tuple) else desc).cpu().numpy())
        raw=np.concatenate(pooled12)
        error=float(np.max(np.abs(raw.astype(np.float32)-old['raw'].astype(np.float32))))
        ru_error=float(np.max(np.abs(np.concatenate(ru_parts)-old['global_descriptors'])))
        if error>.001 or ru_error>1e-5:
            raise ValueError(f'Source extraction does not reproduce: raw={error}, RU={ru_error}')
        parity_max=max(parity_max,error)
        path=shards/f'{batch:05d}.npz'
        save_npz(path,dict(indices=indices.numpy().astype(np.int64),tokens=np.concatenate(parts)))
        validate_layer_shard(path,indices.numpy())
        hashes[path.name]=sha(path)
        write(args.cache/'parity.json',dict(legacy_raw_max_abs_error=parity_max))
    del loader
    write(args.cache/'manifest.json',dict(complete=True,contract_sha256=sha(args.cache/'contract.json'),files=hashes))
    qrows=np.asarray(plan['query_rows'])
    grows=np.asarray(plan['gallery_rows'])
    scores={'L12_g10':old_scores}
    valid={'L12_g10':old_valid}
    for layer_index,layer in enumerate(NEW_LAYERS):
        write(args.output/'progress.json',dict(phase='pooled_matching',layer=layer))
        path=args.output/f'L{layer}_g10.npz'
        if path.exists() and path.with_suffix('.json').exists() and sha(path)==read(path.with_suffix('.json'))['sha256']:
            with np.load(path,allow_pickle=False) as z:
                values,validity=z['scores'],z['valid']
            if values.shape!=candidates.shape or validity.shape!=candidates.shape or not np.isfinite(values).all():
                raise ValueError('Invalid resumed score array')
        else:
            tokens=np.empty((plan['rows'],100,768),dtype=np.float16)
            for batch in range(total):
                indices=shard_rows(plan,batch)
                tokens[indices]=validate_layer_shard(shards/f'{batch:05d}.npz',indices)[:,layer_index]
            values=np.empty(candidates.shape,np.float32)
            validity=np.empty(candidates.shape,bool)
            for start in tqdm(range(0,len(qrows),4),desc=f'Match L{layer} pooled10'):
                end=min(start+4,len(qrows))
                left=tokens[np.repeat(qrows[start:end],candidates.shape[1])]
                right=tokens[grows[candidates[start:end]].flatten()]
                result,ok=pair_scores(left,right)
                values[start:end]=result.reshape(end-start,-1)
                validity[start:end]=ok.reshape(end-start,-1)
            del tokens
            save_npz(path,dict(scores=values,valid=validity))
        scores[f'L{layer}_g10'],valid[f'L{layer}_g10']=values,validity
    coarse,outcomes=summarize_layers(candidates,positive,groups,gap,scores,valid,plan['calibration_places'])
    selected=choose_layer(coarse['calibration'])
    write(args.output/'selection.json',dict(**selected,policy=LAYER_POLICY['selection'],
        source='calibration pooled10 only; native/evaluation results unavailable to selection'))
    write(args.output/'pooled_outcomes.json',outcomes)
    native_scores={f'L{x}_g{grid}':[] for x in LAYER_POLICY['layers'] for grid in (10,20)}
    native_valid={name:[] for name in native_scores}
    fine_dir=args.cache/'fine_scores'
    fine_dir.mkdir(exist_ok=True)
    fine_loader=DataLoader(FinePairs(dataset,plan,candidates,fine_ids),batch_size=1,num_workers=min(args.workers,2))
    fine_error=0.
    for images,q_tensor in tqdm(fine_loader,desc='Native20 vs pooled10; locked RU-stratified subset'):
        q=int(q_tensor[0])
        write(args.output/'progress.json',dict(phase='native_matching',query=q,subset=len(fine_ids)))
        path=fine_dir/f'{q:05d}.npz'
        reused=False
        if path.exists() and path.with_suffix('.json').exists() and sha(path)==read(path.with_suffix('.json'))['sha256']:
            with np.load(path,allow_pickle=False) as z:
                if set(z.files)!={name+s for name in native_scores for s in ('_scores','_valid')}:
                    raise ValueError('Fine score fields differ')
                packed={name:z[name] for name in z.files}
            reused=True
        if not reused:
            encoded={name:[] for name in native_scores}
            with torch.inference_mode():
                for chunk in images[0].cuda().split(4):
                    features,last=capture(chunk)
                    for layer in LAYER_POLICY['layers']:
                        for grid in (10,20):
                            encoded[f'L{layer}_g{grid}'].append(quantized_tokens(features[layer],grid))
            packed={}
            for name,parts in encoded.items():
                tokens=np.concatenate(parts)
                # Candidate sub-batches bound 400x400 attention/similarity memory.
                result,ok=[],[]
                for start in range(1,len(tokens),4):
                    right=tokens[start:start+4]
                    val,flags=pair_scores(np.repeat(tokens[:1],len(right),axis=0),right)
                    result.extend(val.tolist())
                    ok.extend(flags.tolist())
                packed[name+'_scores']=np.asarray(result,np.float32)
                packed[name+'_valid']=np.asarray(ok,bool)
            save_npz(path,packed)
        for name in native_scores:
            value,flags=packed[name+'_scores'],packed[name+'_valid']
            if value.shape!=(candidates.shape[1],) or flags.dtype!=bool or flags.shape!=value.shape or not np.isfinite(value).all():
                raise ValueError('Invalid native matching score array')
            native_scores[name].append(value)
            native_valid[name].append(flags)
            if name.endswith('g10'):
                error=float(np.max(np.abs(value-scores[name][q])))
                fine_error=max(fine_error,error)
                if error>.002 or not np.array_equal(flags,valid[name][q]):
                    raise ValueError(f'Fine subset pooled reproduction differs: {name}, q={q}, error={error}')
    capture.close()
    del fine_loader,visual
    native_scores={name:np.stack(value) for name,value in native_scores.items()}
    native_valid={name:np.stack(value) for name,value in native_valid.items()}
    ids=np.asarray(fine_ids)
    fine,fine_outcomes=summarize_layers(candidates[ids],positive[ids],groups[ids],gap[ids],
                                      native_scores,native_valid,plan['calibration_places'])
    write(args.output/'native_outcomes.json',dict(query_ids=fine_ids,**fine_outcomes))
    for role in fine:
        for layer in LAYER_POLICY['layers']:
            f=fine[role]['variants']
            f[f'L{layer}_g20']['net_vs_same_layer_pooled10']=f[f'L{layer}_g20']['correct']-f[f'L{layer}_g10']['correct']
    chosen=f'L{selected["layer"]}_g10'
    evaluation=coarse['evaluation']['variants'][chosen]
    reference=coarse['evaluation']['variants']['L12_g10']
    verdict=('CALIBRATION_SELECTED_EVALUATION_COMPLEMENT' if selected['layer']!=12 and
             evaluation['hard_net_vs_last']>0 and evaluation['net_vs_last']>=0 else
             'NO_REPLICATED_LAYER_COMPLEMENT')
    if args.smoke:
        verdict='SMOKE_PASS'
    summary=dict(verdict=verdict,selection=selected,pooled=coarse,native_subset=fine,
        native_query_ids=fine_ids,additional_cache_gib=estimated/1024**3,
        layer_cache_manifest_sha256=sha(args.cache/'manifest.json'),legacy_raw_max_abs_error=parity_max,
        fine_pooled_score_max_abs_error=fine_error,selected_evaluation_hard_gain_pp=
        100*evaluation['hard_net_vs_last']/max(1,coarse['evaluation']['hard_queries']),
        selected_evaluation_vs_reference_correct_delta=evaluation['correct']-reference['correct'],
        scope=LAYER_POLICY['scope'],teacher_trained=False,student_trained=False,
        no_midlayer_boq=True,notes='Native subset is RU-hard enriched, not population recall; layer selection is NOT novelty evidence')
    write(args.output/'summary.json',summary)
    write(args.output/'progress.json',dict(phase='complete',verdict=verdict))
    seal(args.output)
    print('Layer/grid diagnostic complete:',verdict,'selected layer',selected['layer'],flush=True)
    print('No teacher/student/CLIP training was launched.',flush=True)


if __name__=='__main__':
    main()
