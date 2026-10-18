"""Fixed-final four-arm CLIP-key evaluation; never trains or reads blind tests.

Separate evaluator so sealed training code/config/source inventories stay intact.
One DINO and one CLIP encoding per image, shared fairly across frozen arms.
Only predictions/results are persisted, not full dense database feature caches.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scripts.clip_key_study import (configuration, contract_for, finish, gpu, identity,
    open_plan, path, source_hashes, verified, PROTECTED)
from scripts.semantic_region_study import AdaptationHoldout, transform
from src.models.clip_key_boq import ARMS, ClipKeyBoQ, permute_patches
from src.models.clip_dense_keys import DenseClipKeys
from src.models.official_boq import load_official
from src.semantic_region_protocol import owned_run, paired, read, sha, state_hash, write

EVAL_SOURCES = ('scripts/eval_clip_key_study.py', 'scripts/launch_clip_key_eval.py',
    'tests/test_clip_key_eval.py', 'src/dataloaders/valid/mapillary_sls.py',
    'src/dataloaders/valid/pittsburgh.py', 'src/utils/config_manager.py')
VARIANTS = ('official', *ARMS)
TAGS = ('gsv-adaptation-holdout','msls-val','pitts30k-val')
OUTPUT = ROOT/'logs/clip_key_boq_eval_v1'


def order_digest(names):
    return hashlib.sha256('\n'.join(str(n) for n in names).encode()).hexdigest()


def gt_digest(ground_truth):
    return hashlib.sha256(json.dumps([np.asarray(g,dtype=np.int64).tolist() for g in ground_truth],
                                    separators=(',',':')).encode()).hexdigest()


class DualImages(Dataset):
    def __init__(self,root,names,expected_hashes=None):
        self.root = Path(root).resolve()
        self.names = [str(n) for n in names]
        self.expected_hashes = expected_hashes
        if expected_hashes is not None and len(expected_hashes) != len(names):
            raise ValueError('Image hash population differs')
        self.student, self.teacher = transform(322), transform(224,teacher=True)

    def __len__(self):
        return len(self.names)

    def __getitem__(self,index):
        p = (self.root/self.names[index]).resolve()
        if not p.is_relative_to(self.root) or not p.is_file():
            raise ValueError('Invalid/missing image path: '+str(p))
        digest = sha(p)
        if self.expected_hashes is not None and digest != self.expected_hashes[index]:
            raise ValueError('Sealed GSV image changed: '+str(p))
        with Image.open(p) as image:
            rgb = image.convert('RGB')
            student, teacher = self.student(rgb), self.teacher(rgb)
        return student,teacher,index,digest


def populations(c,plan):
    from src.dataloaders.valid.mapillary_sls import MapillarySLSDataset
    from src.dataloaders.valid.pittsburgh import PittsburghDataset
    gsv = AdaptationHoldout(c,plan)
    # Deliberately no argument/root selector that can open Pitts test data.
    datasets = ((gsv,path(c,'dataset_root'),[r['image_sha256'] for r in gsv.rows]),
        (MapillarySLSDataset(ROOT/'datasets/msls-val'),ROOT/'datasets/msls-val',None),
        (PittsburghDataset(ROOT/'datasets/pitts30k-val'),ROOT/'datasets/pitts30k-val',None))
    result = []
    for d,root,hashes in datasets:
        names = np.concatenate((d.dbImages,d.qImages)).tolist()
        if d.dataset_name not in TAGS or len(names) != d.num_references+d.num_queries:
            raise ValueError('Unexpected development population')
        if len(d.ground_truth) != d.num_queries or d.num_queries < 1 or d.num_references < 20:
            raise ValueError('Invalid development ground-truth population')
        for g in d.ground_truth:
            value = np.asarray(g)
            if value.ndim != 1 or not len(value) or not np.issubdtype(value.dtype,np.integer):
                raise ValueError('Expected nonempty integer ground-truth arrays')
            if (value < 0).any() or (value >= d.num_references).any():
                raise ValueError('Ground-truth reference index outside population')
        result.append(dict(tag=d.dataset_name,root=root,names=names,hashes=hashes,
            references=d.num_references,queries=d.num_queries,ground_truth=d.ground_truth))
    return result


def training_ready(c):
    base = identity(c)
    initial = None
    inventory = {}
    for arm in ARMS:
        run = path(c,'runs')/arm
        contract = verified(run)
        if contract != contract_for(c,arm,'train'):
            raise ValueError('Training provenance differs: '+arm)
        summary = read(run/'summary.json')
        if not all(summary['checks'].values()) or summary['steps'] != len(open_plan(c)['schedule']):
            raise ValueError('Incomplete/failed training: '+arm)
        current = read(run/'initial.json')
        if initial is not None and current != initial:
            raise ValueError('Three arms have different initialization/capacity')
        initial = current
        inventory[arm] = dict(completed_sha256=sha(run/'completed.json'),checkpoint_sha256=sha(run/'last.pt'),
                              contract_sha256=sha(run/'contract.json'))
    return base,inventory,initial


def evaluation_contract(c,sets):
    base,inventory,initial = training_ready(c)
    return dict(stage='fixed_final_four_arm_development_evaluation',training_identity=base,
        training_artifacts=inventory,matched_initialization=initial,evaluation_code=source_hashes(EVAL_SOURCES),
        protocol=dict(student_size=322,clip_size=224,clip_grid=[14,14],batch_size=16,
            microbatch=c['microbatch'],precision='fp32',key_storage_simulation='float32->float16->float32, same as training cache',
            checkpoint='Fixed epoch5, step2560; no test-set/best-epoch tuning',
            retrieval='Single-stage unit global descriptors, cosine top20; independent per-image extraction'),
        datasets={d['tag']:dict(root=str(d['root'].resolve()),references=d['references'],queries=d['queries'],
            image_order_sha256=order_digest(d['names']),ground_truth_sha256=gt_digest(d['ground_truth']),
            metadata={p.name:sha(p) for p in sorted(d['root'].glob('*.npy'))}) for d in sets},
        training_started=False,blind_test_used=False)


def build_models(c,contract):
    official = load_official(path(c,'official_cache'),path(c,'dino_source'),c['device'])
    if state_hash(official) != contract['matched_initialization']['official_initial_sha256']:
        raise ValueError('Loaded official backbone/BoQ differs from training start')
    models = {}
    for arm in ARMS:
        model = ClipKeyBoQ(official,c['hidden'],c['bias_cap'],c['key_temperature'],c['seed']).to(c['device'])
        # Entire original backbone/BoQ is shared read-only, Values unchanged.
        active = {n:p for n,p in model.named_parameters() if p.requires_grad}
        run = path(c,'runs')/arm
        payload = torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        if payload['state']['contract_sha256'] != contract['training_artifacts'][arm]['contract_sha256']:
            raise ValueError('Checkpoint contract link differs')
        if payload['state']['step'] != 2560 or set(payload['parameters']) != set(active):
            raise ValueError('Final checkpoint step/parameter inventory differs')
        with torch.no_grad():
            for n,p in active.items():
                value = payload['parameters'][n]
                if value.shape != p.shape or not torch.isfinite(value).all():
                    raise ValueError('Invalid trained parameter: '+n)
                p.copy_(value.to(p.device))
        if state_hash(model.routers) != read(run/'summary.json')['final_router_sha256']:
            raise ValueError('Final router tensors not reproduced')
        # Do not disable routers' requires_grad until all shared-base inventory
        # checks are finished. Only the original visual object is shared.
        model.requires_grad_(False).eval()
        models[arm] = model
    teacher = DenseClipKeys(path(c,'clip_checkpoint'),c['device']).requires_grad_(False).eval()
    return official,models,teacher


def retrieval_report(predictions,ground_truth,references):
    p = np.asarray(predictions)
    if p.shape != (len(ground_truth),20) or not np.issubdtype(p.dtype,np.integer):
        raise ValueError('Expected Qx20 integer rankings')
    if (p < 0).any() or (p >= references).any() or any(len(set(row.tolist())) != 20 for row in p):
        raise ValueError('Invalid/duplicate predicted indices')
    hits = np.asarray([[np.isin(row[:k],g).any() for k in (1,5,10,20)]
                       for row,g in zip(p,ground_truth)],bool)
    return dict(queries=len(p),correct=int(hits[:,0].sum()),recall_k=[1,5,10,20],
                recall=hits.mean(0).tolist(),outcomes=hits[:,0].tolist(),predictions=p.tolist())


def paired_report(reference,candidate):
    value = paired(reference['outcomes'],candidate['outcomes'])
    c,r = value['corrections'],value['regressions']
    n = c+r
    # Descriptive only: repeated development use / spatial correlation mean this
    # is not a confirmatory significance or stable-generalization claim.
    value['descriptive_mcnemar_exact_p'] = (min(1.,2*sum(math.comb(n,i) for i in range(min(c,r)+1))/2**n)
                                            if n else 1.)
    a,b = np.asarray(reference['outcomes'],bool),np.asarray(candidate['outcomes'],bool)
    value.update(correction_query_ids=np.flatnonzero(~a & b).tolist(),
                 regression_query_ids=np.flatnonzero(a & ~b).tolist(),
                 top1_changed=int((np.asarray(reference['predictions'])[:,0] != np.asarray(candidate['predictions'])[:,0]).sum()))
    return value


def compare_reports(reports):
    if set(reports) != set(VARIANTS) or len({r['queries'] for r in reports.values()}) != 1:
        raise ValueError('Mismatched variant/query populations')
    return dict(correct={a:r['correct'] for a,r in reports.items()},recall={a:r['recall'] for a,r in reports.items()},
                aligned_vs={a:paired_report(reports[a],reports['aligned']) for a in ('official','visual','shuffled')})


def validate_completed_dataset(report,d,contract):
    expected = contract['datasets'][d['tag']]
    if report['dataset'] != expected or report['tag'] != d['tag']:
        raise ValueError('Finished evaluation metadata differs')
    for arm in VARIANTS:
        r = report['variants'][arm]
        rebuilt = retrieval_report(r['predictions'],d['ground_truth'],d['references'])
        if rebuilt != r:
            raise ValueError('Saved outcomes do not reproduce rankings/GT')
    if report['comparison'] != compare_reports(report['variants']):
        raise ValueError('Saved paired counts differ')


def evaluate(c,args):
    plan = open_plan(c)
    sets = populations(c,plan)
    contract = evaluation_contract(c,sets)
    output = args.output.resolve()
    with owned_run(output,contract,args.resume):
        if (output/'completed.json').exists():
            verified(output)
            print('Evaluation already complete; no updates',flush=True)
            return
        gpu(c)
        official,models,teacher = build_models(c,contract)
        before = state_hash(official)
        timings = {}
        for d in sets:
            tag = d['tag']
            target,ready = output/(tag+'.json'),output/(tag+'.ready.json')
            if ready.exists():
                if sha(target) != read(ready)['sha256']:
                    raise ValueError('Finished dataset result checksum differs')
                saved = read(target)
                validate_completed_dataset(saved,d,contract)
                timings[tag] = saved['elapsed_seconds']
                print('Reuse completed dataset:',tag,flush=True)
                continue
            started = time.monotonic()
            dataset = DualImages(d['root'],d['names'],d['hashes'])
            loader = DataLoader(dataset,batch_size=16,num_workers=c['workers'],shuffle=False)
            # ~3.6 GiB RAM on MSLS for all four descriptor arrays; no disk cache.
            features = {a:np.empty((len(dataset),12288),np.float32) for a in VARIANTS}
            image_hashes = []
            with torch.inference_mode():
                for batch,(images,clip_images,indices,digests) in enumerate(tqdm(loader,desc='Four-arm '+tag)):
                    if indices.tolist() != list(range(batch*16,batch*16+len(images))):
                        raise ValueError('Extraction image order differs')
                    image_hashes.extend(digests)
                    for start in range(0,len(images),c['microbatch']):
                        stop = min(start+c['microbatch'],len(images))
                        student = images[start:stop].to(c['device'])
                        x = official.backbone(student)
                        keys = teacher(clip_images[start:stop].to(c['device'])).half().float()
                        names = [d['names'][i] for i in indices[start:stop].tolist()]
                        wrong = permute_patches(keys,names,c['seed'])
                        values = dict(official=official.aggregator(x)[0],
                            visual=models['visual'].aggregate(x,keys,'visual'),
                            aligned=models['aligned'].aggregate(x,keys,'aligned'),
                            shuffled=models['shuffled'].aggregate(x,wrong,'shuffled'))
                        for arm,value in values.items():
                            if (value.shape != (stop-start,12288) or not torch.isfinite(value).all()
                                    or float((value.norm(dim=1)-1).abs().max()) > 1e-4):
                                raise ValueError('Invalid global descriptors: '+arm)
                            features[arm][indices[start:stop].numpy()] = value.cpu().numpy()
                    write(output/'progress.json',dict(phase='extracting',dataset=tag,images=len(image_hashes),
                        total_images=len(dataset),variants=list(VARIANTS),elapsed_seconds=time.monotonic()-started,
                        peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,training_started=False))
                reports = {}
                for arm in VARIANTS:
                    database = torch.from_numpy(features[arm][:d['references']]).to(c['device'])
                    rankings = []
                    queries = features[arm][d['references']:]
                    for start in range(0,len(queries),16):
                        rankings.append((torch.from_numpy(queries[start:start+16]).to(c['device']) @ database.T)
                                        .topk(20,dim=1).indices.cpu().numpy())
                    reports[arm] = retrieval_report(np.concatenate(rankings),d['ground_truth'],d['references'])
                    del database
            content_digest = order_digest([n+':'+s for n,s in zip(d['names'],image_hashes)])
            report = dict(tag=tag,dataset=contract['datasets'][tag],image_content_sha256=content_digest,
                variants=reports,comparison=compare_reports(reports),elapsed_seconds=time.monotonic()-started,
                scope=('GSV adaptation-place holdout, exact-ID GT; official pretraining may contain these places.'
                       if tag=='gsv-adaptation-holdout' else 'Exposed development dataset, not blind test.'))
            validate_completed_dataset(report,d,contract)
            write(target,report)
            write(ready,dict(sha256=sha(target)))
            timings[tag] = report['elapsed_seconds']
            print(tag,json.dumps(report['comparison'],ensure_ascii=False),flush=True)
            del features,loader,dataset
        checks = dict(original_visual_unchanged=state_hash(official)==before,
            innovation1_sources_unchanged=source_hashes(PROTECTED)==contract['training_identity']['protected_sources'],
            fixed_final_checkpoints_unchanged=all(sha(path(c,'runs')/a/'last.pt') == contract['training_artifacts'][a]['checkpoint_sha256'] for a in ARMS))
        if not all(checks.values()):
            raise ValueError('Read-only evaluation integrity checks failed')
        summary = dict(results={d['tag']:read(output/(d['tag']+'.json'))['comparison'] for d in sets},checks=checks,
            dataset_seconds=timings,training_started=False,blind_test_used=False,
            scope='Fixed-final, matched single-seed development comparison; not stable benefit or confirmatory significance.',
            runtime='Shared encodings accelerate this comparison; NOT deployment latency measurement.')
        write(output/'summary.json',summary)
        write(output/'progress.json',dict(phase='complete',datasets=list(TAGS),training_started=False,blind_test_used=False))
        finish(output)
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluate',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--config',type=Path,default=ROOT/'config/clip_key_boq.yaml')
    p.add_argument('--output',type=Path,default=OUTPUT)
    args = p.parse_args()
    if not args.evaluate:
        p.error('Explicit --evaluate required; this never trains')
    evaluate(configuration(args.config),args)


if __name__=='__main__':
    main()
