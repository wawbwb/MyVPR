"""Replay all locked augmented training batches with RU and fixed-last adapters."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.candidate_set_screen import read, write, sha, complete
from scripts.adaptive_pair_budget import verified
from scripts.train_depth_query import MatchedPlaces, seed, RU_SHA
from scripts.eval_condition_robustness import load_inference_model_from_ckpt
from src.models.depth_query import DepthQueryVPR
from src.models.query_relation import install
from src.dataloaders.train.gsv_cities import GSVCitiesDataset
from src.losses.vpr_losses import VPRLossFunction
from src.query_loss_audit import paired_losses

ARMS = ('mixed_1e-4', 'mixed_1e-5', 'broad_matched_1e-4', 'broad_matched_1e-5')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, default=Path('logs/query_protocol_screen_v1'))
    p.add_argument('--output', type=Path, default=Path('logs/query_protocol_loss_audit_v1'))
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    from torchvision.transforms import v2 as T
    from tqdm import tqdm
    import fcntl
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or torch.cuda.device_count() != 1:
        raise ValueError('Expose only physical GPU1')
    if not a.checkpoint.is_file() or sha(a.checkpoint) != RU_SHA: raise ValueError('Wrong RU')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    lock = (a.output.parent/'query_relation_screen_v1.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.cuda.set_per_process_memory_fraction(.4, 0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    contracts = {}
    for arm in ARMS:
        run = a.runs/arm
        verified(run)
        c = read(run/'contract.json')
        if c['mode'] != 'appearance' or c['smoke'] or c['checkpoint_sha256'] != RU_SHA:
            raise ValueError('Unexpected source')
        for name, digest in c['code'].items():
            if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != digest:
                raise ValueError('Training source changed: '+name)
        for name, digest in c['data'].items():
            if sha(Path(name)) != digest: raise ValueError('Metadata changed: '+name)
        contracts[arm] = c
    contract = dict(sources={arm:sha(a.runs/arm/'completed.json') for arm in ARMS},
                    code={name:sha(ROOT/name) for name in ('scripts/audit_query_protocol_loss.py','src/query_loss_audit.py')},
                    scope='Fixed epoch3 vs RU on all 960 original augmented training batches per arm; not historical online losses or held-out generalization')
    if a.output.exists():
        if not a.resume or read(a.output/'contract.json') != contract: raise ValueError('Audit contract differs')
        if (a.output/'completed.json').exists(): verified(a.output); print('Already complete'); return
    else:
        a.output.mkdir(); write(a.output/'contract.json', contract)
    seed(42)
    transform = T.Compose([T.ToImage(), T.Resize((280,280), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ColorJitter(.4,.4,.4,.1), T.RandomGrayscale(.2), T.ToDtype(torch.float32,scale=True),
        T.Normalize([.485,.456,.406],[.229,.224,.225])])
    dataset = GSVCitiesDataset(dataset_path=Path('datasets/gsv_cities'), cities='all', img_per_place=4, transform=transform)
    id_to_index = {str(v):i for i,v in enumerate(dataset.places_ids)}
    visual = load_inference_model_from_ckpt(a.checkpoint,'cpu').cuda().eval()
    model = DepthQueryVPR(visual,'baseline').cuda().eval().requires_grad_(False)
    seed(42); adapter = install(model.aggregator,'appearance'); model.cuda().eval()
    parameters = {k:v for k,v in model.named_parameters() if v.requires_grad}
    objective = VPRLossFunction()
    summaries = {}
    for arm in ARMS:
        payload = torch.load(a.runs/arm/'last.pt', map_location='cpu', weights_only=True)
        if payload['state']['epoch'] != 3 or payload['state']['cursor'] != 0 or set(payload['parameters']) != set(parameters):
            raise ValueError('Not a complete matched fixed-last checkpoint')
        if payload['state']['contract_sha256'] != sha(a.runs/arm/'contract.json'): raise ValueError('Checkpoint identity differs')
        with torch.no_grad():
            for name,value in parameters.items(): value.copy_(payload['parameters'][name].to(value.device))
        file = a.output/(arm+'.json')
        rows = read(file) if file.exists() else []
        for epoch,batches in enumerate(contracts[arm]['batch_schedule']):
            start = max(0,len(rows)-epoch*320)
            if start >= 320: continue
            indices = [id_to_index[v] for b in batches[start:] for v in b]
            loader = DataLoader(MatchedPlaces(dataset,indices,epoch),batch_size=16,num_workers=4)
            for batch,(images,labels) in enumerate(tqdm(loader,desc=f'{arm} replay e{epoch+1}'),start):
                ru,trained = [],[]
                with torch.no_grad():
                    for chunk in images.flatten(0,1).cuda().split(8):
                        layers = model.features(chunk)
                        adapter.enabled = False; ru.append(model.aggregate(layers))
                        adapter.enabled = True; trained.append(model.aggregate(layers))
                    result = paired_losses(objective,torch.cat(ru),torch.cat(trained),labels.flatten().cuda())
                rows.append(dict(epoch=epoch+1,batch=batch,
                    group='base_broad' if batch%5 != 4 else ('hard' if arm.startswith('mixed') else 'extra_broad'),**result))
                if len(rows)%16 == 0: write(file,rows)
                write(a.output/'progress.json',dict(phase='running',arm=arm,done=len(rows),total=960))
            write(file,rows)
        if len(rows) != 960: raise ValueError('Incomplete replay')
        report = {}
        for epoch in (1,2,3):
            for group in ('all','base_broad','hard','extra_broad'):
                selected = [r for r in rows if r['epoch']==epoch and (group=='all' or r['group']==group)]
                if not selected: continue
                report[f'epoch{epoch}/{group}'] = dict(batches=len(selected),
                    means={k:float(np.mean([r[k] for r in selected])) for k in rows[0] if k not in ('epoch','batch','group')},
                    loss_improved=sum(r['trained_loss']<r['ru_loss']-1e-8 for r in selected),
                    loss_worsened=sum(r['trained_loss']>r['ru_loss']+1e-8 for r in selected))
        summaries[arm] = report
        write(a.output/'summary.json',summaries)
    write(a.output/'progress.json',dict(phase='complete'))
    complete(a.output)
    print('COMPLETE',a.output,flush=True)


if __name__ == '__main__': main()
