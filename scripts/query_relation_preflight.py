"""QR-BoQ real-update preflights and fixed expanded-gallery RU audit; no formal training."""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_depth_query import seed,MatchedPlaces,RU_SHA
from scripts.candidate_set_screen import read,write,sha,complete
from scripts.adaptive_pair_budget import verified
from scripts.eval_condition_robustness import load_inference_model_from_ckpt
from src.models.depth_query import DepthQueryVPR
from src.models.query_relation import MODES,install
from src.dataloaders.train.gsv_cities import GSVCitiesDataset
from src.losses.vpr_losses import VPRLossFunction


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--plan',type=Path,default=Path('doc/local_value_plan_v1'))
    p.add_argument('--output',type=Path,default=Path('logs/query_relation_preflight_v1'))
    a=p.parse_args()
    import os,fcntl
    from torch.utils.data import DataLoader
    from torchvision.transforms import v2 as T
    from tqdm import tqdm
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise ValueError('Physical GPU1 only')
    # Leave room for unrelated services; never terminate their processes.
    torch.cuda.set_per_process_memory_fraction(.4,0)
    if not a.checkpoint.is_file() or sha(a.checkpoint)!=RU_SHA:raise ValueError('Original RU required')
    verified(a.plan);plan=read(a.plan/'plan.json')
    if len(plan['train_places'])!=4096 or len(plan['dev_places'])!=1024 or set(plan['train_places'])&set(plan['dev_places']):raise ValueError('Invalid split')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    lock=(a.output.parent/(a.output.name+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.output.exists():raise ValueError('Use fresh output; old evidence never overwritten')
    a.output.mkdir()
    files=['scripts/query_relation_preflight.py','src/models/query_relation.py','src/models/depth_query.py','scripts/train_depth_query.py','src/models/aggregators/boq.py','src/dataloaders/train/gsv_cities.py','scripts/eval_condition_robustness.py']
    metadata=sorted(Path('datasets/gsv_cities/Dataframes').glob('*.csv'))
    write(a.output/'contract.json',dict(ru_sha256=RU_SHA,plan_sha256=sha(a.plan/'completed.json'),
        code={f:hashlib.sha256((ROOT/f).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for f in files},metadata={str(f):sha(f) for f in metadata},
        policy=dict(gallery='all four clean epoch0 views of4096 train and1024 development places; queries only development; exclude self',
                    ground_truth='same place identity; distinct nearby places may be false negatives; no geographical certification',
                    seed=42,smoke_steps=2,cuda_allocator_fraction=.4,formal_training=False)))
    seed(42);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    clean=T.Compose([T.ToImage(),T.Resize((280,280),interpolation=T.InterpolationMode.BICUBIC,antialias=True),T.ToDtype(torch.float32,scale=True),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    dataset=GSVCitiesDataset(dataset_path=Path('datasets/gsv_cities'),cities='all',img_per_place=4,transform=clean)
    lookup={str(v):i for i,v in enumerate(dataset.places_ids)}
    sample=MatchedPlaces(dataset,[lookup[plan['dev_places'][0]]],0)[0][0].cuda()
    reports={}
    for mode in MODES:
        write(a.output/'progress.json',dict(phase='smoke',mode=mode))
        visual=load_inference_model_from_ckpt(a.checkpoint,'cpu').cuda().eval()
        with torch.inference_mode():expected=visual(sample)
        model=DepthQueryVPR(visual,'baseline').cuda().eval().requires_grad_(False)
        seed(42);adapter=install(model.aggregator,mode);model.cuda().eval()
        active=[p for p in model.parameters() if p.requires_grad]
        names={n for n,p in model.named_parameters() if p.requires_grad}
        frozen={n:t.cpu().clone() for n,t in model.state_dict().items() if n not in names}
        with torch.inference_mode():error=float((model(sample)-expected).abs().max())
        if error>2e-6:raise ValueError('RU zero-start mismatch')
        opt=torch.optim.AdamW(active,lr=1e-4,weight_decay=0);loss_fn=VPRLossFunction();losses=[];probe={}
        for step,batch in enumerate(plan['epochs'][0]['batches'][:2]):
            data=MatchedPlaces(dataset,[lookup[v] for v in batch],0)
            images,labels=next(iter(DataLoader(data,batch_size=16,num_workers=0)))
            opt.zero_grad(set_to_none=True)
            desc=torch.cat([model(part.cuda()) for part in images.flatten(0,1).split(8)])
            loss,_=loss_fn(desc,labels.flatten().cuda())
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            if step==1:
                grads=torch.autograd.grad((desc*torch.linspace(-1,1,desc.shape[1],device='cuda')).sum(),active,retain_graph=True)
                probe={n:float(g.abs().sum()) for (n,v),g in zip([(n,v) for n,v in model.named_parameters() if v.requires_grad],grads)}
                if any(not np.isfinite(g) or g<=0 for g in probe.values()):raise ValueError('Dead upstream branch after first update')
            loss.backward();torch.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True)
            if any(v.grad is not None for v in model.parameters() if not v.requires_grad):raise ValueError('Frozen gradients')
            if loss.item()>0:opt.step()
            losses.append(loss.item())
            del desc,loss
        if not any(v>0 for v in losses) or adapter.output.weight.norm()==0:raise ValueError('No genuine update')
        if any(not torch.equal(t.cpu(),frozen[n]) for n,t in model.state_dict().items() if n in frozen):raise ValueError('Frozen state changed')
        reports[mode]=dict(initial_error=error,losses=losses,trainable_parameters=sum(v.numel() for v in active),probe=probe,diagnostics=adapter.last_diagnostics,frozen_unchanged=True)
        write(a.output/'preflights.json',reports)
        del model,visual,adapter,opt,active,expected,frozen;torch.cuda.empty_cache()
    write(a.output/'progress.json',dict(phase='expanded_gallery_baseline'))
    visual=load_inference_model_from_ckpt(a.checkpoint,'cpu').cuda().eval()
    ids=plan['train_places']+plan['dev_places'];features=[]
    data=MatchedPlaces(dataset,[lookup[v] for v in ids],0)
    with torch.inference_mode():
        for images,_ in tqdm(DataLoader(data,batch_size=8,num_workers=4),desc='Expanded GSV RU gallery'):
            features.append(visual(images.flatten(0,1).cuda()).cpu())
            write(a.output/'progress.json',dict(phase='expanded_gallery_baseline',images=sum(len(f) for f in features),total=20480))
    features=torch.cat(features).cuda();offset=4096*4;outcomes=[];predictions=[];margins=[]
    with torch.inference_mode():
        labels=torch.arange(len(features),device='cuda')//4
        for start in range(offset,len(features),32):
            scores=features[start:start+32]@features.T;rows=torch.arange(len(scores),device='cuda')
            scores[rows,start+rows]=-torch.inf
            pos=labels[start:start+len(scores),None]==labels[None]
            margin=scores.masked_fill(~pos,-torch.inf).amax(1)-scores.masked_fill(pos,-torch.inf).amax(1)
            pred=scores.argmax(1);hit=labels[pred]==labels[start:start+len(scores)]
            outcomes.extend(hit.cpu().tolist());predictions.extend(pred.cpu().tolist());margins.extend(margin.cpu().tolist())
    report=dict(preflights=reports,expanded_gallery=dict(queries=len(outcomes),gallery=20480,correct=sum(outcomes),errors=len(outcomes)-sum(outcomes),r1=sum(outcomes)/len(outcomes)),
        previous_small_gallery_correct=4081,scope='Baseline difficulty audit, NOT QR training results; nearby distinct places may be ambiguous negatives')
    write(a.output/'baseline_outcomes.json',dict(place_ids=ids,query_start=offset,correct=outcomes,predictions=predictions,margins=margins))
    write(a.output/'summary.json',report);write(a.output/'progress.json',dict(phase='complete'));complete(a.output);print(report,flush=True)


if __name__=='__main__':main()
