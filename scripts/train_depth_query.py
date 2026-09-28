"""Small matched DSQ-BoQ screen. Epochs refer to a fixed 4096-place subset."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.candidate_set_screen import read,write,sha,complete
from scripts.adaptive_pair_budget import verified
from scripts.eval_condition_robustness import load_inference_model_from_ckpt
from src.models.depth_query import DepthQueryVPR,MODES
from src.dataloaders.train.gsv_cities import GSVCitiesDataset
from src.dataloaders.valid.mapillary_sls import MapillarySLSDataset
from src.dataloaders.valid.pittsburgh import PittsburghDataset
from src.losses.vpr_losses import VPRLossFunction

RU_SHA='38feab0601f553ed03a1ea4f6955f02bcad82618bc784cab6f4191f30e9c9f3e'
POLICY=dict(seed=42,places=4096,views=4,places_per_batch=16,epochs=3,size=280,
            aggregator_lr=1e-5,new_lr=1e-4,weight_decay=.001,depths=[6,9,12],
            scope='Single-seed fixed GSV subset development screen; MSLS/Pitts are historically exposed, not independent tests',
            selection='Report fixed last epoch and earliest best MSLS epoch; do not select by Pitts')


def seed(value):random.seed(value);np.random.seed(value);torch.manual_seed(value)


class MatchedPlaces(Dataset):
    def __init__(self,dataset,indices,epoch):self.dataset=dataset;self.indices=indices;self.epoch=epoch
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        # Per-place/epoch RNG makes worker scheduling and mid-epoch resume irrelevant.
        i=self.indices[index];r=random.getstate();n=np.random.get_state();t=torch.get_rng_state()
        try:
            value=int(hashlib.sha256(f'dsq42:{self.epoch}:{self.dataset.places_ids[i]}'.encode()).hexdigest()[:8],16)
            random.seed(value);np.random.seed(value);torch.random.default_generator.manual_seed(value)
            return self.dataset[i]
        finally:random.setstate(r);np.random.set_state(n);torch.set_rng_state(t)


def save(path,model,opt,state):
    temp=path.with_suffix('.tmp')
    torch.save(dict(parameters={k:p.detach().cpu() for k,p in model.named_parameters() if p.requires_grad},
                    optimizer=opt.state_dict(),state=state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),temp)
    temp.replace(path)


def restore(payload,model,opt):
    params={k:p for k,p in model.named_parameters() if p.requires_grad}
    if set(params)!=set(payload['parameters']):raise ValueError('Checkpoint parameters differ')
    with torch.no_grad():
        for k,p in params.items():p.copy_(payload['parameters'][k].to(p.device))
    opt.load_state_dict(payload['optimizer']);torch.set_rng_state(payload['torch_rng']);torch.cuda.set_rng_state_all(payload['cuda_rng'])
    return payload['state']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=MODES,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--dataset-root',type=Path,default=Path('datasets'))
    a=p.parse_args()
    import fcntl
    from torchvision.transforms import v2 as T
    from tqdm import tqdm
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Expose only physical GPU1')
    if not a.checkpoint.is_file() or sha(a.checkpoint)!=RU_SHA:raise ValueError('Expected original RU checkpoint')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    lock=(a.output.parent/(a.output.name+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    seed(42)
    clean=T.Compose([T.ToImage(),T.Resize((280,280),interpolation=T.InterpolationMode.BICUBIC,antialias=True),T.ToDtype(torch.float32,scale=True),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    augment=T.Compose([T.ToImage(),T.Resize((280,280),interpolation=T.InterpolationMode.BICUBIC,antialias=True),T.ColorJitter(.4,.4,.4,.1),T.RandomGrayscale(.2),T.ToDtype(torch.float32,scale=True),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    dataset=GSVCitiesDataset(dataset_path=a.dataset_root/'gsv_cities',cities='all',img_per_place=4,transform=augment)
    indices=sorted(range(len(dataset)),key=lambda i:hashlib.sha256(f'dsq-subset42:{dataset.places_ids[i]}'.encode()).hexdigest())[:32 if a.smoke else 4096]
    if len(indices)!=(32 if a.smoke else 4096):raise ValueError('Insufficient GSV places')
    vals=[MapillarySLSDataset(a.dataset_root/'msls-val',clean),PittsburghDataset(a.dataset_root/'pitts30k-val',clean)]
    source_paths=sorted((a.dataset_root/'gsv_cities').glob('Dataframes/*.csv'))
    if not source_paths:raise ValueError('GSV metadata files missing')
    for d in ('msls-val','pitts30k-val'):source_paths+=sorted((a.dataset_root/d).glob('*.npy'))
    contract=dict(mode=a.mode,smoke=a.smoke,policy=POLICY,checkpoint_sha256=RU_SHA,
        places=[str(dataset.places_ids[i]) for i in indices],data={str(f):sha(f) for f in source_paths},
        code={f:hashlib.sha256((ROOT/f).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for f in ['src/models/depth_query.py','scripts/train_depth_query.py']})
    if a.output.exists():
        if not a.resume or read(a.output/'contract.json')!=contract:raise ValueError('Existing run/contract mismatch')
        if (a.output/'completed.json').exists():verified(a.output);print('Already complete');return
    else:a.output.mkdir()
    write(a.output/'contract.json',contract)
    visual=load_inference_model_from_ckpt(a.checkpoint,'cpu').cuda().eval()
    seed(42);model=DepthQueryVPR(visual,a.mode).cuda().eval()
    # Real images, independent historical forward vs manually extracted depths.
    sample=next(iter(DataLoader(vals[0],batch_size=2,num_workers=0)))[0].cuda()
    with torch.inference_mode():
        expected=visual(sample);actual=model(sample);error=float((actual-expected).abs().max())
    if error>2e-6 or not torch.isfinite(actual).all():raise ValueError(f'Initial descriptor mismatch: {error}')
    print('PASS RU descriptor equality',error,'dimension',actual.shape[1],flush=True)
    del visual,actual,expected,sample
    agg=list(model.aggregator.parameters());ids={id(x) for x in agg}
    new=[p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    groups=[dict(params=agg,lr=POLICY['aggregator_lr'])]
    if new:groups.append(dict(params=new,lr=POLICY['new_lr']))
    opt=torch.optim.AdamW(groups,weight_decay=POLICY['weight_decay']);loss_fn=VPRLossFunction()
    state=dict(epoch=0,cursor=0,loss_sum=0.,zero_loss_batches=0,history=[],best_epoch=0,best_correct=-1,contract_sha256=sha(a.output/'contract.json'))
    if (a.output/'last.pt').exists():
        state=restore(torch.load(a.output/'last.pt',map_location='cpu',weights_only=True),model,opt)
        if state['contract_sha256']!=sha(a.output/'contract.json'):raise ValueError('Resume identity differs')
    def evaluate(epoch):
        model.eval();report={}
        for ds in vals:
            descriptors=[];write(a.output/'progress.json',dict(phase='validation',epoch=epoch,dataset=ds.dataset_name,done=0,total=len(ds)))
            with torch.inference_mode():
                for images,_ in tqdm(DataLoader(ds,batch_size=32,num_workers=a.workers),desc=f'{a.mode} e{epoch} {ds.dataset_name}'):
                    descriptors.append(model(images.cuda()).cpu())
                    if len(descriptors)%32==0:write(a.output/'progress.json',dict(phase='validation',epoch=epoch,dataset=ds.dataset_name,done=min(len(descriptors)*32,len(ds)),total=len(ds)))
            features=torch.cat(descriptors).cuda();db=features[:ds.num_references];preds=[]
            for q in features[ds.num_references:].split(64):preds.append((q@db.T).topk(20,dim=1).indices.cpu().numpy())
            predictions=np.concatenate(preds);hits=np.array([[bool(np.isin(row[:k],gt).any()) for k in (1,5,20)] for row,gt in zip(predictions,ds.ground_truth)])
            counts=hits.sum(0).tolist();report[ds.dataset_name]=dict(queries=ds.num_queries,correct=counts[0],r1=counts[0]/ds.num_queries,r5=counts[1]/ds.num_queries,r20=counts[2]/ds.num_queries)
            write(a.output/f'{ds.dataset_name}_epoch{epoch:02d}.json',dict(predictions=predictions.tolist(),correct=hits[:,0].tolist()))
            del features,db,descriptors
        print(report,flush=True);return report
    if not state['history']:
        if not a.smoke:
            initial=evaluate(0)
            if initial['msls-val']['correct']!=675:raise ValueError('Initial MSLS RU recall mismatch')
            state['history']=[dict(epoch=0,validation=initial)];state['best_correct']=675
        save(a.output/'best.pt',model,opt,state);save(a.output/'last.pt',model,opt,state)
    seed(42)
    epochs=1 if a.smoke else 3
    while state['epoch']<epochs:
        epoch=state['epoch'];order=np.random.default_rng(42+epoch).permutation(indices).tolist()
        remaining=order[state['cursor']:];data=MatchedPlaces(dataset,remaining,epoch)
        model.train();bar=tqdm(DataLoader(data,batch_size=16,num_workers=a.workers,drop_last=False),desc=f'{a.mode} train {epoch+1}/{epochs}')
        grad_probe={}
        for images,labels in bar:
            opt.zero_grad(set_to_none=True);desc=model(images.flatten(0,1).cuda());loss,acc=loss_fn(desc,labels.flatten().cuda())
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            if state['cursor']==16 and a.smoke:
                # A miner may return no informative pairs. Probe differentiability
                # separately; this scalar is NEVER optimized or added to the loss.
                probe=(desc*torch.linspace(-1.,1.,desc.shape[1],device=desc.device)).sum()
                for name,module in [('aggregator',model.aggregator),('residuals',model.residuals),('routers',model.routers),('align',model.align)]:
                    active=[p for p in module.parameters() if p.requires_grad]
                    grads=torch.autograd.grad(probe,active,allow_unused=True,retain_graph=True) if active else []
                    grad_probe[name]=sum(float(g.abs().sum()) for g in grads if g is not None)
                required=['aggregator']+(['residuals','align'] if a.mode!='baseline' else [])+(['routers'] if a.mode in ('adaptive_depth','last_repeat') else [])
                if any(grad_probe[k]<=0 for k in required):raise ValueError(f'Dead branch: {grad_probe}')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            if any(p.grad is not None for p in model.backbone.parameters()):raise ValueError('Frozen backbone received gradients')
            state['zero_loss_batches']+=int(float(loss.detach())==0.)
            opt.step();state['cursor']+=len(images);state['loss_sum']+=float(loss.detach())*len(images)
            bar.set_postfix(loss=float(loss.detach()),acc=acc)
            write(a.output/'progress.json',dict(phase='training',epoch=epoch+1,done=state['cursor'],total=len(indices)))
            if state['cursor']%256==0:save(a.output/'last.pt',model,opt,state)
        train_loss=state['loss_sum']/len(indices)
        validation={} if a.smoke else evaluate(epoch+1)
        routes=None if model.last_depth_weights is None else model.last_depth_weights.mean(dim=(0,1,2)).cpu().tolist()
        state['history'].append(dict(epoch=epoch+1,train_loss=train_loss,zero_loss_batches=state['zero_loss_batches'],total_batches=len(indices)//16,validation=validation,last_batch_depth_weights=routes))
        state['epoch']+=1;state['cursor']=0;state['loss_sum']=0.;state['zero_loss_batches']=0
        if not a.smoke and validation['msls-val']['correct']>state['best_correct']:
            state['best_correct']=validation['msls-val']['correct'];state['best_epoch']=state['epoch'];save(a.output/'best.pt',model,opt,state)
        save(a.output/'last.pt',model,opt,state);write(a.output/'history.json',state['history'])
        if a.smoke:
            before={k:p.detach().clone() for k,p in model.named_parameters() if p.requires_grad}
            restore(torch.load(a.output/'last.pt',map_location='cpu',weights_only=True),model,opt)
            if any(not torch.equal(before[k],p) for k,p in model.named_parameters() if p.requires_grad):raise ValueError('Resume roundtrip mismatch')
            write(a.output/'preflight.json',dict(initial_max_error=error,descriptor_probe_gradients=grad_probe,probe_scope='Connectivity probe only, not VPR loss gradients; never optimized',checkpoint_roundtrip=True))
    write(a.output/'summary.json',dict(mode=a.mode,smoke=a.smoke,history=state['history'],best_epoch=state['best_epoch'],trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),scope=POLICY['scope']))
    write(a.output/'progress.json',dict(phase='complete'));complete(a.output);print('COMPLETE',a.output,flush=True)


if __name__=='__main__':main()
