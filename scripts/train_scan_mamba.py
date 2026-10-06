"""SC-Mamba-BoQ matched pilot/full training. No benchmark checkpoint selection."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_depth_query import seed, MatchedPlaces, save, restore, RU_SHA
from scripts.candidate_set_screen import read,write,sha,complete
from scripts.adaptive_pair_budget import verified
from scripts.eval_condition_robustness import load_inference_model_from_ckpt
from src.models.scan_mamba import MODES,ScanMambaVPR,descriptor_consistency
from src.dataloaders.train.gsv_cities import GSVCitiesDataset
from src.dataloaders.valid.mapillary_sls import MapillarySLSDataset
from src.dataloaders.valid.pittsburgh import PittsburghDataset
from src.losses.vpr_losses import VPRLossFunction
from src.query_relation_training import score_development

POLICY=dict(seed=42,size=280,views=4,places_per_batch=16,holdout_places=1024,
            aggregator_lr=1e-5,mixer_lr=1e-4,weight_decay=0.,clip=1.,precision='fp32',
            inner=64,state=8,rank=4,consistency_weight=.1,pilot_steps=128,
            full_epochs=3,microbatch=4,selection='Fixed last only; pilot is not an accuracy gate',
            implementation='Pure PyTorch Mamba-1 style real selective recurrence; logarithmic-depth prefix reference, not fused kernels')


def code_hash(path):return hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=MODES,required=True)
    p.add_argument('--stage',choices=('smoke','pilot','full'),default='pilot')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    from torchvision.transforms import v2 as T
    from tqdm import tqdm
    import fcntl
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1' or torch.cuda.device_count()!=1:raise ValueError('Expose physical GPU1 only')
    if not a.checkpoint.is_file() or sha(a.checkpoint)!=RU_SHA:raise ValueError('Original RU required')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    lock=(a.output.parent/(a.output.name+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.cuda.set_per_process_memory_fraction(.4,0)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    seed(42)
    start=[T.ToImage(),T.Resize((280,280),interpolation=T.InterpolationMode.BICUBIC,antialias=True)]
    end=[T.ToDtype(torch.float32,scale=True),T.Normalize([.485,.456,.406],[.229,.224,.225])]
    augment=T.Compose(start+[T.ColorJitter(.4,.4,.4,.1),T.RandomGrayscale(.2)]+end)
    clean=T.Compose(start+end)
    train=GSVCitiesDataset(dataset_path=Path('datasets/gsv_cities'),cities='all',img_per_place=4,transform=augment)
    seed(42)
    dev=GSVCitiesDataset(dataset_path=Path('datasets/gsv_cities'),cities='all',img_per_place=4,transform=clean)
    if list(train.places_ids)!=list(dev.places_ids):raise ValueError('Dataset order differs')
    order=sorted(range(len(train)),key=lambda i:hashlib.sha256(f'scan-mamba42:{train.places_ids[i]}'.encode()).hexdigest())
    if len(order)<6144:raise ValueError('Insufficient places')
    holdout,training=order[:1024],order[1024:]
    schedule=[]
    for epoch in range(3 if a.stage=='full' else 1):
        indices=np.random.default_rng(62031+epoch).permutation(training).tolist()
        batches=[indices[i:i+16] for i in range(0,len(indices)-15,16)]
        schedule.append(batches[:4 if a.stage=='smoke' else 128] if a.stage!='full' else batches)
    paths=['scripts/train_scan_mamba.py','src/models/scan_mamba.py','src/models/depth_query.py',
           'src/models/aggregators/boq.py','src/losses/vpr_losses.py','scripts/train_depth_query.py',
           'src/dataloaders/train/gsv_cities.py','scripts/eval_condition_robustness.py','src/models/backbones/dinov2.py']
    metadata=sorted(Path('datasets/gsv_cities/Dataframes').glob('*.csv'))
    if not metadata:raise ValueError('Missing metadata')
    contract=dict(mode=a.mode,stage=a.stage,policy=POLICY,ru_sha256=RU_SHA,
        source={name:code_hash(ROOT/name) for name in paths},metadata={str(f):sha(f) for f in metadata},
        versions=dict(torch=torch.__version__,numpy=np.__version__),
        train_places=[str(train.places_ids[i]) for i in training],holdout_places=[str(train.places_ids[i]) for i in holdout],
        schedule=[[[str(train.places_ids[i]) for i in b] for b in epoch] for epoch in schedule])
    if a.output.exists():
        if not a.resume or read(a.output/'contract.json')!=contract:raise ValueError('Run contract differs')
        if (a.output/'completed.json').exists():verified(a.output);print('Already complete');return
    else:a.output.mkdir();write(a.output/'contract.json',contract)
    seed(42)
    visual=load_inference_model_from_ckpt(a.checkpoint,'cpu').cuda().eval()
    sample=MatchedPlaces(dev,holdout[:1],0)[0][0].cuda()
    with torch.no_grad():expected=visual(sample)
    seed(42);model=ScanMambaVPR(visual,a.mode).cuda().eval()
    with torch.no_grad():err=float((model(sample)-expected).abs().max())
    if not np.isfinite(err) or err>2e-6:raise ValueError('Zero-start RU mismatch')
    active={k:v for k,v in model.named_parameters() if v.requires_grad}
    frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k not in active}
    groups=[dict(params=list(model.base.aggregator.parameters()),lr=1e-5)]
    if model.mixer is not None:groups.append(dict(params=list(model.mixer.parameters()),lr=1e-4))
    opt=torch.optim.AdamW(groups,weight_decay=0.)
    lossfn=VPRLossFunction()
    state=dict(epoch=0,cursor=0,steps=0,rows=[],contract_sha256=sha(a.output/'contract.json'),ssm_gradient_seen=False)
    if (a.output/'last.pt').exists():
        state=restore(torch.load(a.output/'last.pt',map_location='cpu',weights_only=True),model,opt)
        if state['contract_sha256']!=sha(a.output/'contract.json'):raise ValueError('Resume mismatch')
    print('MODE',a.mode,'STAGE',a.stage,'zero error',err,'trainable',sum(v.numel() for v in active.values()),flush=True)

    def descriptors(images,training=False):
        result=[]
        for chunk in images.split(4):
            features=model.features(chunk)
            result.append(checkpoint(model.aggregate,features,use_reentrant=False) if training else model.aggregate(features))
        return tuple(torch.cat([r[i] for r in result]) for i in range(3))

    @torch.no_grad()
    def probe(tag):
        # Identical augmented images, labels and batches before and after training.
        indices=[i for b in schedule[0][:4 if a.stage=='smoke' else 8] for i in b]
        results=[]
        for images,labels in DataLoader(MatchedPlaces(train,indices,0),batch_size=16,num_workers=a.workers):
            d,r,c=descriptors(images.flatten(0,1).cuda())
            loss,_=lossfn(d,labels.flatten().cuda())
            results.append(dict(vpr_loss=float(loss),scan_disagreement=float(descriptor_consistency(r,c))))
        write(a.output/f'probe_{tag}.json',results)
        indices=holdout[:8 if a.stage=='smoke' else 128]
        ds=[];disagreement=[]
        for images,_ in DataLoader(MatchedPlaces(dev,indices,0),batch_size=8,num_workers=a.workers):
            d,r,c=descriptors(images.flatten(0,1).cuda());ds.append(d.cpu());disagreement.append(float(descriptor_consistency(r,c)))
        f=torch.cat(ds)
        if tag=='initial':torch.save(f,a.output/'initial_dev.tmp');(a.output/'initial_dev.tmp').replace(a.output/'initial_dev.pt')
        reference=torch.load(a.output/'initial_dev.pt',weights_only=True)
        result=score_development(f.cuda(),0)
        result.update(descriptor_drift=float((f-reference).norm(dim=1).mean()),scan_disagreement=float(np.mean(disagreement)))
        write(a.output/f'holdout_{tag}.json',result)

    if not (a.output/'holdout_initial.json').exists():
        if state['steps']!=0:raise ValueError('Missing initial diagnostic')
        probe('initial');save(a.output/'last.pt',model,opt,state)
    for epoch in range(state['epoch'],len(schedule)):
        indices=[i for b in schedule[epoch][state['cursor']:] for i in b]
        data=MatchedPlaces(train,indices,epoch)
        for images,labels in tqdm(DataLoader(data,batch_size=16,num_workers=a.workers),desc=f'{a.mode} {a.stage} e{epoch+1}'):
            t=time.monotonic();opt.zero_grad(set_to_none=True)
            d,row,col=descriptors(images.flatten(0,1).cuda(),True)
            vpr,_=lossfn(d,labels.flatten().cuda())
            consistency=descriptor_consistency(row,col)
            loss=vpr + (.1*consistency if a.mode=='mamba_consistent' else 0)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward()
            grad=float(torch.nn.utils.clip_grad_norm_(active.values(),1.,error_if_nonfinite=True))
            ssmgrad=None
            if a.mode.startswith('mamba'):
                grad_value=model.mixer.ssm.select.weight.grad
                ssmgrad=float(grad_value.abs().sum()) if grad_value is not None else 0.
                state['ssm_gradient_seen'] |= ssmgrad>0
            if float(loss.detach())!=0:opt.step();state['steps']+=1
            state['cursor']+=1
            state['rows'].append(dict(epoch=epoch+1,batch=state['cursor'],vpr=float(vpr.detach()),
                consistency=float(consistency.detach()),gradient_norm=grad,ssm_gradient=ssmgrad,seconds=time.monotonic()-t))
            write(a.output/'progress.json',dict(phase='training',mode=a.mode,stage=a.stage,epoch=epoch+1,done=state['cursor'],total=len(schedule[epoch]),steps=state['steps']))
            if state['cursor']%16==0:save(a.output/'last.pt',model,opt,state)
        state.update(epoch=epoch+1,cursor=0);save(a.output/'last.pt',model,opt,state)
    if state['steps']==0:raise ValueError('No actual optimizer update')
    if a.mode.startswith('mamba') and not state['ssm_gradient_seen']:raise ValueError('No upstream selective-state gradient')
    probe('final')
    if a.stage=='full':
        with torch.no_grad():
            for ds in (MapillarySLSDataset(Path('datasets/msls-val'),clean),PittsburghDataset(Path('datasets/pitts30k-val'),clean)):
                features=[]
                for images,_ in tqdm(DataLoader(ds,batch_size=16,num_workers=a.workers),desc=ds.dataset_name):features.append(descriptors(images.cuda())[0].cpu())
                f=torch.cat(features).cuda();db=f[:ds.num_references]
                predictions=np.concatenate([(q@db.T).topk(20,dim=1).indices.cpu().numpy() for q in f[ds.num_references:].split(32)])
                hits=np.asarray([[np.isin(row[:k],gt).any() for k in (1,5,20)] for row,gt in zip(predictions,ds.ground_truth)])
                write(a.output/(ds.dataset_name+'.json'),dict(correct=hits[:,0].tolist(),predictions=predictions.tolist(),recall=hits.mean(0).tolist()))
    for k,v in model.state_dict().items():
        if k in frozen and not torch.equal(v.cpu(),frozen[k]):raise ValueError('Frozen parameter or buffer changed: '+k)
    before={k:v.detach().clone() for k,v in active.items()}
    restore(torch.load(a.output/'last.pt',map_location='cpu',weights_only=True),model,opt)
    if any(not torch.equal(v,before[k]) for k,v in active.items()):raise ValueError('Checkpoint roundtrip mismatch')
    write(a.output/'history.json',state['rows'])
    write(a.output/'summary.json',dict(mode=a.mode,stage=a.stage,optimizer_steps=state['steps'],zero_start_error=err,
        frozen_unchanged=True,checkpoint_roundtrip=True,ssm_gradient_seen=state['ssm_gradient_seen'],
        trainable_parameters=sum(v.numel() for v in active.values()),
        mixer_parameters=sum(v.numel() for v in model.mixer.parameters()) if model.mixer else 0,
        max_memory_allocated=torch.cuda.max_memory_allocated(),scope='Pilot is mechanism/learning feasibility, not benchmark success or failure' if a.stage!='full' else 'Fixed last single-seed exploratory benchmark; not independent final tests'))
    write(a.output/'progress.json',dict(phase='complete',stage=a.stage,mode=a.mode));complete(a.output)
    print('COMPLETE',a.output,flush=True)


if __name__=='__main__':main()
