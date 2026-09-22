"""Frozen layer2 CLS probe: small GSV place-disjoint feasibility pilot, no deployment."""
import argparse
from collections import OrderedDict
from pathlib import Path
import hashlib
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.candidate_set_screen import read,write,sha,npz,load_npz,complete,official,image
from scripts import candidate_hard_screen as hard
from scripts.adaptive_pair_budget import verified

POLICY=dict(train=512,calibration=128,evaluation=256,layer=2,topk=44,survivors=12,
    epochs=20,seed=42,lr=.001,batch=16,hidden=128,
    selection='calibration teacher-winner retention@12, earliest tie',
    gate='eval retention>=.99, >=10 tail winners and tail retention>=.9; otherwise do not deploy',
    scope='GSV adaptation pilot, pretrained exposure unknown; cal/eval queries place-disjoint but share dev reference bank')

def code():return hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()

def select(plan):
    hard.ensure_disjoint(plan)
    chosen={}
    for split,names in [('train',[('train',512)]),('dev',[('calibration',128),('evaluation',256)])]:
        q=plan[split]['queries']
        order=sorted(range(len(q)),key=lambda i:hashlib.sha256(('wppr42:'+q[i]['label']).encode()).hexdigest())
        if len(order)<sum(n for _,n in names):raise ValueError('Insufficient places')
        offset=0
        for name,n in names:
            chosen[name]=dict(split=split,indices=order[offset:offset+n]);offset+=n
    labels=[{plan[v['split']]['queries'][i]['label'] for i in v['indices']} for v in chosen.values()]
    if any(labels[i]&labels[j] for i in range(3) for j in range(i)):raise ValueError('Place overlap')
    return chosen

def capture_cls(output):
    import torch
    if not isinstance(output,tuple) or len(output)!=2 or output[0].ndim!=3:raise ValueError('Unexpected decoder interface')
    return torch.nn.functional.layer_norm(output[0][:,0,:].float(),(output[0].shape[-1],)).detach().clone()

def retention(pred,teacher,k=12):
    if pred.shape!=teacher.shape or pred.ndim!=2 or pred.shape[1]!=44 or not np.isfinite(pred).all() or not np.isfinite(teacher).all():raise ValueError('Invalid scores')
    keep=np.argsort(-pred,axis=1,kind='stable')[:,:k]
    winner=np.argmax(teacher,axis=1)
    return (keep==winner[:,None]).any(1),keep,winner

def prepare(a):
    verified(a.plan);p=read(a.plan/'contract.json')
    if p['code']!=hard.codes():raise ValueError('Plan code mismatch')
    chosen=select(p['plan']);sources={}
    for split in ['train','dev']:
        verified(a.cache/split);c=read(a.cache/split/'contract.json')
        if c['code']!=hard.codes() or c['plan_sha256']!=sha(a.plan/'completed.json'):raise ValueError('Cache identity mismatch')
        sources[split]=sha(a.cache/split/'completed.json')
    contract=dict(code=code(),policy=POLICY,selection=chosen,plan_sha256=sha(a.plan/'completed.json'),sources=sources)
    if a.output.exists():
        if read(a.output/'contract.json')!=contract:raise ValueError('Existing run differs')
    else:a.output.mkdir(parents=True);write(a.output/'contract.json',contract)
    return contract,p['plan']

def extract(a,c,plan):
    import torch
    from tqdm import tqdm
    model,identity=official(a)
    if len(model.dec_blocks)!=12:raise ValueError('Expected 12 decoder layers')
    captured=[]
    handle=model.dec_blocks[1].register_forward_hook(lambda m,i,o:captured.append(capture_cls(o)))
    try:
        for split in ['train','dev']:
            cc=read(a.cache/split/'contract.json')
            if identity!=cc['official']:raise ValueError('Frozen checkpoint differs')
            records=plan[split]['database']+plan[split]['queries'];nd=len(plan[split]['database'])
            z=[];hs=[]
            for start in range(0,len(records),128):
                block=load_npz(a.cache/split/'global'/f'{start:07d}.npz');z.append(block['vectors']);hs.extend(block['hashes'].tolist())
            z=np.concatenate(z)
            if z.shape!=(len(records),512) or len(hs)!=len(records):raise ValueError('Bad globals')
            db=torch.from_numpy(z[:nd]).cuda();memo=OrderedDict()
            def dense(i):
                if i in memo:memo.move_to_end(i);return memo[i].cuda()
                x,_=image(a.gsv_root,records[i],hs[i]);f,g=model(x[None].cuda(),None,'global')
                if not np.allclose(g[0].cpu().numpy(),z[i],atol=2e-5,rtol=2e-4):raise ValueError('Encoder reproduction failed')
                memo[i]=f.cpu()
                if len(memo)>64:memo.popitem(last=False)
                return f
            with torch.inference_mode():
                for name,sel in c['selection'].items():
                    if sel['split']!=split:continue
                    out=a.output/name;out.mkdir(exist_ok=True)
                    for qi in tqdm(sel['indices'],desc='Layer2 pilot '+name):
                        path=out/f'{qi:06d}.npz'
                        if path.exists() and path.with_suffix('.sha.json').exists():
                            saved=load_npz(path)
                            if saved['evidence'].shape!=(44,1536) or not np.isfinite(saved['evidence']).all():raise ValueError('Invalid pilot shard')
                            continue
                        start=time.perf_counter();scores=(torch.from_numpy(z[nd+qi]).cuda()@db.T).cpu().numpy()
                        ids=np.argsort(-scores,kind='stable')[:44];old=load_npz(a.cache/split/'pairs'/f'{qi:06d}.npz')
                        if not np.array_equal(ids[:20],old['candidates']):raise ValueError('Old top20 differs')
                        qf=dense(nd+qi);feats=[];teacher=[]
                        for di in ids:
                            df=dense(int(di));captured.clear()
                            u=model(qf,df,'pairvpr');v=model(df,qf,'pairvpr')
                            if len(captured)!=2 or captured[0].shape!=(1,768):raise ValueError('Capture mismatch')
                            feats.append(torch.cat(captured,dim=-1)[0].cpu().numpy());teacher.append(float((u+v).item()))
                        teacher=np.array(teacher,np.float32)
                        if not np.allclose(teacher[:20],old['base'],atol=1e-4,rtol=1e-4):raise ValueError('Teacher reproduction failed')
                        npz(path,evidence=np.stack(feats),teacher=teacher,candidates=ids,
                            labels=np.array([records[i]['label']==records[nd+qi]['label'] for i in ids]),seconds=np.array(time.perf_counter()-start))
                        write(a.output/'progress.json',dict(phase='extract',partition=name,last_query=qi))
                        if a.smoke:
                            print('SMOKE PASS: layer2 bidirectional capture and all20 teacher scores reproduced',flush=True)
                            return False
    finally:handle.remove()
    return True

def train(a,c):
    import torch
    torch.set_num_threads(4);torch.manual_seed(42)
    data={}
    for name,sel in c['selection'].items():
        rows=[load_npz(a.output/name/f'{i:06d}.npz') for i in sel['indices']]
        data[name]=(torch.from_numpy(np.stack([r['evidence'] for r in rows])),
                    torch.from_numpy(np.stack([r['teacher'] for r in rows])),np.stack([r['labels'] for r in rows]))
    net=torch.nn.Sequential(torch.nn.Linear(1536,128),torch.nn.ReLU(),torch.nn.Linear(128,1))
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001);best=-1;history=[]
    x,y,_=data['train'];yc=data['calibration'][1].numpy()
    for epoch in range(20):
        net.train();losses=[]
        for ix in torch.randperm(len(x)).split(16):
            pred=net(x[ix]).squeeze(-1)
            regression=torch.nn.functional.smooth_l1_loss(pred,y[ix])
            ranking=torch.nn.functional.kl_div(torch.log_softmax(pred/2,dim=1),torch.softmax(y[ix]/2,dim=1),reduction='batchmean')*4
            loss=regression+ranking;opt.zero_grad();loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(torch.nn.utils.clip_grad_norm_(net.parameters(),1.)):raise ValueError('Nonfinite training')
            opt.step();losses.append(float(loss))
        net.eval()
        with torch.no_grad():pc=net(data['calibration'][0]).squeeze(-1).numpy()
        rate=float(retention(pc,yc)[0].mean());history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),cal_retention=rate))
        if rate>best:
            best=rate;state={k:v.detach().clone() for k,v in net.state_dict().items()};best_epoch=epoch+1
        print('epoch',epoch+1,'cal winner retention',rate,flush=True)
    net.load_state_dict(state);net.eval();torch.save(state,a.output/'head.pt')
    summaries={}
    with torch.no_grad():
        for name in ['calibration','evaluation']:
            x,y,labels=data[name];teacher=y.numpy();pred=net(x).squeeze(-1).numpy();hit,keep,winner=retention(pred,teacher)
            tail=winner>=20;selected=keep[np.arange(len(keep)),np.argmax(np.take_along_axis(teacher,keep,1),axis=1)]
            summaries[name]=dict(queries=len(hit),winner_retained=int(hit.sum()),winner_retention=float(hit.mean()),
                tail_winners=int(tail.sum()),tail_retained=int(hit[tail].sum()),tail_retention=float(hit[tail].mean()) if tail.any() else None,
                teacher_correct=int(labels[np.arange(len(hit)),winner].sum()),
                selected12_full_correct=int(labels[np.arange(len(hit)),selected].sum()),
                fixed12_correct=int(labels[np.arange(len(hit)),np.argmax(teacher[:,:12],1)].sum()),
                fixed20_correct=int(labels[np.arange(len(hit)),np.argmax(teacher[:,:20],1)].sum()))
            npz(a.output/f'{name}_predictions.npz',prediction=pred,teacher=teacher,labels=labels,keep=keep)
    ev=summaries['evaluation'];verdict='INSUFFICIENT_TAIL_CASES' if ev['tail_winners']<10 else 'PASS_FEASIBILITY_ONLY' if ev['winner_retention']>=.99 and ev['tail_retention']>=.9 else 'FAIL'
    write(a.output/'summary.json',dict(partitions=summaries,selected_epoch=best_epoch,verdict=verdict,
        scope='Shallow representation feasibility only, not measured acceleration. Extraction runs all12 layers. No deployed adaptive decoder.',
        theoretical_decoder_layers_per_direction=44*2+12*10,full20_layers=20*12,full44_layers=44*12))
    write(a.output/'history.json',history);write(a.output/'progress.json',dict(phase='complete',verdict=verdict));complete(a.output)
    print(read(a.output/'summary.json'),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke',action='store_true',help='Generate one new query shard and stop before training')
    for key,value in [('plan','doc/candidate_hard_plan_v1'),('cache','.cache/candidate_hard_v1'),
        ('gsv-root','datasets/gsv_cities'),('official-repo','/home/wt/workspace/Pair-VPR-official'),
        ('audit','doc/pairvpr_official_paired_audit_v1'),('output','.cache/wppr_pilot_v1')]:p.add_argument('--'+key,type=Path,default=Path(value))
    a=p.parse_args()
    import fcntl
    lockpath=a.output.parent/(a.output.name+'.lock');lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c,plan=prepare(a)
        if (a.output/'completed.json').exists():verified(a.output);print('Already complete');return
        if extract(a,c,plan) and not a.smoke:train(a,c)

if __name__=='__main__':main()
