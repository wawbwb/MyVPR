"""Matched four-arm PMD screening. Fixed top20, place supervision, resumable."""
import argparse
from collections import OrderedDict
import hashlib
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.candidate_set_screen import read,write,sha,load_npz,official,image,complete
from scripts.adaptive_pair_budget import verified
from scripts.candidate_hard_screen import ensure_disjoint
from src.models.partial_matching import PMDPair

MODES=('baseline','ordinary','forced','partial')
POLICY=dict(train_queries=1024,dev_queries=256,epochs=3,seed=42,topk=20,
    last_block_lr=1e-5,adapter_lr=1e-4,weight_decay=.001,temperature=10.,
    save_every=32,sampling='hash-selected reachable train queries, one positive and one top-ranked negative; positive rotates by epoch',
    selection='highest dev R1 including epoch0, earliest tie',
    scope='First GSV screen only; no Pitts; all arms unfreeze block12 only, other original parameters frozen; no synthetic correspondence loss yet')


def ordered_ids(records,tag):
    return sorted(range(len(records)),key=lambda i:hashlib.sha256((tag+records[i]['label']).encode()).hexdigest())


def pair_indices(labels,epoch):
    pos=np.flatnonzero(labels);neg=np.flatnonzero(~labels)
    if not len(pos) or not len(neg):raise ValueError('No positive/negative candidate')
    return int(pos[epoch%len(pos)]),int(neg[0])


def trainable_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}


def restore_trainable(model,state):
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(params)!=set(state):raise ValueError('Trainable parameters changed')
    import torch
    with torch.no_grad():
        for name,p in params.items():p.copy_(state[name].to(p.device))


def save_checkpoint(path,model,opt,state):
    import torch
    payload=dict(state=state,parameters=trainable_state(model),optimizer=opt.state_dict(),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)


def load_checkpoint(path,model,opt):
    import torch
    ck=torch.load(path,map_location='cpu',weights_only=True)
    restore_trainable(model,ck['parameters']);opt.load_state_dict(ck['optimizer'])
    torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
    return ck['state']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=MODES,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    for key,val in [('plan','doc/candidate_hard_plan_v1'),('cache','.cache/candidate_hard_v1'),
        ('gsv-root','datasets/gsv_cities'),('official-repo','/home/wt/workspace/Pair-VPR-official'),
        ('audit','doc/pairvpr_official_paired_audit_v1')]:p.add_argument('--'+key,type=Path,default=Path(val))
    a=p.parse_args()
    import torch,fcntl
    from tqdm import tqdm
    torch.set_num_threads(4)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with (a.output.parent/(a.output.name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print('Verifying fixed data/model sources...',flush=True)
        verified(a.plan)
        plan=read(a.plan/'contract.json')['plan'];ensure_disjoint(plan)
        for split in ['train','dev']:
            verified(a.cache/split)
            if read(a.cache/split/'contract.json')['plan_sha256']!=sha(a.plan/'completed.json'):raise ValueError('Cache/plan differs')
        train=[]
        for qi in ordered_ids(plan['train']['queries'],'pmd-train42:'):
            z=load_npz(a.cache/'train'/'pairs'/f'{qi:06d}.npz')
            if z['labels'].any() and (~z['labels']).any():train.append(qi)
            if len(train)==(2 if a.smoke else POLICY['train_queries']):break
        dev=ordered_ids(plan['dev']['queries'],'pmd-dev42:')[:2 if a.smoke else POLICY['dev_queries']]
        if len(train)!=(2 if a.smoke else 1024) or len(dev)!=(2 if a.smoke else 256):raise ValueError('Insufficient data')
        contract=dict(mode=a.mode,smoke=a.smoke,policy=POLICY,train=train,dev=dev,
            sources={str(d):sha(d/'completed.json') for d in [a.plan,a.cache/'train',a.cache/'dev']},
            code={n:hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                  for n in ['scripts/train_pmd.py','src/models/partial_matching.py']})
        if a.output.exists():
            if not a.resume:raise FileExistsError('Run exists; use --resume')
            if read(a.output/'contract.json')!=contract:raise ValueError('Run contract changed')
            if (a.output/'completed.json').exists():verified(a.output);print('Already complete');return
        else:a.output.mkdir(parents=True)
        write(a.output/'contract.json',contract)
        base,identity=official(a)
        for split in ['train','dev']:
            if identity!=read(a.cache/split/'contract.json')['official']:raise ValueError('Model mismatch')
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        model=PMDPair(base,'ordinary' if a.mode=='baseline' else a.mode).cuda()
        for param in base.dec_blocks[-1].parameters():param.requires_grad_(True)
        if a.mode=='baseline':
            for param in model.match.parameters():param.requires_grad_(False)
        elif a.mode!='partial':model.match.bin_score.requires_grad_(False)
        groups=[dict(params=list(base.dec_blocks[-1].parameters()),lr=POLICY['last_block_lr'])]
        adapter=[p for p in model.match.parameters() if p.requires_grad]
        if adapter:groups.append(dict(params=adapter,lr=POLICY['adapter_lr']))
        opt=torch.optim.AdamW(groups,weight_decay=POLICY['weight_decay'])
        memo=OrderedDict();globals_cache={}
        def dense(split,index):
            key=(split,index)
            if key in memo:memo.move_to_end(key);return memo[key].cuda()
            start=index//128*128;gkey=(split,start)
            if gkey not in globals_cache:globals_cache[gkey]=load_npz(a.cache/split/'global'/f'{start:07d}.npz')
            z=globals_cache[gkey];records=plan[split]['database']+plan[split]['queries']
            x,_=image(a.gsv_root,records[index],str(z['hashes'][index-start]))
            with torch.no_grad():f,g=base(x[None].cuda(),None,'global')
            if not np.allclose(g[0].cpu().numpy(),z['vectors'][index-start],atol=2e-5,rtol=2e-4):raise ValueError('Encoder mismatch')
            memo[key]=f.cpu()
            if len(memo)>64:memo.popitem(last=False)
            return f
        def score(q,d):
            u,s=model.finish(model.prefix(q,d),bypass=a.mode=='baseline')
            v,t=model.finish(model.prefix(d,q),bypass=a.mode=='baseline')
            mass=None if not s else (s['matched_mass_x']+t['matched_mass_x'])/2
            return (u+v).squeeze(0),mass
        def evaluate(epoch,check_initial=False):
            outcomes=[];correct=0;max_error=0.
            with torch.no_grad():
                for qi in tqdm(dev,desc=f'{a.mode} validation epoch{epoch}'):
                    z=load_npz(a.cache/'dev'/'pairs'/f'{qi:06d}.npz');q=dense('dev',len(plan['dev']['database'])+qi)
                    values=np.array([float(score(q,dense('dev',int(di)))[0]) for di in z['candidates']])
                    if check_initial:
                        max_error=max(max_error,float(np.max(np.abs(values-z['base']))))
                        if not np.allclose(values,z['base'],atol=1e-4,rtol=1e-4):raise ValueError('Epoch0 does not reproduce original')
                    hit=bool(z['labels'][values.argmax()]);correct+=hit
                    outcomes.append(dict(query=qi,correct=hit,baseline_correct=bool(z['labels'][z['base'].argmax()]),scores=values.tolist()))
            write(a.output/f'validation_epoch{epoch:02d}.json',outcomes)
            return dict(epoch=epoch,correct=correct,queries=len(dev),r1=correct/len(dev),initial_max_error=max_error)
        last=a.output/'last.pt'
        if a.resume and last.exists():
            state=load_checkpoint(last,model,opt)
            if state['contract_sha256']!=sha(a.output/'contract.json'):raise ValueError('Checkpoint identity differs')
            print('Resume epoch/cursor:',state['epoch'],state['cursor'],flush=True)
        else:
            initial=evaluate(0,True)
            state=dict(epoch=0,cursor=0,loss_sum=0.,history=[initial],best_correct=initial['correct'],best_epoch=0,
                contract_sha256=sha(a.output/'contract.json'))
            save_checkpoint(a.output/'best.pt',model,opt,state);save_checkpoint(last,model,opt,state)
        epochs=1 if a.smoke else POLICY['epochs']
        # Keep dropout disabled in all arms; parameter gradients remain enabled.
        model.eval()
        while state['epoch']<epochs:
            epoch=state['epoch'];order=np.random.default_rng(42+epoch).permutation(train).tolist()
            bar=tqdm(range(state['cursor'],len(order)),desc=f'{a.mode} train {epoch+1}/{epochs}')
            for cursor in bar:
                qi=order[cursor];z=load_npz(a.cache/'train'/'pairs'/f'{qi:06d}.npz')
                pi,ni=pair_indices(z['labels'],epoch);q=dense('train',len(plan['train']['database'])+qi)
                positive,pm=score(q,dense('train',int(z['candidates'][pi])))
                negative,nm=score(q,dense('train',int(z['candidates'][ni])))
                loss=torch.nn.functional.softplus((negative-positive)/POLICY['temperature'])
                opt.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                if not torch.isfinite(loss) or not torch.isfinite(grad):raise ValueError('Nonfinite training')
                opt.step();state['loss_sum']+=float(loss.detach());state['cursor']=cursor+1
                bar.set_postfix(loss=float(loss.detach()),mass=None if pm is None else float(pm.detach()))
                if state['cursor']%POLICY['save_every']==0 or state['cursor']==len(order):save_checkpoint(last,model,opt,state)
                if a.smoke:
                    # Real save/load round-trip after every update; optimizer moments included.
                    save_checkpoint(last,model,opt,state);before=trainable_state(model)
                    state=load_checkpoint(last,model,opt)
                    if any(not torch.equal(before[n],v) for n,v in trainable_state(model).items()):raise ValueError('Resume state mismatch')
                write(a.output/'progress.json',dict(phase='training',mode=a.mode,epoch=epoch+1,done=state['cursor'],total=len(order)))
            result=evaluate(epoch+1);result['train_loss']=state['loss_sum']/len(order)
            state['history'].append(result);state['epoch']+=1;state['cursor']=0;state['loss_sum']=0.
            if result['correct']>state['best_correct']:
                state['best_correct']=result['correct'];state['best_epoch']=epoch+1;save_checkpoint(a.output/'best.pt',model,opt,state)
            save_checkpoint(last,model,opt,state)
            write(a.output/'history.json',state['history']);print(result,flush=True)
        write(a.output/'summary.json',dict(mode=a.mode,smoke=a.smoke,history=state['history'],best_epoch=state['best_epoch'],
            best_correct=state['best_correct'],trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            scope=POLICY['scope'],verdict='SMOKE_PASS' if a.smoke else 'SCREEN_COMPLETE_NOT_EFFICACY_VERDICT'))
        write(a.output/'progress.json',dict(phase='complete'));complete(a.output)
        print('Completed',a.output,flush=True)


if __name__=='__main__':main()
