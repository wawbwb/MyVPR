"""Decoupled correspondence/rejection gate; NOT VPR efficacy training."""
import argparse
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.candidate_set_screen import read,write,sha,official,complete
from scripts.adaptive_pair_budget import verified
from scripts.candidate_hard_screen import ensure_disjoint
from scripts.train_pmd import ordered_ids
from src.models.partial_matching import PMDPair
from src.pmd_correspondence import make_views
from src.pmd_decoupled import DecoupledHead as CorrespondenceHead,supervised_loss,counts

MODES=['forced','partial','decoupled']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key,val in [('plan','doc/candidate_hard_plan_v1'),('gsv-root','datasets/gsv_cities'),
        ('official-repo','/home/wt/workspace/Pair-VPR-official'),('audit','doc/pairvpr_official_paired_audit_v1'),
        ('output','logs/pmd/decoupled_v1')]:p.add_argument('--'+key,type=Path,default=Path(val))
    p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    import torch,fcntl
    from PIL import Image
    from torchvision import transforms as T
    from tqdm import tqdm
    torch.set_num_threads(4);verified(a.plan);plan=read(a.plan/'contract.json')['plan'];ensure_disjoint(plan)
    ids={s:ordered_ids(plan[s]['queries'],'pmd-correspondence42:')[:(2 if a.smoke else n)] for s,n in [('train',256),('dev',64)]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with (a.output.parent/(a.output.name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        images={}
        for split in ids:
            for qi in ids[split]:
                path=(a.gsv_root/plan[split]['queries'][qi]['path']).resolve()
                if not path.is_relative_to(a.gsv_root.resolve()):raise ValueError('Unsafe image path')
                images[str(path)]=sha(path)
        contract=dict(ids=ids,smoke=a.smoke,epochs=1 if a.smoke else 3,seed=42,lr=1e-4,plan=sha(a.plan/'completed.json'),images=images,
            modes=MODES,position='after frozen block11; both pair directions',scope='Synthetic crop/flip/occlusion gate, previously explored GSV dev; not VPR improvement or novel matching claim',
            policy='200 Sinkhorn iterations all arms; explicit probabilities; fixed binary rejection0.5; matched location NLL plus class-balanced binary rejection NLL; forced location-only; detached rejection in decoupled; no epoch selection. Partial location NLL also contains matched mass, so objectives are not identical.',
            code={n:hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in ['src/pmd_correspondence.py','src/pmd_decoupled.py','scripts/pmd_decoupled_gate.py','scripts/audit_pmd_rejection.py','src/models/partial_matching.py']})
        if a.output.exists():
            if not a.resume or read(a.output/'contract.json')!=contract:raise ValueError('Existing or changed output')
            if (a.output/'completed.json').exists():verified(a.output);print('Already complete');return
        else:a.output.mkdir();write(a.output/'contract.json',contract)
        base,identity=official(a);model=PMDPair(base).cuda().eval()
        heads={};opts={}
        for mode in MODES:
            torch.manual_seed(42);heads[mode]=CorrespondenceHead(mode).cuda()
            opts[mode]=torch.optim.AdamW([v for v in heads[mode].parameters() if v.requires_grad],lr=1e-4)
        transform=T.Compose([T.Resize((406,406),interpolation=T.InterpolationMode.BILINEAR),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
        def example(split,qi,seed,occlude):
            path=(a.gsv_root/plan[split]['queries'][qi]['path']).resolve()
            with Image.open(path) as im:x=transform(im.convert('RGB'))
            views,targets=make_views(x,seed,occlude)
            with torch.no_grad():
                f,_=base(views.cuda(),None,'global');x,y=model.prefix(f[:1],f[1:]);u,v=model.prefix(f[1:],f[:1])
            return [(x[:,1:],y,targets),(u[:,1:],v,targets[::-1])]
        def evaluate(tag):
            report={}
            with torch.no_grad():
                for occlude in [False,True]:
                    total={m:None for m in MODES}
                    for qi in tqdm(ids['dev'],desc=tag+(' occluded' if occlude else ' crop')):
                        for x,y,targets in example('dev',qi,500000+qi,occlude):
                            for mode in MODES:
                                c=counts(heads[mode](x,y),targets)
                                if total[mode] is None:total[mode]=c
                                else:total[mode]={k:total[mode][k]+c[k] for k in c}
                    for c in total.values():
                        c['location_accuracy']=c['location_correct']/max(1,c['matched'])
                        c['matched_accuracy']=c['matched_correct']/max(1,c['matched'])
                        c['rejection_recall']=c['rejected_unmatched']/max(1,c['unmatched'])
                        c['false_rejection_rate']=c['false_rejected_matched']/max(1,c['matched'])
                    report['occluded' if occlude else 'crop']=total
            write(a.output/(tag+'.json'),report);return report
        state=dict(epoch=0,cursor=0);last=a.output/'last.pt'
        def save():
            temp=last.with_suffix('.tmp');torch.save(dict(state=state,heads={m:h.state_dict() for m,h in heads.items()},
                optimizers={m:o.state_dict() for m,o in opts.items()},contract=sha(a.output/'contract.json'),official=identity),temp);temp.replace(last)
        if last.exists():
            ck=torch.load(last,map_location='cpu',weights_only=True)
            if ck['contract']!=sha(a.output/'contract.json') or ck['official']!=identity:raise ValueError('Checkpoint differs')
            for m in MODES:heads[m].load_state_dict(ck['heads'][m]);opts[m].load_state_dict(ck['optimizers'][m])
            state=ck['state']
        else:evaluate('initial');save()
        while state['epoch']<contract['epochs']:
            epoch=state['epoch'];order=ids['train']
            for cursor in tqdm(range(state['cursor'],len(order)),desc=f'Correspondence epoch{epoch+1}'):
                qi=order[cursor];examples=example('train',qi,42+epoch*100000+qi,cursor%2==0)
                for mode in MODES:
                    loss=sum(supervised_loss(heads[mode](x,y),t,mode!='forced') for x,y,t in examples)/len(examples)
                    opts[mode].zero_grad();loss.backward();g=torch.nn.utils.clip_grad_norm_(heads[mode].match.parameters(),1.)
                    if mode=='decoupled':
                        rg=torch.nn.utils.clip_grad_norm_(heads[mode].reject.parameters(),1.)
                        if not torch.isfinite(rg):raise ValueError('Nonfinite rejection gradient')
                    if not torch.isfinite(loss) or not torch.isfinite(g):raise ValueError('Nonfinite optimization')
                    opts[mode].step()
                state['cursor']=cursor+1
                if state['cursor']%16==0 or state['cursor']==len(order):save()
                write(a.output/'progress.json',dict(phase='training',epoch=epoch+1,done=state['cursor'],total=len(order)))
            state['epoch']+=1;state['cursor']=0;save()
        result=evaluate('final');write(a.output/'summary.json',dict(scope=contract['scope'],metrics=result,verdict='SYNTHETIC_GATE_COMPLETE_NOT_VPR_VERDICT'))
        write(a.output/'progress.json',dict(phase='complete'));complete(a.output);print(result,flush=True)


if __name__=='__main__':main()
