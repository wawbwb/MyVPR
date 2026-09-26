"""Read-only checkpoint audit: correspondence versus rejection and Sinkhorn residual."""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.candidate_set_screen import read,write,sha,official,complete
from scripts.adaptive_pair_budget import verified
from src.pmd_correspondence import make_views,CorrespondenceHead
from src.models.partial_matching import PMDPair


def explicit_assignment(head,a,b,iterations):
    import torch
    m=head.match;a=m.key(m.norm(a));b=m.key(m.norm(b));s=a@b.transpose(1,2)/a.shape[-1]**.5
    batch,n,k=s.shape
    if head.mode=='independent_dustbin':
        def one(x):
            z=torch.cat([x,m.bin_score.expand(*x.shape[:-1],1)],-1).softmax(-1)
            return z[...,:-1],z[...,-1]
        p,d=one(s);q,e=one(s.transpose(1,2));return [(p,d),(q,e)]
    if head.mode=='partial':
        z=torch.cat([torch.cat([s,m.bin_score.expand(batch,n,1)],2),m.bin_score.expand(batch,1,k+1)],1)
        mu=torch.cat([s.new_ones(n),s.new_tensor([float(k)])])/(n+k)
        nu=torch.cat([s.new_ones(k),s.new_tensor([float(n)])])/(n+k)
    else:z=s;mu=s.new_full((n,),1/n);nu=s.new_full((k,),1/k)
    u=torch.zeros_like(z[:,:,0]);v=torch.zeros_like(z[:,0,:])
    for _ in range(iterations):
        u=mu.log()[None]-torch.logsumexp(z+v[:,None,:],2)
        v=nu.log()[None]-torch.logsumexp(z+u[:,:,None],1)
    plan=torch.exp(z+u[:,:,None]+v[:,None,:])
    p=plan[:,:n,:k]/mu[:n][None,:,None];q=plan[:,:n,:k].transpose(1,2)/nu[:k][None,:,None]
    if head.mode=='partial':d=plan[:,:n,k]/mu[:n];e=plan[:,n,:k]/nu[:k]
    else:d=s.new_zeros(batch,n);e=s.new_zeros(batch,k)
    return [(p,d),(q,e)]


def metrics(p,d,t,can_reject):
    import torch
    p=p[0];d=d[0];t=t.to(p.device);valid=t>=0;mass=p.sum(-1);correct=p.argmax(-1)==t
    # Explicit dustbin mass, NEVER 1-row_sum for a model without dustbin.
    categorical=(d>p.max(-1).values) if can_reject else torch.zeros_like(valid)
    binary=(d>mass) if can_reject else torch.zeros_like(valid)
    probability=d/(mass+d).clamp_min(1e-12)
    out=dict(matched=int(valid.sum()),unmatched=int((~valid).sum()),location_correct=int((correct&valid).sum()),
             row_residual_max=float((mass+d-1).abs().max()),legacy_negative_dust_count=int((mass>1+1e-6).sum()),
             brier_sum=float(((probability-(~valid).float())**2).sum()),patches=len(t),
             dust_matched_sum=float(probability[valid].sum()),dust_unmatched_sum=float(probability[~valid].sum()))
    for name,reject in [('categorical',categorical),('binary_half',binary)]:
        out[name+'_correct']=int((correct&valid&~reject).sum())
        out[name+'_false_rejected']=int((reject&valid).sum())
        out[name+'_unmatched_rejected']=int((reject&~valid).sum())
    return out


def combine(rows):
    out={k:(max(r[k] for r in rows) if k.endswith('_max') else sum(r[k] for r in rows)) for k in rows[0]}
    out['location_accuracy']=out['location_correct']/max(1,out['matched'])
    out['brier']=out['brier_sum']/out['patches']
    out['dust_mean_matched']=out['dust_matched_sum']/max(1,out['matched'])
    out['dust_mean_unmatched']=out['dust_unmatched_sum']/max(1,out['unmatched'])
    for name in ['categorical','binary_half']:
        out[name+'_accuracy']=out[name+'_correct']/max(1,out['matched'])
        out[name+'_false_rejection_rate']=out[name+'_false_rejected']/max(1,out['matched'])
        out[name+'_rejection_recall']=out[name+'_unmatched_rejected']/max(1,out['unmatched'])
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key,val in [('run','logs/pmd/correspondence_v1'),('plan','doc/candidate_hard_plan_v1'),('gsv-root','datasets/gsv_cities'),
        ('official-repo','/home/wt/workspace/Pair-VPR-official'),('audit','doc/pairvpr_official_paired_audit_v1'),('output','doc/pmd_rejection_audit_v1')]:
        p.add_argument('--'+key,type=Path,default=Path(val))
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose new output; old results are preserved')
    import torch
    from PIL import Image
    from torchvision import transforms as T
    from tqdm import tqdm
    torch.set_num_threads(4);verified(a.run);verified(a.plan);contract=read(a.run/'contract.json')
    if contract['plan']!=sha(a.plan/'completed.json'):raise ValueError('Plan changed')
    for name,h in contract['code'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=h:raise ValueError('Original code changed')
    base,identity=official(a);model=PMDPair(base).cuda().eval()
    ck=torch.load(a.run/'last.pt',map_location='cpu',weights_only=True)
    if ck['contract']!=sha(a.run/'contract.json') or ck['official']!=identity:raise ValueError('Checkpoint changed')
    heads={}
    for mode in contract['modes']:
        h=CorrespondenceHead(mode).cuda().eval();h.load_state_dict(ck['heads'][mode],strict=True);heads[mode]=h
    transform=T.Compose([T.Resize((406,406),interpolation=T.InterpolationMode.BILINEAR),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    plan=read(a.plan/'contract.json')['plan']['dev'];rows=[];reproduction=0.
    with torch.no_grad():
        for occlude in [False,True]:
            for qi in tqdm(contract['ids']['dev'],desc='Rejection audit '+str(occlude)):
                path=(a.gsv_root/plan['queries'][qi]['path']).resolve()
                if not path.is_relative_to(a.gsv_root.resolve()) or sha(path)!=contract['images'][str(path)]:raise ValueError('Image changed')
                with Image.open(path) as im:views,targets=make_views(transform(im.convert('RGB')),500000+qi,occlude)
                f,_=base(views.cuda(),None,'global')
                for direction in [0,1]:
                    x,y=model.prefix(f[direction:direction+1],f[1-direction:2-direction]);tt=targets if direction==0 else targets[::-1]
                    for mode,h in heads.items():
                        native=h(x[:,1:],y)
                        for iterations in ([20] if mode=='independent_dustbin' else [20,200]):
                            plans=explicit_assignment(h,x[:,1:],y,iterations)
                            if iterations==20:
                                for old,(new,dust) in zip(native,plans):
                                    error=float((old-new).abs().max());reproduction=max(reproduction,error)
                                    if not torch.allclose(old,new,atol=1e-6,rtol=1e-5):raise ValueError('Assignment reproduction failed')
                            stats=combine([metrics(pp,dd,t,mode!='forced') for (pp,dd),t in zip(plans,tt)])
                            # Store raw sufficient statistics only; rates recomputed after aggregation.
                            raw=[metrics(pp,dd,t,mode!='forced') for (pp,dd),t in zip(plans,tt)]
                            rows.append(dict(query=qi,occluded=occlude,direction=direction,mode=mode,iterations=iterations,raw=raw))
    summary={}
    for occlude in [False,True]:
        condition='occluded' if occlude else 'crop';summary[condition]={}
        for mode in heads:
            for iterations in ([20] if mode=='independent_dustbin' else [20,200]):
                raw=[r for row in rows if row['occluded']==occlude and row['mode']==mode and row['iterations']==iterations for r in row['raw']]
                summary[condition][mode+'_'+str(iterations)]=combine(raw)
    a.output.mkdir(parents=True)
    write(a.output/'summary.json',dict(metrics=summary,reproduction_max_error=reproduction,
        scope='Post-hoc synthetic diagnostic. Fixed categorical and binary_half rules; no threshold fitting. 200 iterations is numerical sensitivity, NOT a new trained model. Forced rejection always zero by definition. Correlated patches are not independent samples.'))
    write(a.output/'per_image.json',rows);write(a.output/'provenance.json',dict(run=sha(a.run/'completed.json'),checkpoint=sha(a.run/'last.pt'),script=sha(Path(__file__))))
    complete(a.output);print(summary,flush=True)


if __name__=='__main__':main()
