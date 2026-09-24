"""Post-hoc PMD mechanism audit; no training or hyperparameter selection."""
import argparse
import hashlib
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.candidate_set_screen import read,write,sha,load_npz,official,image,complete
from scripts.adaptive_pair_budget import verified
from scripts.train_pmd import restore_trainable,ordered_ids
from src.models.partial_matching import PMDPair


def ranking(scores,labels):
    scores=np.asarray(scores);labels=np.asarray(labels,dtype=bool)
    if scores.shape!=labels.shape or scores.ndim!=1 or not np.isfinite(scores).all():raise ValueError('Invalid scores')
    pos=scores[labels];neg=scores[~labels]
    return dict(correct=bool(labels[scores.argmax()]),reachable=bool(len(pos)),
                margin=float(pos.max()-neg.max()) if len(pos) and len(neg) else None,
                winner=int(scores.argmax()))


def normalize_mass(p):
    return p/p.sum(-1,keepdim=True).clamp_min(1e-12)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key,val in [('runs','logs/pmd'),('plan','doc/candidate_hard_plan_v1'),('cache','.cache/candidate_hard_v1/dev'),
        ('gsv-root','datasets/gsv_cities'),('official-repo','/home/wt/workspace/Pair-VPR-official'),
        ('audit','doc/pairvpr_official_paired_audit_v1'),('output','doc/pmd_mechanism_v1')]:
        p.add_argument('--'+key,type=Path,default=Path(val))
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose a new report directory')
    import torch
    from tqdm import tqdm
    torch.set_num_threads(4)
    verified(a.plan);verified(a.cache)
    plan=read(a.plan/'contract.json')['plan']['dev'];runs={};contract=None;sources={}
    for mode in ['baseline','ordinary','forced','partial']:
        root=a.runs/(mode+'_screen_v1');verified(root);c=read(root/'contract.json');s=read(root/'summary.json')
        if c.pop('mode')!=mode or c['smoke']:raise ValueError('Invalid run')
        if contract is not None and c!=contract:raise ValueError('Unmatched runs')
        contract=c;sources[str(root)]=sha(root/'completed.json')
        runs[mode]={stage:{r['query']:r for r in read(root/f'validation_epoch{epoch:02d}.json')}
                    for stage,epoch in [('initial',0),('best',s['best_epoch']),('last',3)]}
    if contract['sources'][str(a.cache)]!=sha(a.cache/'completed.json') or contract['sources'][str(a.plan)]!=sha(a.plan/'completed.json'):
        raise ValueError('Sources changed')
    for name,h in contract['code'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=h:raise ValueError('Training code changed')
    rows=[];shards={}
    for qi in contract['dev']:
        z=load_npz(a.cache/'pairs'/f'{qi:06d}.npz');shards[qi]=z
        row=dict(query=qi,frozen=ranking(z['base'],z['labels']),arms={})
        for mode in runs:
            row['arms'][mode]={stage:ranking(runs[mode][stage][qi]['scores'],z['labels']) for stage in ['best','last']}
        rows.append(row)
    errors=[r['query'] for r in rows if not r['frozen']['correct']]
    controls=[qi for qi in ordered_ids(plan['queries'],'pmd-diagnostic-controls42:') if qi in contract['dev'] and qi not in errors][:16]
    selected=errors+controls
    base,identity=official(a)
    if identity!=read(a.cache/'contract.json')['official']:raise ValueError('Model mismatch')
    model=PMDPair(base).cuda().eval()
    for param in base.dec_blocks[-1].parameters():param.requires_grad_(True)
    root=a.runs/'partial_screen_v1';feature_cache={};chunks={}
    records=plan['database']+plan['queries'];nd=len(plan['database'])
    def dense(index):
        if index in feature_cache:return feature_cache[index].cuda()
        start=index//128*128
        if start not in chunks:chunks[start]=load_npz(a.cache/'global'/f'{start:07d}.npz')
        z=chunks[start];x,_=image(a.gsv_root,records[index],str(z['hashes'][index-start]))
        f,g=base(x[None].cuda(),None,'global')
        if not np.allclose(g[0].cpu().numpy(),z['vectors'][index-start],atol=2e-5,rtol=2e-4):raise ValueError('Encoder mismatch')
        feature_cache[index]=f.cpu()
        if len(feature_cache)>64:feature_cache.pop(next(iter(feature_cache)))
        return f
    def finish(state):
        x,y=state;m=model.match;u=m.norm(x[:,1:]);v=m.norm(y)
        transport,reverse=m.assignment((m.key(u)@m.key(v).transpose(1,2)/math.sqrt(m.key.out_features)).float())
        masses=torch.cat([transport.sum(-1).flatten(),reverse.sum(-1).flatten()])
        values={}
        for name in ['native','unit_mass','bypass']:
            xx,yy=x,y
            if name!='bypass':
                pp,qq=(transport,reverse) if name=='native' else (normalize_mass(transport),normalize_mass(reverse))
                xx=torch.cat([x[:,:1],x[:,1:]+m.output(pp@m.value(v))],1)
                yy=y+m.output(qq@m.value(u))
            xx,_=base.dec_blocks[-1](xx,yy)
            values[name]=float(base.classvprmodule(base.dec_norm(xx)[:,0]).flatten()[0])
        return values,dict(mean=float(masses.mean()),low_mass_fraction=float((masses<.1).float().mean()),
                           high_mass_fraction=float((masses>.99).float().mean()))
    details={};max_error=0.
    with torch.no_grad():
        for stage,filename in [('best','best.pt'),('last','last.pt')]:
            ck=torch.load(root/filename,map_location='cpu',weights_only=True)
            if ck['state']['contract_sha256']!=sha(root/'contract.json'):raise ValueError('Checkpoint identity mismatch')
            restore_trainable(model,ck['parameters']);output=[]
            for qi in tqdm(selected,desc='PMD diagnostic '+stage):
                z=shards[qi];q=dense(nd+qi);pairs=[]
                for j,di in enumerate(z['candidates']):
                    d=dense(int(di));f,fs=finish(model.prefix(q,d));b,bs=finish(model.prefix(d,q))
                    scores={k:f[k]+b[k] for k in f};expected=runs['partial'][stage][qi]['scores'][j]
                    error=abs(scores['native']-expected);max_error=max(max_error,error)
                    if not np.isclose(scores['native'],expected,atol=1e-4,rtol=1e-4):raise ValueError('Native score mismatch')
                    pairs.append(dict(candidate=int(di),positive=bool(z['labels'][j]),scores=scores,
                                      mass={k:(fs[k]+bs[k])/2 for k in fs}))
                output.append(dict(query=qi,group='frozen_error' if qi in errors else 'correct_control',pairs=pairs,
                    rankings={k:ranking([r['scores'][k] for r in pairs],z['labels']) for k in ['native','unit_mass','bypass']}))
            details[stage]=output
    summary=dict(queries=len(rows),frozen_errors=len(errors),reachable_errors=sum(shards[q]['labels'].any().item() for q in errors),
        error_ids=errors,control_ids=controls,native_max_error=max_error,stages={},
        scope='Post-hoc development diagnostic, not new validation. unit_mass removes attenuation but retains dustbin-influenced assignment; NOT retrained forced matching. bypass retains trained final block; NOT original baseline.')
    for stage,items in details.items():
        groups={}
        for group in ['frozen_error','correct_control']:
            subset=[r for r in items if r['group']==group]
            mass_means={}
            for positive in [True,False]:
                vals=[np.mean([r['mass']['mean'] for r in q['pairs'] if r['positive']==positive]) for q in subset if any(r['positive']==positive for r in q['pairs'])]
                mass_means['positive' if positive else 'negative']=dict(queries=len(vals),mean=float(np.mean(vals)) if vals else None)
            groups[group]=dict(queries=len(subset),correct={k:sum(q['rankings'][k]['correct'] for q in subset) for k in ['native','unit_mass','bypass']},query_balanced_matched_mass=mass_means)
        summary['stages'][stage]=groups
    a.output.mkdir(parents=True)
    write(a.output/'per_query.json',rows);write(a.output/'mechanism.json',details);write(a.output/'summary.json',summary)
    write(a.output/'provenance.json',dict(runs=sources,cache=sha(a.cache/'completed.json'),plan=sha(a.plan/'completed.json'),script=sha(Path(__file__))))
    complete(a.output);print(summary,flush=True)


if __name__=='__main__':main()
