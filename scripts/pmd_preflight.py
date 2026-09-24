"""PMD real-image gradient/identity preflight. NOT a retrieval efficacy test."""
import argparse
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.candidate_set_screen import read,write,sha,load_npz,official,image,complete
from scripts.adaptive_pair_budget import verified
from src.models.partial_matching import PMDPair


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key,val in [('plan','doc/candidate_hard_plan_v1'),('cache','.cache/candidate_hard_v1/train'),
        ('gsv-root','datasets/gsv_cities'),('official-repo','/home/wt/workspace/Pair-VPR-official'),
        ('audit','doc/pairvpr_official_paired_audit_v1'),('output','doc/pmd_preflight_v1')]:
        p.add_argument('--'+key,type=Path,default=Path(val))
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use new report directory')
    verified(a.plan);verified(a.cache)
    if read(a.cache/'contract.json')['plan_sha256']!=sha(a.plan/'completed.json'):raise ValueError('Plan mismatch')
    plan=read(a.plan/'contract.json')['plan']['train'];nd=len(plan['database']);records=plan['database']+plan['queries']
    # Two first eligible train queries, each positive and highest-ranked negative.
    chosen=[]
    for qi in range(64):
        z=load_npz(a.cache/'pairs'/f'{qi:06d}.npz')
        pos=np.flatnonzero(z['labels']);neg=np.flatnonzero(~z['labels'])
        if len(pos) and len(neg):chosen.append((qi,[int(z['candidates'][pos[0]]),int(z['candidates'][neg[0]])]))
        if len(chosen)==2:break
    if len(chosen)!=2:raise ValueError('Need two train positive/negative pairs')
    import torch
    torch.set_num_threads(4)
    base,identity=official(a)
    if identity!=read(a.cache/'contract.json')['official']:raise ValueError('Model mismatch')
    model=PMDPair(base).cuda();features={}
    with torch.no_grad():
        for qi,ds in chosen:
            for index in [nd+qi]+ds:
                if index in features:continue
                start=index//128*128;cache=load_npz(a.cache/'global'/f'{start:07d}.npz')
                x,_=image(a.gsv_root,records[index],str(cache['hashes'][index-start]))
                f,g=base(x[None].cuda(),None,'global')
                if not np.allclose(g[0].cpu().numpy(),cache['vectors'][index-start],atol=2e-5,rtol=2e-4):raise ValueError('Encoder mismatch')
                features[index]=f
    states=[];reference=[]
    with torch.no_grad():
        for qi,ds in chosen:
            for di in ds:
                q,d=features[nd+qi],features[di]
                states.append((model.prefix(q,d),model.prefix(d,q)))
                reference.append((base(q,d,'pairvpr')+base(d,q,'pairvpr')).flatten())
    reference=torch.cat(reference)
    def score():
        values=[];mass=[]
        for f,b in states:
            u,s=model.finish(f);v,t=model.finish(b);values.append(u+v);mass.append(s['matched_mass_x'])
        return torch.cat(values),torch.stack(mass).mean()
    with torch.no_grad():initial,_=score()
    error=float((initial-reference).abs().max())
    if error>1e-4:raise ValueError('Zero-start differs from original')
    opt=torch.optim.AdamW(model.match.parameters(),lr=1e-4,weight_decay=.01);history=[]
    for step in range(5):
        scores,mass=score();pairs=scores.reshape(2,2)
        # Fixed temperature avoids saturation on these easy implementation pairs.
        loss=torch.nn.functional.softplus((pairs[:,1]-pairs[:,0])/10).mean()
        opt.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.match.parameters(),1.)
        if not torch.isfinite(loss) or not torch.isfinite(grad):raise ValueError('Nonfinite loss/gradient')
        history.append(dict(step=step+1,loss=float(loss),gradient=float(grad),mass=float(mass),
            key_gradient=float(model.match.key.weight.grad.abs().sum()),
            dustbin_gradient=float(model.match.bin_score.grad.abs())))
        opt.step();print(history[-1],flush=True)
    with torch.no_grad():final,_=score()
    drift=float((final-reference).abs().max())
    if drift<=1e-7 or not any(r['key_gradient']>0 and r['dustbin_gradient']>0 for r in history[1:]):
        raise ValueError('Matching path did not learn')
    if any(p.grad is not None for p in base.parameters()):raise ValueError('Frozen base received parameter gradients')
    a.output.mkdir(parents=True)
    write(a.output/'summary.json',dict(verdict='PASS_IMPLEMENTATION_ONLY',zero_start_max_error=error,
        score_drift=drift,history=history,queries=chosen,official=identity,
        source_hashes={str(a.plan):sha(a.plan/'completed.json'),str(a.cache):sha(a.cache/'completed.json')},
        scope='Two train queries,5 updates; verifies interface/gradient only, NOT accuracy or novelty. No deployable checkpoint saved.'))
    complete(a.output);print('PASS_IMPLEMENTATION_ONLY',flush=True)


if __name__=='__main__':main()
