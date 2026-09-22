"""Evaluate the frozen WPPR pilot head on ALL remaining GSV dev queries."""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import wppr_pilot as pilot
from scripts.candidate_set_screen import read,write,sha,load_npz,complete,npz
from scripts.adaptive_pair_budget import verified


def remaining_indices(selection,n=2048):
    cal=selection['calibration']['indices'];ev=selection['evaluation']['indices']
    if len(cal)!=128 or len(ev)!=256 or len(set(cal+ev))!=384 or any(type(i)!=int or not 0<=i<n for i in cal+ev):
        raise ValueError('Invalid prior partition')
    return sorted(set(range(n))-set(cal+ev))


def metrics(pred,teacher,labels,ids):
    hit,keep,winner=pilot.retention(pred,teacher)
    n=len(hit);ix=np.arange(n)
    selected=keep[ix,np.argmax(np.take_along_axis(teacher,keep,1),axis=1)]
    reference=labels[ix,winner];adaptive=labels[ix,selected]
    base=labels[ix,np.argmax(teacher[:,:20],axis=1)];tail=winner>=20
    result=dict(queries=n,winner_retained=int(hit.sum()),winner_retention=float(hit.mean()),
        tail_winners=int(tail.sum()),tail_retained=int(hit[tail].sum()),tail_retention=float(hit[tail].mean()) if tail.any() else None,
        teacher_correct=int(reference.sum()),selected12_full_correct=int(adaptive.sum()),
        fixed12_correct=int(labels[ix,np.argmax(teacher[:,:12],axis=1)].sum()),fixed20_correct=int(base.sum()),
        corrections_vs20=[ids[i] for i in np.flatnonzero(adaptive & ~base)],
        regressions_vs20=[ids[i] for i in np.flatnonzero(~adaptive & base)],
        corrections_vs44=[ids[i] for i in np.flatnonzero(adaptive & ~reference)],
        regressions_vs44=[ids[i] for i in np.flatnonzero(~adaptive & reference)])
    cases=[dict(query_index=ids[i],teacher_winner_rank=int(winner[i]+1),selected_rank=int(selected[i]+1),
        teacher_correct=bool(reference[i]),selected_correct=bool(adaptive[i]),
        full_score_gap=float(teacher[i,winner[i]]-teacher[i,selected[i]])) for i in np.flatnonzero(~hit)]
    return result,cases,keep


def predict(net,rows):
    import torch
    result=[]
    with torch.inference_mode():
        for start in range(0,len(rows),16):
            x=torch.from_numpy(np.stack([r['evidence'] for r in rows[start:start+16]]))
            result.append(net(x).squeeze(-1).numpy())
    return np.concatenate(result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key,val in [('pilot','.cache/wppr_pilot_v1'),('plan','doc/candidate_hard_plan_v1'),
        ('cache','.cache/candidate_hard_v1'),('gsv-root','datasets/gsv_cities'),
        ('official-repo','/home/wt/workspace/Pair-VPR-official'),('audit','doc/pairvpr_official_paired_audit_v1'),
        ('full','doc/candidate_top44_eval_v1'),('output','.cache/wppr_extension_v1'),('report','doc/wppr_extension_v1')]:
        p.add_argument('--'+key,type=Path,default=Path(val))
    a=p.parse_args();a.smoke=False
    import fcntl
    lockpath=a.output.parent/(a.output.name+'.lock');lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print('Verify completed pilot, frozen head and source caches...',flush=True)
        verified(a.pilot);verified(a.plan);verified(a.full)
        pc=read(a.pilot/'contract.json');plan_c=read(a.plan/'contract.json');plan=plan_c['plan']
        if pc['code']!=pilot.code() or pc['policy']!=pilot.POLICY or plan_c['code']!=pilot.hard.codes():raise ValueError('Source code mismatch')
        if pc['plan_sha256']!=sha(a.plan/'completed.json') or pc['selection']!=pilot.select(plan):raise ValueError('Original plan changed')
        for split in ['train','dev']:
            verified(a.cache/split)
            if sha(a.cache/split/'completed.json')!=pc['sources'][split]:raise ValueError('Cache changed')
        if len(plan['dev']['queries'])!=2048:raise ValueError('Expected2048')
        remaining=remaining_indices(pc['selection'])
        if len(remaining)!=1664:raise ValueError('Expected1664')
        contract=dict(code_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
            pilot_code=pilot.code(),pilot_completed_sha256=sha(a.pilot/'completed.json'),head_sha256=sha(a.pilot/'head.pt'),
            full_sha256=sha(a.full/'completed.json'),plan_sha256=sha(a.plan/'completed.json'),
            selection={'remaining':dict(split='dev',indices=remaining)},
            policy='Frozen selected epoch1 head, no training/no selection; primary remaining1664; prior256 descriptive; calibration128 excluded; historically exposed GSV')
        if a.output.exists():
            if read(a.output/'contract.json')!=contract:raise ValueError('Extension contract changed')
        else:a.output.mkdir(parents=True);write(a.output/'contract.json',contract)
        if (a.report/'completed.json').exists():verified(a.report);print('Already complete');return
        import torch
        torch.set_num_threads(4)
        net=torch.nn.Sequential(torch.nn.Linear(1536,128),torch.nn.ReLU(),torch.nn.Linear(128,1))
        net.load_state_dict(torch.load(a.pilot/'head.pt',map_location='cpu',weights_only=True),strict=True);net.eval()
        for param in net.parameters():param.requires_grad_(False)
        oldids=pc['selection']['evaluation']['indices'];oldrows=[load_npz(a.pilot/'evaluation'/f'{i:06d}.npz') for i in oldids]
        oldpred=predict(net,oldrows);saved=load_npz(a.pilot/'evaluation_predictions.npz')
        if not np.allclose(oldpred,saved['prediction'],atol=1e-5,rtol=1e-5):raise ValueError('Frozen head not reproduced')
        oldmetric,oldcases,_=metrics(oldpred,saved['teacher'],saved['labels'],oldids)
        expected=read(a.pilot/'summary.json')['partitions']['evaluation']
        if any(oldmetric[k]!=expected[k] for k in expected):raise ValueError('Pilot metrics not reproduced')
        write(a.output/'prior_case_audit.json',dict(summary=oldmetric,missing_winner_cases=oldcases))
        print('Prior missing-winner cases:',oldcases,flush=True)
        pilot.extract(a,contract,plan)
        rows=[load_npz(a.output/'remaining'/f'{i:06d}.npz') for i in remaining]
        # All44 teacher scores/candidate IDs must reproduce the previous full dev evaluation.
        for qi,r in zip(remaining,rows):
            original=load_npz(a.full/'pairs'/f'{qi:06d}.npz')
            if not np.array_equal(r['candidates'],original['candidates']) or not np.array_equal(r['labels'],original['labels']) or not np.allclose(r['teacher'],original['scores'],atol=1e-4,rtol=1e-4):
                raise ValueError('Full44 teacher changed')
        if sha(a.pilot/'head.pt')!=contract['head_sha256']:raise ValueError('Frozen head changed')
        pred=predict(net,rows);teacher=np.stack([r['teacher'] for r in rows]);labels=np.stack([r['labels'] for r in rows])
        result,cases,keep=metrics(pred,teacher,labels,remaining)
        verdict='INSUFFICIENT_TAIL_CASES' if result['tail_winners']<10 else 'PASS_FEASIBILITY_ONLY' if result['winner_retention']>=.99 and result['tail_retention']>=.9 else 'FAIL'
        a.report.mkdir(parents=True,exist_ok=True)
        write(a.report/'contract.json',contract)
        write(a.report/'summary.json',dict(remaining=result,prior256=oldmetric,verdict=verdict,
            scope='No retraining. Remaining queries share historically used dev bank; no independent generalization or latency claim.'))
        write(a.report/'missing_winner_cases.json',dict(remaining=cases,prior256=oldcases))
        npz(a.report/'predictions.npz',query_ids=np.array(remaining),prediction=pred,teacher=teacher,labels=labels,keep=keep)
        complete(a.report);write(a.output/'progress.json',dict(phase='complete',verdict=verdict,report=str(a.report)))
        print(read(a.report/'summary.json'),flush=True)

if __name__=='__main__':main()
