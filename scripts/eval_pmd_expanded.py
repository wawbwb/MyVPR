"""Locked-checkpoint PMD evaluation on remaining GSV dev queries (exploratory)."""
import argparse
from collections import OrderedDict
import copy
import hashlib
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.candidate_set_screen import read,write,sha,load_npz,npz,official,image,complete
from scripts.adaptive_pair_budget import verified
from scripts.train_pmd import MODES,restore_trainable
from scripts.diagnose_pmd import ranking
from src.models.partial_matching import PMDPair

SCOPE='Remaining GSV dev, historically explored pool; fixed best checkpoints and original top20; NOT independent confirmation, no training or selection on this evaluation.'


def remaining_ids(total,excluded):
    if len(set(excluded))!=len(excluded) or any(type(i) is not int or not 0<=i<total for i in excluded):raise ValueError('Invalid excluded IDs')
    return [i for i in range(total) if i not in set(excluded)]


def paired_counts(reference,variant,ids):
    reference=np.asarray(reference,bool);variant=np.asarray(variant,bool);ids=np.asarray(ids)
    if reference.shape!=variant.shape or reference.shape!=ids.shape or not len(ids):raise ValueError('Invalid paired arrays')
    fixes=ids[~reference & variant].tolist();losses=ids[reference & ~variant].tolist()
    return dict(corrections=fixes,regressions=losses,net=len(fixes)-len(losses),delta_r1_pp=100*(len(fixes)-len(losses))/len(ids))


def validate_row(z,qi,source):
    if z['query'].shape!=() or int(z['query'])!=qi or not np.array_equal(z['candidates'],source['candidates']) or not np.array_equal(z['labels'],source['labels']):raise ValueError('Row identity mismatch')
    if z['scores'].shape!=(4,20) or not np.isfinite(z['scores']).all():raise ValueError('Invalid scores')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key,val in [('runs','logs/pmd'),('plan','doc/candidate_hard_plan_v1'),('cache','.cache/candidate_hard_v1/dev'),
        ('gsv-root','datasets/gsv_cities'),('official-repo','/home/wt/workspace/Pair-VPR-official'),
        ('audit','doc/pairvpr_official_paired_audit_v1'),('output','doc/pmd_expanded_v1')]:p.add_argument('--'+key,type=Path,default=Path(val))
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    import torch,fcntl
    from tqdm import tqdm
    torch.set_num_threads(4);a.output.parent.mkdir(parents=True,exist_ok=True)
    with (a.output.parent/(a.output.name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print('Verifying completed runs and cache; CPU/disk only...',flush=True)
        verified(a.plan);verified(a.cache);plan=read(a.plan/'contract.json')['plan']['dev']
        common=None;checkpoints={};summaries={};contracts={}
        for mode in MODES:
            root=a.runs/(mode+'_screen_v1');verified(root);c=read(root/'contract.json');s=read(root/'summary.json')
            contracts[mode]=sha(root/'contract.json')
            if c.pop('mode')!=mode or c['smoke'] or s['history'][-1]['epoch']!=3:raise ValueError('Incomplete/wrong run')
            if common is not None and common!=c:raise ValueError('Unmatched runs')
            common=c;summaries[mode]=s
            checkpoints[mode]=dict(path=str(root/'best.pt'),sha256=sha(root/'best.pt'),epoch=s['best_epoch'])
        for path in [a.plan,a.cache]:
            if common['sources'][str(path)]!=sha(path/'completed.json'):raise ValueError('Source changed')
        for name,h in common['code'].items():
            if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=h:raise ValueError('Training code changed')
        ids=remaining_ids(len(plan['queries']),common['dev'])
        if len(ids)!=1792 or len(common['dev'])!=256:raise ValueError('Unexpected evaluation size')
        contract=dict(scope=SCOPE,query_ids=ids,excluded_selection_ids=common['dev'],topk=20,modes=list(MODES),checkpoints=checkpoints,
            plan=sha(a.plan/'completed.json'),cache=sha(a.cache/'completed.json'),
            code={n:hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in
                  ['scripts/eval_pmd_expanded.py','scripts/diagnose_pmd.py','scripts/candidate_set_screen.py','src/models/partial_matching.py']})
        if a.output.exists():
            if not a.resume or read(a.output/'contract.json')!=contract:raise ValueError('Existing/changed output contract')
            if (a.output/'completed.json').exists():verified(a.output);print('Already complete');return
        else:
            a.output.mkdir();write(a.output/'contract.json',contract)
        (a.output/'rows').mkdir(exist_ok=True)
        missing=[];sources={}
        for qi in ids:
            source=load_npz(a.cache/'pairs'/f'{qi:06d}.npz');sources[qi]=source
            target=a.output/'rows'/f'{qi:06d}.npz'
            if target.exists() and target.with_suffix('.sha.json').exists():validate_row(load_npz(target),qi,source)
            else:missing.append(qi)
        print(f'Valid rows reused: {len(ids)-len(missing)}; pending: {len(missing)}',flush=True)
        if missing:
            base,identity=official(a)
            if identity!=read(a.cache/'contract.json')['official']:raise ValueError('Official identity mismatch')
            models={}
            for mode in MODES:
                model=PMDPair(copy.deepcopy(base),'ordinary' if mode=='baseline' else mode).cuda().eval()
                for param in model.base.dec_blocks[-1].parameters():param.requires_grad_(True)
                if mode=='baseline':
                    for param in model.match.parameters():param.requires_grad_(False)
                elif mode!='partial':model.match.bin_score.requires_grad_(False)
                ck=torch.load(checkpoints[mode]['path'],map_location='cpu',weights_only=True)
                if ck['state']['contract_sha256']!=contracts[mode] or ck['state']['best_epoch']!=checkpoints[mode]['epoch']:raise ValueError('Checkpoint identity mismatch')
                restore_trainable(model,ck['parameters']);models[mode]=model
            del ck
            # The encoder and blocks1..11 are frozen and identical in all arms.
            # Reuse their computation, but apply each trained block12 separately.
            features=OrderedDict();chunks={};records=plan['database']+plan['queries'];nd=len(plan['database'])
            def dense(index):
                if index in features:features.move_to_end(index);return features[index].cuda()
                start=index//128*128
                if start not in chunks:chunks[start]=load_npz(a.cache/'global'/f'{start:07d}.npz')
                z=chunks[start];x,_=image(a.gsv_root,records[index],str(z['hashes'][index-start]))
                f,g=base(x[None].cuda(),None,'global')
                if not np.allclose(g[0].cpu().numpy(),z['vectors'][index-start],atol=2e-5,rtol=2e-4):raise ValueError('Encoder mismatch')
                features[index]=f.cpu()
                if len(features)>64:features.popitem(last=False)
                return f
            def scores(qi,source):
                q=dense(nd+qi);result=[]
                for di in source['candidates']:
                    d=dense(int(di));forward=models['baseline'].prefix(q,d);reverse=models['baseline'].prefix(d,q)
                    result.append([float((model.finish(forward,bypass=mode=='baseline')[0]+model.finish(reverse,bypass=mode=='baseline')[0])[0]) for mode,model in models.items()])
                return np.asarray(result,dtype=np.float64).T
            with torch.no_grad():
                errors={mode:0. for mode in MODES}
                for qi in common['dev'][:2]:
                    source=load_npz(a.cache/'pairs'/f'{qi:06d}.npz');values=scores(qi,source)
                    for mi,mode in enumerate(MODES):
                        saved=read(a.runs/(mode+'_screen_v1')/f"validation_epoch{checkpoints[mode]['epoch']:02d}.json")
                        reference=next(r['scores'] for r in saved if r['query']==qi)
                        errors[mode]=max(errors[mode],float(np.max(np.abs(values[mi]-reference))))
                        if not np.allclose(values[mi],reference,atol=1e-4,rtol=1e-4):raise ValueError('Shared-prefix score reproduction failed')
                write(a.output/'reproduction.json',errors);print('PASS score reproduction:',errors,flush=True)
                for i,qi in enumerate(tqdm(missing,desc='PMD four-arm fixed top20')):
                    source=sources[qi];values=scores(qi,source)
                    z=dict(query=np.array(qi),candidates=source['candidates'],labels=source['labels'],scores=values)
                    validate_row(z,qi,source);npz(a.output/'rows'/f'{qi:06d}.npz',**z)
                    write(a.output/'progress.json',dict(phase='evaluation',completed=len(ids)-len(missing)+i+1,total=len(ids)))
        rows=[]
        for qi in ids:
            z=load_npz(a.output/'rows'/f'{qi:06d}.npz');validate_row(z,qi,sources[qi])
            rows.append(dict(query=qi,frozen=ranking(sources[qi]['base'],z['labels']),
                arms={mode:ranking(z['scores'][i],z['labels']) for i,mode in enumerate(MODES)}))
        frozen=[r['frozen']['correct'] for r in rows];hits={m:[r['arms'][m]['correct'] for r in rows] for m in MODES}
        report=dict(scope=SCOPE,queries=len(rows),frozen_correct=sum(frozen),
            reachable_errors=sum(not r['frozen']['correct'] and r['frozen']['reachable'] for r in rows),
            unreachable_errors=sum(not r['frozen']['reachable'] for r in rows),arms={},partial_vs_controls={})
        for mode in MODES:
            report['arms'][mode]=dict(epoch=checkpoints[mode]['epoch'],correct=sum(hits[mode]),r1=sum(hits[mode])/len(rows),vs_frozen=paired_counts(frozen,hits[mode],ids))
        for mode in ['baseline','ordinary','forced']:report['partial_vs_controls'][mode]=paired_counts(hits[mode],hits['partial'],ids)
        write(a.output/'per_query.json',rows);write(a.output/'summary.json',report)
        write(a.output/'progress.json',dict(phase='complete',completed=len(ids),total=len(ids)));complete(a.output);print(report,flush=True)


if __name__=='__main__':main()
