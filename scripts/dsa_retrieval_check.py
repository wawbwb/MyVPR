"""Read-only phase0 source, train-only bases, fixed cross-view development retrieval."""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.candidate_set_screen import read,write,sha,complete
from scripts.adaptive_pair_budget import verified
from src.discriminative_subspace import MODES,fit,shuffle_places,overlap
from src.dsa_retrieval import cross_view,paired


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('logs/dsa_phase0_v1'))
    p.add_argument('--output',type=Path,default=Path('logs/dsa_retrieval_v1'))
    a=p.parse_args()
    import fcntl
    torch.set_num_threads(4)
    verified(a.source)
    c=read(a.source/'contract.json'); n=len(c['train_places'])
    if n!=4096 or len(c['dev_places'])!=1024 or set(c['train_places'])&set(c['dev_places']):
        raise ValueError('Unexpected source split')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    lock=(a.output.parent/(a.output.name+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.output.exists(): raise ValueError('Use a fresh output directory')
    a.output.mkdir()
    policy=dict(primary='train-mean centered cosine; Euclidean orthonormal rank16, no whitening',
        secondary='uncentered cosine, descriptive only',views='all 12 ordered distinct view pairs; one reference per place per pair',
        halves='split true training places even/odd first, then shuffle within each half',
        decision='positive paired delta vs both matched controls at BOTH layers for full/half0/half1 primary; not a formal significance gate',
        scope='GSV exposed development, mean intermediate features NOT RU descriptors; no formal VPR training')
    files=['scripts/dsa_retrieval_check.py','src/dsa_retrieval.py','src/discriminative_subspace.py']
    write(a.output/'contract.json',dict(source_completed_sha256=sha(a.source/'completed.json'),policy=policy,
        code={f:hashlib.sha256((ROOT/f).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for f in files}))
    paths=sorted((a.source/'vectors').glob('*.npz'))
    if [f.name for f in paths]!=[f'{i:04d}.npz' for i in range(640)]: raise ValueError('Unexpected source shards')
    arrays=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as z: arrays.append(z['vectors'].copy())
    x=torch.from_numpy(np.concatenate(arrays)).double()
    if x.shape!=(5120,4,2,768) or not torch.isfinite(x).all(): raise ValueError('Invalid vectors')
    saved=torch.load(a.source/'bases.pt',map_location='cpu',weights_only=True)
    summary={}; outcomes={}; decisions=[]
    for layer in range(2):
        train,dev=x[:n,:,layer],x[n:,:,layer]; layer_report={}
        for split in ('full','half0','half1'):
            fitting=train if split=='full' else train[int(split[-1])::2]
            mean=fitting.mean((0,1)); bases={}; stats={}
            for mode in MODES:
                data=shuffle_places(fitting) if mode=='shuffled_fisher' else fitting
                u,_=fit(data,fisher=mode!='pca');bases[mode]=u
                if split=='full' and overlap(u,saved[mode][layer])<.99999: raise ValueError('Saved basis not reproduced')
            for centering in ('centered','uncentered'):
                evaluated=dev-mean if centering=='centered' else dev
                hitsets={}; table={}
                for mode in ('original',)+MODES:
                    hits,ranks=cross_view(evaluated if mode=='original' else evaluated@bases[mode])
                    key=f'l{layer+11}_{split}_{centering}_{mode}'
                    outcomes[key+'_ranks']=ranks; hitsets[mode]=hits
                    table[mode]=dict(r1=float(hits.mean()),r5=float((ranks<=5).mean()),
                                     r10=float((ranks<=10).mean()),correct=int(hits.sum()),queries=int(hits.size))
                comparisons={m:paired(hitsets['place_fisher'],hitsets[m]) for m in ('original','pca','shuffled_fisher')}
                stats[centering]=dict(metrics=table,true_vs=comparisons)
                if centering=='centered': decisions.extend(comparisons[m]['delta_r1_pp']>0 for m in ('pca','shuffled_fisher'))
            layer_report[split]=stats
            write(a.output/'progress.json',dict(phase='retrieval',layer=layer+11,split=split))
            print('Finished',layer+11,split,stats['centered']['metrics'],flush=True)
        summary[str(layer+11)]=layer_report
    np.savez_compressed(a.output/'per_place_ranks.npz',**outcomes)
    report=dict(layers=summary,policy=policy,dev_place_ids=c['dev_places'],
        diagnostic='CONSISTENT_CONTROL_ADVANTAGE' if all(decisions) else 'CONTROL_ADVANTAGE_NOT_CONSISTENT',
        original_phase0_verdict=read(a.source/'summary.json')['verdict'],optimizer_steps=0)
    write(a.output/'summary.json',report);write(a.output/'progress.json',dict(phase='complete',diagnostic=report['diagnostic']))
    complete(a.output);print(report['diagnostic'],flush=True)


if __name__=='__main__':main()
