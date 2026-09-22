"""Small frozen GSV runtime gate. No training; no local dense-feature cache."""
import argparse
import hashlib
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.candidate_set_screen import read, write, sha, load_npz, complete, official, image
from scripts.adaptive_pair_budget import verified
from src.wppr_runtime import prefix, finish, progressive, full


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key, default in [('pilot', '.cache/wppr_pilot_v1'), ('extension', '.cache/wppr_extension_v1'),
                         ('plan', 'doc/candidate_hard_plan_v1'), ('cache', '.cache/candidate_hard_v1'),
                         ('gsv-root', 'datasets/gsv_cities'), ('official-repo', '/home/wt/workspace/Pair-VPR-official'),
                         ('audit', 'doc/pairvpr_official_paired_audit_v1'), ('output', 'doc/wppr_runtime_v1')]:
        p.add_argument('--'+key, type=Path, default=Path(default))
    a = p.parse_args()
    import torch
    import fcntl
    lockpath = a.output.parent / (a.output.name+'.lock')
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    with lockpath.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verified(a.pilot); verified(a.plan); verified(a.cache/'dev')
        contract = dict(head=sha(a.pilot/'head.pt'), plan=sha(a.plan/'completed.json'),
                        cache=sha(a.cache/'dev'/'completed.json'), extension=sha(a.extension/'contract.json'),
                        code={n: hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                              for n in ['src/wppr_runtime.py', 'scripts/wppr_runtime_benchmark.py']},
                        policy='first32 remaining sorted IDs, 3 rotated repeats, FP32 batch1/direction; fixed frozen head')
        if a.output.exists():
            if read(a.output/'contract.json') != contract: raise ValueError('Output contract differs')
            if (a.output/'completed.json').exists(): verified(a.output); print('Already complete'); return
        else: a.output.mkdir(parents=True)
        write(a.output/'contract.json', contract)
        ec = read(a.extension/'contract.json')
        if ec['head_sha256'] != contract['head'] or ec['plan_sha256'] != contract['plan']:
            raise ValueError('Extension/head identity mismatch')
        ids = ec['selection']['remaining']['indices'][:32]
        plan = read(a.plan/'contract.json')['plan']['dev']
        records = plan['database'] + plan['queries']; nd = len(plan['database'])
        model, identity = official(a)
        if identity != read(a.cache/'dev'/'contract.json')['official']: raise ValueError('Model changed')
        torch.set_num_threads(4)
        head = torch.nn.Sequential(torch.nn.Linear(1536,128), torch.nn.ReLU(), torch.nn.Linear(128,1)).cuda().eval()
        head.load_state_dict(torch.load(a.pilot/'head.pt', map_location='cuda', weights_only=True))
        vectors=[]; hashes=[]
        for start in range(0, len(records), 128):
            z=load_npz(a.cache/'dev'/'global'/f'{start:07d}.npz')
            vectors.extend(z['vectors']); hashes.extend(z['hashes'].tolist())
        def encode(i):
            x,_=image(a.gsv_root, records[i], hashes[i])
            f,g=model(x[None].cuda(),None,'global')
            if not np.allclose(g[0].cpu().numpy(), vectors[i], atol=2e-5, rtol=2e-4):
                raise ValueError('Encoder changed')
            return f
        rows=[]
        with torch.inference_mode():
            for position, qi in enumerate(ids):
                old=load_npz(a.extension/'remaining'/f'{qi:06d}.npz')
                query=encode(nd+qi)
                database=torch.cat([encode(int(i)) for i in old['candidates']])
                # Untimed warm-up and exact prefix/continuation correctness gate.
                direct=full(model,query,database,44)
                resumed=finish(model,prefix(model,query,database[:1]))
                if not torch.allclose(resumed, model(query,database[:1],'pairvpr').flatten(),atol=1e-4,rtol=1e-4):
                    raise ValueError('Continuation differs')
                if not np.allclose(direct.cpu().numpy(),old['teacher'],atol=1e-4,rtol=1e-4):
                    raise ValueError('Full44 changed')
                keep,score,pred=progressive(model,head,query,database)
                expected=np.argsort(-head(torch.from_numpy(old['evidence']).cuda()).flatten().cpu().numpy(),kind='stable')[:12]
                if not np.array_equal(keep.cpu().numpy(),expected): raise ValueError('Screen differs')
                if not torch.allclose(score,direct[keep],atol=1e-4,rtol=1e-4): raise ValueError('Selected scores differ')
                winner=int(keep[score.argmax()]); labels=old['labels']
                row=dict(query=qi,selected_correct=bool(labels[winner]),
                         fixed20_correct=bool(labels[int(direct[:20].argmax())]),
                         full44_correct=bool(labels[int(direct.argmax())]),
                         max_score_error=float((score-direct[keep]).abs().max()),timings={})
                methods={'full20':lambda:full(model,query,database,20),
                         'full44':lambda:full(model,query,database,44),
                         'progressive':lambda:progressive(model,head,query,database)}
                del keep,score,pred,direct,resumed
                names=list(methods)
                for repeat in range(3):
                    shift=(position+repeat)%3
                    for name in names[shift:]+names[:shift]:
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                        start=time.perf_counter(); result=methods[name]();torch.cuda.synchronize()
                        elapsed=time.perf_counter()-start
                        memory=torch.cuda.max_memory_allocated()
                        del result
                        row['timings'].setdefault(name,[]).append(dict(seconds=elapsed,peak_bytes=memory))
                rows.append(row);write(a.output/'rows.json',rows)
                write(a.output/'progress.json',dict(phase='benchmark',done=len(rows),total=len(ids)))
                print('runtime',len(rows),'/',len(ids),flush=True)
        stats={}
        for name in ['full20','full44','progressive']:
            seconds=[np.median([t['seconds'] for t in r['timings'][name]]) for r in rows]
            stats[name]=dict(mean_seconds=float(np.mean(seconds)),median_seconds=float(np.median(seconds)),
                             p95_seconds=float(np.quantile(seconds,.95)),
                             peak_mib=max(t['peak_bytes'] for r in rows for t in r['timings'][name])/2**20)
        write(a.output/'summary.json',dict(queries=len(rows),continuation_pass=True,timings=stats,
            correct={k:sum(r[k] for r in rows) for k in ['selected_correct','fixed20_correct','full44_correct']},
            speedup_vs20=stats['full20']['mean_seconds']/stats['progressive']['mean_seconds'],
            speedup_vs44=stats['full44']['mean_seconds']/stats['progressive']['mean_seconds'],
            scope='Decoder-only cached GPU dense inputs, includes screening/state retention, excludes image encoding/retrieval; not end-to-end latency. First32 GSV only; no Pitts claim.',
            environment=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0))))
        write(a.output/'progress.json',dict(phase='complete'));complete(a.output)
        print(read(a.output/'summary.json'))


if __name__=='__main__': main()
