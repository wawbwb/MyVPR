"""Frozen Pair-VPR admission audit, using the existing Pitts30k-val NPY indices."""
import argparse
from collections import OrderedDict
import hashlib
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.candidate_set_screen import read,write,sha,npz,load_npz,official,image,complete,codes as previous_codes
from src.candidate_set_utils import summary,stable_key


def codes():
    return {**previous_codes(),'scripts/pitts_pair_admission.py':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()}


def load_index(root,query_limit=1024):
    names=['pitts30k_val_dbImages.npy','pitts30k_val_qImages.npy','pitts30k_val_gt_25m.npy']
    db,queries=[np.load(root/n,allow_pickle=False) for n in names[:2]]
    gt=np.load(root/names[2],allow_pickle=True)
    def paths(values):
        if values.ndim!=1:raise ValueError('Expected one-dimensional image index')
        out=[]
        for value in values:
            if not isinstance(value,(str,bytes,np.str_,np.bytes_)):raise ValueError('Invalid image path type')
            name=(value.decode() if isinstance(value,bytes) else str(value)).replace('\\','/')
            path=(root/name).resolve()
            if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):raise ValueError('Path escapes dataset: '+name)
            if not path.is_file():raise FileNotFoundError(path)
            out.append(name)
        return out
    db,queries=paths(db),paths(queries)
    if not db or not queries or len(gt)!=len(queries):raise ValueError('Empty or mismatched dataset')
    if len(set(db+queries))!=len(db)+len(queries):raise ValueError('Duplicate query/database paths')
    positives=[]
    for row in gt:
        row=np.asarray(row)
        if row.ndim!=1 or not len(row) or row.dtype.kind not in 'iu':raise ValueError('GT must contain nonempty integer index arrays')
        if (row<0).any() or (row>=len(db)).any() or len(np.unique(row))!=len(row):raise ValueError('Invalid GT indices')
        positives.append(row.astype(np.int64).tolist())
    if query_limit<1:raise ValueError('query-limit must be positive')
    selected=sorted(range(len(queries)),key=lambda i:stable_key('pitts-admission-v1:'+queries[i]))[:query_limit]
    return {'database':db,'queries':[queries[i] for i in selected],'query_indices':selected,
        'gt':[positives[i] for i in selected],'all_queries':len(queries),
        'index_sha256':{n:sha(root/n) for n in names},'selection':'Fixed path hash before inference; full reference database'}


def verify_files(out):
    done=read(out/'completed.json')
    if not done['complete']:raise ValueError('Incomplete output')
    for n,h in done['files'].items():
        if sha(out/n)!=h:raise ValueError('Changed output '+n)


def check(a):
    index=load_index(a.dataset,a.query_limit)
    if a.output.exists():
        verify_files(a.output)
        if read(a.output/'contract.json')!={'code':codes(),'index':index}:raise ValueError('Index or code changed')
    else:
        a.output.mkdir(parents=True);write(a.output/'contract.json',{'code':codes(),'index':index});complete(a.output)
    print('PASS NPY indices, multi-positive GT, all indexed image paths',flush=True)
    print({'database':len(index['database']),'all_queries':index['all_queries'],'fixed_selected_queries':len(index['queries'])},flush=True)


def run(a):
    import torch
    index=load_index(a.dataset,a.query_limit)
    # Existing loader uses gsv_root only to construct an unused dataset-parent configuration field.
    a.gsv_root=a.dataset
    model,identity=official(a)
    contract={'code':codes(),'index':index,'official':identity,'topk':20,
        'scope':'Pitts30k-val development admission, fixed query sample, full database, frozen official Pair-VPR. No training or test-set claim.'}
    out=a.output
    if out.exists():
        if read(out/'contract.json')!=contract:raise ValueError('Output contract changed')
        if (out/'completed.json').exists():verify_files(out);verify_files(out/'report');print('Completed audit verified');return
    else:
        out.mkdir(parents=True);write(out/'contract.json',contract);(out/'global').mkdir();(out/'pairs').mkdir()
    ndb=len(index['database']);nq=len(index['queries']);records=[{'path':p} for p in index['database']+index['queries']]
    k=min(20,ndb);all_vectors=[];hashes=[]
    with torch.inference_mode():
        for start in range(0,len(records),128):
            stop=min(start+128,len(records));file=out/'global'/f'{start:07d}.npz'
            if file.exists() and file.with_suffix('.sha.json').exists():
                data=load_npz(file)
                if len(data['hashes'])!=stop-start:raise ValueError('Bad hash count')
                for r,h in zip(records[start:stop],data['hashes']):
                    if sha(a.dataset/r['path'])!=str(h):raise ValueError('Image changed')
            else:
                vectors=[];hs=[]
                for j in range(start,stop,4):
                    inputs=[image(a.dataset,r) for r in records[j:min(j+4,stop)]]
                    _,z=model(torch.stack([x[0] for x in inputs]).cuda(),None,'global')
                    vectors.append(z.cpu().numpy());hs.extend([x[1] for x in inputs])
                data={'vectors':np.concatenate(vectors),'hashes':np.array(hs)};npz(file,**data)
            if data['vectors'].shape!=(stop-start,512) or not np.isfinite(data['vectors']).all() or not np.allclose(np.linalg.norm(data['vectors'],axis=1),1,atol=2e-4):raise ValueError('Invalid global vectors')
            all_vectors.append(data['vectors']);hashes.extend(data['hashes'].tolist())
            print(f'Pitts global {stop}/{len(records)}',flush=True)
        z=np.concatenate(all_vectors);db=torch.from_numpy(z[:ndb]).cuda();memo=OrderedDict()
        def dense(i):
            if i in memo:memo.move_to_end(i);return memo[i].cuda()
            x,_=image(a.dataset,records[i],hashes[i]);f,_=model(x[None].cuda(),None,'global');memo[i]=f.cpu()
            if len(memo)>64:memo.popitem(last=False)
            return f
        captured=[];handle=model.classvprmodule.register_forward_pre_hook(lambda m,i:captured.append(i[0].detach().clone()))
        try:
            for qi in range(nq):
                file=out/'pairs'/f'{qi:06d}.npz'
                if file.exists() and file.with_suffix('.sha.json').exists():load_npz(file);continue
                scores=(torch.from_numpy(z[ndb+qi]).cuda()@db.T).cpu().numpy()
                ids=np.argsort(-scores,kind='stable')[:k];q=dense(ndb+qi);evidence=[];base=[]
                for di in ids:
                    d=dense(int(di));captured.clear();s1=model(q,d,'pairvpr');s2=model(d,q,'pairvpr')
                    if len(captured)!=2 or captured[0].shape!=(1,768):raise ValueError('Official CLS interface changed')
                    if not torch.allclose(s1,model.classvprmodule(captured[0]),atol=1e-6,rtol=1e-6):raise ValueError('Pair score reconstruction failed')
                    evidence.append(torch.cat(captured[:2],-1)[0].cpu().numpy());base.append(float((s1+s2).item()))
                values=dict(evidence=np.stack(evidence),base=np.array(base,np.float32),candidates=ids,
                    db_vectors=z[ids],global_scores=scores[ids],labels=np.isin(ids,index['gt'][qi]))
                if not all(np.isfinite(values[key]).all() for key in ['evidence','base','global_scores']):raise ValueError('Nonfinite pair scores')
                npz(file,**values)
                if (qi+1)%16==0 or qi+1==nq:print(f'Pitts bidirectional pairs {qi+1}/{nq} x {k}',flush=True)
        finally:handle.remove()
    shards=[load_npz(out/'pairs'/f'{i:06d}.npz') for i in range(nq)]
    base=np.stack([r['base'] for r in shards]);glob=np.stack([r['global_scores'] for r in shards]);labels=np.stack([r['labels'] for r in shards])
    global_stats=summary(glob,labels,glob);pair_stats=summary(base,labels,glob)
    errors=pair_stats['reachable']-pair_stats['correct']
    report=out/'report';report.mkdir(exist_ok=True)
    write(report/'contract.json',contract)
    write(report/'decision.json',{'passed':errors>=10,'global':global_stats,'pair':pair_stats,'reachable_errors':errors,
        'unreachable':nq-pair_stats['reachable'],'minimum_reachable_errors':10,
        'note':'Fixed feasibility threshold only; passing does not start training or establish candidate-context value.'})
    npz(report/'outcomes.npz',base=base,global_scores=glob,labels=labels,candidates=np.stack([r['candidates'] for r in shards]),query_indices=np.array(index['query_indices']))
    complete(report);complete(out)
    print({'global_correct':global_stats['correct'],'pair_correct':pair_stats['correct'],'reachable_errors':errors,'queries':nq},flush=True)
    print('Download:',report,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['check','run'])
    p.add_argument('--dataset',type=Path,default=Path('datasets/pitts30k-val'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--query-limit',type=int,default=1024)
    p.add_argument('--official-repo',type=Path,default=Path('/home/wt/workspace/Pair-VPR-official'))
    p.add_argument('--audit',type=Path,default=Path('doc/pairvpr_official_paired_audit_v1'))
    a=p.parse_args();globals()[a.stage](a)


if __name__=='__main__':main()
