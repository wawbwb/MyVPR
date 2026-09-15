"""GSV-only head training; fixed Pitts admission queries are development data."""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.candidate_set_screen import (read,write,sha,npz,load_npz,complete,
    verify,seed,cuda,predict,MODES)
from scripts.pitts_pair_admission import codes as cache_codes,verify_files
from src.candidate_set_utils import summary

SCOPE='GSV training / fixed Pitts30k-val development selection. Not independent test performance.'


def codes():
    return {**cache_codes(),'scripts/candidate_set_pitts.py':hashlib.sha256(
        Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()}


def verify_run(path):
    verify_files(path)
    c=read(path/'contract.json')
    if c['code']!=codes():raise ValueError('Training/report code changed')
    return c


def validate_row(row,positives,ndb,k=20):
    shapes={'evidence':(k,1536),'base':(k,),'db_vectors':(k,512),
        'labels':(k,),'candidates':(k,),'global_scores':(k,)}
    for key,shape in shapes.items():
        if key not in row or row[key].shape!=shape or not np.isfinite(row[key]).all():
            raise ValueError('Invalid pair row: '+key)
    ids=row['candidates']
    if ids.dtype.kind not in 'iu' or (ids<0).any() or (ids>=ndb).any() or len(np.unique(ids))!=k:
        raise ValueError('Invalid candidate indices')
    if row['labels'].dtype!=np.bool_ or not np.array_equal(row['labels'],np.isin(ids,positives)):
        raise ValueError('Cached labels disagree with original multi-positive GT')
    if (np.diff(row['global_scores'])>1e-6).any():raise ValueError('Wrong candidate order')
    if not np.allclose(np.linalg.norm(row['db_vectors'],axis=1),1,atol=2e-4):
        raise ValueError('Unnormalized database vectors')


def validate_outcomes(data,outcomes,qids,decision):
    for key in ['base','labels','candidates','global_scores']:
        if not np.array_equal(outcomes[key],data[key]):raise ValueError('Pitts report/query order mismatch: '+key)
    if not np.array_equal(outcomes['query_indices'],qids):raise ValueError('Pitts query mapping mismatch')
    result=summary(data['base'],data['labels'],data['global_scores'])
    if result!=decision['pair'] or not decision['passed'] or result['reachable']-result['correct']<10:
        raise ValueError('Pitts admission not passed or changed')


def load_cache(path,kind,plan_path=None):
    verify_files(path);c=read(path/'contract.json')
    if kind=='pitts':
        if c['code']!=cache_codes() or c['topk']!=20:raise ValueError('Pitts cache source changed')
        index=c['index'];n=len(index['queries']);ndb=len(index['database'])
        if n!=1024 or n!=len(index['gt']) or len(index['query_indices'])!=n:
            raise ValueError('Expected the original 1024-query development sample')
        qids=np.asarray(index['query_indices'])
        if qids.dtype.kind not in 'iu' or len(np.unique(qids))!=n or (qids<0).any() or (qids>=index['all_queries']).any():
            raise ValueError('Invalid original query mapping')
        positives=index['gt']
    else:
        c=verify(path);plan=verify(plan_path)
        if c['split']!='train' or c['plan_sha256']!=sha(plan_path/'completed.json') or c['topk']!=20:
            raise ValueError('Wrong GSV training cache or plan')
        part=plan['plan']['train'];n=len(part['queries']);ndb=len(part['database'])
        if c['queries']!=n or c['ndb']!=ndb:raise ValueError('GSV cache size mismatch')
        places={}
        for i,r in enumerate(part['database']):places.setdefault(r['label'],[]).append(i)
        positives=[places.get(r['label'],[]) for r in part['queries']]
    rows=[]
    for i,gt in enumerate(positives):
        row=load_npz(path/'pairs'/f'{i:06d}.npz');validate_row(row,gt,ndb);rows.append(row)
    data={key:np.stack([r[key] for r in rows]) for key in rows[0]}
    if kind=='pitts':
        verify_files(path/'report')
        if read(path/'report'/'contract.json')!=c:raise ValueError('Pitts report contract mismatch')
        outcomes=load_npz(path/'report'/'outcomes.npz')
        validate_outcomes(data,outcomes,qids,read(path/'report'/'decision.json'))
    return data,c


def as_tensors(data):
    import torch
    return {k:torch.from_numpy(data[k]).cuda() for k in ['evidence','base','db_vectors','labels']}


def check(a):
    data,c=load_cache(a.dev_cache,'pitts')
    verify(a.plan)
    result=summary(data['base'],data['labels'],data['base'])
    print('PASS fixed Pitts cache, multi-positive GT, query mapping, code and file hashes')
    print('Frozen correct:',result['correct'],'/',result['queries'],
        'reachable errors:',result['reachable']-result['correct'],flush=True)


def preflight(a):
    import unittest
    import torch
    from src.models.candidate_set import CandidateSet,list_loss
    cuda();seed()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in
        ['tests.test_candidate_set_utils','tests.test_candidate_set_torch','tests.test_candidate_set_pitts'])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:raise ValueError('Preflight failed or skipped tests')
    x=torch.randn(2,20,1536,device='cuda');s=torch.randn(2,20,device='cuda')
    z=torch.nn.functional.normalize(torch.randn(2,20,512,device='cuda'),dim=-1)
    y=torch.zeros(2,20,dtype=torch.bool,device='cuda');y[:,[1,3]]=True
    for mode in MODES:
        model=CandidateSet(mode).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        before=model(x,s,z);torch.testing.assert_close(before,s,rtol=0,atol=0)
        loss,count=list_loss(before,y);loss.backward()
        if count!=2 or not torch.isfinite(loss) or not model.out.weight.grad.abs().sum()>0:
            raise ValueError('Invalid CUDA gradient')
        optimizer.step()
        if torch.equal(model(x,s,z),s):raise ValueError('CUDA head did not update')
    print('PASS five CUDA heads: frozen initialization, multi-positive loss, backward and update')


def train(a):
    import torch
    from src.models.candidate_set import CandidateSet,list_loss,duplicate_consistency
    cuda();seed()
    tr_np,tc=load_cache(a.train_cache,'gsv',a.plan);dev_np,dc=load_cache(a.dev_cache,'pitts')
    if tc['official']!=dc['official']:raise ValueError('GSV/Pitts frozen model identity mismatch')
    tr=as_tensors(tr_np);dev=as_tensors(dev_np)
    train_valid=int((tr_np['labels'].any(1)&(~tr_np['labels']).any(1)).sum())
    base=summary(dev['base'].cpu().numpy(),dev['labels'].cpu().numpy(),dev['base'].cpu().numpy())
    reachable_errors=base['reachable']-base['correct']
    contract={'code':codes(),'train_sha256':sha(a.train_cache/'completed.json'),
        'dev_sha256':sha(a.dev_cache/'completed.json'),'plan_sha256':tc['plan_sha256'],'official':tc['official'],
        'epochs':5,'seed':42,'lr':.0001,'modes':MODES,'scope':SCOPE,'weight_decay':.001,'batch_size':8,'residual_bound':4,'density_temperature':.02,'selection':'Best Pitts dev R1 per mode, earliest epoch on ties; no epoch-zero candidate','consistency_weight':1,'duplicate_count':5}
    if a.output.exists():
        if not a.resume or read(a.output/'contract.json')!=contract:raise ValueError('Existing run requires matching --resume')
        if (a.output/'completed.json').exists():
            verify_run(a.output)
            if not read(a.output/'gate.json')['passed']:sys.exit(3)
            print('Run already complete');return
    else:
        a.output.mkdir(parents=True);write(a.output/'contract.json',contract)
    if reachable_errors<10 or train_valid<128:
        write(a.output/'gate.json',{'passed':False,'dev_baseline':base,'dev_reachable_errors':reachable_errors,
            'reason':'Insufficient non-saturated retrieval supervision. No training; do not tune this threshold to pass.'})
        complete(a.output);print('SCREEN STOP: download gate.json');sys.exit(3)
    write(a.output/'gate.json',{'passed':True,'dev_baseline':base,'dev_reachable_errors':reachable_errors,
        'train_queries':len(tr_np['base']),'train_valid_queries':train_valid,
        'train_unreachable_queries':int((~tr_np['labels'].any(1)).sum())})
    valid=torch.where(tr['labels'].any(1)&(~tr['labels']).any(1))[0].cpu().numpy()
    logs=read(a.output/'training_logs.json') if (a.output/'training_logs.json').exists() else {}
    selection=read(a.output/'selection.json') if (a.output/'selection.json').exists() else {}
    for mode in MODES:
        if len(logs.get(mode,[]))==5 and mode in selection and (a.output/f'{mode}.pt').exists():
            if sha(a.output/f'{mode}.pt')!=selection[mode]['checkpoint_sha256']:raise ValueError('Changed saved checkpoint')
            continue
        seed();model=CandidateSet(mode).cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.001)
        best=-1;logs[mode]=[]
        for epoch in range(5):
            model.train();order=np.random.default_rng(42+epoch).permutation(valid);epoch_logs=[]
            for start in range(0,len(order),8):
                ids=order[start:start+8];x=tr['evidence'][ids];s=tr['base'][ids];z=tr['db_vectors'][ids];y=tr['labels'][ids]
                original=model(x,s,z);loss,_=list_loss(original,y)
                # Same duplicate augmentation in all arms; no candidate labels enter the network.
                rng=np.random.default_rng(100000*epoch+start+42);extra=rng.integers(0,s.shape[1],size=5)
                ix=torch.tensor(list(range(s.shape[1]))+extra.tolist(),device='cuda')
                augmented=model(x[:,ix],s[:,ix],z[:,ix]);aug_loss,_=list_loss(augmented[:,:s.shape[1]],y)
                consistency=duplicate_consistency(original,augmented)
                total=.5*(loss+aug_loss)+(consistency if mode in ['consistency','competition'] else consistency*0)
                opt.zero_grad(set_to_none=True);total.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
                if not torch.isfinite(total):raise ValueError('Nonfinite training loss')
                epoch_logs.append({'loss':float(loss),'consistency':float(consistency),'grad_norm':float(norm)})
            scores=predict(model,dev).cpu().numpy();result=summary(scores,dev['labels'].cpu().numpy(),dev['base'].cpu().numpy())
            logs[mode].append({'epoch':epoch+1,'batches':epoch_logs,'dev':result})
            if result['correct']>best:
                best=result['correct'];temp=a.output/f'{mode}.tmp';torch.save(model.state_dict(),temp);temp.replace(a.output/f'{mode}.pt')
                selection[mode]={'epoch':epoch+1,'dev':result,'checkpoint_sha256':sha(a.output/f'{mode}.pt')}
            print(f'{mode} epoch{epoch+1}: dev {result["correct"]}/{len(scores)}',flush=True)
        write(a.output/'training_logs.json',logs);write(a.output/'selection.json',selection)
    complete(a.output)


def report(a):
    import torch
    from src.models.candidate_set import CandidateSet
    cuda();seed();runs=verify_run(a.runs)
    if not read(a.runs/'gate.json')['passed']:raise ValueError('Training gate failed; inspect report first')
    raw,c=load_cache(a.dev_cache,'pitts')
    if sha(a.dev_cache/'completed.json')!=runs['dev_sha256'] or c['official']!=runs['official']:
        raise ValueError('Wrong development cache')
    contract={'code':codes(),'dev_cache_sha256':sha(a.dev_cache/'completed.json'),
        'training_sha256':sha(a.runs/'completed.json'),'scope':SCOPE}
    if a.output.exists():
        if read(a.output/'contract.json')!=contract:raise ValueError('Report contract changed')
        if (a.output/'completed.json').exists():verify_run(a.output);print('Report already complete');return
    else:a.output.mkdir(parents=True)
    write(a.output/'contract.json',contract)
    data=as_tensors(raw)
    base=data['base'].cpu().numpy();labels=data['labels'].cpu().numpy();results={'frozen':summary(base,labels,base)};stress={}
    outputs={'labels':labels,'frozen':base,'candidates':raw['candidates'],'query_indices':np.array(c['index']['query_indices'])}
    for mode in MODES:
        model=CandidateSet(mode).cuda();model.load_state_dict(torch.load(a.runs/f'{mode}.pt',map_location='cuda',weights_only=True));model.eval()
        scores=predict(model,data)
        if float((scores-data['base']).abs().max())>4.0001:raise ValueError('Residual bound violated')
        outputs[mode]=scores.cpu().numpy();results[mode]=summary(outputs[mode],labels,base)
        if results[mode]!=read(a.runs/'selection.json')[mode]['dev']:raise ValueError('Selected checkpoint does not reproduce dev score')
        perm=torch.arange(base.shape[1]-1,-1,-1,device='cuda');differences=[];dup=[]
        with torch.no_grad():
            for start in range(0,len(base),16):
                x=data['evidence'][start:start+16];s=data['base'][start:start+16];z=data['db_vectors'][start:start+16]
                original=scores[start:start+16]
                shuffled=model(x[:,perm],s[:,perm],z[:,perm])[:,perm]
                differences.append(float((original-shuffled).abs().max()))
                ix=torch.tensor(list(range(s.shape[1]))+[0]*5,device='cuda')
                repeated=model(x[:,ix],s[:,ix],z[:,ix])[:,:s.shape[1]]
                dup.append(repeated.cpu().numpy())
        if max(differences)>2e-4:raise ValueError('Candidate-order invariance failed')
        dup=np.concatenate(dup);stress[mode]={'permutation_max_error':max(differences),
            'duplicating_global_top1_five_times':summary(dup,labels,outputs[mode]),
            'unique_top1_changed':int((dup.argmax(1)!=outputs[mode].argmax(1)).sum()),
            'score_max_change':float(np.abs(dup-outputs[mode]).max())}
    npz(a.output/'scores.npz',**outputs)
    write(a.output/'summary.json',results);write(a.output/'stress.json',stress)
    write(a.output/'training_selection.json',read(a.runs/'selection.json'))
    write(a.output/'training_logs.json',read(a.runs/'training_logs.json'))
    write(a.output/'training_contract.json',runs)
    write(a.output/'gate.json',read(a.runs/'gate.json'))
    comparisons={}
    for left,right in [('set','independent'),('density','set'),('consistency','set'),('competition','density'),('competition','consistency'),('competition','independent')]:
        comparisons[left+'_vs_'+right]=summary(outputs[left],labels,outputs[right])
    write(a.output/'ablation_comparisons.json',comparisons)
    write(a.output/'query_mapping.json',{'sample_rows':list(range(len(base))),
        'original_query_indices':c['index']['query_indices'],'query_paths':c['index']['queries'],
        'note':'Corrected/regressed arrays in summaries are sample rows, not original query indices.'})
    complete(a.output);print({m:r['correct'] for m,r in results.items()});print('Download:',a.output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['check','preflight','train','report'])
    p.add_argument('--plan',type=Path,default=Path('doc/candidate_set_plan_v1'))
    p.add_argument('--train-cache',type=Path,default=Path('.cache/candidate_set_v1/train'))
    p.add_argument('--dev-cache',type=Path,default=Path('.cache/pitts_pair_admission_v1'))
    p.add_argument('--output',type=Path,default=Path('doc/candidate_set_pitts_train_v1'))
    p.add_argument('--runs',type=Path,default=Path('doc/candidate_set_pitts_train_v1'))
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();globals()[a.stage](a)


if __name__=='__main__':main()
