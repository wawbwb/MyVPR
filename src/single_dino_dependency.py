"""Standard-library-only, fail-closed dependency verification. No CUDA imports."""
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify_seal(directory):
    root=Path(directory).resolve();d=read(root/'completed.json')
    if d.get('complete') is not True or d.get('schema')!='hetero_interleave_v1':
        raise ValueError('Upstream artifact incomplete or wrong schema')
    for name,digest in d['files'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or not p.is_file() or sha(p)!=digest:
            raise ValueError('Upstream artifact changed: '+str(p))
    return read(root/'contract.json')


def dependency_status(root,config,check_process=True):
    """Ready ONLY after successful training AND evaluation, regardless of gain."""
    root=Path(root)
    run=root/config['wait_for_run'];launcher=root/config['wait_for_launcher']
    contract=run/'train/contract.json'
    if not contract.is_file() or sha(contract)!=config['wait_for_train_contract_sha256']:
        return dict(state='blocked',reason='UPSTREAM_TRAIN_CONTRACT_MISSING_OR_CHANGED')
    progress=launcher/'progress.json'
    if not progress.is_file():return dict(state='blocked',reason='UPSTREAM_LAUNCHER_STATUS_MISSING')
    d=read(progress)
    if d.get('phase')=='failed':return dict(state='blocked',reason='UPSTREAM_FAILED',detail=d)
    if d.get('phase')!='complete':
        return dict(state='waiting',reason='UPSTREAM_NOT_COMPLETE',phase=d.get('phase'))
    if d.get('completed_stages')!=['cpu_tests','plan','smoke','train','eval']:
        return dict(state='blocked',reason='UPSTREAM_STAGE_INVENTORY_MISMATCH')
    if check_process and (launcher/'launcher.json').is_file():
        pid=read(launcher/'launcher.json')['pid'];proc=Path(f'/proc/{pid}/cmdline')
        try:
            if proc.exists() and b'launch_hetero_interleave.py' in proc.read_bytes():
                return dict(state='waiting',reason='UPSTREAM_LAUNCHER_EXITING')
        except FileNotFoundError:pass
    try:
        train=verify_seal(run/'train');evaluation=verify_seal(run/'eval')
        ts,es=read(run/'train/summary.json'),read(run/'eval/summary.json')
        if (train['stage']!='train' or evaluation['stage']!='eval' or ts['steps']!=1024
                or not ts['checks'] or not es['checks']
                or any(v is not True for v in ts['checks'].values()) or any(v is not True for v in es['checks'].values())
                or evaluation['checkpoint_sha256']!=sha(run/'train/last.pt')):
            raise ValueError('Upstream mechanics/checkpoint linkage mismatch')
    except (ValueError,KeyError,FileNotFoundError) as error:
        return dict(state='blocked',reason='UPSTREAM_COMPLETION_INVALID',error=str(error))
    return dict(state='ready',reason='UPSTREAM_TRAIN_AND_EVAL_VERIFIED',
        train_completed_sha256=sha(run/'train/completed.json'),
        eval_completed_sha256=sha(run/'eval/completed.json'),
        upstream_train_contract_sha256=config['wait_for_train_contract_sha256'])


def require_ready(root,config):
    status=dependency_status(root,config)
    if status['state']!='ready':
        raise RuntimeError('Single-DINO GPU work is forbidden until dependency ready: '+json.dumps(status))
    return status
