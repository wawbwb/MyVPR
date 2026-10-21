"""Explicitly authorized V2 pilot; detached finite pipeline, no recurring job."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'.cache/single_dino_supervision_v2_launcher'


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(path,value):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def commands(python,shared):
    base=[python,'-u','scripts/single_dino_supervision.py']
    extra=['--allow-shared-gpu'] if shared else []
    return [('cpu_tests',[python,'-m','pytest','-q','tests/test_single_dino_supervision.py',
        'tests/test_single_dino_diagnosis.py','tests/test_single_dino_interaction.py','tests/test_hetero_interleave.py']),
        ('plan',[*base,'plan']),('smoke',[*base,'smoke','--allow-training','--resume',*extra]),
        ('train',[*base,'train','--allow-training','--resume',*extra]),('eval',[*base,'eval','--resume',*extra])]


def worker(shared):
    with (STATE/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        completed=[]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
            OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        for stage,cmd in commands(sys.executable,shared):
            status=dict(phase=stage,pid=os.getpid(),updated_at=now(),completed_stages=list(completed),
                physical_gpu=1,allow_shared_gpu=shared,log=str(STATE/(stage+'.log')))
            write(STATE/'progress.json',status);print(now(),'START',stage,flush=True)
            with (STATE/(stage+'.log')).open('ab',buffering=0) as f:
                result=subprocess.run(cmd,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES='') if stage in ('cpu_tests','plan') else env,
                    stdout=f,stderr=subprocess.STDOUT)
            if result.returncode:
                status.update(phase='failed',failed_stage=stage,returncode=result.returncode,updated_at=now())
                write(STATE/'progress.json',status);raise RuntimeError(stage+' failed; see '+status['log'])
            completed.append(stage);print(now(),'DONE',stage,flush=True)
        write(STATE/'progress.json',dict(phase='complete',completed_stages=completed,updated_at=now(),
            output='logs/single_dino_supervision_v2/eval',allow_shared_gpu=shared))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-training',action='store_true');p.add_argument('--allow-shared-gpu',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if not a.allow_training:p.error('Explicit --allow-training required')
    STATE.mkdir(parents=True,exist_ok=True)
    if a.worker:worker(a.allow_shared_gpu);return
    with (STATE/'.launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record=STATE/'launcher.json'
        if record.exists():
            old=json.loads(record.read_text());proc=Path(f"/proc/{old['pid']}/cmdline")
            if proc.exists() and b'launch_single_dino_supervision.py' in proc.read_bytes():raise RuntimeError('Already running')
            if old.get('allow_shared_gpu')!=a.allow_shared_gpu:raise RuntimeError('Original resource policy differs')
        if (STATE/'progress.json').exists() and json.loads((STATE/'progress.json').read_text())['phase']=='complete':
            print('Already complete');return
        inventory={f:hashlib.sha256((ROOT/f).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for f in
            ('src/models/single_dino_supervision.py','src/single_dino_grad_balance.py','scripts/single_dino_supervision.py',
             'scripts/launch_single_dino_supervision.py','config/single_dino_supervision.json','tests/test_single_dino_supervision.py')}
        if record.exists() and old.get('source_hashes')!=inventory:raise RuntimeError('Launcher source changed since first launch')
        cmd=[sys.executable,str(Path(__file__).resolve()),'--allow-training','--worker']
        if a.allow_shared_gpu:cmd.append('--allow-shared-gpu')
        with (STATE/'driver.log').open('ab',buffering=0) as f:
            child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(pid=child.pid,started_at=now(),command=cmd,allow_shared_gpu=a.allow_shared_gpu,
            source_hashes=inventory,policy='Finite CPU tests -> plan -> smoke -> fresh train -> eval. No jobs killed.')
        write(record,value);print(json.dumps(value,indent=2))


if __name__=='__main__':main()
