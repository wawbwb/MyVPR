"""Deferred single-DINO GPU work after pinned dual training AND eval complete.

The waiting process imports only standard-library/protocol code; no torch/CUDA.
No upstream edits, kills, environment installs, or concurrent training.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.single_dino_dependency import dependency_status

STATE=ROOT/'.cache/single_dino_interaction_queue_v1'


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(path,value):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def wait_dependency(config):
    while True:
        dependency=dependency_status(ROOT,config)
        status=dict(phase='waiting_for_dual',pid=os.getpid(),updated_at=now(),
            dependency=dependency,physical_gpu=1,gpu_initialized=False,real_training_started=False)
        if dependency['state']=='blocked':
            status['phase']='blocked_dependency';write(STATE/'progress.json',status)
            raise RuntimeError('Upstream dependency failed or changed; no GPU work started: '+json.dumps(dependency))
        if dependency['state']=='ready':
            # The existing experiment lock remains the independent interlock.
            # Also avoid competing with unrelated GPU1 users after dependency.
            q=subprocess.run(['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits'],
                check=True,text=True,stdout=subprocess.PIPE,timeout=20)
            if int(q.stdout.strip())<=4096:
                write(STATE/'dependency_ready.json',dict(verified_at=now(),dependency=dependency))
                return
            status['phase']='waiting_for_gpu_after_dual'
        write(STATE/'progress.json',status)
        time.sleep(30)


def worker():
    with (STATE/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=json.loads((ROOT/'config/single_dino_interaction.json').read_text())
        wait_dependency(config)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
            OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        base=[sys.executable,'-u','scripts/single_dino_interaction.py']
        stages=[('cpu_tests',[sys.executable,'-m','pytest','-q','tests/test_single_dino_interaction.py']),
            ('smoke',[*base,'smoke','--allow-training','--resume']),
            ('train',[*base,'train','--allow-training','--resume']),('eval',[*base,'eval','--resume'])]
        completed=[]
        for stage,command in stages:
            dependency=dependency_status(ROOT,config)
            if dependency['state']!='ready':raise RuntimeError('Dependency no longer ready')
            status=dict(phase=stage,pid=os.getpid(),updated_at=now(),completed_stages=list(completed),dependency=dependency)
            write(STATE/'progress.json',status);print(now(),'START',stage,flush=True)
            with (STATE/(stage+'.log')).open('ab',buffering=0) as stream:
                result=subprocess.run(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES='') if stage=='cpu_tests' else env,
                    stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:
                status.update(phase='failed',failed_stage=stage,returncode=result.returncode,updated_at=now())
                write(STATE/'progress.json',status);raise RuntimeError(stage+' failed; see '+str(STATE/(stage+'.log')))
            completed.append(stage);print(now(),'DONE',stage,flush=True)
        write(STATE/'progress.json',dict(phase='complete',completed_stages=completed,updated_at=now(),
            output='logs/single_dino_interaction_v1/eval'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-queued-training',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if not a.allow_queued_training:p.error('Explicit --allow-queued-training required')
    STATE.mkdir(parents=True,exist_ok=True)
    if a.worker:worker();return
    with (STATE/'.launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record=STATE/'queue.json'
        if record.exists():
            old=json.loads(record.read_text());proc=Path(f"/proc/{old['pid']}/cmdline")
            if proc.exists() and b'queue_single_dino_interaction.py' in proc.read_bytes():raise RuntimeError('Already queued/running')
        if (STATE/'progress.json').exists() and json.loads((STATE/'progress.json').read_text())['phase']=='complete':
            print('Already complete');return
        command=[sys.executable,str(Path(__file__).resolve()),'--allow-queued-training','--worker']
        with (STATE/'driver.log').open('ab',buffering=0) as stream:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(pid=child.pid,queued_at=now(),command=command,
            policy='No real-image smoke/train/eval until pinned dual train+eval seals validate and upstream launcher exits')
        write(record,value);print(json.dumps(value,indent=2))


if __name__=='__main__':main()
