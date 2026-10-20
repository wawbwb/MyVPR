"""Authorized GPU1 pilot: CPU tests -> file check -> smoke -> train -> eval."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'.cache/hetero_interleave_launcher_v1'


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(path,value):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def worker():
    with (STATE/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
            OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        base=[sys.executable,'-u','scripts/hetero_interleave_study.py']
        stages=[('cpu_tests',[sys.executable,'-m','pytest','-q','tests/test_hetero_interleave.py']),
            ('plan',[*base,'plan']),('smoke',[*base,'smoke','--allow-training','--resume']),
            ('train',[*base,'train','--allow-training','--resume']),('eval',[*base,'eval','--resume'])]
        completed=[]
        for stage,command in stages:
            status=dict(phase=stage,pid=os.getpid(),updated_at=now(),completed_stages=list(completed),
                physical_gpu=1,log=str(STATE/(stage+'.log')))
            write(STATE/'progress.json',status);print(now(),'START',stage,flush=True)
            with (STATE/(stage+'.log')).open('ab',buffering=0) as stream:
                result=subprocess.run(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES='') if stage in ('cpu_tests','plan') else env,
                    stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:
                status.update(phase='failed',failed_stage=stage,returncode=result.returncode,updated_at=now())
                write(STATE/'progress.json',status);raise RuntimeError(stage+' failed; see '+status['log'])
            completed.append(stage);print(now(),'DONE',stage,flush=True)
        write(STATE/'progress.json',dict(phase='complete',completed_stages=completed,updated_at=now(),output='logs/hetero_interleave_v1/eval'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-training',action='store_true');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if not a.allow_training:p.error('Explicit --allow-training required')
    STATE.mkdir(parents=True,exist_ok=True)
    if a.worker:worker();return
    with (STATE/'.launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record=STATE/'launcher.json'
        if record.exists():
            old=json.loads(record.read_text());proc=Path(f"/proc/{old['pid']}/cmdline")
            if proc.exists() and b'launch_hetero_interleave.py' in proc.read_bytes():raise RuntimeError('Pilot already running')
        if (STATE/'progress.json').exists() and json.loads((STATE/'progress.json').read_text())['phase']=='complete':
            print('Already complete');return
        gpu=subprocess.run(['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits'],
            check=True,text=True,stdout=subprocess.PIPE,timeout=20)
        if int(gpu.stdout.strip())>4096:raise RuntimeError('GPU1 busy; not starting, no tasks killed')
        command=[sys.executable,str(Path(__file__).resolve()),'--allow-training','--worker']
        with (STATE/'driver.log').open('ab',buffering=0) as stream:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(pid=child.pid,started_at=now(),command=command,physical_gpu=1)
        write(record,value);print(json.dumps(value,indent=2))


if __name__=='__main__':main()
