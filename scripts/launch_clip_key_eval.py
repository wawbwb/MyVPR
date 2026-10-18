"""Explicit detached GPU1 evaluator; no training, no stopping other jobs."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'.cache/clip_key_eval_launcher_v1'
OUTPUT = ROOT/'logs/clip_key_boq_eval_v1'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(p,value):
    temp = p.with_name(p.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(p)


def worker():
    with (STATE/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env = dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        steps = [('cpu_tests',[sys.executable,'-m','pytest','-q','tests/test_clip_key_eval.py']),
                 ('evaluation',[sys.executable,'scripts/eval_clip_key_study.py','--evaluate'])]
        done=[]
        for name,command in steps:
            if name=='evaluation' and (OUTPUT/'contract.json').exists():
                command=command+['--resume']
            log=STATE/(name+'.log')
            status=dict(phase=name,pid=os.getpid(),updated_at=now(),completed_stages=list(done),
                command=command,log=str(log),physical_gpu=1,training_started=False,blind_test_used=False)
            write(STATE/'progress.json',status)
            print(now(),'START',name,flush=True)
            with log.open('ab',buffering=0) as stream:
                result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:
                status.update(phase='failed',failed_stage=name,returncode=result.returncode,updated_at=now())
                write(STATE/'progress.json',status)
                raise RuntimeError(f'{name} failed; see {log}')
            done.append(name)
            print(now(),'DONE',name,flush=True)
        write(STATE/'progress.json',dict(phase='complete',pid=os.getpid(),updated_at=now(),
            completed_stages=done,summary=str(OUTPUT/'summary.json'),training_started=False,blind_test_used=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluate',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if not args.evaluate:p.error('Explicit --evaluate required')
    STATE.mkdir(parents=True,exist_ok=True)
    if args.worker:
        worker();return
    with (STATE/'.launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior=STATE/'launcher.json'
        if prior.exists():
            old=json.loads(prior.read_text())
            process=Path('/proc')/str(old['pid'])/'cmdline'
            if process.exists() and b'launch_clip_key_eval.py' in process.read_bytes():
                raise RuntimeError('Evaluator already alive; refusing duplicate')
        command=[sys.executable,str(Path(__file__).resolve()),'--evaluate','--worker']
        with (STATE/'driver.log').open('ab',buffering=0) as log:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
        record=dict(pid=child.pid,started_at=now(),physical_gpu=1,command=command,
            status=str(STATE/'progress.json'),log=str(STATE/'evaluation.log'),training_started=False,blind_test_used=False)
        write(prior,record)
        print(json.dumps(record,indent=2))


if __name__=='__main__':main()
