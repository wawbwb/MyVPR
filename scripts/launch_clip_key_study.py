"""Explicit detached serial GPU1 launcher; no evaluation and no process kills."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'.cache/clip_key_launcher_v1'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(path,value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def worker():
    import fcntl
    import yaml
    with (STATE/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config = yaml.safe_load((ROOT/'config/clip_key_boq.yaml').read_text())
        env = dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1',
                   OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        stages = [('cpu_tests',[sys.executable,'-m','pytest','-q','tests/test_clip_key_boq.py'],None),
                  ('preflight',[sys.executable,'scripts/clip_key_study.py','preflight'],None),
                  ('cache',[sys.executable,'scripts/clip_key_study.py','cache'],ROOT/config['key_cache'])]
        for stage in ('smoke','train'):
            for arm in ('visual','aligned','shuffled'):
                output = ROOT/config['runs']/(('smoke_' if stage=='smoke' else '')+arm)
                stages.append((stage+'_'+arm,[sys.executable,'scripts/clip_key_study.py',stage,
                               '--arm',arm,'--allow-training'],output))
        done = []
        for name,command,output in stages:
            if output is not None and (output/'contract.json').exists():
                command = command+['--resume']
            log = STATE/(name+'.log')
            status = dict(phase=name,pid=os.getpid(),updated_at=now(),completed_stages=list(done),
                          command=command,log=str(log),physical_gpu=1,automatic_evaluation=False)
            write(STATE/'progress.json',status)
            print(now(),'START',name,flush=True)
            with log.open('ab',buffering=0) as stream:
                result = subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:
                status.update(phase='failed',failed_stage=name,returncode=result.returncode,updated_at=now())
                write(STATE/'progress.json',status)
                raise RuntimeError(f'{name} failed; see {log}')
            done.append(name)
            print(now(),'DONE',name,flush=True)
        write(STATE/'progress.json',dict(phase='complete',completed_stages=done,updated_at=now(),
              training_complete=True,evaluation_started=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-training',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = p.parse_args()
    if not args.allow_training:
        p.error('Explicit --allow-training is required')
    STATE.mkdir(parents=True,exist_ok=True)
    if args.worker:
        worker()
        return
    import fcntl
    with (STATE/'.launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record = STATE/'launcher.json'
        if record.exists():
            old = json.loads(record.read_text())
            process = Path('/proc')/str(old['pid'])/'cmdline'
            if process.exists() and b'launch_clip_key_study.py' in process.read_bytes():
                raise RuntimeError('Launcher already alive; do not start a duplicate')
        command = [sys.executable,str(Path(__file__).resolve()),'--allow-training','--worker']
        with (STATE/'driver.log').open('ab',buffering=0) as log:
            child = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
        value = dict(pid=child.pid,started_at=now(),physical_gpu=1,command=command,
                     status=str(STATE/'progress.json'),driver_log=str(STATE/'driver.log'))
        write(record,value)
        print(json.dumps(value,indent=2))


if __name__ == '__main__':
    main()
