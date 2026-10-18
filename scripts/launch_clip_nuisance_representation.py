"""Detached GPU1 frozen extraction -> CPU probes, NEVER resumes VPR training."""
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
STATE=ROOT/'.cache/clip_nuisance_representation_launcher_v1'


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(p,value):
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)


def worker():
    import yaml
    with (STATE/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c=yaml.safe_load((ROOT/'config/clip_nuisance_representation.yaml').read_text())
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        script=[sys.executable,'scripts/clip_nuisance_representation.py']
        stages=[('cpu_tests',[sys.executable,'-m','pytest','-q','tests/test_clip_nuisance_representation.py'],None),
                ('extract',script+['extract'],ROOT/c['cache']),
                ('probe',script+['probe','--allow-probe-fit'],ROOT/c['output'])]
        done=[]
        for name,command,output in stages:
            if output is not None and (output/'contract.json').exists():command=command+['--resume']
            log=STATE/(name+'.log')
            progress=dict(phase=name,pid=os.getpid(),updated_at=now(),completed_stages=list(done),command=command,
                          log=str(log),physical_gpu=1,no_vpr_training=True,blind_test_used=False)
            write(STATE/'progress.json',progress);print(now(),'START',name,flush=True)
            with log.open('ab',buffering=0) as stream:
                result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:
                progress.update(phase='failed',failed_stage=name,returncode=result.returncode,updated_at=now())
                write(STATE/'progress.json',progress);raise RuntimeError(f'{name} failed; see {log}')
            done.append(name);print(now(),'DONE',name,flush=True)
        write(STATE/'progress.json',dict(phase='complete',updated_at=now(),completed_stages=done,
            summary=str(ROOT/c['output']/'summary.json'),no_vpr_training=True,blind_test_used=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-diagnostic',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if not a.run_diagnostic:p.error('Explicit --run-diagnostic required')
    STATE.mkdir(parents=True,exist_ok=True)
    if a.worker:worker();return
    with (STATE/'.launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record=STATE/'launcher.json'
        if record.exists():
            previous=json.loads(record.read_text());proc=Path('/proc')/str(previous['pid'])/'cmdline'
            if proc.exists() and b'launch_clip_nuisance_representation.py' in proc.read_bytes():raise RuntimeError('Diagnostic already running')
        command=[sys.executable,str(Path(__file__).resolve()),'--run-diagnostic','--worker']
        with (STATE/'driver.log').open('ab',buffering=0) as log:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(pid=child.pid,started_at=now(),physical_gpu=1,command=command,status=str(STATE/'progress.json'),no_vpr_training=True)
        write(record,value);print(json.dumps(value,indent=2))


if __name__=='__main__':main()
