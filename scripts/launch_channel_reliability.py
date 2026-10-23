"""Explicit finite frozen-descriptor reliability experiment on physical GPU1, no automatic expansion."""
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
STATE=ROOT/'.cache/channel_reliability_v1_launcher'
FILES=('scripts/channel_reliability.py','scripts/launch_channel_reliability.py','tests/test_channel_reliability.py','doc/CHANNEL_RELIABILITY_V1.md')


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(path,value):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def worker(shared):
    with (STATE/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base=[sys.executable,'-u','scripts/channel_reliability.py'];extra=['--allow-shared-gpu'] if shared else []
        stages=[('cpu_tests',[sys.executable,'-m','pytest','-q','tests/test_channel_reliability.py','tests/test_channel_rank_fusion.py']),
            ('cache', [*base,'cache','--resume',*extra]),
            ('retrieval', [*base,'retrieval','--resume',*extra]),
            ('analyze', [*base,'analyze','--resume'])]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
            PYTHONUNBUFFERED='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        done=[]
        for stage,cmd in stages:
            status=dict(phase=stage,pid=os.getpid(),updated_at=now(),completed_stages=list(done),
                physical_gpu=1,allow_shared_gpu=shared,log=str(STATE/(stage+'.log')))
            write(STATE/'progress.json',status);print(now(),'START',stage,flush=True)
            with (STATE/(stage+'.log')).open('ab',buffering=0) as f:
                result=subprocess.run(cmd,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES='') if stage in ('cpu_tests','analyze') else env,
                    stdout=f,stderr=subprocess.STDOUT)
            if result.returncode:
                status.update(phase='failed',failed_stage=stage,returncode=result.returncode,updated_at=now())
                write(STATE/'progress.json',status);raise RuntimeError(stage+' failed; see '+status['log'])
            done.append(stage);print(now(),'DONE',stage,flush=True)
        write(STATE/'progress.json',dict(phase='complete',completed_stages=done,updated_at=now(),
            output='doc/channel_reliability_v1',allow_shared_gpu=shared))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-training',action='store_true');p.add_argument('--allow-shared-gpu',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    if not a.allow_training:p.error('Explicit training consent required')
    STATE.mkdir(parents=True,exist_ok=True)
    if a.worker:worker(a.allow_shared_gpu);return
    with (STATE/'.launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record=STATE/'launcher.json'
        inventory={n:hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in FILES}
        if record.exists():
            old=json.loads(record.read_text());proc=Path(f"/proc/{old['pid']}/cmdline")
            if proc.exists() and b'launch_channel_reliability.py' in proc.read_bytes():raise RuntimeError('Already running')
            if old['allow_shared_gpu']!=a.allow_shared_gpu or old['source_hashes']!=inventory:raise RuntimeError('Resource/source policy changed')
        if (STATE/'progress.json').exists() and json.loads((STATE/'progress.json').read_text())['phase']=='complete':print('Already complete');return
        cmd=[sys.executable,str(Path(__file__).resolve()),'--allow-training','--worker']
        if a.allow_shared_gpu:cmd.append('--allow-shared-gpu')
        with (STATE/'driver.log').open('ab',buffering=0) as f:
            child=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(pid=child.pid,started_at=now(),command=cmd,allow_shared_gpu=a.allow_shared_gpu,source_hashes=inventory,
            policy='Finite frozen training cache and CPU reliability admission. Physical GPU1 allocator20%; no jobs killed; no posthoc budget expansion.')
        write(record,value);print(json.dumps(value,indent=2))


if __name__=='__main__':main()
