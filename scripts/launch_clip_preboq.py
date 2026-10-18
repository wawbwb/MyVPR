"""Explicit serial GPU1 launch: CPU tests, three smokes, three trainings, evaluation."""
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
STATE = ROOT/'.cache/clip_preboq_launcher_v1'


def now(): return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(p, value):
    temp = p.with_name(p.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8'); temp.replace(p)


def worker():
    with (STATE/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        script = [sys.executable, 'scripts/clip_preboq_study.py']
        runs = ROOT/'logs/clip_preboq_v1'
        stages = [('cpu_tests', [sys.executable, '-m', 'pytest', '-q', 'tests/test_clip_preboq.py'], None)]
        for stage in ('smoke', 'train'):
            for arm in ('visual', 'aligned', 'shuffled'):
                stages.append((stage+'_'+arm, script+[stage, '--arm', arm, '--allow-training'],
                               runs/(('smoke_' if stage == 'smoke' else '')+arm)))
        stages.append(('evaluation', script+['eval', '--evaluate'], runs/'evaluation'))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        complete = []
        for name, command, output in stages:
            if output is not None and (output/'contract.json').exists(): command += ['--resume']
            log = STATE/(name+'.log')
            status = dict(phase=name, pid=os.getpid(), updated_at=now(), completed_stages=list(complete),
                          log=str(log), physical_gpu=1, blind_test_used=False)
            write(STATE/'progress.json', status); print(now(), 'START', name, flush=True)
            with log.open('ab', buffering=0) as stream:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                status.update(phase='failed', failed_stage=name, returncode=result.returncode, updated_at=now())
                write(STATE/'progress.json', status); raise RuntimeError(name+' failed; see '+str(log))
            complete.append(name); print(now(), 'DONE', name, flush=True)
        write(STATE/'progress.json', dict(phase='complete', completed_stages=complete, updated_at=now(),
            summary=str(runs/'evaluation/summary.json'), physical_gpu=1, blind_test_used=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-training', action='store_true'); p.add_argument('--evaluate', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if not a.allow_training or not a.evaluate: p.error('Need explicit --allow-training AND --evaluate')
    STATE.mkdir(parents=True, exist_ok=True)
    if a.worker: worker(); return
    with (STATE/'.launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = STATE/'launcher.json'
        if record.exists():
            previous = json.loads(record.read_text()); proc = Path('/proc')/str(previous['pid'])/'cmdline'
            if proc.exists() and b'launch_clip_preboq.py' in proc.read_bytes(): raise RuntimeError('Already alive')
        command = [sys.executable, str(Path(__file__).resolve()), '--allow-training', '--evaluate', '--worker']
        with (STATE/'driver.log').open('ab', buffering=0) as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        value = dict(pid=child.pid, started_at=now(), command=command, physical_gpu=1, progress=str(STATE/'progress.json'))
        write(record, value); print(json.dumps(value, indent=2))


if __name__ == '__main__': main()
