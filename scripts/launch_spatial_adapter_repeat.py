"""Explicit detached GPU1 paired-seed experiment; no process kills or GPU0 fallback."""
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
STATE = ROOT/'.cache/spatial_adapter_repeat_launcher_v1'


def now(): return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(p, value):
    temp = p.with_name(p.name+'.tmp'); temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8'); temp.replace(p)


def worker():
    with (STATE/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        script = [sys.executable, 'scripts/spatial_adapter_repeat.py']; runs = ROOT/'logs/spatial_adapter_repeat_v1'
        stages = [('cpu_tests', [sys.executable, '-m', 'pytest', '-q', 'tests/test_spatial_adapter_repeat.py'], None),
                  ('paired_smoke', script+['smoke', '--allow-training'], runs/'smoke_seed42')]
        for seed in (42, 43, 44):
            stages.append((f'train_seed{seed}', script+['train', '--seed', str(seed), '--allow-training'], runs/f'train_seed{seed}'))
            stages.append((f'eval_seed{seed}', script+['eval', '--seed', str(seed), '--evaluate'], runs/f'eval_seed{seed}'))
        stages.append(('aggregate', script+['aggregate', '--evaluate'], runs/'aggregate'))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        completed = []
        for name, command, output in stages:
            if output is not None and (output/'contract.json').exists(): command += ['--resume']
            log = STATE/(name+'.log')
            status = dict(phase=name, pid=os.getpid(), updated_at=now(), completed_stages=list(completed),
                          log=str(log), physical_gpu=1, no_clip=True, blind_test_used=False)
            write(STATE/'progress.json', status); print(now(), 'START', name, flush=True)
            with log.open('ab', buffering=0) as stream:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                status.update(phase='failed', failed_stage=name, returncode=result.returncode, updated_at=now())
                write(STATE/'progress.json', status); raise RuntimeError(name+' failed; see '+str(log))
            completed.append(name); print(now(), 'DONE', name, flush=True)
        write(STATE/'progress.json', dict(phase='complete', completed_stages=completed, updated_at=now(),
            summary=str(runs/'aggregate/summary.json'), physical_gpu=1, blind_test_used=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-training', action='store_true'); p.add_argument('--evaluate', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS); a = p.parse_args()
    if not a.allow_training or not a.evaluate: p.error('Explicit --allow-training AND --evaluate required')
    STATE.mkdir(parents=True, exist_ok=True)
    if a.worker: worker(); return
    with (STATE/'.launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = STATE/'launcher.json'
        if record.exists():
            previous = json.loads(record.read_text()); proc = Path('/proc')/str(previous['pid'])/'cmdline'
            if proc.exists() and b'launch_spatial_adapter_repeat.py' in proc.read_bytes(): raise RuntimeError('Experiment already alive')
        command = [sys.executable, str(Path(__file__).resolve()), '--allow-training', '--evaluate', '--worker']
        with (STATE/'driver.log').open('ab', buffering=0) as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        value = dict(pid=child.pid, started_at=now(), command=command, physical_gpu=1, progress=str(STATE/'progress.json'))
        write(record, value); print(json.dumps(value, indent=2))


if __name__ == '__main__': main()
