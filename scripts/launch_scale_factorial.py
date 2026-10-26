"""Finite six-cell official/F1 scale and gate ablation; all results retained."""
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
ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'.cache/scale_factorial_v1_launcher'
FILES = ('scripts/scale_factorial.py', 'scripts/launch_scale_factorial.py',
    'src/scale_factorial_statistics.py', 'config/scale_factorial.json',
    'tests/test_scale_factorial.py', 'doc/SCALE_FACTORIAL_V1.md')


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(path, value):
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def worker(shared):
    with (STATE/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base = [sys.executable, '-u', 'scripts/scale_factorial.py']
        extra = ['--allow-shared-gpu'] if shared else []
        stages = [('cpu_tests', [sys.executable, '-m', 'pytest', '-q', 'tests/test_scale_factorial.py', 'tests/test_scale_agreement_fusion.py']),
            ('preflight', [*base, 'preflight', '--resume']),
            ('probe', [*base, 'probe', '--resume', *extra]),
            ('cache_train', [*base, 'cache_train', '--resume', *extra]),
            ('smoke', [*base, 'smoke', '--allow-training', '--resume', *extra]),
            ('train', [*base, 'train', '--allow-training', '--resume', *extra]),
            ('cache_eval', [*base, 'cache_eval', '--resume', *extra]),
            ('eval', [*base, 'eval', '--resume', *extra]),
            ('aggregate', [*base, 'aggregate', '--resume'])]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
            PYTHONUNBUFFERED='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        done = []
        for stage, cmd in stages:
            state = dict(phase=stage, pid=os.getpid(), completed_stages=list(done), updated_at=now(), log=str(STATE/(stage+'.log')))
            write(STATE/'progress.json', state)
            print(now(), 'START', stage, flush=True)
            with (STATE/(stage+'.log')).open('ab', buffering=0) as f:
                result = subprocess.run(cmd, cwd=ROOT,
                    env=dict(env, CUDA_VISIBLE_DEVICES='') if stage in ('cpu_tests', 'preflight', 'aggregate') else env,
                    stdout=f, stderr=subprocess.STDOUT)
            if result.returncode:
                state.update(phase='failed', failed_stage=stage, returncode=result.returncode, updated_at=now())
                write(STATE/'progress.json', state)
                raise RuntimeError(stage+' failed; see '+state['log'])
            done.append(stage)
            print(now(), 'DONE', stage, flush=True)
        write(STATE/'progress.json', dict(phase='complete', completed_stages=done, updated_at=now(), output='logs/scale_factorial_v1/aggregate'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-training', action='store_true')
    p.add_argument('--allow-shared-gpu', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if not a.allow_training:
        p.error('Explicit training consent required')
    STATE.mkdir(parents=True, exist_ok=True)
    if a.worker:
        worker(a.allow_shared_gpu)
        return
    with (STATE/'.launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = STATE/'launcher.json'
        inventory = {n: hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for n in FILES}
        if record.exists():
            old = json.loads(record.read_text())
            proc = Path(f"/proc/{old['pid']}/cmdline")
            if proc.exists() and b'launch_scale_factorial.py' in proc.read_bytes():
                raise RuntimeError('Already running')
            if old['source_hashes'] != inventory or old['allow_shared_gpu'] != a.allow_shared_gpu:
                raise RuntimeError('Source/resource policy changed')
        if (STATE/'progress.json').exists() and json.loads((STATE/'progress.json').read_text())['phase'] == 'complete':
            print('Already complete')
            return
        cmd = [sys.executable, str(Path(__file__).resolve()), '--allow-training', '--worker']
        if a.allow_shared_gpu:
            cmd.append('--allow-shared-gpu')
        with (STATE/'driver.log').open('ab', buffering=0) as f:
            child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        value = dict(pid=child.pid, started_at=now(), source_hashes=inventory, allow_shared_gpu=a.allow_shared_gpu,
            command=cmd, policy='Paired43,44,45; official two-scale encoding; four128parameter heads1024 steps; F1 row reused; no search or expansion.')
        write(record, value)
        print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
