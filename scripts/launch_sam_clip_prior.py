"""GPU1-only detached frozen experiment: CPU tests -> smoke -> cache -> eval."""
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
STATE = ROOT/'.cache/sam_clip_prior_launcher_v1'
PROTECTED = ('src/models/conditional_complement.py', 'src/conditional_complement_protocol.py',
    'scripts/conditional_complement.py', 'src/models/hcen_ablation.py',
    'src/models/hcen_vpr.py', 'src/models/conv_residual_ablation.py', 'src/models/official_boq.py')
CHECKPOINT = 'logs/dinov2_vitb14/BoQ_semantic_region_repeatability_uniqueness_only/version_0/checkpoints/epoch(26)_step(42201)_R1[0.9122]_R5[0.9514].ckpt'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(path, value):
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def protected():
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in PROTECTED}


def worker():
    with (STATE/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
            OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        initial = protected()
        write(STATE/'protected_initial.json', initial)
        cache = '.cache/sam_clip_prior_v1'
        output = 'doc/sam_clip_prior_eval_v1'
        base = [sys.executable, '-u', 'scripts/sam_clip_prior.py']
        common = ['--checkpoint', CHECKPOINT, '--output', cache, '--resume']
        stages = [('cpu_tests', [sys.executable, '-m', 'pytest', '-q', 'tests/test_sam_clip_prior.py']),
            ('smoke', [*base, 'cache', *common, '--limit-images', '4']),
            ('cache', [*base, 'cache', *common]),
            ('eval', [*base, 'eval', '--cache', cache, '--output', output])]
        completed = []
        for name, command in stages:
            status = dict(phase=name, pid=os.getpid(), updated_at=now(), completed_stages=list(completed),
                log=str(STATE/(name+'.log')), physical_gpu=1, parameter_updates=0)
            write(STATE/'progress.json', status)
            print(now(), 'START', name, flush=True)
            stage_env = dict(env, CUDA_VISIBLE_DEVICES='') if name in ('cpu_tests', 'eval') else env
            with (STATE/(name+'.log')).open('ab', buffering=0) as stream:
                result = subprocess.run(command, cwd=ROOT, env=stage_env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode or protected() != initial:
                status.update(phase='failed', failed_stage=name, returncode=result.returncode,
                    protected_unchanged=protected() == initial, updated_at=now())
                write(STATE/'progress.json', status)
                raise RuntimeError(name+' failed; see '+status['log'])
            completed.append(name)
            print(now(), 'DONE', name, flush=True)
        write(STATE/'progress.json', dict(phase='complete', updated_at=now(), completed_stages=completed,
            physical_gpu=1, parameter_updates=0, protected_unchanged=True, output=output))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-experiment', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if not a.allow_experiment:
        p.error('Explicit --allow-experiment is required')
    STATE.mkdir(parents=True, exist_ok=True)
    if a.worker:
        worker()
        return
    with (STATE/'.launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (STATE/'progress.json').exists():
            status = json.loads((STATE/'progress.json').read_text())
            if status['phase'] == 'complete':
                print('Experiment already complete.'); return
        record = STATE/'launcher.json'
        if record.exists():
            old = json.loads(record.read_text())
            proc = Path(f"/proc/{old['pid']}/cmdline")
            if proc.exists() and b'launch_sam_clip_prior.py' in proc.read_bytes():
                raise RuntimeError('Experiment already running')
        query = subprocess.run(['nvidia-smi', '-i', '1', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            check=True, text=True, stdout=subprocess.PIPE, timeout=20)
        if int(query.stdout.strip()) > 4096:
            raise RuntimeError('GPU1 is busy (>4GiB used); not starting, no other tasks killed')
        command = [sys.executable, str(Path(__file__).resolve()), '--allow-experiment', '--worker']
        with (STATE/'driver.log').open('ab', buffering=0) as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        value = dict(pid=child.pid, started_at=now(), command=command, physical_gpu=1)
        write(record, value)
        print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
