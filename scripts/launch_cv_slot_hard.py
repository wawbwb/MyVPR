"""Explicit GPU1 chain: CPU tests -> hard teacher -> gated smoke/train.

Any failed check stops. Never patch a report, change a threshold, or override
a FAIL. This launcher does NOT start benchmark evaluation.
"""
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
STATE = ROOT/'.cache/cv_slot_hard_launcher_v2'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write(path, value):
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def worker():
    with (STATE/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
            PYTHONUNBUFFERED='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        output = ROOT/'logs/cv_slot_teacher_hard_v2'
        config = 'config/cv_slot_boq_hard.yaml'
        stages = [('cpu_tests', [sys.executable, '-m', 'pytest', '-q',
            'tests/test_cv_slot_teacher_hard.py', 'tests/test_cv_slot_boq.py', 'tests/test_cv_slot_teacher_audit.py']),
            ('hard_teacher', [sys.executable, '-u', 'scripts/cv_slot_teacher_hard.py', '--resume']),
            ('smoke', [sys.executable, '-u', 'scripts/cv_slot_study.py', 'smoke', '--config', config, '--allow-training', '--resume']),
            ('train', [sys.executable, '-u', 'scripts/cv_slot_study.py', 'train', '--config', config, '--allow-training', '--resume'])]
        completed = []
        for name, command in stages:
            status = dict(phase=name, pid=os.getpid(), updated_at=now(), completed_stages=list(completed),
                          log=str(STATE/(name+'.log')), physical_gpu=1)
            write(STATE/'progress.json', status)
            stage_env = dict(env, CUDA_VISIBLE_DEVICES='') if name == 'cpu_tests' else env
            print(now(), 'START', name, flush=True)
            with (STATE/(name+'.log')).open('ab', buffering=0) as stream:
                result = subprocess.run(command, cwd=ROOT, env=stage_env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                status.update(phase='failed', failed_stage=name, returncode=result.returncode, updated_at=now())
                write(STATE/'progress.json', status)
                raise RuntimeError(name+' failed; inspect '+status['log'])
            completed.append(name)
            if name == 'hard_teacher':
                # The student checks the full sealed hashes again, independently.
                summary = json.loads((output/'summary.json').read_text())
                if summary['verdict'] != 'PASS':
                    write(STATE/'progress.json', dict(phase='stopped_teacher_not_supported',
                        verdict=summary['verdict'], completed_stages=completed, updated_at=now(),
                        student_updates=0, summary=str(output/'summary.json')))
                    print(now(), 'STOP; no student updates', flush=True)
                    return
            print(now(), 'DONE', name, flush=True)
        write(STATE/'progress.json', dict(phase='trained_not_evaluated', updated_at=now(),
            completed_stages=completed, physical_gpu=1, output='logs/cv_slot_boq_hard_v2'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allow-training', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if not a.allow_training:
        p.error('Explicit --allow-training required for the conditional chain')
    STATE.mkdir(parents=True, exist_ok=True)
    if a.worker:
        worker()
        return
    with (STATE/'.launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = STATE/'launcher.json'
        if record.exists():
            old = json.loads(record.read_text())
            cmdline = Path(f"/proc/{old['pid']}/cmdline")
            if cmdline.exists() and b'launch_cv_slot_hard.py' in cmdline.read_bytes():
                raise RuntimeError('Conditional experiment already running')
        command = [sys.executable, str(Path(__file__).resolve()), '--allow-training', '--worker']
        with (STATE/'driver.log').open('ab', buffering=0) as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        record_value = dict(pid=child.pid, started_at=now(), command=command, physical_gpu=1)
        write(record, record_value)
        print(json.dumps(record_value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
