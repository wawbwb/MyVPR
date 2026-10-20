"""Launch only the frozen-teacher audit on physical GPU1, never VPR training."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.semantic_region_protocol import read, write


def main():
    import fcntl
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'config/cv_slot_teacher_audit.yaml')
    a = p.parse_args()
    target = ROOT/'logs/cv_slot_teacher_audit_launcher_v1'
    target.mkdir(parents=True, exist_ok=True)
    with (target/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = target/'process.json'
        if state.exists():
            old = read(state)
            command = Path(f"/proc/{old['pid']}/cmdline")
            if command.exists() and b'cv_slot_teacher_audit.py' in command.read_bytes():
                print('Audit already running:', old, flush=True)
                return
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        log = target/f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
        with log.open('ab') as stream:
            child = subprocess.Popen([sys.executable, '-u', str(ROOT/'scripts/cv_slot_teacher_audit.py'),
                '--config', str(a.config.resolve()), '--resume'], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        value = dict(pid=child.pid, log=str(log), started=time.strftime('%Y-%m-%d %H:%M:%S'),
                     scope='Frozen-teacher audit only; no automatic student training')
        write(state, value)
        print(value, flush=True)


if __name__ == '__main__':
    main()
