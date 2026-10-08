"""Mechanical matched gate; neither benchmark accuracy nor tuned weights."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.candidate_set_screen import read, write
from scripts.adaptive_pair_budget import verified
from src.models.protected_mamba import MODES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('logs/protected_mamba_v1'))
    a = p.parse_args()
    checks, contracts = {}, []
    for mode in MODES:
        for stage, steps in (('smoke', 4), ('pilot', 128)):
            run = a.root/f'{mode}_{stage}'
            verified(run)
            s, c = read(run/'summary.json'), read(run/'contract.json')
            h = read(run/'holdout_final.json')
            checks[f'{mode}/{stage}'] = bool(s['optimizer_steps'] == steps and s['matched_optimizer_clock']
                and s['nonzero_vpr_batches'] > 0 and s['frozen_ru_unchanged'] and s['checkpoint_roundtrip']
                and s['zero_start_error'] <= 2e-6 and h['bypass_matches_ru'] and 0 < h['delta_l2_mean']
                and h['delta_l2_max'] <= c['policy']['cap']+1e-6)
            if mode.startswith('mamba'):
                checks[f'{mode}/{stage}/ssm_active'] = s['ssm_gradient_seen']
            if stage == 'pilot':
                contracts.append(c)
                grads = read(run/'gradient_final.json')['batches']
                checks[f'{mode}/preservation_gradient'] = any(g['preservation_gradient_norm'] > 0 for g in grads)
    for key in ('schedule', 'train_places', 'holdout_places', 'policy', 'ru_sha256', 'metadata', 'source'):
        checks['matched_'+key] = all(c[key] == contracts[0][key] for c in contracts[1:])
    write(a.root/'gate.json', dict(checks=checks, passed=all(checks.values()),
        scope='Finite learning, frozen original RU and matched protocol only; NOT evidence of accuracy gain'))
    if not all(checks.values()):
        raise ValueError('Mechanical gate failed: '+str(checks))
    print('PROTECTED MAMBA MECHANICAL GATE PASS', flush=True)


if __name__ == '__main__':
    main()
