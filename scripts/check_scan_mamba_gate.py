"""Mechanical checks only: do not select hyperparameters from pilot outcomes."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.candidate_set_screen import read,write
from scripts.adaptive_pair_budget import verified


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('logs/scan_mamba_v2'))
    a=p.parse_args()
    checks={};contracts=[]
    for mode in ('boq','conv','mamba','mamba_consistent'):
        for stage,steps in (('smoke',4),('pilot',128)):
            run=a.root/f'{mode}_{stage}'
            verified(run)
            summary=read(run/'summary.json');c=read(run/'contract.json')
            checks[f'{mode}/{stage}']=bool(summary['optimizer_steps']==steps and summary['matched_optimizer_clock']
                and summary['nonzero_vpr_batches']>0 and summary['frozen_unchanged'] and summary['checkpoint_roundtrip']
                and summary['zero_start_error']<=2e-6 and c['policy']['revision']==2)
            if mode.startswith('mamba'):
                grad=read(run/'gradient_final.json')['batches']
                checks[f'{mode}/{stage}/ssm_active']=summary['ssm_gradient_seen']
                checks[f'{mode}/{stage}/consistency_gradient']=any(v['gradients']['mixer']['weighted_consistency_norm']>0 for v in grad)
                checks[f'{mode}/{stage}/bypass_active']=read(run/'holdout_final.json')['mixer_bypass_l2_mean']>0
            if stage=='pilot':contracts.append(c)
    for key in ('schedule','train_places','holdout_places','policy','ru_sha256','metadata','source'):
        checks['matched_'+key]=all(c[key]==contracts[0][key] for c in contracts[1:])
    write(a.root/'gate.json',dict(checks=checks,passed=all(checks.values()),
        scope='Implementation and matched protocol only. No accuracy gate, checkpoint choice, or tuned consistency weight.'))
    if not all(checks.values()):raise ValueError('Mechanical preflight failed: '+str(checks))
    print('MATCHED MECHANICAL GATE PASS; full training may start fresh RU',flush=True)


if __name__=='__main__':main()
