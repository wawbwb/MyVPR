"""Matched initialization sensitivity; never pool repeated-query trials as iid."""
from pathlib import Path
import numpy as np
from scripts.candidate_set_screen import read, write, sha, complete
from scripts.adaptive_pair_budget import verified
from scripts.summarize_depth_query import paired

MODES = ('local_contrast', 'shuffled_contrast')


def comparable(c):
    c = dict(c)
    c.pop('mode')
    c['policy'] = dict(c['policy'])
    c['policy'].pop('initialization_seed', None)
    c['code'] = {k:v for k,v in c['code'].items() if k not in ('scripts/train_local_value.py', 'scripts/train_local_value_repeat.py')}
    return c


def main():
    output = Path('doc/local_value_repeat_v1')
    if output.exists():
        verified(output); print('Already complete'); return
    contracts, summaries, paths, provenance = {}, {}, {}, {}
    reference = None
    for seed in (42, 43, 44):
        for mode in MODES:
            key = f'{mode}_seed{seed}'
            p = Path('logs/local_value')/(mode+'_screen_v1') if seed == 42 else Path('logs/local_value_repeat')/(key+'_screen_v1')
            verified(p)
            c, s = read(p/'contract.json'), read(p/'summary.json')
            if c['smoke'] or c['mode'] != mode or s['history'][-1]['epoch'] != 3:
                raise ValueError('Incomplete/mismatched formal run')
            if seed != 42 and c['policy'].get('initialization_seed') != seed:
                raise ValueError('Wrong initialization seed')
            common = comparable(c)
            if reference is not None and common != reference: raise ValueError('Training conditions changed')
            reference = common
            paths[key], summaries[key], contracts[key] = p, s, c
            provenance[key] = dict(complete=sha(p/'completed.json'), contract=sha(p/'contract.json'), trainer_code=c['code'])
    report, stability = {}, {}
    for stage in ('gsv_selected', 'last'):
        report[stage], stability[stage] = {}, {}
        for ds in ('msls-val', 'pitts30k-val'):
            baseline = read(paths['local_contrast_seed42']/f'{ds}_epoch00.json')['correct']
            entries = []
            for seed in (42,43,44):
                rows = {}
                for mode in MODES:
                    key = f'{mode}_seed{seed}'
                    if read(paths[key]/f'{ds}_epoch00.json')['correct'] != baseline: raise ValueError('Baseline outcomes differ')
                    epoch = 3 if stage == 'last' else summaries[key]['best_epoch']
                    r = read(paths[key]/f'{ds}_epoch{epoch:02d}.json')
                    rows[mode] = dict(epoch=epoch, result=r, vs_ru=paired(baseline, r['correct']))
                control = paired(rows['shuffled_contrast']['result']['correct'], rows['local_contrast']['result']['correct'])
                entries.append(dict(seed=seed, arms={m:dict(epoch=r['epoch'], metrics=r['result']['metrics'], vs_ru=r['vs_ru']) for m,r in rows.items()}, aligned_vs_shuffled=control))
            report[stage][ds] = entries
            values = [r['arms']['local_contrast']['vs_ru']['net'] for r in entries]
            advantage = [r['aligned_vs_shuffled']['net'] for r in entries]
            stability[stage][ds] = dict(aligned_net_vs_ru=values, aligned_net_vs_shuffled=advantage,
                mean_net_vs_ru=float(np.mean(values)), range_net_vs_ru=[min(values),max(values)],
                mean_net_vs_shuffled=float(np.mean(advantage)), wins_vs_ru=sum(v>0 for v in values),
                wins_vs_shuffled=sum(v>0 for v in advantage))
    output.mkdir(parents=True)
    write(output/'summary.json', dict(results=report, stability=stability, sources=provenance,
        scope='Initialization sensitivity only: same training subset, same batch schedules, augmentations and fixed shuffle permutation. Original seed42 was exploratory; added seeds43/44 are confirmation. Repeated queries are not independent samples. No automatic stable-benefit claim.'))
    complete(output)
    print('COMPLETE', output)


if __name__ == '__main__': main()
