"""Paired development-only PMD report; never claims confirmed improvement."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.candidate_set_screen import read,write,complete
from scripts.adaptive_pair_budget import verified


def paired(reference,variant):
    if [x['query'] for x in reference]!=[x['query'] for x in variant]:raise ValueError('Different query order')
    fixes=[b['query'] for a,b in zip(reference,variant) if not a['correct'] and b['correct']]
    losses=[b['query'] for a,b in zip(reference,variant) if a['correct'] and not b['correct']]
    return dict(corrections=fixes,regressions=losses,net=len(fixes)-len(losses))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('logs/pmd'))
    p.add_argument('--output',type=Path,default=Path('doc/pmd_screen_v1'))
    a=p.parse_args();runs={};contract=None
    for mode in ['baseline','ordinary','forced','partial']:
        root=a.root/(mode+'_screen_v1');verified(root)
        c=read(root/'contract.json');s=read(root/'summary.json')
        if c.pop('mode')!=mode or c['smoke'] or s['smoke']:raise ValueError('Wrong mode or smoke run')
        if contract is not None and c!=contract:raise ValueError('Unmatched contracts')
        contract=c
        last=s['history'][-1]['epoch']
        if last!=c['policy']['epochs']:raise ValueError('Training incomplete')
        runs[mode]=dict(summary=s,best=read(root/f"validation_epoch{s['best_epoch']:02d}.json"),
                        last=read(root/f'validation_epoch{last:02d}.json'),initial=read(root/'validation_epoch00.json'))
    report=dict(scope='Single-seed GSV development screen; no independent confirmation',arms={})
    for mode,r in runs.items():
        if paired(runs['baseline']['initial'],r['initial'])['corrections'] or paired(runs['baseline']['initial'],r['initial'])['regressions']:
            raise ValueError('Initial outcomes differ')
        report['arms'][mode]=dict(best_epoch=r['summary']['best_epoch'],trainable_parameters=r['summary']['trainable_parameters'])
        for stage in ['best','last']:
            report['arms'][mode][stage]=dict(correct=sum(x['correct'] for x in r[stage]),queries=len(r[stage]),
                vs_frozen=paired(r['initial'],r[stage]),vs_continued_baseline=paired(runs['baseline'][stage],r[stage]))
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'summary.json',report);complete(a.output);print(report)


if __name__=='__main__':main()
