"""Verify matched contracts and report paired last/GSV-selected results."""
from pathlib import Path
from scripts.candidate_set_screen import read, write, complete
from scripts.adaptive_pair_budget import verified
from scripts.summarize_depth_query import paired
from src.models.competitive_query import MODES


def main():
    root = Path('logs/competitive_query')
    output = Path('doc/competitive_query_screen_v1')
    contract, results = None, {}
    for mode in MODES:
        run = root/(mode+'_screen_v1')
        verified(run)
        c = read(run/'contract.json')
        c.pop('mode')
        if c['smoke']: raise ValueError('Smoke is not formal training')
        if contract is not None and c != contract: raise ValueError('Unmatched contracts')
        contract = c
        s = read(run/'summary.json')
        if s['history'][-1]['epoch'] != 3: raise ValueError('Incomplete epochs')
        results[mode] = dict(summary=s, comparisons={})
    for mode in MODES:
        for stage, epoch in [('last', 3), ('gsv_selected', results[mode]['summary']['best_epoch'])]:
            report = {}
            for ds in ('msls-val', 'pitts30k-val'):
                initial = read(root/'temperature_screen_v1'/f'{ds}_epoch00.json')['correct']
                own = read(root/(mode+'_screen_v1')/f'{ds}_epoch00.json')['correct']
                if initial != own: raise ValueError('Frozen baseline outcomes differ')
                result = read(root/(mode+'_screen_v1')/f'{ds}_epoch{epoch:02d}.json')
                controls = {}
                for ref in MODES:
                    e = 3 if stage == 'last' else results[ref]['summary']['best_epoch']
                    controls[ref] = paired(read(root/(ref+'_screen_v1')/f'{ds}_epoch{e:02d}.json')['correct'], result['correct'])
                report[ds] = dict(metrics=result['metrics'], vs_frozen=paired(initial, result['correct']), vs_controls=controls)
            results[mode]['comparisons'][stage] = report
    output.mkdir(parents=True, exist_ok=False)
    write(output/'summary.json', dict(arms=results, policy=contract['policy'], verdict='REVIEW_PAIRED_RESULTS_NOT_AUTOMATIC_SUCCESS'))
    complete(output)
    print('Summary:', output)


if __name__ == '__main__': main()
