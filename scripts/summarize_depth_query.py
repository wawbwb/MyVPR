"""Matched DSQ development screen. Preserve last and MSLS-selected outcomes."""
from pathlib import Path
from scripts.candidate_set_screen import read,write,complete
from scripts.adaptive_pair_budget import verified
from src.models.depth_query import MODES


def paired(reference,variant):
    if len(reference)!=len(variant):raise ValueError('Query counts differ')
    fixes=[i for i,(a,b) in enumerate(zip(reference,variant)) if not a and b]
    losses=[i for i,(a,b) in enumerate(zip(reference,variant)) if a and not b]
    return dict(corrections=fixes,regressions=losses,net=len(fixes)-len(losses))


def main():
    root=Path('logs/depth_query');output=Path('doc/depth_query_screen_v1')
    summaries={};contract=None
    for mode in MODES:
        run=root/(mode+'_screen_v1');verified(run)
        c=read(run/'contract.json');c.pop('mode')
        if c['smoke']:raise ValueError('Smoke is not an experiment')
        if contract is not None and c!=contract:raise ValueError('Unmatched arm contracts')
        contract=c;summaries[mode]=read(run/'summary.json')
        if summaries[mode]['history'][-1]['epoch']!=3:raise ValueError('Incomplete training')
    report=dict(scope=contract['policy']['scope'],arms={})
    for mode in MODES:
        s=summaries[mode];entry=dict(best_msls_epoch=s['best_epoch'],trainable_parameters=s['trainable_parameters'],history=s['history'],comparisons={})
        for ds in ('msls-val','pitts30k-val'):
            initial=read(root/'baseline_screen_v1'/f'{ds}_epoch00.json')['correct']
            own_initial=read(root/(mode+'_screen_v1')/f'{ds}_epoch00.json')['correct']
            if initial!=own_initial:raise ValueError('Initial query outcomes differ')
            entry['comparisons'][ds]={}
            for stage,ep in [('last',3),('msls_selected',s['best_epoch'])]:
                result=read(root/(mode+'_screen_v1')/f'{ds}_epoch{ep:02d}.json')['correct']
                baseline_epoch=3 if stage=='last' else summaries['baseline']['best_epoch']
                ref=read(root/'baseline_screen_v1'/f'{ds}_epoch{baseline_epoch:02d}.json')['correct']
                entry['comparisons'][ds][stage]=dict(correct=sum(result),queries=len(result),vs_frozen=paired(initial,result),vs_baseline=paired(ref,result))
        report['arms'][mode]=entry
    output.mkdir(parents=True,exist_ok=False);write(output/'summary.json',report);complete(output);print(report)


if __name__=='__main__':main()
