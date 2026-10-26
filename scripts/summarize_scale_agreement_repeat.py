"""Repair report export only; preserve sealed training/eval and failed aggregate."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts import scale_agreement_repeat as study
from src.scale_repeat_statistics import contrast, positive_components
from src.semantic_region_protocol import owned_run, read, write, sha


def json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    c = study.configuration()
    contract = dict(**study.stage_contract(c, 'aggregate_fixed'),
        original_aggregate_contract_sha256=sha(study.root(c)/'aggregate/contract.json'),
        report_sources=study.source.old.key.source_hashes(('scripts/summarize_scale_agreement_repeat.py', 'tests/test_scale_repeat_report.py')),
        repair='Convert NumPy scalar types at JSON boundary only; original statistics and numerical policy unchanged')
    with owned_run(study.root(c)/'aggregate_fixed', contract, args.resume) as out:
        if (out/'completed.json').exists():
            study.verified(out)
            return
        for seed in c['seeds']:
            study.verified(study.root(c)/f'seed{seed}/eval')
        results = {}
        for d in study.source.populations(study.source.configuration()):
            reports = {s: read(study.root(c)/f'seed{s}/eval'/(d['tag']+'.json'))['variants'] for s in c['seeds']}
            candidate = [reports[s]['group_gate']['outcomes'] for s in c['seeds']]
            clusters = positive_components(d['ground_truth'])
            comparisons = {a: contrast(candidate, [reports[s][a]['outcomes'] for s in c['seeds']], clusters,
                c['bootstrap_repetitions'], c['bootstrap_seed']) for a in study.source.VARIANTS if a != 'group_gate'}
            counts = {a: [reports[s][a]['correct'] for s in c['seeds']] for a in study.source.VARIANTS}
            results[d['tag']] = dict(seed_correct=counts,
                correct_mean_std={a: dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)), minimum=min(v), maximum=max(v)) for a, v in counts.items()},
                contrasts=comparisons, dependency_components=len(np.unique(clusters)),
                historical_seed42=read(ROOT/'logs/scale_agreement_fusion_v1/eval'/(d['tag']+'.json'))['comparison']['correct'])
        package = (results['pitts30k-val']['contrasts']['innovation1']['positive_seeds'] == 3 and
            all(results[t]['contrasts']['innovation1']['mean_net_queries'] >= 0 for t in ('gsv-adaptation-holdout', 'msls-val')))
        summary = dict(results=results, new_seeds=c['seeds'], historical_seed_excluded_from_inference=42,
            repeatable_package_count_signal=package,
            note='Count repetition, conditional query uncertainty, and mechanism contrasts are separate evidence. No winner selection.',
            independent_test_used=False, encoder_retrained=False, novelty_proven=False, verdict='REPLICATION_COMPLETE',
            report_export_repaired=True, training_restarted_for_repair=False)
        write(out/'summary.json', json_ready(summary))
        study.finish(out)
        print('REPLICATION_COMPLETE', json_ready({t: r['seed_correct'] for t, r in results.items()}), flush=True)


if __name__ == '__main__':
    main()
