"""Read-only preregistered SLGD gates, not a full/stable-gain declaration."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_slgd import POLICY, SOURCES, code_hash, read, verify, write, teacher_checks


def compare(root, stage):
    names = ('teacher', 'retrieval', 'local_distill')
    contracts = {}
    summaries = {}
    finals = {}
    checks = {}
    for mode in names:
        output = root/f'{mode}_{stage}'
        contract = verify(output)
        if contract['policy'] != POLICY or contract['code'] != {n: code_hash(ROOT/n) for n in SOURCES}:
            raise ValueError('Current code/protocol differs from completed run')
        if contract['mode'] != mode or contract['stage'] != stage:
            raise ValueError('Mode/stage differs')
        contracts[mode] = contract
        summaries[mode] = read(output/'summary.json')
        finals[mode] = read(output/'final.json')
        summary = summaries[mode]
        checks[mode+'_mechanical'] = (summary['steps'] == POLICY[stage+'_steps']
                                     and summary['checkpoint_roundtrip']
                                     and summary['parameters_changed']
                                     and summary['nonzero_gradient_steps'] > 0)
    for mode in names[1:]:
        for field in ('stage','policy','checkpoint_sha256','train_places','holdout_places','metadata','code','versions'):
            if contracts[mode][field] != contracts['teacher'][field]:
                raise ValueError('Unmatched experiment: '+field)
    a, b = 'retrieval', 'local_distill'
    checks['identical_initial_parameters'] = summaries[a]['initial_parameter_sha256'] == summaries[b]['initial_parameter_sha256']
    checks['identical_frozen_teacher'] = contracts[a]['teacher_sha256'] == contracts[b]['teacher_sha256']
    checks['identical_initial_retrieval'] = read(root/f'{a}_{stage}'/'initial.json') == read(root/f'{b}_{stage}'/'initial.json')
    for field in ('query_indices','positive_indices','negative_indices','ru_correct','descriptor_dimension'):
        if finals[a][field] != finals[b][field]:
            raise ValueError('Unmatched heldout pairs: '+field)
    corrections = sum(not x and y for x, y in zip(finals[a]['outcomes'], finals[b]['outcomes']))
    regressions = sum(x and not y for x, y in zip(finals[a]['outcomes'], finals[b]['outcomes']))
    metrics = dict(queries=finals[a]['queries'], frozen_ru_correct=finals[a]['ru_correct'],
                   a_correct=finals[a]['correct'], b_correct=finals[b]['correct'],
                   corrections=corrections, regressions=regressions, net=corrections-regressions,
                   local_pair_gain_pp=100*(finals[b]['student_local_pair_accuracy']-finals[a]['student_local_pair_accuracy']),
                   dimension=finals[a]['descriptor_dimension'])
    if stage == 'pilot':
        checks.update({'teacher_'+k: v for k, v in teacher_checks(finals['teacher']).items()})
        checks['retrieval_gain'] = metrics['net'] >= POLICY['student_gate']['min_correct_gain']
        checks['local_gain'] = metrics['local_pair_gain_pp'] >= POLICY['student_gate']['min_local_gain_pp']
        checks['b_not_below_frozen_ru'] = metrics['b_correct'] >= metrics['frozen_ru_correct']
    return dict(stage=stage, verdict='PASS' if all(checks.values()) else 'FAIL', checks=checks,
                metrics=metrics, scope='Smoke PASS means mechanical only; pilot PASS is exploratory feasibility, not stable gain or semantic benefit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('logs/slgd_v1'))
    parser.add_argument('--stage', choices=('teacher','smoke','pilot'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Gate reports are immutable; use a new filename')
    if args.stage == 'teacher':
        out = args.root/'teacher_pilot'
        contract = verify(out)
        if contract['mode'] != 'teacher' or contract['stage'] != 'pilot' or contract['policy'] != POLICY:
            raise ValueError('Teacher contract differs')
        if contract['code'] != {n: code_hash(ROOT/n) for n in SOURCES}:
            raise ValueError('Teacher code changed')
        checks = teacher_checks(read(out/'final.json'))
        report = dict(stage='teacher', checks=checks, verdict='PASS' if all(checks.values()) else 'FAIL',
                      scope='Heldout pseudo-MNN usefulness only, not final VPR performance')
    else:
        report = compare(args.root, args.stage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, report)
    print(report, flush=True)
    if report['verdict'] != 'PASS':
        print('GATE STOP: inspect result; no full training or CLIP stage was started.', flush=True)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
