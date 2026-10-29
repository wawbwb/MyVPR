"""CPU-only, post-hoc archival export. Never modify sealed training sources.

The original aggregate() has a duplicate 'checks' keyword. This independent
exporter separates mechanical_checks from gain_checks; no model is retrained.
It creates a NEW curated archive, excluding tensors, images and patch caches.
"""
import argparse
import ast
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.train_slgd import read, write, sha, code_hash, verify
from scripts.conditional_complement import SOURCES
from src.conditional_complement_protocol import (
    ARMS, POLICY, WARM_PARAMETER_SHA, matched_contract, compare_outcomes, candidate_verdict)
from src.models.official_boq import COMMIT, WEIGHT_SHA, WEIGHT_URL


def require(condition, message):
    if not condition:
        raise ValueError(message)


def project_dependencies(initial):
    """Capture statically imported local dependencies without importing them.

    Pinned training sources remain labelled separately from supplemental
    dependency snapshots. External packages/DINO cache are NOT vendored.
    """
    pending, found = list(initial), set()
    def resolve(module):
        if module.split('.')[0] not in ('src', 'scripts', 'argparser'):
            return []
        base = Path(*module.split('.'))
        return [p.as_posix() for p in (base.with_suffix('.py'), base / '__init__.py')
                if (ROOT / p).is_file()]
    while pending:
        name = pending.pop()
        if name in found:
            continue
        found.add(name)
        if not name.endswith('.py'):
            continue
        parts = list(Path(name).with_suffix('').parts)
        package = parts[:-1]
        nodes = ast.walk(ast.parse((ROOT / name).read_text(encoding='utf-8-sig')))
        for node in nodes:
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = (package[:len(package) - node.level + 1] if node.level else [])
                base = '.'.join(prefix + ([node.module] if node.module else []))
                modules = [base] + [base + '.' + alias.name for alias in node.names]
            for module in modules:
                pending.extend(resolve(module))
        for parent in Path(name).parents:
            init = parent / '__init__.py'
            if parent != Path('.') and (ROOT / init).is_file():
                pending.append(init.as_posix())
    return found


def export(args):
    paths = dict(official=args.reference / 'baseline', warm_start=args.warm_start,
                 **{arm: args.source / ('train_' + arm) for arm in ARMS})
    contracts = {name: verify(path) for name, path in paths.items()}
    originals = {name: read(path / 'completed.json') for name, path in paths.items()}
    first = contracts[ARMS[0]]
    require(first['policy'] == POLICY, 'Training policy changed')
    require(first['code'] == {p: code_hash(ROOT / p) for p in SOURCES}, 'Sealed source changed')
    for arm in ARMS:
        require(contracts[arm]['arm'] == arm, 'Wrong registered arm')
        require(matched_contract(contracts[arm]) == matched_contract(first), 'Unmatched contracts')
        require(contracts[arm]['code'] == first['code'], 'Different code between arms')
    require(first['official_baseline_completed_sha256'] == sha(paths['official'] / 'completed.json'),
            'Official baseline seal changed')
    require(first['warm_completed_sha256'] == sha(paths['warm_start'] / 'completed.json'),
            'Warm-start seal changed')
    require(first['warm_checkpoint_sha256'] == sha(paths['warm_start'] / 'last.pt'),
            'Warm-start checkpoint changed')
    require(read(paths['warm_start'] / 'summary.json')['final_parameter_sha256'] == WARM_PARAMETER_SHA,
            'Unexpected warm-start parameters')
    initials = {arm: read(paths[arm] / 'initial.json') for arm in ARMS}
    summaries = {arm: read(paths[arm] / 'summary.json') for arm in ARMS}
    mechanical = dict(
        source_seals_verified=True,
        source_code_verified=True,
        matched_contracts=True,
        matched_anchor_initialization=len({r['anchor_parameter_sha256'] for r in initials.values()}) == 1,
        matched_bc_initialization=initials[ARMS[1]]['initial_parameter_sha256'] == initials[ARMS[2]]['initial_parameter_sha256'],
        matched_bc_capacity=initials[ARMS[1]]['trainable_parameters'] == initials[ARMS[2]]['trainable_parameters'],
        all_run_checks=all(all(r['checks'].values()) for r in summaries.values()),
        all_2048_updates=all(r['additional_updates'] == POLICY['steps'] for r in summaries.values()))
    require(all(mechanical.values()), 'Mechanical fairness checks failed')
    metrics, outcomes = {}, {}
    for size in POLICY['eval_sizes']:
        for ds in ('msls-val', 'pitts30k-val'):
            tag = f'{ds}_{size}'
            rows = {name: read(path / (tag + '.json')) for name, path in paths.items()}
            for name, row in rows.items():
                ok = np.asarray(row['outcomes'])
                require(ok.ndim == 1 and ok.dtype == np.bool_, 'Invalid Boolean outcomes')
                require(len(ok) == row['queries'] and int(ok.sum()) == row['correct'], 'Wrong count')
                require(len(row['recall']) == 4 and abs(row['recall'][0] - float(ok.mean())) < 1e-12,
                        'Recall does not match outcome vector')
            outcomes[tag] = {name: row['outcomes'] for name, row in rows.items()}
            metrics[tag] = compare_outcomes(
                {arm: rows[arm]['outcomes'] for arm in ARMS},
                rows['official']['outcomes'], rows['warm_start']['outcomes'])
            metrics[tag]['reference_counts'] = {
                name: dict(correct=rows[name]['correct'], queries=rows[name]['queries'],
                           r1=rows[name]['recall'][0]) for name in ('official', 'warm_start')}
            metrics[tag]['recall_k'] = [1, 5, 10, 20]
            metrics[tag]['recall'] = {name: row['recall'] for name, row in rows.items()}
            metrics[tag]['bc_r1_outcomes_identical'] = rows[ARMS[1]]['outcomes'] == rows[ARMS[2]]['outcomes']
            metrics[tag]['bc_top20_predictions_identical'] = rows[ARMS[1]]['predictions'] == rows[ARMS[2]]['predictions']
    verdict = candidate_verdict(metrics)
    training = {}
    for arm, report in summaries.items():
        rows = report['rows']
        late = rows[-128:]
        keys = sorted(set.intersection(*(set(r['conditional']) for r in late)))
        training[arm] = dict(
            **{k: v for k, v in report.items() if k != 'rows'},
            first_training_row=rows[0], final_training_row=rows[-1],
            last128_loss_mean=float(np.mean([r['loss'] for r in late])),
            last128_relative_residual_mean=float(np.mean([r['relative_residual_mean'] for r in late])),
            last128_conditional_means={key: float(np.mean([r['conditional'][key] for r in late]))
                                       for key in keys})
    require(not args.output.exists(), 'Archive output exists; choose a NEW directory')
    args.output.mkdir(parents=True)
    evidence = args.output / 'results'
    evidence.mkdir()
    write(evidence / 'summary.json', dict(
        archive_title='创新点1：官方BoQ权重的末层锚定条件补充实验',
        archived_at=datetime.now().astimezone().isoformat(),
        verdict=verdict['verdict'], mechanical_checks=mechanical,
        gain_checks=verdict['checks'], candidate_assessment=verdict,
        metrics=metrics, policy=POLICY,
        aggregation_recovery=dict(
            original_error="dict() got multiple values for keyword argument 'checks'",
            recovery='Separate read-only post-hoc export; mechanical_checks and gain_checks are distinct',
            source_runs_modified=False, original_aggregate_modified=False, retrained=False),
        scope=POLICY['scope']))
    write(evidence / 'query_outcomes.json', outcomes)
    write(evidence / 'study_contract.json', first)
    write(evidence / 'initial_states.json', initials)
    write(evidence / 'training_summary.json', training)
    write(evidence / 'original_completions.json', originals)
    failure = args.source / 'progress.json'
    write(evidence / 'original_root_status.json', read(failure))
    provenance = dict(
        project_root=str(ROOT), git_head=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        note='Snapshot hashes, not Git HEAD alone, identify the executed uncommitted code',
        official=dict(commit=COMMIT, weight_sha256=WEIGHT_SHA, url=WEIGHT_URL),
        checkpoints_excluded=True, images_excluded=True, patch_caches_excluded=True,
        raw_top20_predictions_excluded=True,
        evidence_note='R1 outcomes and published recall retained; full top20 and image diagnostics remain at the sealed remote paths',
        sources={})
    for name, path in paths.items():
        provenance['sources'][name] = dict(
            path=str(path.resolve()), completed_sha256=sha(path / 'completed.json'),
            completed_at=datetime.fromtimestamp((path / 'completed.json').stat().st_mtime).astimezone().isoformat())
        if (path / 'last.pt').is_file():
            provenance['sources'][name]['checkpoint'] = dict(
                path=str((path / 'last.pt').resolve()), sha256=sha(path / 'last.pt'))
    write(evidence / 'provenance.json', provenance)
    code = args.output / 'code'
    inventory = {}
    # Preserve exact executed sources, including the historical aggregation bug.
    # The separate exporter is clearly post-hoc and never part of the training.
    selected_sources = project_dependencies(set(SOURCES) | {'scripts/archive_conditional_complement.py'})
    for name in sorted(selected_sources):
        source = (ROOT / name).resolve()
        require(source.is_relative_to(ROOT), 'Unsafe source path')
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        role = ('posthoc_archive_exporter' if name == 'scripts/archive_conditional_complement.py'
                else 'sealed_training_dependency' if name in SOURCES else 'static_project_dependency_snapshot')
        inventory[name] = dict(sha256=sha(target), normalized_code_sha256=code_hash(target), role=role)
    # Package init files and project dependency/license declarations are small.
    extras = {'requirements.txt', 'environment.yml', 'LICENSE'}
    for name in list(inventory):
        for parent in Path(name).parents:
            candidate = parent / '__init__.py'
            if parent != Path('.') and (ROOT / candidate).is_file():
                extras.add(candidate.as_posix())
    for name in sorted(extras):
        source = ROOT / name
        if source.is_file() and name not in inventory:
            target = code / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            inventory[name] = dict(sha256=sha(target), role='package_or_environment_context')
    write(args.output / 'code_manifest.json', inventory)
    write(args.output / 'archive_manifest.json', dict(
        complete=True, kind='curated_posthoc_archive_not_a_new_training_run',
        files={p.relative_to(args.output).as_posix(): sha(p)
               for p in sorted(args.output.rglob('*')) if p.is_file()},
        source_completed_sha256={name: sha(path / 'completed.json') for name, path in paths.items()}))
    # Confirm the export did not disturb any of the original seals.
    for path in paths.values():
        verify(path)
    print(json.dumps(dict(output=str(args.output.resolve()), code_files=len(inventory),
        total_mib=sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()) / 1024**2,
        mechanical_checks=mechanical, verdict=verdict['verdict'],
        primary_counts={tag: {arm: metrics[tag][arm]['correct'] for arm in ARMS}
                        for tag in metrics if tag.endswith('_322')},
        training_modified=False), ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('logs/conditional_complement_v1'))
    p.add_argument('--reference', type=Path, default=Path('logs/conv_official_boq_v1'))
    p.add_argument('--warm-start', type=Path, default=Path('logs/conv_official_layers_v1/last_only'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    protected = [args.source, args.reference, args.warm_start]
    resolved = args.output.resolve()
    require(resolved.is_relative_to(ROOT / 'doc') and resolved != ROOT / 'doc',
            'Use a new archive child under project doc')
    require(not any(resolved.is_relative_to(x.resolve()) or x.resolve().is_relative_to(resolved)
                    for x in protected), 'Never export over a source run')
    export(args)


if __name__ == '__main__':
    main()
