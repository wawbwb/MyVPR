"""Recompute GT/counts, verify checkpoint/seals, and snapshot small reports."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import scale_reliability_gate as study
from src.models.scale_reliability_gate import ScaleReliabilityGate, ARMS
from src.semantic_region_protocol import read, write, sha, owned_run, state_hash


def rebuild(report, truth, references):
    ranks = np.asarray(report['predictions'])
    if ranks.shape != (len(truth), 20) or ranks.min() < 0 or ranks.max() >= references:
        raise ValueError('Wrong prediction shape/range')
    if any(len(set(row)) != 20 for row in ranks):
        raise ValueError('Duplicate predictions')
    hit = np.array([[bool(set(row[:k]).intersection(set(g))) for k in (1, 5, 10, 20)] for row, g in zip(ranks, truth)])
    assert hit[:, 0].tolist() == report['outcomes']
    assert int(hit[:, 0].sum()) == report['correct']
    assert hit.mean(0).tolist() == report['recall']
    return hit[:, 0]


def main():
    c = study.configuration()
    sets = study.source.populations(study.source.configuration())
    result = dict(checks={}, seeds={}, datasets={}, audited_version_reports=0)
    for stage in ('preflight', 'aggregate'):
        assert study.verified(study.root(c)/stage) == study.identity(c, stage)
        result['checks'][stage+'_seal'] = True
    final = read(study.root(c)/'aggregate/summary.json')
    for seed in c['seeds']:
        b = study.base_value(seed)
        seeds = {}
        for stage in ('smoke', 'train', 'eval'):
            p = study.root(c)/f'seed{seed}'/stage
            contract = study.verified(p)
            assert all(read(p/'summary.json')['checks'].values())
            if stage != 'eval':
                assert contract == study.identity(c, stage, seed)
            seeds[stage+'_seal'] = sha(p/'completed.json')
        p = study.root(c)/f'seed{seed}/train'
        payload = torch.load(p/'last.pt', map_location='cpu', weights_only=True)
        state = payload['state']
        summary = read(p/'summary.json')
        assert state['step'] == c['pilot_steps'] and len(state['history']) == c['pilot_steps']
        assert state['contract_sha256'] == sha(p/'contract.json')
        for a in ARMS:
            m = ScaleReliabilityGate(a, b)
            m.load_state_dict(payload['heads'][a], strict=True)
            assert state_hash(m) == summary['final_head_sha256'][a]
            assert sum(p.numel() for p in m.parameters()) == 128
        assert sha(p/'last.pt') == read(p.parent/'eval/contract.json')['checkpoint_sha256']
        assert read(p/'initial.json')['heads'] == read(p.parent/'smoke/initial.json')['heads']
        history = state['history']
        seeds.update(frozen_scalar=b, radius=min(.35, .5-abs(b)),
            initial=read(p/'initial.json')['heads'], final=summary['final_head_sha256'],
            effective_ms_steps=summary['effective_ms_steps'],
            mean_preservation_error=max(r[a]['mean_preservation_error'] for r in history for a in ARMS if a != 'quality_global'),
            maximum_gate=max(r[a]['gate_max'] for r in history for a in ARMS),
            teacher_target_std=float(np.mean([r['teacher']['target_within_image_std'] for r in history])),
            training_diagnostics={a: dict(
                quality_first128=float(np.mean([r[a]['quality_loss'] for r in history[:128]])),
                quality_last128=float(np.mean([r[a]['quality_loss'] for r in history[-128:]])),
                within_image_std_last128=float(np.mean([r[a]['within_image_std'] for r in history[-128:]])),
                main_last128=float(np.mean([r[a]['main_loss'] for r in history[-128:]]))) for a in ARMS})
        result['seeds'][str(seed)] = seeds
        del payload
    from scripts import eval_clip_key_study as metrics
    for d in sets:
        reports = {s: read(study.root(c)/f'seed{s}/eval'/(d['tag']+'.json')) for s in c['seeds']}
        for seed, report in reports.items():
            assert report['dataset'] == study.source.population_contract(d)
            assert set(report['variants']) == set(study.VARIANTS)
            for name, r in report['variants'].items():
                rebuild(r, d['ground_truth'], d['references'])
                result['audited_version_reports'] += 1
                if name in study.source.VARIANTS:
                    old = read(study.prior.root(study.prior.configuration())/f'seed{seed}/eval'/(d['tag']+'.json'))
                    assert r == old['variants'][name]
            pairs = {a: metrics.paired_report(r, report['variants']['quality_group']) for a, r in report['variants'].items() if a != 'quality_group'}
            assert pairs == report['comparison']['candidate_vs_others']
            assert report['comparison']['correct'] == {a: r['correct'] for a, r in report['variants'].items()}
        f = final['results'][d['tag']]
        assert f['seed_correct'] == {a: [reports[s]['variants'][a]['correct'] for s in c['seeds']] for a in study.VARIANTS}
        for a, comparison in f['contrasts'].items():
            x = np.array([reports[s]['variants']['quality_group']['outcomes'] for s in c['seeds']], int)
            y = np.array([reports[s]['variants'][a]['outcomes'] for s in c['seeds']], int)
            assert comparison['seed_net_queries'] == (x-y).sum(1).tolist()
            assert comparison['mean_net_queries'] == float((x-y).sum(1).mean())
            if d['tag'] == 'pitts30k-val' and a in ('frozen_scalar', 'same_head_shuffled'):
                # Independent direct query resampling (not component/bin sums).
                delta = (x-y).mean(0)
                rng = np.random.default_rng(c['bootstrap_seed'])
                estimates = []
                for i in range(0, c['bootstrap_repetitions'], 64):
                    ids = rng.integers(0, len(delta), (min(64, c['bootstrap_repetitions']-i), len(delta)))
                    estimates.extend((100*delta[ids].mean(1)).tolist())
                np.testing.assert_allclose(np.quantile(estimates, [.025, .975]), comparison['query_paired']['percentile_95_ci_pp'], atol=1e-12)
        result['datasets'][d['tag']] = dict(correct=f['seed_correct'], all_GT_and_pairs_recomputed=True,
            gate_diagnostics={str(s): reports[s]['gate_diagnostics'] for s in c['seeds']})
    for s in c['seeds']:
        for t in ('smoke', 'train', 'eval'):
            study.prior.verified(study.prior.root(study.prior.configuration())/f'seed{s}'/t)
    result['checks'].update(historical_seals_unchanged=True, fixed_final_states=True, all_GT_and_pairs_recomputed=True,
        independent_query_intervals_match=True, no_encoder_loaded=True)
    contract = dict(study.identity(c, 'audit'), audit_source_sha256=sha(Path(__file__)), aggregate_seal_sha256=sha(study.root(c)/'aggregate/completed.json'))
    with owned_run(study.root(c)/'audit', contract, resume=True) as out:
        if (out/'completed.json').exists():
            study.verified(out)
        else:
            write(out/'summary.json', result)
            study.finish(out)
    import tarfile
    manifest = {}
    payloads = {}
    for p in sorted(study.root(c).rglob('*')):
        if p.is_file() and p.suffix in ('.json', '.npz'):
            payloads[p.relative_to(study.root(c)).as_posix()] = p
    launcher = ROOT/'.cache/scale_reliability_gate_v1_launcher'
    for p in sorted(launcher.glob('*')):
        if p.suffix in ('.json', '.log'):
            payloads['launcher/'+p.name] = p
    for name, p in payloads.items():
        manifest[name] = sha(p)
    snapshot = study.root(c)/'snapshot_manifest.json'
    write(snapshot, dict(files=manifest, excludes='Large checkpoints and caches remain on server; seals may reference them.'))
    archive = Path('/tmp/scale_reliability_gate_v1_reports.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        for name, p in payloads.items():
            tar.add(p, arcname=name)
        tar.add(snapshot, arcname='snapshot_manifest.json')
    print(json.dumps(dict(checks=result['checks'], audited_version_reports=result['audited_version_reports'],
        snapshot_files=len(payloads), archive=str(archive), sha256=sha(archive)), indent=2))


if __name__ == '__main__':
    main()
