"""Independent saved-GT/count/state audit; never encode images or train heads."""
from datetime import datetime
from pathlib import Path
import sys
import tarfile
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts import scale_factorial as study
from src.models.scale_agreement_fusion import ScaleAgreementFusion, ARMS
from src.semantic_region_protocol import read, write, sha, state_hash, owned_run


def rebuild(report, positives, references):
    p = np.asarray(report['predictions'])
    assert p.shape == (len(positives), 20) and np.issubdtype(p.dtype, np.integer)
    assert p.min() >= 0 and p.max() < references
    assert all(len(set(row)) == 20 for row in p)
    hits = np.array([[bool(set(row[:k]).intersection(g)) for k in (1, 5, 10, 20)] for row, g in zip(p, positives)])
    assert report['queries'] == len(positives)
    assert report['correct'] == int(hits[:, 0].sum())
    assert report['outcomes'] == hits[:, 0].tolist() and report['recall_k'] == [1, 5, 10, 20]
    assert report['recall'] == hits.mean(0).tolist()


def paired(reference, candidate, saved):
    a, b = np.asarray(reference['outcomes'], bool), np.asarray(candidate['outcomes'], bool)
    fixes, harms = np.flatnonzero(~a & b).tolist(), np.flatnonzero(a & ~b).tolist()
    assert saved['queries'] == len(a) and saved['correction_query_ids'] == fixes and saved['regression_query_ids'] == harms
    assert saved['corrections'] == len(fixes) and saved['regressions'] == len(harms) and saved['net'] == len(fixes)-len(harms)
    assert saved['top1_changed'] == int((np.asarray(reference['predictions'])[:, 0] != np.asarray(candidate['predictions'])[:, 0]).sum())
    return dict(corrections=len(fixes), regressions=len(harms), top1_changed=saved['top1_changed'])


def interval(delta, expected, c):
    per_query = np.asarray(delta, float).mean(0)
    rng = np.random.default_rng(c['bootstrap_seed'])
    estimates = []
    for start in range(0, c['bootstrap_repetitions'], 64):
        ids = rng.integers(0, len(per_query), (min(64, c['bootstrap_repetitions']-start), len(per_query)))
        estimates.extend((100*per_query[ids].mean(1)).tolist())
    np.testing.assert_allclose(np.quantile(estimates, [.025, .975]), expected['percentile_95_ci_pp'], atol=1e-12)
    np.testing.assert_allclose(100*per_query.mean(), expected['delta_r1_pp'], atol=1e-12)


def main():
    c = study.configuration()
    base = study.source.configuration()
    root = study.root(c)
    final = read(root/'aggregate/summary.json')
    result = dict(audited_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), checks={}, seeds={}, datasets={}, audited_version_reports=0)
    for stage in ('preflight', 'probe', 'aggregate'):
        assert study.verified(root/stage) == study.identity(c, stage)
    probe = read(root/'probe/summary.json')
    assert all(probe['checks'].values())
    plan = study.source.old.key.open_plan(base)
    train_rows = study.source.rows_for(plan, base)
    cache = study.cache_root(c)/'train_official'
    study.verified(cache)
    assert read(cache/'summary.json')['official_state_sha256'] == probe['official_state_sha256']
    for i in range(512):
        with np.load(cache/f'{i:04d}.npz') as d:
            rows = train_rows[i*64:(i+1)*64]
            assert d['indices'].tolist() == list(range(i*64, (i+1)*64)) and d['row_ids'].tolist() == rows
            assert d['hashes'].tolist() == [plan['images'][r]['image_sha256'] for r in rows]
    study.source.rm.verified(study.source.LOW_CACHE)
    study.source.verified(ROOT/'.cache/scale_agreement_fusion_v1/train_high')
    for seed in c['seeds']:
        seed_result = {}
        for family in ('official', 'innovation1'):
            parent = (root if family == 'official' else study.prior.root(study.prior.configuration()))/f'seed{seed}'
            verifier = study.verified if family == 'official' else study.prior.verified
            for stage in ('smoke', 'train'):
                verifier(parent/stage)
                assert all(read(parent/stage/'summary.json')['checks'].values())
                if family == 'official':
                    assert read(parent/stage/'contract.json') == study.identity(c, stage, seed)
            evaluation = root/family/f'seed{seed}/eval'
            eval_contract = study.verified(evaluation)
            assert all(read(evaluation/'summary.json')['checks'].values())
            assert eval_contract['checkpoint_sha256'] == sha(parent/'train/last.pt')
            state = torch.load(parent/'train/last.pt', map_location='cpu', weights_only=True)
            summary = read(parent/'train/summary.json')
            assert state['state']['step'] == 1024 and len(state['state']['history']) == 1024
            assert state['state']['contract_sha256'] == sha(parent/'train/contract.json')
            for a in ARMS:
                model = ScaleAgreementFusion(a)
                model.load_state_dict(state['heads'][a], strict=True)
                assert state_hash(model) == summary['final_head_sha256'][a]
                assert sum(p.numel() for p in model.parameters()) == 128
            assert read(parent/'smoke/initial.json')['heads'] == read(parent/'train/initial.json')['heads']
            for stage in ('smoke', 'train'):
                assert read(root/f'seed{seed}'/stage/'initial.json')['heads'] == read(study.prior.root(study.prior.configuration())/f'seed{seed}'/stage/'initial.json')['heads']
            with np.load(parent/'train/train_mean_gate.npz') as d:
                mean = d['mean'].astype(float)
                image_std = d['between_image_std'].astype(float)
            seed_result[family] = dict(steps=summary['steps'], effective_steps=summary['effective_steps'], final_head_sha256=summary['final_head_sha256'],
                mean_gate=float(mean.mean()), between_group_mean_gate_std=float(mean.std()), mean_between_image_gate_std=float(image_std.mean()),
                group_gate_last128_within_image_std=float(np.mean([h['group_gate']['within_image_std'] for h in state['state']['history'][-128:]])))
            if family == 'innovation1':
                study.prior.verified(parent/'eval')
            del state
        result['seeds'][str(seed)] = seed_result
    mapping = dict(A=('official', 'base_322'), B=('official', 'descriptor_average'), C=('official', 'group_gate'),
        D=('innovation1', 'base_322'), E=('innovation1', 'descriptor_average'), F=('innovation1', 'group_gate'))
    comparisons = dict(C_minus_A=('C', 'A'), B_minus_A=('B', 'A'), C_minus_B=('C', 'B'), F_minus_D=('F', 'D'),
        E_minus_D=('E', 'D'), F_minus_E=('F', 'E'), D_minus_A=('D', 'A'), E_minus_B=('E', 'B'), F_minus_C=('F', 'C'))
    for d in study.source.populations(base):
        tag = d['tag']
        official_cache = study.cache_root(c)/('eval_'+tag)
        study.verified(official_cache)
        old_cache = ROOT/'.cache/scale_agreement_fusion_v1'/('eval_'+tag)
        study.source.verified(old_cache)
        assert read(official_cache/'summary.json')['official_state_sha256'] == probe['official_state_sha256']
        for i in range((len(d['names'])+63)//64):
            with np.load(official_cache/f'{i:04d}.npz') as a, np.load(old_cache/f'{i:04d}.npz') as b:
                assert a['indices'].tolist() == b['indices'].tolist() and a['hashes'].tolist() == b['hashes'].tolist()
        positives = [set(map(int, g)) for g in d['ground_truth']]
        families, pairs = {}, {}
        for family in ('official', 'innovation1'):
            families[family], pairs[family] = [], {}
            for seed in c['seeds']:
                report = read(root/family/f'seed{seed}/eval'/(tag+'.json'))
                assert report['dataset'] == study.source.population_contract(d) and set(report['variants']) == set(study.VARIANTS)
                r = report['variants']
                for v in r.values():
                    rebuild(v, positives, d['references'])
                    result['audited_version_reports'] += 1
                pairs[family][str(seed)] = {a: paired(v, r['group_gate'], report['comparison']['candidate_vs_others'][a]) for a, v in r.items() if a != 'group_gate'}
                assert report['comparison']['correct'] == {a: v['correct'] for a, v in r.items()}
                if family == 'official':
                    assert r['base_322'] == read(study.reference(tag))['variants']['official']
                else:
                    old = study.remap_historical(read(study.prior.root(study.prior.configuration())/f'seed{seed}/eval'/(tag+'.json'))['variants'])
                    assert all(r[a] == v for a, v in old.items())
                families[family].append(r)
        matrices = {k: np.asarray([r[name]['outcomes'] for r in families[family]], int) for k, (family, name) in mapping.items()}
        f = final['results'][tag]
        assert f['cell_correct'] == {k: v.sum(1).tolist() for k, v in matrices.items()}
        for name, (left, right) in comparisons.items():
            delta = matrices[left]-matrices[right]
            z = f['contrasts'][name]
            assert z['seed_net_queries'] == delta.sum(1).tolist() and z['mean_net_queries'] == float(delta.sum(1).mean())
            if name in ('C_minus_A', 'C_minus_B', 'F_minus_D', 'F_minus_E'):
                interval(delta, z['query_paired'], c)
        for name, delta in dict(package_interaction=matrices['F']-matrices['D']-matrices['C']+matrices['A'],
                gate_interaction=matrices['F']-matrices['E']-matrices['C']+matrices['B']).items():
            assert f[name]['seed_net_queries'] == delta.sum(1).tolist()
            interval(delta, f[name]['query_paired'], c)
        result['datasets'][tag] = dict(cell_correct=f['cell_correct'], pairs=pairs,
            recall_counts={family: {a: [np.rint(np.asarray(r[a]['recall'])*d['queries']).astype(int).tolist() for r in rs] for a in study.VARIANTS} for family, rs in families.items()})
    result['checks'] = dict(all_stage_and_cache_seals_verified=True, matched_image_contents_and_train_rows=True,
        matched_head_initialization=True, final1024_states_verified=True, historical_official_Top20_exact=True,
        historical_innovation1_Top20_exact=True, all_GT_and_paired_IDs_recomputed=True,
        four_main_query_intervals_and_both_interactions_independently_recomputed=True, encoder_not_loaded=True, training_not_started=True)
    contract = dict(study.identity(c, 'independent_audit'), audit_source_sha256=sha(Path(__file__)), aggregate_seal_sha256=sha(root/'aggregate/completed.json'))
    with owned_run(root/'independent_audit', contract, resume=True) as out:
        if (out/'completed.json').exists():
            study.verified(out)
        else:
            write(out/'summary.json', result)
            study.finish(out)
    payloads = {p.relative_to(root).as_posix(): p for p in sorted(root.rglob('*'))
        if p.is_file() and p.suffix in ('.json', '.npz') and p.name != 'snapshot_manifest.json'}
    for p in sorted((ROOT/'.cache/scale_factorial_v1_launcher').glob('*')):
        if p.suffix in ('.json', '.log'):
            payloads['launcher/'+p.name] = p
    for name in ('train_official', *('eval_'+t for t in study.source.TAGS)):
        for suffix in ('contract.json', 'summary.json', 'completed.json'):
            payloads['cache/'+name+'/'+suffix] = study.cache_root(c)/name/suffix
    for name in (*study.FILES, 'scripts/audit_scale_factorial.py'):
        payloads['source/'+name] = ROOT/name
    manifest = root/'snapshot_manifest.json'
    write(manifest, dict(snapshot_at=result['audited_at'], files={name: sha(p) for name, p in payloads.items()},
        note='Checkpoints and descriptor cache shards remain on server; source seals reference them. No training/reencoding in this audit.'))
    archive = Path('/tmp/scale_factorial_v1_reports.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        for name, p in payloads.items():
            tar.add(p, arcname=name)
        tar.add(manifest, arcname='snapshot_manifest.json')
    print(dict(audited_at=result['audited_at'], audited_version_reports=result['audited_version_reports'],
        snapshot_files=len(payloads), archive=str(archive), sha256=sha(archive), checks=result['checks']))


if __name__ == '__main__':
    main()
