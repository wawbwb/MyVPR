#!/usr/bin/env python
"""Build portable HTML case review from saved rankings; no model or GPU."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil

import numpy as np


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--msls-path', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--success-samples', type=int, default=12)
    a = p.parse_args()
    if a.success_samples < 0:
        p.error('success-samples must be nonnegative')
    completed = json.loads((a.audit/'completed.json').read_text())
    summary = json.loads((a.audit/'summary.json').read_text())
    provenance = json.loads((a.audit/'provenance.json').read_text())
    assert completed['complete'] and summary['complete']
    assert sha(a.audit/'summary.json') == completed['summary_sha256']
    assert sha(a.audit/'predictions.npz') == summary['predictions_sha256']
    root = a.msls_path.resolve()
    for name, expected in provenance['index_sha256'].items():
        assert sha(root/name) == expected, f'Index changed: {name}'
    db = np.load(root/'msls_val_dbImages.npy')
    queries = np.load(root/'msls_val_qImages.npy')
    gt = np.load(root/'msls_val_gt_25m.npy', allow_pickle=True)
    with np.load(a.audit/'predictions.npz') as data:
        arrays = {k: data[k] for k in ['global_ids', 'refined_ids', 'ru_ids']}
    assert len(queries) == summary['queries'] and len(db) == summary['references']
    for ids in arrays.values():
        assert ids.ndim == 2 and len(ids) == len(queries)
        assert np.issubdtype(ids.dtype, np.integer)
        assert (ids >= 0).all() and (ids < len(db)).all()
    g, r, u = (arrays[k] for k in ['global_ids', 'refined_ids', 'ru_ids'])
    hits = {k: np.asarray([np.isin(row, positives) for row, positives in zip(ids, gt)]) for k, ids in arrays.items()}
    gc, rc, uc = (hits[k][:, 0] for k in ['global_ids', 'refined_ids', 'ru_ids'])
    reachable = hits['global_ids'][:, :100].any(1)
    assert int(rc.sum()) == summary['refined_correct']
    assert int(gc.sum()) == summary['global_correct']
    assert int(uc.sum()) == summary['ru_correct']
    assert all(set(x[:100]) == set(y) for x, y in zip(g, r))
    groups = {
        'reachable_errors': np.flatnonzero(~rc & reachable).tolist(),
        'unreachable_errors': np.flatnonzero(~rc & ~reachable).tolist(),
        'global_regressions': np.flatnonzero(gc & ~rc).tolist(),
        'ru_only': np.flatnonzero(uc & ~rc).tolist(),
        'refined_only_vs_ru': np.flatnonzero(~uc & rc).tolist(),
        'global_corrections': np.flatnonzero(~gc & rc).tolist(),
    }
    sample = sorted(groups['global_corrections'], key=lambda i: hashlib.sha256(f'pairvpr_review_v1:{i}'.encode()).hexdigest())[:a.success_samples]
    display = {k: v for k, v in groups.items() if k not in ['global_corrections', 'refined_only_vs_ru']}
    display['successful_correction_sample'] = sample
    selected = sorted(set(i for values in display.values() for i in values))
    a.output.mkdir(parents=True, exist_ok=False)
    assets = a.output/'assets'
    assets.mkdir()
    copied = set()

    def figure(label, role, index, path):
        path = str(path.decode() if isinstance(path, bytes) else path)
        source = (root/path).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError(f'Invalid image path: {source}')
        name = f'{role}_{index}{source.suffix.lower()}'
        if name not in copied:
            shutil.copy2(source, assets/name)
            copied.add(name)
        return f'<figure><img loading="lazy" src="assets/{html.escape(name)}"><figcaption>{html.escape(label)}<br>{html.escape(path)}</figcaption></figure>'

    style = '<style>body{font:16px sans-serif;margin:24px}section{display:flex;flex-wrap:wrap}figure{width:360px;margin:10px}img{width:100%;height:240px;object-fit:contain;background:#eee}figcaption{overflow-wrap:anywhere}li{margin:8px}</style>'
    annotations = []
    for i in selected:
        positive = set(int(x) for x in gt[i])
        # Choose best GT according to saved global retrieval. Otherwise use a
        # deterministic annotated GT, explicitly not a scored nearest positive.
        ranked_positive = [int(x) for x in g[i] if int(x) in positive]
        best = ranked_positive[0] if ranked_positive else min(positive)
        global_rank = next((j+1 for j, x in enumerate(g[i]) if x in positive), None)
        refined_rank = next((j+1 for j, x in enumerate(r[i]) if x in positive), None)
        cards = figure('Query', 'q', i, queries[i])
        for j in range(3):
            idx = int(r[i, j])
            cards += figure(f'Refined #{j+1}: GT={idx in positive}', 'db', idx, db[idx])
        for label, idx in [('Global #1', int(g[i, 0])), ('RU #1', int(u[i, 0])),
                           ('Best GT in saved global list' if ranked_positive else 'Annotated GT (not ranked in saved list)', best)]:
            cards += figure(f'{label}: GT={idx in positive}', 'db', idx, db[idx])
        content = f'<h1>Query {i}</h1><a href="index.html">Index</a><p>Global first-positive rank: {global_rank}; refined first-positive rank: {refined_rank}. GT means dataset label, not guaranteed pixel overlap.</p><section>{cards}</section>'
        (a.output/f'q_{i:04d}.html').write_text('<!doctype html><meta charset="utf-8">'+style+content, encoding='utf8')
        annotations.append({'query_index': i, 'groups': [k for k, ids in display.items() if i in ids],
            'global_positive_rank': global_rank, 'refined_positive_rank': refined_rank,
            'visible_overlap': 'unreviewed', 'semantic_contradiction': 'unreviewed',
            'appearance_or_geometry_issue': 'unreviewed', 'gt_ambiguity': 'unreviewed', 'notes': ''})
    sections = '<h1>Pair-VPR case review</h1><p>Descriptive review only. Do not tune MSLS thresholds or treat these labels as independent validation.</p>'
    for name, ids in display.items():
        sections += f'<h2>{html.escape(name)} ({len(ids)})</h2><ul>'
        sections += ''.join(f'<li><a href="q_{i:04d}.html">Query {i}</a> {html.escape(str(queries[i]))}</li>' for i in ids)
        sections += '</ul>'
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8">'+style+sections, encoding='utf8')
    write_json(a.output/'annotations_template.json', annotations)
    write_json(a.output/'review_manifest.json', {'complete': True, 'audit_summary_sha256': sha(a.audit/'summary.json'),
        'script_sha256': sha(Path(__file__)), 'groups': groups, 'display_groups': display,
        'pages': len(selected), 'copied_original_images': len(copied), 'no_model_inference': True})
    print(json.dumps({k: len(v) for k, v in groups.items()}, indent=2))
    print('Open:', a.output/'index.html')


if __name__ == '__main__':
    main()
