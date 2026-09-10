"""CPU-only portable HTML review of changed 17Places predictions (trusted pickles)."""
import argparse
from contextlib import ExitStack
import html
import json
import os
from pathlib import Path
import pickle
import sys

if __package__:
    from .segvlad_official import COMMIT, git, literal_config, sha
else:
    from segvlad_official import COMMIT, git, literal_config, sha


def window_info(query_id, reference_id):
    offset = reference_id - query_id
    return {'offset': offset, 'official_hit': abs(offset) <= 15,
            'distance_to_window_boundary': abs(abs(offset) - 15)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--paired', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    repo, base, paired, out = a.repo.resolve(), a.data_root.resolve()/'17places', a.paired.resolve(), a.output.resolve()
    if out.exists():
        p.error('Choose a new output directory')
    if git(repo, 'rev-parse', 'HEAD') != COMMIT or git(repo, 'status', '--porcelain', '--untracked-files=no'):
        p.error('Expected clean pinned upstream')
    def read(name):
        return json.loads((paired/name).read_text())
    if not read('completed.json').get('complete'):
        p.error('Paired run is incomplete')
    audit, summary, rows, provenance = read('protocol_audit.json'), read('summary.json'), read('per_query.json'), read('provenance.json')
    selected = sorted(summary['correction_query_ids'] + summary['regression_query_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['MPLBACKEND'] = 'Agg'
    sys.path.insert(0, str(repo))
    sys.argv = [str(repo/'place_rec_main.py')]
    os.chdir(repo)
    import numpy as np
    import h5py
    from natsort import natsorted
    from PIL import Image
    import func_vpr
    cfg, exp = literal_config(repo)
    names = [audit['references'], audit['queries']]
    for side, image_names in zip(('ref', 'query'), names):
        if natsorted(os.listdir(base/side)) != image_names:
            raise ValueError('Image order changed since paired run')
    match_paths = [Path(s) for s in provenance['asset_sha256'] if 'matches_sims' in Path(s).name]
    if len(match_paths) != 1:
        raise ValueError('Ambiguous saved matches')
    match_path = match_paths[0]
    for path in [match_path] + [base/'out'/cfg[f'masks_h5_filename_{s}'] for s in ('r', 'q')]:
        if sha(path) != provenance['asset_sha256'].get(str(path)):
            raise ValueError(f'Artifact changed: {path}')
    with match_path.open('rb') as f:
        saved = pickle.load(f)
    matches, sims = saved['matches'][:, :50], 2-saved['sims'][:, :50]
    if not np.isfinite(sims).all() or sims.max() <= sims.min():
        raise ValueError('Invalid similarity range')
    weights = (sims-sims.min())/(sims.max()-sims.min())
    out.mkdir(parents=True)
    (out/'images').mkdir()
    style = '<meta charset="utf-8"><style>body{font:16px sans-serif;margin:24px}img{max-width:440px;width:100%}.cards{display:flex;flex-wrap:wrap;gap:12px}figure{width:440px;margin:8px}pre{white-space:pre-wrap}</style>'
    def card(filename, caption):
        return f'<figure><img src="images/{filename}"><figcaption>{html.escape(caption)}</figcaption></figure>'
    def render(side, image_id, tag, mask=None):
        filename = f'{side}_{image_id}_{tag}.jpg'
        with Image.open(base/('ref' if side == 'r' else 'query')/names[0 if side == 'r' else 1][image_id]) as im:
            im = im.convert('RGB').resize((640, 480))
            if mask is not None:
                m = np.asarray(Image.fromarray(np.asarray(mask, dtype=np.uint8)*255).resize(im.size, Image.Resampling.NEAREST)) > 0
                pixels = np.asarray(im).copy()
                pixels[~m] = (pixels[~m]*0.25).astype('uint8')
                im = Image.fromarray(pixels)
            im.save(out/'images'/filename)
        return filename
    reports = []
    with ExitStack() as stack:
        handles = [stack.enter_context(h5py.File(base/'out'/cfg[f'masks_h5_filename_{s}'], 'r')) for s in ('r','q')]
        counts = [[len(h[f'{n}/masks']) for n in ns] for h, ns in zip(handles, names)]
        offsets = [np.concatenate(([0], np.cumsum(c))) for c in counts]
        ref_ids = np.repeat(np.arange(len(names[0])), counts[0])
        if len(matches) != offsets[1][-1] or matches.min() < 0 or matches.max() >= len(ref_ids):
            raise ValueError('Segment index mismatch')
        cache = {}
        def region(side, image_id, local):
            key = (side, image_id)
            if key not in cache:
                j = 0 if side == 'r' else 1
                masks = func_vpr.preload_masks(handles[j], names[j][image_id])
                adj = func_vpr.nbrMasksAGGFastSingle(masks, exp['order']).numpy()
                cache[key] = (masks, adj)
            masks, adj = cache[key]
            # Union is a spatial illustration, not residual magnitude/attention.
            return np.logical_or.reduce([masks[i] for i in np.flatnonzero(adj[local])])
        for q in selected:
            row = rows[q]
            if row['query_id'] != q or row['gt'] != list(range(q-15,q+16)):
                raise ValueError('Query/GT mismatch')
            lo, hi = offsets[1][q:q+2]
            votes = {}
            # Same rank-major traversal and summed normalized similarity as upstream voting.
            evidence = {}
            for rank in range(50):
                for s in range(lo, hi):
                    rseg = int(matches[s,rank]); rid = int(ref_ids[rseg]); w = float(weights[s,rank])
                    votes[rid] = votes.get(rid, 0.) + w
                    evidence.setdefault(rid, []).append((w, int(s-lo), int(rseg-offsets[0][rid]), rank+1))
            ranking = sorted(votes, key=votes.get, reverse=True)
            if ranking[:5] != row['segvlad_top5']:
                raise ValueError(f'Vote reconstruction mismatch q={q}')
            body = [style, f'<h1>Query {q}: {html.escape(row["query"])}</h1>',
                    '<p>GT = natural-sorted index ±15, not verified physical distance. '
                    'Masks below show order-3 neighbor union support, NOT attention or exact token weights.</p>',
                    '<div class="cards">', card(render('q',q,'original'),'Query')]
            report = {'query_id':q, 'type':'correction' if q in summary['correction_query_ids'] else 'regression', 'candidates':[]}
            for label, rid in [('AnyLoc',row['anyloc_top5'][0]), ('SegVLAD',row['segvlad_top5'][0])]:
                info = {'method':label, 'reference_id':rid, **window_info(q,rid), 'vote':votes.get(rid,0.)}
                report['candidates'].append(info)
                body.append(card(render('r',rid,'original'),str(info)))
            body.append('</div><h2>Strongest supporting region pairs for EACH candidate</h2>')
            for info in report['candidates']:
                rid = info['reference_id']
                pairs = sorted(evidence.get(rid,[]), reverse=True)[:3]
                info['top_region_pairs'] = pairs
                body.append(f'<h3>{info["method"]}: reference {rid}</h3><div class="cards">')
                for w, qs, rs, rank in pairs:
                    body.append(card(render('q',q,f'region{qs}',region('q',q,qs)),f'Query region {qs}; vote contribution {w:.5f}'))
                    body.append(card(render('r',rid,f'region{rs}',region('r',rid,rs)),f'Ref region {rs}; region-neighbor rank {rank}'))
                if not pairs:
                    body.append('<p>No votes in the saved top-50 region neighbors.</p>')
                body.append('</div>')
            body.append('<h2>GT boundary context: reference indices q−16, q−15, q, q+15, q+16</h2><div class="cards">')
            for rid in (q-16,q-15,q,q+15,q+16):
                if 0 <= rid < len(names[0]):
                    body.append(card(render('r',rid,'original'),f'Reference {rid}: {window_info(q,rid)}'))
            body.append('</div><p>Manual review: same physical scene? repeated objects? GT-window boundary effect? uncertain?</p>')
            (out/f'q_{q:04d}.html').write_text('\n'.join(body), encoding='utf8')
            reports.append(report)
            print(f'Review written: q={q}', flush=True)
    (out/'index.html').write_text(style+'<h1>17Places changed-case review</h1>'+''.join(f'<p><a href="q_{r["query_id"]:04d}.html">{r["query_id"]}: {r["type"]}</a></p>' for r in reports), encoding='utf8')
    (out/'review.json').write_text(json.dumps(reports,indent=2),encoding='utf8')
    (out/'completed.json').write_text(json.dumps({'complete':True,'paired':str(paired),'cases':selected,'matches_sha256':sha(match_path)}),encoding='utf8')
    print(out, flush=True)


if __name__ == '__main__':
    main()
