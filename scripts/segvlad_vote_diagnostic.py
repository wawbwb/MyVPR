"""Offline GT export and vote-cap sensitivity; no training or GT replacement."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

if __package__:
    from .segvlad_official import COMMIT, git, literal_config, sha
    from .segvlad_paired import recall_counts, compare_predictions
else:
    from segvlad_official import COMMIT, git, literal_config, sha
    from segvlad_paired import recall_counts, compare_predictions


def vote_rank(reference_ids, weights):
    """Rank-major original sum; cap keeps max per (query segment, reference image)."""
    total, per_pair, hits = {}, {}, {}
    for rank in range(len(reference_ids[0])):
        for qseg in range(len(reference_ids)):
            rid, w = int(reference_ids[qseg][rank]), float(weights[qseg][rank])
            total[rid] = total.get(rid, 0.) + w
            hits[rid] = hits.get(rid, 0) + 1
            key = qseg, rid
            per_pair[key] = max(per_pair.get(key, float('-inf')), w)
    capped = {rid: 0. for rid in total}  # Preserve original first-seen tie order.
    unique = {rid: 0 for rid in total}
    for (_, rid), w in per_pair.items():
        capped[rid] += w
        unique[rid] += 1
    return (sorted(total, key=total.get, reverse=True),
            sorted(capped, key=capped.get, reverse=True),
            {rid: {'original_vote': total[rid], 'capped_vote': capped[rid],
                   'vote_entries': hits[rid], 'distinct_query_segments': unique[rid],
                   'repeated_entries': hits[rid]-unique[rid]} for rid in total})


def json_value(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {'dict_entries': [[json_value(k), json_value(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return [json_value(x) for x in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {'uninterpreted_type': type(value).__name__, 'repr': repr(value)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--paired', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    repo, base, paired, out = a.repo.resolve(), a.data_root.resolve()/'17places', a.paired.resolve(), a.output.resolve()
    if out.exists():
        p.error('Output exists; use a new directory')
    if git(repo, 'rev-parse', 'HEAD') != COMMIT or git(repo, 'status', '--porcelain', '--untracked-files=no'):
        p.error('Expected clean pinned upstream')
    def read(name):
        return json.loads((paired/name).read_text())
    def save(name, value):
        (out/name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')
    if not read('completed.json').get('complete'):
        p.error('Paired run incomplete')
    audit, rows, provenance = read('protocol_audit.json'), read('per_query.json'), read('provenance.json')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['MPLBACKEND'] = 'Agg'
    sys.path.insert(0, str(repo))
    sys.argv = [str(repo/'place_rec_main.py')]
    os.chdir(repo)
    import numpy as np
    import h5py
    from natsort import natsorted
    import func_vpr
    cfg, exp = literal_config(repo)
    out.mkdir(parents=True)
    # Only load trusted user-downloaded author artifacts: object NPY uses pickle.
    gt_exports = {}
    for filename in ('ground_truth_new.npy', 'my_ground_truth_new.npy'):
        path = base/filename
        if not path.exists():
            gt_exports[filename] = {'missing': True}
            continue
        digest = sha(path)
        expected = audit['packaged_annotations'].get(filename, {}).get('sha256')
        if digest != expected:
            raise ValueError(f'Annotation hash changed or absent: {filename}')
        array = np.load(path, allow_pickle=True)
        gt_exports[filename] = {'sha256': digest, 'shape': list(array.shape),
                                'dtype': str(array.dtype), 'content': json_value(array)}
    save('packaged_gt.json', {'annotations': gt_exports,
        'readme': (base/'ReadMe.txt').read_text(errors='replace') if (base/'ReadMe.txt').exists() else None,
        'interpretation': 'Raw content only. No assumption about index base, query/reference direction, '
                          'or physical-place labels. Official GT unchanged.'})
    paths = [Path(s) for s in provenance['asset_sha256'] if 'matches_sims' in Path(s).name]
    if len(paths) != 1:
        raise ValueError('Ambiguous match artifact')
    for path in paths + [base/'out'/cfg[f'masks_h5_filename_{s}'] for s in ('r','q')]:
        if sha(path) != provenance['asset_sha256'].get(str(path)):
            raise ValueError(f'Artifact changed: {path}')
    with paths[0].open('rb') as f:
        data = pickle.load(f)
    matches, sims = data['matches'][:, :50], 2-data['sims'][:, :50]
    if not np.isfinite(sims).all() or sims.max() <= sims.min():
        raise ValueError('Invalid similarities')
    weights = (sims-sims.min())/(sims.max()-sims.min())
    offsets = []
    names = [audit['references'], audit['queries']]
    for side, ns in zip(('r','q'), names):
        with h5py.File(base/'out'/cfg[f'masks_h5_filename_{side}'], 'r') as h:
            if natsorted(h.keys()) != ns:
                raise ValueError('Mask image order mismatch')
            offsets.append(np.concatenate(([0], np.cumsum([len(h[f'{n}/masks']) for n in ns]))))
    ref_ids = np.repeat(np.arange(len(names[0])), np.diff(offsets[0]))
    if len(matches) != offsets[1][-1] or matches.min() < 0 or matches.max() >= len(ref_ids):
        raise ValueError('Segment indices incompatible')
    original, capped, details = [], [], []
    gt = audit['official_gt']
    if len(rows) != 406 or len(gt) != 406:
        raise ValueError('Expected 406 queries')
    for q, row in enumerate(rows):
        if row['query_id'] != q or row['gt'] != gt[q]:
            raise ValueError('Per-query order or GT mismatch')
        lo, hi = offsets[1][q:q+2]
        raw, cap, stats = vote_rank(ref_ids[matches[lo:hi]], weights[lo:hi])
        if raw[:5] != row['segvlad_top5']:
            raise ValueError(f'Original top5 mismatch q={q}; stop')
        original.append(raw[:5]); capped.append(cap[:5])
        candidates = set([raw[0], cap[0], row['anyloc_top5'][0]])
        details.append({'query_id':q, 'query':row['query'], 'original_top5':raw[:5],
                        'capped_top5':cap[:5], 'gt':gt[q],
                        'candidate_votes':{str(r):stats.get(r) for r in candidates}})
    save('per_query.json', details)
    anyloc = [r['anyloc_top5'] for r in rows]
    summary = {'queries':406, 'original_counts':recall_counts(original, gt),
        'capped_counts':recall_counts(capped, gt), 'anyloc_counts':recall_counts(anyloc, gt),
        'capped_vs_original':compare_predictions(original,capped,gt),
        'capped_vs_anyloc':compare_predictions(anyloc,capped,gt),
        'top1_changed':[q for q in range(406) if original[q][0] != capped[q][0]],
        'scope':'Post-hoc sensitivity only. Max one vote per query-segment/reference-image. '
                'Different query segments may still overlap. No claim of improved method or independent evidence.'}
    save('summary.json', summary)
    print(json.dumps(summary,indent=2), flush=True)
    # Exact spatial-support duplicates for changed-case images only, not all 812 images.
    selected = read('summary.json')['correction_query_ids'] + read('summary.json')['regression_query_ids']
    images = {('q',q) for q in selected}
    for q in selected:
        images.update(('r',r) for r in (original[q][0],capped[q][0],anyloc[q][0]))
    supports = []
    for side, i in sorted(images):
        ns = names[0 if side == 'r' else 1]
        with h5py.File(base/'out'/cfg[f'masks_h5_filename_{side}'], 'r') as h:
            masks = func_vpr.preload_masks(h,ns[i])
        adj = func_vpr.nbrMasksAGGFastSingle(masks, exp['order']).numpy()
        groups = {}
        for j in range(len(masks)):
            union = np.logical_or.reduce([masks[k] for k in np.flatnonzero(adj[j])])
            digest = hashlib.sha256(np.packbits(union).tobytes()).hexdigest()
            groups.setdefault(digest,[]).append(j)
        supports.append({'side':side,'image_index':i,'image':ns[i], 'regions':len(masks),
            'unique_union_supports':len(groups), 'redundant_fraction':1-len(groups)/len(masks),
            'duplicate_groups':[g for g in groups.values() if len(g)>1]})
        print(f'Spatial-support audit: {side}/{i}',flush=True)
    save('support_duplicates.json', {'images':supports,
        'note':'Exact mask-union equality only, not high-IoU overlap; not proof of identical VLAD descriptors.'})
    save('completed.json', {'complete':True,'paired':str(paired), 'matches_sha256':sha(paths[0])})


if __name__ == '__main__':
    main()
