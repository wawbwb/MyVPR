#!/usr/bin/env python
"""Read-only Pair-VPR ranking/metadata audit. No model, GPU or GT changes."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(value):
    value = value.decode() if isinstance(value, bytes) else str(value)
    parts = value.replace('\\', '/').split('/')
    if parts[0] == 'train_val':
        parts = parts[1:]
    require(len(parts) == 4 and parts[1] in ('query', 'database')
            and parts[2] == 'images', f'Unexpected image path: {value}')
    return parts[0], parts[1], Path(parts[3]).stem


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--msls-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.msls_path.resolve()
    summary = json.loads((args.audit / 'summary.json').read_text(encoding='utf8'))
    completed = json.loads((args.audit / 'completed.json').read_text(encoding='utf8'))
    provenance = json.loads((args.audit / 'provenance.json').read_text(encoding='utf8'))
    require(completed['complete'] and summary['complete'], 'Incomplete ranking audit')
    require(sha(args.audit / 'summary.json') == completed['summary_sha256'], 'Summary changed')
    require(sha(args.audit / 'predictions.npz') == summary['predictions_sha256'], 'Predictions changed')
    for name, expected in provenance['index_sha256'].items():
        require(sha(root / name) == expected, f'Index changed: {name}')
    db = np.load(root / 'msls_val_dbImages.npy')
    queries = np.load(root / 'msls_val_qImages.npy')
    # Trusted local dataset GT contains ragged arrays.
    gt = np.load(root / 'msls_val_gt_25m.npy', allow_pickle=True)
    with np.load(args.audit / 'predictions.npz') as saved:
        ranks = {name: saved[name].copy() for name in ('global_ids', 'refined_ids', 'ru_ids')}
    require(len(queries) == len(gt) == summary['queries'] and len(db) == summary['references'], 'Index sizes differ')
    for name, ids in ranks.items():
        require(ids.ndim == 2 and ids.shape[0] == len(queries) and ids.shape[1] > 0
                and np.issubdtype(ids.dtype, np.integer)
                and (ids >= 0).all() and (ids < len(db)).all(), f'Invalid {name}')

    metadata, hashes = {}, {}
    tables = {
        'postprocessed.csv': ('easting', 'northing', 'view_direction'),
        'raw.csv': ('lon', 'lat', 'ca', 'captured_at', 'pano'),
        'seq_info.csv': ('sequence_key', 'frame_number'),
    }
    partitions = sorted({identity(p)[:2] for p in list(queries) + list(db)})
    for city, split in partitions:
        for filename, fields in tables.items():
            path = root / city / split / filename
            hashes[str(path.relative_to(root))] = sha(path)
            with path.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                require({'key', *fields}.issubset(reader.fieldnames or []), f'Missing columns: {path}')
                seen = set()
                for row in reader:
                    key = row['key']
                    require(key and key not in seen, f'Duplicate/empty key in {path}: {key}')
                    seen.add(key)
                    record = metadata.setdefault((city, split, key), {})
                    record.update({field: row[field] for field in fields})
    for path in list(queries) + list(db):
        record = metadata.get(identity(path), {})
        require(all(field in record for fields in tables.values() for field in fields), f'Missing metadata: {path}')

    def pair(qpath, dpath):
        qc, _, _ = identity(qpath)
        dc, _, _ = identity(dpath)
        q, d = metadata[identity(qpath)], metadata[identity(dpath)]
        xy = [number(x) for x in (q['easting'], q['northing'], d['easting'], d['northing'])]
        dist = math.hypot(xy[0] - xy[2], xy[1] - xy[3]) if qc == dc and all(x is not None for x in xy) else None
        angles = [number(q['ca']), number(d['ca'])]
        valid_angles = all(x is not None and 0 <= x <= 360 for x in angles)
        angle = abs((angles[0] - angles[1] + 180) % 360 - 180) if valid_angles else None
        result = {'same_city': qc == dc, 'projected_distance_m': dist,
                  'distance_minus_25m': None if dist is None else dist - 25,
                  'raw_ca_difference_deg': angle,
                  'same_sequence': qc == dc and bool(q['sequence_key']) and q['sequence_key'] == d['sequence_key']}
        for prefix, record in [('query', q), ('db', d)]:
            result.update({f'{prefix}_{field}': record[field] for fields in tables.values() for field in fields})
        return result

    rows, query_rows = [], []
    for i, qpath in enumerate(queries):
        positives = {int(x) for x in gt[i]}
        require(positives and min(positives) >= 0 and max(positives) < len(db), f'Invalid GT: {i}')
        positive_pairs = {j: pair(qpath, db[j]) for j in sorted(positives)}
        measurable = [j for j in positives if positive_pairs[j]['projected_distance_m'] is not None]
        nearest = min(measurable, key=lambda j: (positive_pairs[j]['projected_distance_m'], j)) if measurable else None
        ranked_gt = next((int(j) for j in ranks['global_ids'][i] if int(j) in positives), min(positives))
        flags = {name.replace('_ids', '_correct'): int(ids[i, 0]) in positives for name, ids in ranks.items()}
        flags['top100_reachable'] = any(int(j) in positives for j in ranks['global_ids'][i, :100])
        selected = [('global_top1', int(ranks['global_ids'][i, 0])),
                    ('refined_top1', int(ranks['refined_ids'][i, 0])),
                    ('ru_top1', int(ranks['ru_ids'][i, 0])), ('review_gt', ranked_gt)]
        if nearest is not None:
            selected.append(('nearest_gt_by_metadata', nearest))
        selected.extend(('all_gt', j) for j in sorted(positives))
        for role, j in selected:
            rows.append({'query_index': i, 'query_path': str(qpath), 'role': role,
                         'db_index': j, 'db_path': str(db[j]), 'official_gt': j in positives,
                         **flags, **pair(qpath, db[j])})
        top = pair(qpath, db[int(ranks['refined_ids'][i, 0])])
        query_rows.append({'query_index': i, **flags,
                           'refined_distance_m': top['projected_distance_m'],
                           'refined_raw_ca_difference_deg': top['raw_ca_difference_deg'],
                           'nearest_gt_distance_m': None if nearest is None else positive_pairs[nearest]['projected_distance_m']})
    for name in ('global', 'refined', 'ru'):
        require(sum(row[f'{name}_correct'] for row in query_rows) == summary[f'{name}_correct'], f'{name} recall mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in [('pairs.csv', rows), ('per_query.csv', query_rows)]:
        with (args.output / name).open('x', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    report = {'complete': True, 'queries': len(queries), 'pair_rows': len(rows),
              'summary_sha256': sha(args.audit / 'summary.json'), 'metadata_sha256': hashes,
              'script_sha256': sha(Path(__file__)),
              'notes': ['Distances use supplied projected coordinates only within the same city; cross-city distances are blank.',
                        '25m difference is descriptive, not a replacement GT rule.',
                        'Raw ca circular difference is metadata only, not verified viewing overlap; pano/invalid angles require caution.',
                        'review_gt matches HTML choice: first GT in saved global list, else smallest GT index.',
                        'No model inference, GT changes, or threshold selection.']}
    (args.output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print('query  correct  distance_m  ca_diff_deg  nearest_gt_m')
    for row in query_rows:
        if not row['refined_correct']:
            print(row['query_index'], row['refined_correct'], row['refined_distance_m'],
                  row['refined_raw_ca_difference_deg'], row['nearest_gt_distance_m'])
    print('Wrote:', args.output.resolve())


if __name__ == '__main__':
    main()
