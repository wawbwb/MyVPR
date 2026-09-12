"""CPU-only attribution of cached CLIP category competition to existing polygons."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import clip_dynamic_screen as base
from scripts.clip_dynamic_localization import FINE_BOXES, maps
from scripts.dynamic_mechanism_probe import polygon_mask


def region_diagnostic(cosines, boxes, target):
    """Weights reproduce overlap averaging, without counting overlapping pixels twice."""
    c = np.asarray(cosines, dtype=np.float64)
    maps(c, boxes)  # Validate shape, finite values and complete coverage.
    target = np.asarray(target, dtype=bool)
    base.require(target.shape == (280, 280) and target.any(), 'Empty/invalid region')
    counts = np.zeros((280, 280))
    for x, y, x2, y2 in boxes:
        counts[y*14:y2*14, x*14:x2*14] += 1
    weights = np.array([(target/counts)[y*14:y2*14, x*14:x2*14].sum()
                        for x, y, x2, y2 in boxes]) / target.sum()
    d, e = len(base.DYNAMIC), len(base.EGO)
    dynamic = c[:, :d].max(1)
    ego = c[:, d:d+e].max(1)
    static = c[:, d+e:].max(1)
    margin = dynamic - np.maximum(ego, static)
    positive = np.maximum(2/(1+np.exp(-np.clip(margin/base.TEMPERATURE, -60, 60)))-1, 0)
    labels = base.DYNAMIC + base.EGO + base.STATIC
    winner = c.argmax(1)
    tied = (c == c.max(1, keepdims=True)).sum(1) > 1
    groups = {'dynamic': dynamic, 'ego': ego, 'static': static}
    shares = {name: float(weights[(value > np.maximum.reduce([v for n, v in groups.items() if n != name]))].sum())
              for name, value in groups.items()}
    shares['group_tie'] = max(0., 1-sum(shares.values()))
    rows = []
    for i, weight in enumerate(weights):
        if weight == 0:
            continue
        x, y, x2, y2 = boxes[i]
        rows.append({'crop_index': i, 'box_grid20': list(boxes[i]), 'region_weight': float(weight),
                     'region_fraction_of_crop': float(target[y*14:y2*14, x*14:x2*14].mean()),
                     'top_label': 'tie' if tied[i] else labels[winner[i]],
                     'best_dynamic_label': labels[c[i, :d].argmax()],
                     'dynamic_cosine': float(dynamic[i]), 'ego_cosine': float(ego[i]),
                     'static_cosine': float(static[i]), 'margin': float(margin[i]),
                     'positive_score': float(positive[i]), 'cosines': c[i].tolist()})
    winners = {label: float(weights[(winner == i) & ~tied].sum()) for i, label in enumerate(labels)}
    winners['tie'] = float(weights[tied].sum())
    return {'weighted_group_wins': shares, 'weighted_label_wins': winners,
            'mean_dynamic_cosine': float(weights @ dynamic),
            'mean_ego_cosine': float(weights @ ego), 'mean_static_cosine': float(weights @ static),
            'mean_margin': float(weights @ margin), 'positive_score_inside': float(weights @ positive),
            'crops': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-cache', type=Path, default=Path('.cache/clip_dynamic_msls_v1'))
    p.add_argument('--localization', type=Path, default=Path('doc/clip_dynamic_localization_v2'))
    p.add_argument('--annotations', type=Path, default=Path('doc/dynamic_mechanism_cases_v1/annotations_validated.json'))
    p.add_argument('--output', type=Path, default=Path('doc/clip_competition_audit_v1'))
    a = p.parse_args()
    base.require(not a.output.exists(), 'Output exists; use a new directory')
    read = lambda f: json.loads(f.read_text(encoding='utf-8'))
    old, new = read(a.old_cache/'contract.json'), read(a.localization/'contract.json')
    base.require(old['inputs'] == new['inputs'], 'Input identity mismatch')
    base.require(old['dynamic_text'] == base.DYNAMIC and old['ego_text'] == base.EGO and
                 old['static_text'] == base.STATIC and old['temperature'] == base.TEMPERATURE and
                 old['crop_boxes_grid20'] == [list(b) for b in base.BOXES] and
                 new['fine_boxes'] == [list(b) for b in FINE_BOXES], 'Category/grid configuration mismatch')
    base.require(base.sha(a.annotations) == new['annotation_sha'], 'Annotations differ from V2 audit')
    base.require(base.sha(a.old_cache/'teacher.json') == new['old_teacher_sha'], 'Teacher identity mismatch')
    inputs = {}
    for root, filename, hashkey in [(a.old_cache, 'masks.npz', 'masks_sha256'),
                                    (a.localization, 'maps.npz', 'maps_sha256')]:
        completed = read(root/'completed.json')
        base.require(completed['complete'] and base.sha(root/filename) == completed[hashkey], 'Incomplete/changed cache')
        inputs[str(root/filename)] = completed[hashkey]
    with np.load(a.old_cache/'masks.npz', allow_pickle=False) as z:
        paths = z['query_paths'].tolist()
    with np.load(a.localization/'maps.npz', allow_pickle=False) as z:
        base.require(z['query_paths'].tolist() == paths, 'Query order mismatch')
        stored = {k: z[k].copy() for k in ['coarse_positive', 'fine_positive']}
    results = []
    old_hashes = {Path(k.replace('\\', '/')).name: v for k, v in new['old_shards'].items()}
    for case in read(a.annotations)['cases']:
        if case['decision'] != 'include':
            continue
        q = case['query_index']
        base.require(0 <= q < len(paths) and paths[q] == case['query_path'], 'Annotation query identity mismatch')
        pair = {}
        for scale, root, boxes in [('coarse', a.old_cache, base.BOXES), ('fine', a.localization, FINE_BOXES)]:
            file = root/'shards'/f'{q:04d}.npz'
            inputs[str(file)] = base.sha(file)
            if scale == 'coarse':
                base.require(inputs[str(file)] == old_hashes[file.name], 'Old shard hash mismatch')
            with np.load(file, allow_pickle=False) as z:
                pair[scale] = (z['cosines'].copy(), str(z['image_sha']))
            expected = [v for k, v in new['inputs'].items() if k.replace('\\', '/').endswith('/'+paths[q])]
            base.require(expected == [pair[scale][1]], 'Shard image identity mismatch')
            base.require(np.array_equal(maps(pair[scale][0], boxes)['external_positive'], stored[scale+'_positive'][q]), 'Map/shard mismatch')
        polygons = case['external']['polygons']
        for region, poly in [('union', polygons)] + [(f'polygon_{i}', [v]) for i, v in enumerate(polygons)]:
            target = polygon_mask(poly).astype(bool)
            if not target.any():
                continue
            row = {'query_index': q, 'region': region, 'query_path': paths[q], 'region_pixels': int(target.sum())}
            for scale, boxes in [('coarse', base.BOXES), ('fine', FINE_BOXES)]:
                row[scale] = region_diagnostic(pair[scale][0], boxes, target)
            results.append(row)
    base.require(results, 'No included external polygons')
    report = {'scope': 'Existing external polygons only; weights allocate overlap-averaged region pixels to crops. Shares are not segmentation accuracy. Raw cosine is not calibrated confidence. No training/inference/retrieval.',
              'labels': base.DYNAMIC+base.EGO+base.STATIC, 'inputs': inputs,
              'annotations_sha256': base.sha(a.annotations), 'script_sha256': base.sha(__file__), 'regions': results}
    lines = ['# CLIP 类别竞争诊断', '', '仅复用已有 external 多边形。窗口权重按其对区域像素的重叠平均贡献计算，不重复累计像素。',
             '动态胜率表示动态类别严格胜过自车和静态类别的加权窗口份额，不是检测召回率。余弦绝对值不能直接解释为识别置信度。', '',
             '| Query / 区域 | 窗口 | 动态胜率 | 自车胜率 | 静态胜率 | 平均 margin | 正分均值 | 最大赢家份额 |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in results:
        for scale in ('coarse', 'fine'):
            s = r[scale]; g = s['weighted_group_wins']
            winner, share = max(s['weighted_label_wins'].items(), key=lambda t: t[1])
            lines.append(f"| {r['query_index']} / {r['region']} | {scale} | {g['dynamic']:.1%} | {g['ego']:.1%} | {g['static']:.1%} | {s['mean_margin']:.4f} | {s['positive_score_inside']:.5f} | {winner} ({share:.1%}) |")
    lines += ['', '逐窗口类别余弦、区域占比、最佳动态类别和 margin 见 report.json。',
              '窗口内存在物体并不表示整块 crop 应由该物体主导；低区域占比提示背景混合，但不构成因果证明。',
              '正分为零只能证明固定提示词下动态相对竞争失败，不能证明模型完全没有车辆表征。']
    a.output.mkdir(parents=True)
    base.write(a.output/'report.json', report)
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Saved:', a.output)


if __name__ == '__main__':
    main()
