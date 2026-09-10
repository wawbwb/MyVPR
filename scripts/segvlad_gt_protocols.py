"""Score fixed predictions under three GT protocols; stdlib only, no pickle/model."""
import argparse
import json
from pathlib import Path

if __package__:
    from .segvlad_official import sha
    from .segvlad_paired import recall_counts, compare_predictions
else:
    from segvlad_official import sha
    from segvlad_paired import recall_counts, compare_predictions


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_annotations(content, count):
    require(isinstance(content, list) and len(content) == count, 'GT row count mismatch')
    mapped = {}
    for row in content:
        require(isinstance(row, list) and len(row) == 2, 'Expected [query ID, reference IDs]')
        q, refs = row
        require(type(q) is int and 0 <= q < count and q not in mapped, 'Invalid/duplicate query ID')
        require(isinstance(refs, list) and all(type(r) is int for r in refs), 'Noninteger GT reference IDs')
        mapped[q] = refs
    return [mapped[q] for q in range(count)]


def check_names(names, directory):
    require(len(names) == len(set(names)), 'Duplicate image names')
    for i, name in enumerate(names):
        require(Path(name).name == name and Path(name).stem == str(i),
                'Expected zero-based numeric image names aligned with indices')
        require((directory/name).is_file(), f'Missing image {directory/name}')


def validate_predictions(pred, count, method, query_id):
    context = f'Invalid top5: method={method}, query={query_id}, predictions={pred!r}'
    require(isinstance(pred, list) and 1 <= len(pred) <= 5, context)
    require(all(type(x) is int and 0 <= x < count for x in pred), context)
    require(len(set(pred)) == len(pred), context)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paired', type=Path, required=True)
    p.add_argument('--diagnostic', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    paired, diag, base, out = a.paired.resolve(), a.diagnostic.resolve(), a.data_root.resolve()/'17places', a.output.resolve()
    require(not out.exists(), 'Output exists; choose a new directory')
    def read(root, name):
        return json.loads((root/name).read_text(encoding='utf8'))
    require(read(paired,'completed.json').get('complete') and read(diag,'completed.json').get('complete'), 'Incomplete input')
    audit = read(paired,'protocol_audit.json')
    original, diagnosed = read(paired,'per_query.json'), read(diag,'per_query.json')
    annotations = read(diag,'packaged_gt.json')
    n = len(original)
    require(n == 406 and len(diagnosed) == n, 'Expected 406 queries')
    check_names(audit['references'],base/'ref')
    check_names(audit['queries'],base/'query')
    require(len(audit['references']) == n and len(audit['queries']) == n, 'Image count mismatch')
    provenance = read(paired,'provenance.json')['asset_sha256']
    match_hashes = [h for path,h in provenance.items() if 'matches_sims' in Path(path).name]
    require(match_hashes == [read(diag,'completed.json')['matches_sha256']], 'Mismatched matching artifacts')
    predictions = {'anyloc':[], 'segvlad':[], 'capped':[]}
    official = audit['official_gt']
    require(official == [list(range(i-15,i+16)) for i in range(n)], 'Unexpected official GT')
    for q,(r,d) in enumerate(zip(original,diagnosed)):
        require(r['query_id'] == d['query_id'] == q and r['query'] == d['query'] == audit['queries'][q], 'Query mapping mismatch')
        require(r['gt'] == d['gt'] == official[q], 'GT mismatch between runs')
        require(r['segvlad_top5'] == d['original_top5'], 'Original predictions changed')
        for name, pred in [('anyloc',r['anyloc_top5']),('segvlad',r['segvlad_top5']),('capped',d['capped_top5'])]:
            validate_predictions(pred, n, name, q)
            predictions[name].append(pred)
    raw_gts = {'official_window15':official}
    for label, filename in [('packaged_original','ground_truth_new.npy'),('packaged_revised','my_ground_truth_new.npy')]:
        entry = annotations['annotations'][filename]
        require(sha(base/filename) == entry['sha256'] == audit['packaged_annotations'][filename]['sha256'], 'GT artifact mismatch')
        raw_gts[label] = parse_annotations(entry['content'],n)
    out.mkdir(parents=True)
    def save(name,value):
        (out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8')
    audit_report = {'index_mapping':'All image stems equal zero-based indices. NPY first column used as query ID; second as reference IDs. This does not certify physical-place correctness.',
        'readme':annotations.get('readme'), 'protocols':{},
        'short_candidate_lists':{name:[{'query_id':q,'length':len(pred)} for q,pred in enumerate(preds) if len(pred)<5]
                                 for name,preds in predictions.items()},
        'short_list_policy':'Retain original candidates and order. Recall@K uses available prefix; no padding, no query exclusion.'}
    protocols = {}
    for label, gt in raw_gts.items():
        valid = [sorted(set(x for x in row if 0 <= x < n)) for row in gt]
        audit_report['protocols'][label] = {'out_of_range_ids':[{'query':q,'ids':[x for x in row if not 0 <= x < n]} for q,row in enumerate(gt) if any(not 0 <= x < n for x in row)],
            'empty_valid_queries':[q for q,row in enumerate(valid) if not row],
            'valid_positive_counts':[len(row) for row in valid],
            'different_from_official_queries':[q for q in range(n) if set(valid[q]) != set(x for x in official[q] if 0 <= x < n)]}
        protocols[label] = valid
    save('mapping_audit.json',audit_report)
    require(all(all(row for row in gt) for gt in protocols.values()), 'Empty valid GT rows; inspect mapping audit, no silent denominator changes')
    summary = {'queries':n,'protocols':{},'scope':'Fixed-prediction post-hoc GT sensitivity. All protocols retained; not model selection or independent validation.'}
    cases = []
    for label,gt in protocols.items():
        metrics = {}
        for name,preds in predictions.items():
            counts = recall_counts(preds,gt)
            metrics[name] = {'correct_at_1_to_5':counts,'recall_percent':[x/n*100 for x in counts]}
        summary['protocols'][label] = {'methods':metrics,
            'segvlad_vs_anyloc':compare_predictions(predictions['anyloc'],predictions['segvlad'],gt),
            'capped_vs_anyloc':compare_predictions(predictions['anyloc'],predictions['capped'],gt),
            'capped_vs_segvlad':compare_predictions(predictions['segvlad'],predictions['capped'],gt)}
    expected = read(diag,'summary.json')
    for name,key in [('anyloc','anyloc_counts'),('segvlad','original_counts'),('capped','capped_counts')]:
        require(summary['protocols']['official_window15']['methods'][name]['correct_at_1_to_5'] == expected[key], 'Official count reproduction failed')
    for q in range(n):
        cases.append({'query_id':q,'query':audit['queries'][q],
            'predictions':{name:ps[q] for name,ps in predictions.items()},
            'protocols':{label:{'positive_reference_ids':gt[q], 'top1_hits':{name:ps[q][0] in gt[q] for name,ps in predictions.items()}} for label,gt in protocols.items()}})
    save('summary.json',summary)
    save('per_query.json',cases)
    save('completed.json',{'complete':True,'input_sha256':{str(root/name):sha(root/name) for root,name in [(paired,'per_query.json'),(paired,'protocol_audit.json'),(diag,'per_query.json'),(diag,'packaged_gt.json')]}})
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
