"""Fixed GSV-only hard/ordinary batches with conservative geographic exclusion."""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def distance(a, b):
    a, b = np.radians(a), np.radians(b)
    d = b-a
    h = np.sin(d[..., 0]/2)**2+np.cos(a[..., 0])*np.cos(b[..., 0])*np.sin(d[..., 1]/2)**2
    return 6371000*2*np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def safe_pair(i, j, centers, radii):
    # Triangle inequality: all recorded locations separated by at least 100m.
    return i != j and distance(centers[i], centers[j])-radii[i]-radii[j] >= 100


def make_batches(pairs, centers, radii, seed, count=256):
    rng = np.random.default_rng(seed)
    if not pairs: raise ValueError('No eligible hard pairs; do not silently use random batches')
    pairs = [pairs[i] for i in rng.permutation(len(pairs))]
    batches, planted = [], []
    cursor = 0
    for _ in range(count):
        group, links = [], []
        for _ in range(len(pairs)):
            i, j = pairs[cursor % len(pairs)]
            cursor += 1
            if i in group or j in group or not safe_pair(i, j, centers, radii): continue
            if all(safe_pair(k, old, centers, radii) for k in (i, j) for old in group):
                group.extend((i, j)); links.append([i, j])
            if len(links) == 4: break
        if len(links) != 4: raise ValueError('Cannot construct four geographically safe hard pairs')
        for k in rng.permutation(len(centers)):
            k = int(k)
            if all(safe_pair(k, old, centers, radii) for old in group): group.append(k)
            if len(group) == 16: break
        if len(group) != 16: raise ValueError('Insufficient safe ordinary places')
        batches.append(group); planted.append(links)
    return batches, planted


def main():
    from scripts.candidate_set_screen import read, write, sha, complete
    from scripts.adaptive_pair_budget import verified
    from scripts.train_depth_query import seed
    from src.dataloaders.train.gsv_cities import GSVCitiesDataset
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, default=Path('doc/second_order_training_audit_v1'))
    p.add_argument('--run', type=Path, default=Path('logs/second_order_query/mean_outer_screen_v1'))
    p.add_argument('--output', type=Path, default=Path('doc/local_value_plan_v1'))
    a = p.parse_args()
    verified(a.audit); verified(a.run)
    c = read(a.run/'contract.json')
    places = c['train_places']
    if read(a.audit/'places.json')['places'] != places: raise ValueError('Place order differs')
    if read(a.audit/'contract.json')['sources']['mean_outer']['contract'] != sha(a.run/'contract.json'):
        raise ValueError('Audit source mismatch')
    for path, h in c['data'].items():
        if 'Dataframes' in Path(path).parts and sha(Path(path)) != h: raise ValueError('Metadata mismatch')
    policy = dict(source=sha(a.audit/'completed.json'), training_contract=sha(a.run/'contract.json'),
        threshold=.02, min_distance_m=100, hard_pairs_per_batch=4, ordinary_places_per_batch=8,
        batches_per_epoch=256, epochs=3, seeds=[42,43,44],
        scope='Post-hoc GSV training-only hard sampling; view0 clean RU negatives, not guaranteed hard after augmentation; no new independent test',
        code=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    if a.output.exists():
        verified(a.output)
        if read(a.output/'contract.json') != policy: raise ValueError('Existing plan differs')
        print('Already complete'); return
    seed(42)
    dataset = GSVCitiesDataset(dataset_path=Path('datasets/gsv_cities'), cities='all', img_per_place=4)
    centers, radii = [], []
    for place in places:
        coords = dataset.dataframe.loc[int(place), ['lat', 'lon']].to_numpy(dtype=float).reshape(-1, 2)
        if not np.isfinite(coords).all() or (np.abs(coords[:, 0]) > 90).any() or (np.abs(coords[:, 1]) > 180).any():
            raise ValueError('Invalid coordinates; cannot certify geographic filtering')
        center = coords[0]
        centers.append(center); radii.append(float(distance(center, coords).max()))
    centers, radii = np.array(centers), np.array(radii)
    rows = read(a.audit/'ru.json')
    pairs, seen, rejected = [], set(), 0
    for row in sorted(rows, key=lambda r: r['margin']):
        if row['correct'] and row['margin'] > .02: continue
        i, j = row['query']//4, row['negative']//4
        pair = tuple(sorted((i, j)))
        if pair in seen: continue
        seen.add(pair)
        if safe_pair(i, j, centers, radii): pairs.append([i, j])
        else: rejected += 1
    epochs, counts = [], []
    for epoch in range(3):
        batches, links = make_batches(pairs, centers, radii, 42+epoch)
        epochs.append(dict(batches=[[places[i] for i in batch] for batch in batches], planted_pairs=links))
        counts.append(np.bincount(np.array(batches).ravel(), minlength=len(places)).tolist())
    a.output.mkdir(parents=True)
    write(a.output/'contract.json', policy)
    write(a.output/'plan.json', dict(train_places=places, dev_places=c['dev_places'], epochs=epochs))
    write(a.output/'summary.json', dict(eligible_pairs=len(pairs), geographic_rejections=rejected,
        exposure_counts=counts, unique_places_per_epoch=[sum(x > 0 for x in row) for row in counts],
        warning='Place identities are labels; 100m exclusion reduces but does not prove absence of false negatives. Hard places deliberately repeat.'))
    complete(a.output)
    print('PLAN COMPLETE eligible pairs', len(pairs), 'rejected', rejected, flush=True)


if __name__ == '__main__': main()
