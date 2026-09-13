"""Deterministic region controls and adaptation-only city split (NumPy)."""
import hashlib
from collections import defaultdict
import numpy as np

SETTINGS = dict(seed=42, rank=4, alpha=4, blocks=2, epochs=2, places=4, views=4,
                lr=1e-4, weight_decay=.001, consistency=1., preserve=1.,
                coverage_threshold=.9, max_area=.3, train_places=256, dev_places=256)


def rng_for(key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big'))


def masks(fraction, identity):
    f = np.asarray(fraction)
    if f.shape != (20,20) or not np.isfinite(f).all() or (f<0).any() or (f>1).any():
        raise ValueError('Invalid coverage')
    m = f >= SETTINGS['coverage_threshold']
    if not m.any() or m.mean() > SETTINGS['max_area']:
        return np.zeros_like(m), np.zeros_like(m), 'empty_or_large'
    y,x = np.where(m)
    choices = [(dy,dx) for dy in range(-int(y.min()),20-int(y.max()))
               for dx in range(-int(x.min()),20-int(x.max())) if dy or dx]
    if not choices:
        return np.zeros_like(m), np.zeros_like(m), 'unshiftable'
    dy,dx = choices[int(rng_for('shift:'+identity).integers(len(choices)))]
    shifted = np.zeros_like(m); shifted[y+dy,x+dx] = True
    return m, shifted, 'active'


def make_split(records):
    groups = defaultdict(list)
    cities = defaultdict(list)
    for i,(path,label) in enumerate(records):
        parts = path.replace('\\','/').split('/')
        if len(parts)<3 or parts[0]!='Images':
            raise ValueError('Expected GSV Images/city/file')
        groups[int(label)].append(i)
    for label,ids in groups.items():
        names = {records[i][0].replace('\\','/').split('/')[1] for i in ids}
        if len(names)!=1 or len(ids)<4: raise ValueError('Invalid GSV place')
        cities[next(iter(names))].append(label)
    eligible = [c for c,ids in cities.items() if len(ids)>=SETTINGS['dev_places']]
    eligible.sort(key=lambda c: hashlib.sha256(('dev:'+c).encode()).hexdigest())
    if not eligible or len(cities)<2: raise ValueError('Need multiple cities and enough development places')
    dev_city = eligible[0]
    train = sorted(l for c,ids in cities.items() if c!=dev_city for l in ids)
    dev = sorted(cities[dev_city])
    if len(train)<SETTINGS['train_places']: raise ValueError('Too few training places')
    train = rng_for('train:42').choice(train, SETTINGS['train_places'], replace=False).tolist()
    dev = rng_for('dev:42').choice(dev, SETTINGS['dev_places'], replace=False).tolist()
    return {'dev_city':dev_city, 'train':[[l,groups[l]] for l in train],
            'dev':[[l, sorted(groups[l],key=lambda i: records[i][0])[:4]] for l in dev],
            'warning':'City held out from this adaptation only; may have been seen by pretrained RU.'}


def views(rgb, mask, identity, epoch):
    """Outside mask stays bitwise unchanged. No resize, geometry change or inpainting."""
    x = np.asarray(rgb, dtype=np.float32)
    if x.shape!=(3,280,280): raise ValueError('Expected RGB CHW 280 image')
    pixel = np.repeat(np.repeat(mask,14,0),14,1)[None]
    outputs = []
    for v in range(2):
        rng = rng_for(f'appearance:{identity}:{epoch}:{v}')
        gain = rng.uniform(.65,1.35,(3,1,1))
        bias = rng.uniform(-.12,.12,(3,1,1))
        texture = rng.normal(0,.025,(1,280,280))
        modified = np.clip(x*gain+bias+texture,0,1).astype(np.float32)
        outputs.append(np.where(pixel,modified,x))
    return outputs
