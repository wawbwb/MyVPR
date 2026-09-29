"""Fixed broad-coverage batches plus a minority of previously locked hard batches."""
import numpy as np


def schedules(plan):
    places=plan['train_places']
    if len(places)!=4096 or len(set(places))!=4096 or set(places)&set(plan['dev_places']):
        raise ValueError('Invalid train/development split')
    if len(plan['epochs'])!=3: raise ValueError('Expected three source schedules')
    result=[]
    for epoch,source in enumerate(plan['epochs']):
        hard=source['batches']
        if len(hard)!=256 or any(len(b)!=16 or len(set(b))!=16 or not set(b)<=set(places) for b in hard):
            raise ValueError('Invalid source hard schedule')
        order=np.random.default_rng(42031+epoch).permutation(4096)
        broad=[[places[int(i)] for i in order[j:j+16]] for j in range(0,4096,16)]
        mixed=[]
        for block in range(64):
            mixed.extend(broad[block*4:block*4+4]);mixed.append(hard[block])
        result.append(mixed)
    return result
