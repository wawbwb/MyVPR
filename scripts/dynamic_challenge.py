"""Automatic synthetic dynamic-occlusion challenge; frozen models only."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import sys
from contextlib import nullcontext
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.clip_token_drop import Cache,read,write,sha,seed,device,load_ru,image_tensor,codes as base_codes
from scripts.dynamic_challenge_utils import SEEDS,LOADS,SCHEMES,rng,shifted,shift_scores,scaled_object,composite,summary


def code():
    paths=['scripts/dynamic_challenge.py','scripts/dynamic_challenge_utils.py','scripts/cache_dynamic_category_masks.py']
    return {**base_codes(),**{p:hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in paths}}


def verify(folder):
    done=read(folder/'completed.json')
    if not done['complete']: raise ValueError('Incomplete stage')
    for name,digest in done['files'].items():
        if sha(folder/name)!=digest: raise ValueError('Changed stage file: '+name)
    c=read(folder/'contract.json')
    if c['code']!=code(): raise ValueError('Stage code changed')
    return c


def complete(out):
    write(out/'completed.json',{'complete':True,'files':{str(p.relative_to(out)):sha(p)
          for p in out.rglob('*') if p.is_file() and p.name!='completed.json'}})


def read_rgb(root,path,expected):
    # Recover the exact uint8 resize used by old RU's torchvision v2 transform.
    x,_=image_tensor(root,path,expected)
    rgb=x.numpy()*np.array([.229,.224,.225],np.float32)[:,None,None]+np.array([.485,.456,.406],np.float32)[:,None,None]
    return np.clip(np.round(rgb.transpose(1,2,0)*255),0,255).astype('uint8')


def tensor(rgb):
    from scripts.eval_condition_robustness import build_transform
    return build_transform((280,280))(Image.fromarray(rgb))


def bank(a):
    import torch
    from scipy.ndimage import label,find_objects,binary_erosion
    from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights
    from urllib.parse import urlparse
    from scripts.cache_dynamic_category_masks import build_teacher
    cached=Cache(a.gsv_cache)
    if cached.contract['split']!='train': raise ValueError('Donor cache must be GSV train')
    weight=a.seg_weights or Path(torch.hub.get_dir())/'checkpoints'/Path(urlparse(DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT.url).path).name
    if not weight.is_file(): raise FileNotFoundError('Existing DeepLab weight missing; provide --seg-weights: '+str(weight))
    seed(42); d=device(a.device); model,categories,identity=build_teacher(weight,d)
    wanted=[c for c in ['car','bus','person','bicycle','motorbike'] if c in categories]
    order=sorted(range(len(cached.records)),key=lambda i:hashlib.sha256(cached.records[i][0].encode()).hexdigest())[:1024]
    a.output.mkdir(parents=True,exist_ok=False); objects=[]; scanned=[]
    with torch.inference_mode():
        for i in order:
            path=cached.records[i][0]; rgb=read_rgb(a.gsv_root,path,cached.image_hashes[i])
            prob=model(tensor(rgb)[None].to(d))['out'].softmax(1)[0].cpu().numpy()
            best=prob.argmax(0); scanned.append(path)
            for name in wanted:
                cid=categories.index(name); components,n=label((best==cid)&(prob[cid]>=.9))
                for k,box in enumerate(find_objects(components),1):
                    if box is None: continue
                    y,x=box
                    if y.start<=0 or x.start<=0 or y.stop>=260 or x.stop>=280: continue
                    m=binary_erosion(components[box]==k,iterations=1)
                    if m.sum()<196 or m.mean()<.3: continue
                    patch=rgb[box].copy()
                    try: scaled_object(patch,m,.15)
                    except ValueError: continue
                    filename=f'object_{len(objects):03d}.npz'
                    np.savez_compressed(a.output/filename,rgb=patch,mask=m)
                    objects.append({'file':filename,'category':name,'source':path,'source_sha256':cached.image_hashes[i],
                                    'pixels':int(m.sum()),'box':[x.start,y.start,x.stop,y.stop]})
                    break
                if len(objects)>=32: break
            if len(scanned)%25==0: print(f'Donor scan {len(scanned)}/1024; objects {len(objects)}/32',flush=True)
            if len(objects)>=32: break
    if len(objects)<8: raise ValueError('Fewer than eight eligible objects; inspect segmentation before challenge')
    write(a.output/'contract.json',{'code':code(),'objects':objects,'scanned':scanned,'teacher':identity,
          'source_cache_sha256':cached.hash,'classes':wanted,'confidence':.9,
          'warning':'Automatic connected semantic components may be imperfect objects. No MSLS GT used.'})
    cards=[]
    for j,obj in enumerate(objects):
        with np.load(a.output/obj['file']) as z:
            rgb=z['rgb']; m=z['mask']; rgba=np.dstack([rgb,m.astype('uint8')*255])
        filename=f'donor_{j:03d}.png'; Image.fromarray(rgba).save(a.output/filename)
        cards.append(f'<p>{html.escape(obj["category"]+": "+obj["source"])}</p><img src="{filename}">')
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Automatically extracted donors</h1>'+''.join(cards),encoding='utf8')
    complete(a.output); print('Donors saved:',a.output)


def build(a):
    import torch
    cached=Cache(a.msls_cache); donors=verify(a.donors)['objects']
    if cached.contract['split']!='msls' or len(cached.records)!=19611: raise ValueError('Wrong MSLS cache')
    ndb=cached.contract['info']['ndb']; queries=cached.records[ndb:]
    selected=sorted(sorted(range(len(queries)),key=lambda q:hashlib.sha256(queries[q][0].encode()).hexdigest())[:128])
    a.output.mkdir(parents=True,exist_ok=False); (a.output/'images').mkdir(); cases=[]; preview=[]
    objects=[]
    for obj in donors:
        with np.load(a.donors/obj['file']) as z: objects.append((z['rgb'].copy(),z['mask'].copy()))
    for qi in selected:
        index=ndb+qi; path=cached.records[index][0]
        background=read_rgb(a.msls_root,path,cached.image_hashes[index])
        original,_=image_tensor(a.msls_root,path,cached.image_hashes[index])
        if not torch.allclose(tensor(background),original,atol=1e-6,rtol=0):
            raise ValueError('PNG round-trip changed baseline preprocessing')
        cleanid=f'q{qi:04d}_clean'
        Image.fromarray(background).save(a.output/'images'/f'{cleanid}.png')
        np.savez_compressed(a.output/'images'/f'{cleanid}.npz',mask=np.zeros((280,280),bool))
        cases.append({'id':cleanid,'query':qi,'path':path,'content':'clean','seed':0,'load':0.,'actual_area':0.})
        for s in SEEDS:
            donor=int(rng(f'donor:{qi}:{s}').integers(len(objects))); rgb,m=objects[donor]
            for area in LOADS:
                obj,flat,support=composite(background,rgb,m,area,f'{qi}:{s}')
                for content,img in [('object',obj),('flat',flat)]:
                    key=f'q{qi:04d}_s{s}_a{int(area*100):02d}_{content}'
                    Image.fromarray(img).save(a.output/'images'/f'{key}.png')
                    np.savez_compressed(a.output/'images'/f'{key}.npz',mask=support)
                    cases.append({'id':key,'query':qi,'path':path,'content':content,'seed':s,'load':area,
                                  'actual_area':float(support.mean()),'donor':donor})
                if len(preview)<24 and s==SEEDS[0]:
                    name=f'preview_{len(preview):02d}.jpg'
                    Image.fromarray(np.concatenate([background,obj,flat],1)).save(a.output/name)
                    preview.append(f'<p>q{qi}, target area {area}</p><img width="840" src="{name}">')
        if len(cases)%104==0: print(f'Challenge images {len(cases)}/1664',flush=True)
    write(a.output/'contract.json',{'code':code(),'cases':cases,'queries':selected,'seeds':SEEDS,'loads':LOADS,
          'msls_cache_sha256':cached.hash,'donors_completed_sha256':sha(a.donors/'completed.json'),
          'warning':'Fixed synthetic MSLS subset selected by path hash, not independent real-world validation. Flat control keeps object silhouette.'})
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Original / object / matched flat occlusion</h1>'+''.join(preview),encoding='utf8')
    complete(a.output); print('Challenge saved:',a.output)


def clip(a):
    import torch
    from src.models.clip_token_drop import DenseCLIPTeacher
    c=verify(a.challenge); seed(42); d=device(a.device); teacher=DenseCLIPTeacher(d)
    a.output.mkdir(parents=True,exist_ok=False); maps=[]
    with torch.inference_mode():
        for start in range(0,len(c['cases']),4):
            batch=[]
            for case in c['cases'][start:start+4]:
                with Image.open(a.challenge/'images'/f'{case["id"]}.png') as im: batch.append(tensor(np.array(im.convert('RGB'))))
            maps.append(teacher.fractions(torch.stack(batch).to(d)))
            if (start+4)%64==0: print(f'CLIP challenge {start+4}/{len(c["cases"])}',flush=True)
    np.save(a.output/'fractions.npy',np.concatenate(maps))
    write(a.output/'contract.json',{'code':code(),'challenge_completed_sha256':sha(a.challenge/'completed.json'),
          'ids':[v['id'] for v in c['cases']],'teacher':teacher.identity})
    complete(a.output); print('CLIP saved:',a.output)


def evaluate(a):
    import torch
    from scripts.clip_dynamic_screen import outcome
    c=verify(a.challenge); cc=verify(a.clip_cache); cached=Cache(a.msls_cache)
    if c['msls_cache_sha256']!=cached.hash or cc['challenge_completed_sha256']!=sha(a.challenge/'completed.json'):
        raise ValueError('Challenge/CLIP identity mismatch')
    cases=c['cases']; maps=np.load(a.clip_cache/'fractions.npy')
    if cc['ids']!=[v['id'] for v in cases] or maps.shape!=(len(cases),20,20): raise ValueError('CLIP map order/shape mismatch')
    if not np.isfinite(maps).all() or (maps<0).any() or (maps>1).any(): raise ValueError('Invalid CLIP maps')
    source=read(a.source/'completed.json')
    if not source['complete'] or not source['frozen_top1_matches_previous']: raise ValueError('Need completed clean MSLS evaluation')
    for name in ['frozen_outcomes.json','provenance.json']:
        if sha(a.source/'report'/name)!=source['files'][name]: raise ValueError('Changed reference report')
    file=a.source/'frozen_descriptors.npy'
    if sha(file)!=source['arrays'][file.name]: raise ValueError('Changed frozen descriptors')
    desc=np.load(file,mmap_mode='r')
    if desc.shape!=(19611,12288): raise ValueError('Wrong database shape')
    seed(42); d=device(a.device); model,_,digest=load_ru(a,d)
    provenance=read(a.source/'report/provenance.json')
    if digest!=provenance['init_sha256'] or provenance['cache_sha256']!=cached.hash: raise ValueError('Wrong RU/reference space')
    old={r['query_index']:r for r in read(a.source/'report/frozen_outcomes.json')}
    database=torch.from_numpy(np.array(desc[:18871])).to(d)
    gt=cached.contract['info']['gt']; rows=[]; diagnostics=[]
    a.output.mkdir(parents=True,exist_ok=False); report=a.output/'report'; report.mkdir()
    with torch.inference_mode():
        for i in [0,9435,18870]:
            x,_=image_tensor(a.msls_root,cached.records[i][0],cached.image_hashes[i])
            if not np.allclose(model(x[None].to(d)).cpu().numpy()[0],desc[i],atol=2e-5,rtol=2e-4):
                raise ValueError('Database sentinel mismatch')
        vectors=np.lib.format.open_memmap(a.output/'descriptors.npy',mode='w+',dtype='float32',shape=(len(cases),5,12288))
        scores_all=np.lib.format.open_memmap(a.output/'scores.npy',mode='w+',dtype='float32',shape=(len(cases),5,18871))
        for i,case in enumerate(cases):
            with Image.open(a.challenge/'images'/f'{case["id"]}.png') as im: x=tensor(np.array(im.convert('RGB')))
            with np.load(a.challenge/'images'/f'{case["id"]}.npz') as z: known=z['mask'].copy()
            clipmap=np.repeat(np.repeat(maps[i].astype('float32'),14,0),14,1)
            clipshift=np.repeat(np.repeat(shift_scores(maps[i],case['id']),14,0),14,1)
            knownshift=shifted(known,case['id'])
            masks=[np.zeros((280,280),np.float32),clipmap,clipshift,known,knownshift]
            images=torch.stack([x*(1-.5*torch.from_numpy(m.astype('float32'))[None]) for m in masks]).to(d)
            ds=torch.cat([model(images[j:j+1]) for j in range(5)])
            if not torch.isfinite(ds).all() or not torch.allclose(ds.norm(dim=1),torch.ones(5,device=d),atol=2e-4):
                raise ValueError('Invalid descriptors')
            scores=(ds@database.T).cpu().numpy(); vectors[i]=ds.cpu().numpy(); scores_all[i]=scores
            for j,scheme in enumerate(SCHEMES):
                row={'case':case['id'],'query':case['query'],'content':case['content'],'seed':case['seed'],
                     'load':case['load'],'scheme':scheme,**outcome(scores[j],gt[case['query']])}
                if case['content']=='clean' and scheme=='none':
                    if row['top1_reference_index']!=old[case['query']]['top1_reference_index']:
                        raise ValueError('Clean subset baseline changed')
                    if not np.allclose(vectors[i,0],desc[18871+case['query']],atol=2e-5,rtol=2e-4):
                        raise ValueError('Clean preprocessing mismatch')
                rows.append(row)
            diagnostics.append({'case':case['id'],'known_area':float(known.mean()),'clip_mean':float(clipmap.mean()),
                'clip_inside':float(clipmap[known].mean()) if known.any() else None,
                'clip_outside':float(clipmap[~known].mean()),
                'clip_mass_on_paste':float(clipmap[known].sum()/max(float(clipmap.sum()),1e-12)),
                'known_shift_overlap':float((known&knownshift).sum()/max(int(known.sum()),1)),
                'clip_shift_soft_overlap':float(np.minimum(clipmap,clipshift).sum()/max(float(clipmap.sum()),1e-12))})
            if (i+1)%16==0: print(f'Challenge evaluation {i+1}/{len(cases)}',flush=True)
        vectors.flush(); scores_all.flush()
    write(report/'outcomes.json',rows); write(report/'summary.json',summary(rows)); write(report/'localization.json',diagnostics)
    write(report/'contract.json',{'code':code(),'challenge_completed_sha256':sha(a.challenge/'completed.json'),
          'clip_completed_sha256':sha(a.clip_cache/'completed.json'),'reference_completed_sha256':sha(a.source/'completed.json'),
          'ru_sha256':digest,'strength':.5,'schemes':SCHEMES,'queries':c['queries'],
          'note':'Same attenuation operation, full pixel paste support versus 20x20 CLIP coverage. Not equal suppression budget. CLIP shift wraps; known shift does not. Clean known masks empty.'})
    complete(report)
    write(a.output/'completed.json',{'complete':True,'report_completed_sha256':sha(report/'completed.json'),
          'arrays':{p.name:sha(p) for p in a.output.glob('*.npy')}})
    print('Download:',report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['bank','build','clip','evaluate'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--donors',type=Path,default=Path('.cache/dynamic_challenge_donors_v1'))
    p.add_argument('--challenge',type=Path,default=Path('doc/dynamic_challenge_images_v1'))
    p.add_argument('--clip-cache',type=Path,default=Path('.cache/dynamic_challenge_clip_v1'))
    p.add_argument('--gsv-cache',type=Path,default=Path('.cache/clearclip_token_drop_train_v1'))
    p.add_argument('--msls-cache',type=Path,default=Path('.cache/clearclip_token_drop_msls_v1'))
    p.add_argument('--gsv-root',type=Path,default=Path('datasets/gsv_cities'))
    p.add_argument('--msls-root',type=Path,default=Path('datasets/msls-val'))
    p.add_argument('--source',type=Path,default=Path('doc/dynamic_invariance_msls_v1'))
    p.add_argument('--run',type=Path,default=Path('doc/dynamic_category_prior_screen_b0.5_full_db_condition_queries'))
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--seg-weights',type=Path)
    p.add_argument('--device',default='cuda:1')
    a=p.parse_args()
    if a.output.exists(): p.error('Output exists; choose a fresh directory')
    globals()[a.stage](a)


if __name__=='__main__': main()
