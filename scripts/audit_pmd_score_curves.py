"""Post-hoc rejection ROC/PR diagnostic; no deployment threshold fitting."""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.candidate_set_screen import read,write,sha,official,complete,npz
from scripts.adaptive_pair_budget import verified
from src.pmd_correspondence import make_views
from src.pmd_decoupled import DecoupledHead
from src.models.partial_matching import PMDPair

MODES=['partial','decoupled']
CAPS=[.01,.05,.10,.20]


def curve_report(labels,scores):
    from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score,precision_recall_curve
    labels=np.asarray(labels,bool);scores=np.asarray(scores,float)
    if labels.shape!=scores.shape or labels.ndim!=1 or not np.isfinite(scores).all() or len(np.unique(labels))!=2:raise ValueError('Invalid binary scores')
    fpr,tpr,thresholds=roc_curve(labels,scores,drop_intermediate=False)
    precision,recall,pr_thresholds=precision_recall_curve(labels,scores)
    matched={}
    for cap in CAPS:
        eligible=np.flatnonzero(fpr<=cap);j=eligible[np.argmax(tpr[eligible])]
        matched[str(cap)]=dict(actual_false_rejection=float(fpr[j]),rejection_recall=float(tpr[j]))
    # Deliberately do not export a selected deployment threshold.
    return dict(auroc=float(roc_auc_score(labels,scores)),average_precision=float(average_precision_score(labels,scores)),
        unmatched_prevalence=float(labels.mean()),recall_at_false_rejection_caps=matched),dict(fpr=fpr,tpr=tpr,precision=precision,recall=recall)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key,val in [('run','logs/pmd/decoupled_v1'),('plan','doc/candidate_hard_plan_v1'),('gsv-root','datasets/gsv_cities'),
        ('official-repo','/home/wt/workspace/Pair-VPR-official'),('audit','doc/pairvpr_official_paired_audit_v1'),('output','doc/pmd_score_curves_v1')]:
        p.add_argument('--'+key,type=Path,default=Path(val))
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new output directory')
    import torch
    from PIL import Image
    from torchvision import transforms as T
    from tqdm import tqdm
    torch.set_num_threads(4);verified(a.run);verified(a.plan);c=read(a.run/'contract.json')
    if c['plan']!=sha(a.plan/'completed.json'):raise ValueError('Plan mismatch')
    for name,h in c['code'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=h:raise ValueError('Original code changed')
    ck=torch.load(a.run/'last.pt',map_location='cpu',weights_only=True)
    base,identity=official(a);model=PMDPair(base).cuda().eval()
    if ck['contract']!=sha(a.run/'contract.json') or ck['official']!=identity:raise ValueError('Model mismatch')
    heads={}
    for mode in MODES:
        h=DecoupledHead(mode).cuda().eval();h.load_state_dict(ck['heads'][mode],strict=True);heads[mode]=h
    plan=read(a.plan/'contract.json')['plan']['dev']
    transform=T.Compose([T.Resize((406,406),interpolation=T.InterpolationMode.BILINEAR),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    data={};rows=[];report={};original=read(a.run/'final.json')
    with torch.no_grad():
        for occlude in [False,True]:
            condition='occluded' if occlude else 'crop';images=[]
            for qi in tqdm(c['ids']['dev'],desc='Score curves '+condition):
                path=(a.gsv_root/plan['queries'][qi]['path']).resolve()
                if not path.is_relative_to(a.gsv_root.resolve()) or sha(path)!=c['images'][str(path)]:raise ValueError('Image changed')
                with Image.open(path) as im:views,targets=make_views(transform(im.convert('RGB')),500000+qi,occlude)
                f,_=base(views.cuda(),None,'global');ys=[];ss={m:[] for m in MODES};locations={m:[] for m in MODES}
                for direction in [0,1]:
                    x,y=model.prefix(f[direction:direction+1],f[1-direction:2-direction]);tt=targets if direction==0 else targets[::-1]
                    ys.extend([(t<0).numpy() for t in tt])
                    for mode,h in heads.items():
                        for (assignment,dust),t in zip(h(x[:,1:],y),tt):
                            ss[mode].append(dust[0].cpu().numpy());locations[mode].append((assignment[0].argmax(-1).cpu()==t).numpy())
                labels=np.concatenate(ys);entry=dict(query=qi,condition=condition,metrics={})
                for mode in MODES:
                    ss[mode]=np.concatenate(ss[mode]);locations[mode]=np.concatenate(locations[mode])
                    entry['metrics'][mode]=curve_report(labels,ss[mode])[0]
                rows.append(entry);images.append((qi,labels,ss,locations))
            labels=np.concatenate([r[1] for r in images]);ids=np.concatenate([np.full(len(r[1]),r[0]) for r in images]);report[condition]={}
            for mode in MODES:
                scores=np.concatenate([r[2][mode] for r in images]);location=np.concatenate([r[3][mode] for r in images])
                # Reproduce original fixed .5 counts before interpreting curves.
                counts=dict(matched_correct=int((~labels&location&(scores<=.5)).sum()),rejected_unmatched=int((labels&(scores>.5)).sum()),false_rejected_matched=int((~labels&(scores>.5)).sum()))
                if any(original[condition][mode][k]!=v for k,v in counts.items()):raise ValueError('Original metrics do not reproduce')
                result,curves=curve_report(labels,scores);result['reproduced_counts']=counts
                report[condition][mode]=result
                data[condition+'_'+mode]=dict(query_ids=ids,unmatched=labels,scores=scores,location_correct=location,**curves)
            # Resample source images, not individual correlated patches.
            subset=[r for r in rows if r['condition']==condition]
            delta=np.array([r['metrics']['decoupled']['auroc']-r['metrics']['partial']['auroc'] for r in subset])
            rng=np.random.default_rng(42);boot=delta[rng.integers(0,len(delta),(2000,len(delta)))].mean(1)
            report[condition]['image_macro_auroc_difference']=dict(decoupled_minus_partial=float(delta.mean()),
                descriptive_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),images=len(delta),
                note='Source-image resampling within this exposed synthetic sample, not population or VPR confirmation')
    a.output.mkdir(parents=True)
    for name,values in data.items():npz(a.output/(name+'.npz'),**values)
    write(a.output/'summary.json',dict(metrics=report,scope='Descriptive curves on existing exposed dev. FPR caps are empirical ROC operating points, NOT calibrated deployable thresholds. No model/threshold selection or training. AP positive class is unmatched.',false_rejection_caps=CAPS))
    write(a.output/'per_image.json',rows);write(a.output/'provenance.json',dict(run=sha(a.run/'completed.json'),checkpoint=sha(a.run/'last.pt'),script=sha(Path(__file__))))
    complete(a.output);print(report,flush=True)


if __name__=='__main__':main()
