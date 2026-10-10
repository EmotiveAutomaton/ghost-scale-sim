"""C03 common-support ladder from one unchanged converged direct predictor."""
import numpy as np
from pathlib import Path
from ..v18_3.io import read,file_digest
from . import world as W,oracle as O
from .studies import retain,controls
from .readers import Direct

def run(root,d,pulse):
    src=Path(d['forecast_root']);s=read(src/'SUMMARY.json')
    if not s.get('screen_admitted'):
        return dict(card='G21-C03',population=[],state=s['screen_disposition'],reason='GS-4 exact headroom admission gate',controls=controls())
    complete=read(src/'COMPLETE.json');population=[]
    for fit in d.get('fit_seeds',[0,1]):
        train=np.load(src/'training'/f'{fit}_points.npz');x=train['evidence'];y=train['labels']
        model=Direct.__new__(Direct);saved=np.load(src/'training'/f'{fit}-direct-model_points.npz');model.params=[saved[f'parameter_{i}'] for i in range(4)]
        learned={}
        for row,label in zip(x,y):learned.setdefault(row[:13].tobytes(),set()).add(int(label))
        for lineage in d['lineages']:
            pulse(phase='support-ladder',fit=fit,lineage=lineage)
            xx=np.load(src/'reader'/f'{lineage}_points.npz')['evidence'];e=np.load(src/'evaluator'/f'{lineage}_points.npz');yy=e['truth']
            raw=model.predict(xx);exact=np.isfinite(e['supplied_logp']);broad=np.ones_like(exact)
            no_tool=xx[:,5]==0;broad[no_tool]=W.LABELS[:,0]==0
            learned_mask=np.array([[j in learned.get(row[:13].tobytes(),range(128)) for j in range(128)] for row in xx])
            for name,mask in [('none',np.ones_like(exact)),('physical',broad),('learned',learned_mask),('supplied-exact',exact)]:
                lp=O.normalize_log(np.where(mask,raw,-np.inf));a=retain(root,f'{fit}-{lineage}-{name}',lp,yy)
                population.append(dict(fit=fit,lineage=lineage,support=name,metrics=a,truth_exclusion=float((~mask[np.arange(len(yy)),yy]).mean())))
            for rate in (0,.01,.05,.2):
                # Pruning is frozen by candidate identity, never chosen by test truth.
                rng=np.random.default_rng(W.seed('v21-support-omission',rate,fit));omitted=rng.random(128)<rate;mask=exact&~omitted
                empty=~mask.any(1);mask[empty]=exact[empty]
                a=retain(root,f'{fit}-{lineage}-omission-{rate}',O.normalize_log(np.where(mask,raw,-np.inf)),yy)
                population.append(dict(fit=fit,lineage=lineage,support='corrupted-exact',omission=rate,metrics=a,truth_exclusion=float((~mask[np.arange(len(yy)),yy]).mean())))
    return dict(card='G21-C03',population=population,controls=controls(),scope='Same converged direct predictor; support interaction diagnostic, not a structured-reader superiority result')
