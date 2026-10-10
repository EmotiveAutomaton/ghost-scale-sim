"""A02 exact observation partitions; coarse class success never means exact history."""
import numpy as np
from ..v18_3.io import write
from ..v20.checker import verify_world
from . import world as W
from .studies import controls

def run(root,d,pulse):
    rows_out=[];w=dict(route=4,select=4,revise=4,goal=4,inspect=4,skill=6.5,shift=d.get('shift','native'))
    rows=W.enumerate_law(w)
    for tier in W.TIERS:
        for observed in (False,True):
            x=W.project(rows,tier,observed);keys,inv=np.unique(x,axis=0,return_inverse=True)
            classes=[]
            for i,key in enumerate(keys):
                ids=np.flatnonzero(inv==i);weight=rows[ids,-1];total=weight.sum()
                mass=np.bincount(rows[ids,10].astype(int),weights=weight,minlength=128)/total
                classes.append(dict(observed=key.tolist(),class_probability=float(total),trajectories=len(ids),
                    target_mass=mass.tolist(),supported_targets=int((mass>0).sum()),
                    exact_trajectory_modal_accuracy=float(weight.max()/total),target_modal_accuracy=float(mass.max()),
                    equivalence_class_accuracy=1.0,warning='class membership is not exact target or trajectory recovery'))
            write(root/'partitions'/f'{tier}-{observed}.json',classes)
            rows_out.append(dict(tier=tier,inspection_observed=observed,classes=len(classes),
                expected_target_accuracy=sum(c['class_probability']*c['target_modal_accuracy'] for c in classes),
                expected_history_accuracy=sum(c['class_probability']*c['exact_trajectory_modal_accuracy'] for c in classes),
                ambiguous_probability=sum(c['class_probability'] for c in classes if c['supported_targets']>1)))
            pulse(phase='exact-observation-partition',tier=tier,inspection_observed=observed)
    return dict(card='G21-A02',population=rows_out,controls=controls(),
        scope='Exact H=0 law-marginal observation classes; positive-prior law family shares support; no learned or human inference claim')
