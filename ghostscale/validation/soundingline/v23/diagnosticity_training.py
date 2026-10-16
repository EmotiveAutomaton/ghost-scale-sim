"""N2 training-only table cache with the original random draw order unchanged."""
from functools import lru_cache
import numpy as np
from . import diagnosticity as reference
from .common import rng


@lru_cache(None)
def prior(kind,share):
    return reference.class_prior(dict(settings=dict(mark_class=kind,nondiagnostic=share)))


@lru_cache(None)
def tables(beta,slope,law,previous,context):
    v=dict(settings=dict(beta=beta,rt_slope=slope));opponent=np.linspace(.15,.85,16)[law]
    probability=reference.probabilities(v,opponent,previous,context)
    return opponent,probability,reference.rt_prob(v,probability)


def sample(v,phase,world,case,law=None):
    r=rng(v['id'],phase,world,case);law=world%16 if law is None else law;s=v['settings']
    preference=int(r.integers(2));history=[];truth=[];previous=.5
    for _ in range(s['records']):
        kind=int(r.choice(6,p=prior(s['mark_class'],s['nondiagnostic'])))
        context=int(r.choice([-1,1]))
        opponent,probability,timing=tables(s['beta'],s['rt_slope'],law,previous,context)
        chosen=int(r.random()<probability[kind,preference]);latency=int(r.choice(5,p=timing[kind]))
        observed=kind if r.random()<.7 else int(r.choice([k for k in range(6) if k!=kind]))
        visible=bool(r.integers(2))
        d=abs(np.log(probability[kind,1] if chosen else 1-probability[kind,1])-
              np.log(probability[kind,0] if chosen else 1-probability[kind,0]))
        history.append(dict(choice=chosen,latency_bin=latency,context=context,class_cue=observed,
            opponent=opponent if visible else None,previous=previous))
        truth.append(dict(kind=kind,diagnosticity=float(d)));previous=chosen
    return dict(history=history),dict(y=[preference],marks=truth,law=int(law))
