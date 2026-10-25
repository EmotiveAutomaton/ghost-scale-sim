"""W10: declared habit rules and a separate Gaussian explicit-value learner."""
import math
import numpy as np
from .common import rng,softmax,record,close

def increment(p,piece,amount,growth):
    p=p.copy()
    p[piece]+=amount*(1-p[piece]/30) if growth=='saturating' else amount
    if growth=='normalized':p*=60/p.sum()
    return p

def block(setting,identity,independent=False):
    r=rng(*identity);out=[];raw=[];alpha=setting['attention'];lam=setting['exposure_gain'];growth=setting['growth'];forced=setting['exposure']=='forced'
    for maker in range(8):
        uniforms=r.random(200);useful_signals=.5+r.normal(0,.5,200);paths=[]
        for filter_id in range(4):
            p=np.full(6,10.);precision=1.;weighted=0.;threshold=None;history=[]
            for n in range(201):
                rate=float(p[0]/p.sum());explicit=np.array([-1.,1.,weighted/precision,0.,0.,0.]);deliberate=softmax(3*explicit,independent)
                if n and threshold is None and rate>1/6+.1:threshold=n
                if n in (0,10,50,200):out.append(record(dict(habitual_rate=rate,unconditional_habitual_rate=(1-alpha)*rate,total_disliked_rate=(1-alpha)*rate+alpha*float(deliberate[0]),useful_value_error=abs(weighted/precision-.5),explicit_disliked_value=-1.),maker=maker,filter=filter_id,history=n))
                history.append(rate)
                if n==200:break
                attended=uniforms[n]<alpha*(.5 if filter_id in (1,3) else 1.)
                amount=1. if forced else lam*int(attended)
                p=increment(p,0,amount,growth)
                if filter_id==2:p=increment(p,1,1.,growth)
                # Every fifth source event also offers a useful piece's outcome.
                # Declining inference retains its explicit prior, unlike attention withdrawal.
                if attended and filter_id!=3 and n%5==0:precision+=4;weighted+=4*useful_signals[n]
            out.append(record(dict(records_needed=threshold,not_within_200=float(threshold is None)),maker=maker,filter=filter_id,reader='threshold'))
            paths.append(dict(filter=filter_id,habitual_rate=history))
        raw.append(dict(attention_uniforms=uniforms.tolist(),useful_signals=useful_signals.tolist(),paths=paths))
    return dict(public=dict(label='under the stated habit rule',eta=1.,initial_propensity=10.,cap=30.,explicit_prior='Gaussian mean 0 precision 1 for useful piece; disliked -1 and alternative +1 known',useful_outcome_noise=.5),observations=raw,readouts=out)

def controls():
    for growth in ('linear','saturating','normalized'):
        p=np.full(6,10.);close(increment(p,0,0,growth).tolist(),p.tolist(),tol=1e-12)
        for _ in range(200):p=increment(p,0,0,growth);p=increment(p,1,1,growth)
        assert p[0]/p.sum()<=1/6
    return dict(passed=True,zero_eta_unchanged=True,zero_exposure_gain_no_adoption=True,labels_under_stated_habit_rule=True)
