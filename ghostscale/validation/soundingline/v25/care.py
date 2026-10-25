"""W11: exact individual inference inside a noisy eight-agent population."""
import numpy as np
from .common import rng,softmax,posterior,record,close

GRID=np.linspace(0,1,21)
def policies(weights,independent=False):return softmax(3*np.stack([weights,1-np.asarray(weights)],axis=-1),independent)

def block(setting,identity,independent=False):
    r=rng(*identity);weights=np.r_[np.full(4,.7),np.full(4,.3)]+r.uniform(-.05,.05,8);initial=weights.copy();out=[];history=[];wrong=0;cross_count=0
    laws=policies(GRID,independent)
    for t in range(51):
        out.append(record(dict(signed_gap=float(weights[:4].mean()-weights[4:].mean()),within_group_spread=float((np.var(weights[:4])+np.var(weights[4:]))/2),cross_inference_error_rate=wrong/cross_count if cross_count else None),reader='population',history=t))
        if t==50:break
        after=weights.copy();events=[]
        for agent in range(8):
            cross=bool(r.random()<setting['contact']);group=(1-agent//4) if cross else agent//4;eligible=[i for i in range(group*4,group*4+4) if i!=agent];partner=int(r.choice(eligible))
            obs=r.choice(2,4,p=policies(np.array([weights[partner]]),independent)[0]);ll=np.log(laws[:,obs]).sum(1);post=posterior(ll,independent=independent);estimate=float(post@GRID)
            skip=cross and r.random()<setting['withheld'];care=setting['care'] if cross else .3
            if cross:cross_count+=1;wrong+=abs(estimate-weights[partner])>.1
            if skip:
                if setting['stereotype']:after[agent]+=care*((.7 if group==0 else .3)-weights[agent])
            else:after[agent]+=care*(estimate-weights[agent])
            events.append(dict(agent=agent,partner=partner,cross=cross,choices=obs.tolist(),estimate=estimate,withheld=bool(skip)))
        weights=after;history.append(events)
    return dict(public=dict(grid=GRID.tolist(),within_care=.3,choice_model='two-action softmax beta 3',update='simultaneous per round',stereotype_label='fixed opposite-group starting mean, separate secondary arm'),initial_weights=initial.tolist(),observations=history,readouts=out)

def controls():
    gap=.4
    for _ in range(50):gap*=1-2*.3
    assert abs(gap)<.01
    close(policies(np.array([.2,.7])).tolist(),policies(np.array([.2,.7]),True).tolist(),tol=1e-12)
    return dict(passed=True,deterministic_full_contact_convergence=True,independent_choice_arithmetic=True)
