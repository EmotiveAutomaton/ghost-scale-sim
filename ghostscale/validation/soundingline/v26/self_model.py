"""E2: matched evidence under self, population and ideal priors."""
import math,itertools
import numpy as np
from .common import rng,softmax,posterior,scores,record,close,simplex
from ..v25.pieces import library,TRAIN

def laws(e,w,independent=False):
    if independent:return np.array([softmax([3*math.fsum(float(x*y) for x,y in zip(weight,ee)) for ee in e],True) for weight in w])
    return softmax(3*w@e.T)

def block(setting,identity,independent=False):
    r=rng(*identity);dims=setting.get('ends',3);weights=np.array([v for v in itertools.product(range(5),repeat=dims) if sum(v)==4],float)/4;nh=len(weights)
    e=library('partial',r) if dims==3 else r.choice([0.,.5,1.],(6,dims));own=e.copy();overlap=setting['overlap'];kappa=setting['kappa']
    for i in range(overlap,6):own[i]=r.choice([0.,.5,1.],dims)
    if setting.get('missing_piece'):own[-1]=np.mean(own[:-1],axis=0) # declared unknown-piece outcome proxy, with positive softmax support
    true=[laws(e[list(a)],weights,independent) for a in TRAIN];wrong=[laws(own[list(a)],weights,independent) for a in TRAIN];out=[];raw=[]
    for distance in (0.,.25,.5,.75):
        for maker in range(4):
            truth=int(r.integers(nh));tv=np.abs(weights-weights[truth]).sum(1)/2
            candidates=np.flatnonzero(tv>=.75-1e-12) if distance==.75 else np.flatnonzero(abs(tv-distance)<1e-12)
            reader=int(r.choice(candidates));selfprior=softmax(-kappa*np.abs(weights-weights[reader]).sum(1)/2);popprior=softmax(-kappa*np.abs(weights-1/dims).sum(1)/2);mix=(selfprior+popprior)/2
            ll=np.zeros(nh);sl=np.zeros(nh);obs=[]
            for t in range(32):
                si=t%4;y=int(r.choice(len(TRAIN[si]),p=true[si][truth]));ll+=np.log(true[si][:,y]);sl+=np.log(wrong[si][:,y]);obs.append(y)
                if t+1 not in (1,2,4,8,16,32):continue
                ps=posterior(sl,selfprior,independent);pp=posterior(ll,popprior,independent);pi=posterior(ll,independent=independent);pm=posterior(sl,mix,independent)
                for name,p,prior in [('S',ps,selfprior),('P',pp,popprior),('I',pi,np.ones(nh)/nh),('mixture',pm,mix)]:
                    # Comparisons use posterior true-ends log score, not unequal-prior gain.
                    out.append(record(scores(p,truth,prior),maker=maker,distance=distance,reader=name,history=t+1))
                toward=weights[reader]-weights[truth];norm=float(toward@toward)
                out.append(record(dict(primary=math.log(ps[truth]/pp[truth]),egocentric_projection=float((ps@weights-weights[truth])@toward/norm) if norm else 0.,mixture_minus_population=math.log(pm[truth]/pp[truth])),maker=maker,distance=distance,reader='contrasts',history=t+1))
            raw.append(dict(distance=distance,maker=maker,truth=truth,reader_weight=reader,choices=obs))
    return dict(public=dict(true_library=e.tolist(),own_library=own.tolist(),weights=weights.tolist(),choice_model='softmax beta3, no habits; four public menus',matching_pieces=overlap),truth_and_observations=raw,readouts=out)

def controls():
    w=simplex(4);ll=np.arange(15)/10;p=posterior(ll);close(p.tolist(),posterior(ll,np.ones(15)/15,True).tolist(),tol=1e-12)
    for k in (0,2,8):
        prior=softmax(-k*np.abs(w-1/3).sum(1)/2);close(posterior(ll,prior).tolist(),posterior(ll,prior,True).tolist(),tol=1e-12)
    assert np.max(abs(softmax(np.zeros(15))-1/15))<1e-12
    return dict(passed=True,flat_prior_ideal=True,self_at_mean_equals_population=True,zero_kappa_flat=True)

def design_check(settings):
    w=simplex(4)
    for maker in range(15):
        tv=np.abs(w-w[maker]).sum(1)/2
        assert all(np.any(abs(tv-d)<1e-12) for d in (0,.25,.5)) and np.any(tv>=.75)
    return dict(passed=True,seed_domain='enumerated-E2',bin_shares=[.25]*4,all_15_makers_have_each_bin=True,floor=.2)
