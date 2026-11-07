"""Midpoint diffusion with integrated exact first-passage bins.

DESIGN CHECK: image and spectral expansions agree in their overlap; integrated
survival gives the analytic mean at zero and nonzero drift. Review integrates
the independent density expression over each observed bin. No Euler paths.
Personal parameters have a finite uniform 3x3x3 prior over the stated ranges.
Ordinary binary menus use a fixed public utility vector (.5,.3,.2).
"""
import math,itertools
from functools import lru_cache
import numpy as np
from scipy.special import ndtr,expit,logsumexp
from scipy.integrate import quad
from scipy.optimize import brentq
from .common import rng,measure,posteriors,margin,sustained,close

TYPES=('A90','A100','L','Ls','Ah','N')
PARAMETERS=tuple(itertools.product((.5,1.25,2.),(.8,1.2,1.6),(.2,.3,.4)))


def short_cdf(t,v,a):
    if t<=0:return 0.
    v=abs(v);answer=0.;factor=2*math.cosh(v*a/2)
    for k in range(10000):
        b=(2*k+1)*a/2
        value=factor*(math.exp(-b*v)*ndtr((v*t-b)/math.sqrt(t))+math.exp(min(700,b*v))*ndtr((-v*t-b)/math.sqrt(t)))
        answer+=(-1)**k*value
        if value<1e-14:return answer
    raise ArithmeticError('image series did not converge')


def survival(t,v,a):
    if t<=0:return 1.
    if t/a**2<.12:return 1-short_cdf(t,v,a)
    factor=2*math.pi/a**2*math.cosh(v*a/2);answer=0.
    for j in range(10000):
        k=2*j+1;lam=(v*v+(k*math.pi/a)**2)/2
        value=factor*k*math.exp(-lam*t)/lam;answer+=(-1)**j*value
        # Positive terms decrease here; alternating remainder is bounded by
        # the next term. This tighter threshold is below the packet's 1e-10.
        if value<1e-14:return answer
    raise ArithmeticError('spectral series did not converge')


def density(t,v,a):
    if t<=0:return 0.
    if t/a**2<.12:
        total=0.
        for j in range(100):
            b=(2*j+1)*a/2;value=2*b/math.sqrt(2*math.pi*t**3)*math.exp(-b*b/(2*t))
            total+=(-1)**j*value
            if value<1e-16:break
        return math.cosh(v*a/2)*math.exp(-v*v*t/2)*total
    total=0.
    for j in range(100):
        k=2*j+1;value=2*math.pi/a**2*math.cosh(v*a/2)*k*math.exp(-(v*v+(k*math.pi/a)**2)*t/2)
        total+=(-1)**j*value
        if value<1e-16:break
    return total


def interval(lo,hi,v,a,independent=False):
    lo=max(0.,lo);hi=max(0.,hi)
    if hi<=lo:return 0.
    if independent:
        cut=.12*a*a
        return quad(lambda t:density(t,v,a),lo,hi,epsabs=2e-14,epsrel=2e-12,points=[cut] if lo<cut<hi else None)[0]
    if hi/a**2<.12:return short_cdf(hi,v,a)-short_cdf(lo,v,a)
    return survival(lo,v,a)-survival(hi,v,a)


@lru_cache(maxsize=250000)
def bin_mass(index,v,a,t0,jitter,independent=False):
    lo=index*.01;hi=lo+.01
    if not jitter:return interval(lo-t0,hi-t0,v,a,independent)
    half=jitter*t0
    return quad(lambda shift:interval(lo-t0-shift,hi-t0-shift,v,a,independent),-half,half,
                epsabs=1e-13,epsrel=1e-10,points=[x for x in (lo-t0,hi-t0) if -half<x<half])[0]/(2*half)


def draw_time(r,v,a,t0,jitter):
    u=r.random();hi=a*a
    while survival(hi,v,a)>u:hi*=2
    t=brentq(lambda t:survival(t,v,a)-u,0,hi,xtol=1e-13)
    return int(math.floor((t+t0+r.uniform(-jitter*t0,jitter*t0))/.01))


def drift(kind,price,w,speed,ordinary=None):
    if ordinary is not None:return speed*ordinary*(3 if kind=='Ah' else 1),False
    locked=kind in ('L','Ls') or (kind=='N' and w)
    if locked:return speed*(.3 if kind=='Ls' else 3),True
    weight=1. if kind=='A100' else .5 if kind=='N' else .9
    return speed*(weight-(1-weight)*price)*(3 if kind=='Ah' else 1),False


def probability(kind,price,w,param,law,ordinary=None):
    speed,a,t0=param;v,locked=drift(kind,price,w,speed,ordinary)
    return v,(.99 if locked and law=='locked_choices' else .01+.98*expit(v*a))


def generate(s,identity):
    if s.get('choice_law') not in ('locked_choices','diffusion_choices'):raise ValueError('K1 choice-law decision still required')
    r=rng(identity);param=PARAMETERS[int(r.integers(len(PARAMETERS)))];kind=s['truth']
    if s.get('shared_speed'):param=(1.25,param[1],param[2])
    if s.get('upper_boundary_shed'):
        param=(param[0],float(r.choice([.8,1.2])),param[2])
    offers=[];ordinary=[]
    for i in range(64):
        menu=r.integers(0,3,(2,3))/2;du=float((menu[0]-menu[1])@np.array([.5,.3,.2]))
        v,p=probability(kind,0,False,param,s['choice_law'],du)
        ordinary.append([du,int(r.random()<p),draw_time(r,v,param[1],param[2],s['jitter'])])
    for i in range(s['records']):
        price=float(s['ladder'][i%len(s['ladder'])]);w=bool(r.random()<s['n_share']);v,p=probability(kind,price,w,param,s['choice_law'])
        offers.append([price,w,int(r.random()<p),draw_time(r,v,param[1],param[2],s['jitter'])])
    return dict(truth=kind,parameters=list(param),ordinary=ordinary,offers=offers)


def paths(raw,s,reader,ordinary_n=64,independent=False):
    kinds=tuple(t for t in TYPES if reader!='R2' or t not in ('Ls','Ah'))
    if s.get('type_mix')=='no-Ls-Ah':kinds=tuple(t for t in kinds if t not in ('Ls','Ah'))
    pars=PARAMETERS if reader=='R3' else (tuple(raw['parameters']),)
    if s.get('shared_speed') and reader=='R3':pars=tuple(p for p in pars if p[0]==1.25)
    if s.get('upper_boundary_shed') and reader=='R3':pars=tuple(p for p in pars if p[1]<=1.2)
    hs=list(itertools.product(kinds,pars));prior=np.array([2. if s.get('type_mix')=='double-N' and k=='N' else 1. for k,p in hs]);prior/=prior.sum()
    ll=np.zeros(len(hs))
    if reader=='R3':
        for du,choice,index in raw['ordinary'][:ordinary_n]:
            for j,(kind,p) in enumerate(hs):
                v,prob=probability(kind,0,False,p,s['choice_law'],du);b=bin_mass(index,v,p[1],p[2],s['jitter'],independent)
                ll[j]+=math.log(prob if choice else 1-prob)+(math.log(b) if b>0 else -math.inf)
    hist=[]
    for price,w,choice,index in raw['offers']:
        for j,(kind,p) in enumerate(hs):
            v,prob=probability(kind,price,w,p,s['choice_law']);ll[j]+=math.log(prob if choice else 1-prob)
            if reader!='R0':
                b=bin_mass(index,v,p[1],p[2],s['jitter'],independent);ll[j]+=math.log(b) if b>0 else -math.inf
        hist.append(ll.copy())
    logs=np.array(hist);posterior=posteriors(logs,prior,independent)
    mass=np.array([[posterior[t,[k==kind for k,p in hs]].sum() for kind in TYPES] for t in range(len(hist))])
    yes=np.array([k=='L' for k,p in hs]);no=np.array([k=='A100' for k,p in hs])
    return mass,sustained(margin(logs,prior,yes,no))


def analyze(raw,s,independent=False):
    out={};needed={}
    for reader in ('R0','R1','R2','R3'):
        prefixes=(0,16,64) if reader=='R3' else (s['ordinary'],)
        for prefix in prefixes:
            p,n=paths(raw,s,reader,prefix,independent);label=reader+('-'+str(prefix) if reader=='R3' else '')
            needed[label]=n;out[label]=dict(records=n)
            for t in (8,16,64):
                q=p[min(t,len(p))-1];locked=q[[2,3]].sum()
                out[label]['compensatory_call_'+str(t)]=float(locked<.5)
                out[label]['locked_call_'+str(t)]=float(locked>.5)
            if raw['truth'] in TYPES:
                mass=p[-1,TYPES.index(raw['truth'])];out[label]['true_probability']=float(mass)
            for j,k in enumerate(TYPES):out[label]['first_posterior_'+k]=float(p[0,j])
    out['contrasts']=dict(timing_ratio=[needed['R1'],needed['R0']])
    for t in (8,16,64):
        out['contrasts']['slow_false_compensatory_'+str(t)]=out['R2']['compensatory_call_'+str(t)]-out['R1']['compensatory_call_'+str(t)]
        out['contrasts']['practised_false_locked_'+str(t)]=out['R2']['locked_call_'+str(t)]-out['R1']['locked_call_'+str(t)]
    return out


def block(s,identity,independent=False):
    raw=[];values={};out=[]
    for k in range(s['makers']):
        o=generate(s,[identity,k]);raw.append(o)
        for reader,stats in analyze(o,s,independent).items():
            for metric,value in stats.items():values.setdefault((reader,metric),[]).append(value)
    for (reader,key),v in values.items():measure(out,key,v,'median_ratio' if key=='timing_ratio' else 'median' if key=='records' else 'mean',reader=reader)
    return dict(observations=raw,measures=out)


def controls():
    errors=[]
    for v,a in itertools.product((0.,.15,1.,3.,6.),(.8,1.2,1.6)):
        expected=a*a/4 if v==0 else a/(2*v)*math.tanh(v*a/2)
        got=quad(lambda t:survival(t,v,a),0,20*a*a,epsabs=2e-11,epsrel=2e-11,points=[.12*a*a])[0]
        errors.append(abs(got-expected));assert abs(got-expected)<1e-8
        t=.15*a*a;assert abs(short_cdf(t,v,a)+survival(t,v,a)-1)<1e-10
        for index in (25,40,70):
            x=bin_mass(index,v,a,.2,0.);y=bin_mass(index,v,a,.2,0.,True);assert abs(x-y)<1e-12
    assert expit(0)==.5
    return dict(passed=True,zero_drift_choice=.5,largest_mean_error=max(errors),overlapping_series=True,independent_density_bins=True)


def design_check(s,identity):
    weight=.5 if s['truth']=='N' else .9
    close_share=sum(abs(weight-(1-weight)*p)<=.2+1e-12 for p in s['ladder'])/len(s['ladder'])
    applies=s['truth'] in ('A90','Ah','N')
    return dict(passed=not applies or close_share>=.25,case_share=1. if close_share>=.25 else 0.,floor=.25,near_tie_offer_share=close_share,
        expected_signs=dict(R0='choices-only baseline',R1='extra timing cannot lose in expectation',R2='misspecified library',R3='learned nuisance; scale open'),scientific_draws=0)
