"""K4: exact finite laws, nuisance-scale marginalization and separated channels.

DESIGN CHECK: enumerate the designed/observational mixes; fixed-outcome and
trained-policy uncertainty are exactly zero. Finite nuisance grids are checked
for actual likelihood aliasing rather than labelled equal from prose alone.
"""
import itertools,math
import numpy as np
from scipy.special import expit,logsumexp
from .common import rng,measure,posteriors,sustained,margin,scores,close

SCALES=np.geomspace(.1,10,25)


def laws(p,trained,information,magnitude=1.):
    return np.array([p*magnitude,(p>0)*magnitude]),np.array([0. if trained else math.log(3),p*(1-p),information])


def information_gain(p,accuracy):
    # One binary diagnostic action of known accuracy; the maker does not observe
    # the experimenter's latent information gain. Stress S3 may depend on it.
    q=p*accuracy+(1-p)*(1-accuracy)
    if q in (0,1):return 0.
    after1=p*accuracy/q;after0=p*(1-accuracy)/(1-q)
    return max(0.,p*(1-p)-q*after1*(1-after1)-(1-q)*after0*(1-after0))


def episode(r,s):
    trained=bool(r.random()<s['trained_share'])
    if s['mix']=='designed':p=.5 if trained else float(r.choice([0.,1.]))
    elif s['mix']=='observational':p=float(r.choice([0.,1.])) if trained else .5
    else:p=float(r.choice([0.,.1,.5,.9,1.]))
    accuracy=float(r.choice([.6,.8,.95]));info=information_gain(p,accuracy)
    return p,trained,accuracy,info


def generate(s,identity):
    r=rng('K4-world',identity);scale=float(r.choice(SCALES));records=[]
    for t in range(s['records']):
        p,trained,accuracy,info=episode(r,s);fear,stress=laws(p,trained,info,s['magnitude'])
        f=fear[s['fear']];v=stress[s['stress']]*scale
        records.append([p,int(trained),accuracy,int(r.random()<expit(2*f-1)),
            float(r.normal(v,s['latency_noise'])),int(r.random()<expit(2*v-1)),float(r.normal(f+v,1.))])
    return dict(fear=s['fear'],stress=s['stress'],scale=scale,records=records)


def likelihoods(raw,s,independent=False,channel='all'):
    hs=list(itertools.product(range(2),range(3),SCALES));ll=np.zeros(len(hs));history=[]
    for p,trained,accuracy,avoid,latency,seek,arousal in raw['records']:
        fear,stress=laws(p,bool(trained),information_gain(p,accuracy),s['magnitude'])
        if independent:
            step=[]
            for f,h,scale in hs:
                v=stress[h]*scale;value=-.5*(arousal-fear[f]-v)**2-.5*math.log(2*math.pi)
                if channel!='arousal':
                    for z in (-(2*avoid-1)*(2*fear[f]-1),-(2*seek-1)*(2*v-1)):
                        value-=max(z,0)+math.log1p(math.exp(-abs(z)))
                    value+=-.5*((latency-v)/s['latency_noise'])**2-math.log(s['latency_noise']*math.sqrt(2*math.pi))
                step.append(value)
            ll+=step
        else:
            f=np.array([fear[h[0]] for h in hs]);v=np.array([stress[h[1]]*h[2] for h in hs])
            ll+=-.5*(arousal-f-v)**2-.5*math.log(2*math.pi)
            if channel!='arousal':
                # log-sigmoid forms remain finite at the declared large scales.
                ll+=-np.logaddexp(0,-(2*avoid-1)*(2*f-1))-np.logaddexp(0,-(2*seek-1)*(2*v-1))
                ll+=-.5*((latency-v)/s['latency_noise'])**2-math.log(s['latency_noise']*math.sqrt(2*math.pi))
        history.append(ll.copy())
    return np.array(history),hs


def analyze(raw,s,independent=False):
    result={}
    for channel in (('arousal',) if s.get('channel')=='arousal' else ('all','arousal')):
        ll,hs=likelihoods(raw,s,independent,channel);prior=np.full(len(hs),1/len(hs));p=posteriors(ll,prior,independent)
        stressids=np.array([h[1] for h in hs]);fearids=np.array([h[0] for h in hs])
        ps=np.array([p[-1,stressids==i].sum() for i in range(3)]);pf=np.array([p[-1,fearids==i].sum() for i in range(2)])
        row=dict(stress_logscore=float(np.log(ps[raw['stress']])),stress_half_brier=scores(ps,raw['stress'])['half_brier'],
            fear_logscore=float(np.log(pf[raw['fear']])),fear_half_brier=scores(pf,raw['fear'])['half_brier'])
        if raw['stress'] in (0,1):row['policy_outcome_records']=sustained(margin(ll,prior,stressids==raw['stress'],stressids==1-raw['stress']))
        row['fear_records']=sustained(margin(ll,prior,fearids==raw['fear'],fearids==1-raw['fear']))
        result[channel]=row
    return result


def block(s,identity,independent=False):
    raw=[];values={};out=[]
    for k in range(s['makers']):
        o=generate(s,[identity,k]);raw.append(o)
        for reader,stats in analyze(o,s,independent).items():
            for metric,v in stats.items():values.setdefault((reader,metric),[]).append(v)
    for (reader,metric),v in values.items():measure(out,metric,v,'median' if metric.endswith('_records') else 'mean',reader=reader,fear=s['fear'],stress=s['stress'])
    return dict(observations=raw,measures=out)


def controls():
    assert laws(0,False,0)[1][1]==laws(1,False,0)[1][1]==0
    assert laws(.5,True,.1)[1][0]==0
    assert information_gain(.5,.5)==0 and abs(information_gain(.5,1)-.25)<1e-12
    s=dict(records=4,fear=1,stress=1,magnitude=1.,trained_share=.5,mix='designed',latency_noise=1.)
    o=generate(s,['excluded-fixture']);a,_=likelihoods(o,s);b,_=likelihoods(o,s,True);close(a.tolist(),b.tolist(),tol=1e-12)
    return dict(passed=True,fixed_outcome_zero=True,trained_policy_zero=True,binary_diagnostic_variance_identity=True,independent_log_likelihood=True)


def design_check(s,identity):
    pairs=[];r=rng('excluded-case',identity)
    for _ in range(4096):
        p,tr,acc,info=episode(r,s);pairs.append(laws(p,tr,info,s['magnitude'])[1][:2])
    pairs=np.array(pairs);decorrelated=bool(np.corrcoef(pairs.T)[0,1]<0)
    # The observational patterns are proportional, but finite allowed scales
    # do not necessarily contain the exact rescaling needed for equality.
    grid_residual=float(np.min(np.abs(math.log(3)*SCALES[:,None]-.25*SCALES[None,:])))
    return dict(passed=decorrelated,case_share=1. if decorrelated else 0.,floor=.25,
        expected_signs={f'F{f+1}-S{k+1}':('designed S1/S2 channels differ' if k<2 else 'S3 is a separate rival; no forced S1/S2 primary') for f in range(2) for k in range(3)},
        minimum_observational_scale_residual=grid_residual,exact_observational_alias=grid_residual<1e-12,scientific_draws=0)
