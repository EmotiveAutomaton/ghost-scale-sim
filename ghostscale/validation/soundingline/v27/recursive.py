"""K6: hard latent-cause iteration with each observation counted once.

DESIGN CHECK: no-difficulty S=J and goal-independent occurrence/redraw gives the
prior. Exact joint inference is the score ceiling in expectation, not a forced
finite-sample verdict. The gain-share denominator is retained without flooring.
"""
import math
import numpy as np
from scipy.special import logsumexp
from ..v26 import pauses
from .common import rng,measure,posteriors,scores,close


def components(h,w,s,independent=False):
    d=s['d'];q=s['q'];hard=np.arange(8)<s['hard']
    if s.get('correlated'):q=q*(.5+.5*(w<1))
    if s.get('independent_delib'):redraw=h;delib=np.full(8,d)
    else:redraw=h*w/(h@w);delib=d*(w<1)
    if not independent:
        return np.stack((h*(1-delib)*(1-q*hard),h*(1-delib)*q*hard,(h@delib)*redraw))
    out=np.zeros((3,8))
    for old in range(8):
        out[0,old]+=h[old]*(1-delib[old])*(1-q[old]*hard[old] if np.ndim(q) else 1-q*hard[old])
        out[1,old]+=h[old]*(1-delib[old])*(q[old] if np.ndim(q) else q)*hard[old]
        for y in range(8):out[2,y]+=h[old]*delib[old]*redraw[y]
    return out


def library(s,identity):
    w=pauses.weights(['v27',identity],n=s['goals']);w=np.exp(np.log(w)*s['information'])
    if s.get('independent_delib'):w[:]=1
    return w,pauses.HABITS.copy()


def generate(s,identity):
    w,habits=library(s,identity);r=rng('K6-world',identity);g=int(r.integers(s['goals']));hi=int(r.integers(3))
    probs=components(habits[hi],w[g],s);obs=[];causes=[]
    for _ in range(s['records']):
        code=int(r.choice(24,p=probs.ravel()));cause,y=divmod(code,8);obs.append([y,int(cause>0)]);causes.append(cause)
    return dict(weights=w.tolist(),habits=habits.tolist(),goal=g,habit=hi,observations=obs,causes=causes)


def analyze(raw,s,independent=False):
    w=np.array(raw['weights']);habits=np.array(raw['habits']);ng=len(w)
    comp=np.array([components(h,g,s,independent) for g in w for h in habits]);prior=np.full(ng*3,1/(ng*3))
    obs=np.array(raw['observations']);paused=np.flatnonzero(obs[:,1]);y=obs[:,0]
    def infer(classes=None):
        ll=np.zeros(len(prior))
        for t,item in enumerate(y):
            prob=comp[:,0,item] if not obs[t,1] else comp[:,1:,item].sum(1) if classes is None else comp[:,2 if classes[t] else 1,item]
            with np.errstate(divide='ignore'):ll+=np.log(prob)
        if np.all(np.isneginf(ll)):raise ValueError('all hypotheses excluded by hard classification')
        return posteriors(ll[None,:],prior,independent)[0]
    def classify(p,mode):
        if mode=='point':p=np.eye(1,len(p),int(np.argmax(p)))[0]
        classes=np.zeros(len(obs),int)
        for t in paused:
            classes[t]=int(p@comp[:,2,y[t]]>=p@comp[:,1,y[t]])
        return classes
    joint=infer();initial=classify(prior,'weighted');sequential=infer(initial)
    posts={'J':joint,'S':sequential};nonconvergence={};rounds={}
    for reader,mode in [('I','weighted'),('Ic','point')]:
        classes=initial.copy();p=sequential.copy();ok=False
        for step in range(s['rounds']):
            updated=classify(p,mode)
            if np.array_equal(updated,classes):ok=True;break
            classes=updated;p=infer(classes)
        posts[reader]=p;nonconvergence[reader]=int(not ok);rounds[reader]=step+1
    out={}
    for reader,p in posts.items():
        gp=p.reshape(ng,3).sum(1);out[reader]=scores(gp,raw['goal'])|dict(log_score=float(np.log(gp[raw['goal']])),
            nonconvergence=nonconvergence.get(reader,0),rounds=rounds.get(reader,1))
    numerator=out['I']['log_score']-out['S']['log_score'];denom=out['J']['log_score']-out['S']['log_score']
    out['contrasts']=dict(primary=[numerator,denom],iterative_gain=numerator,joint_gain=denom,
        confirmation_excess=out['Ic']['half_brier']-out['J']['half_brier'])
    return out


def block(s,identity,independent=False):
    raw=[];values={};out=[]
    for k in range(s['makers']):
        o=generate(s,[identity,k]);raw.append(o)
        for reader,stats in analyze(o,s,independent).items():
            for metric,v in stats.items():values.setdefault((reader,metric),[]).append(v)
    for (reader,metric),v in values.items():measure(out,metric,v,'ratio_of_sums' if metric=='primary' else 'mean',reader=reader)
    return dict(observations=raw,measures=out)


def controls():
    s=dict(records=16,goals=4,d=.3,q=0.,hard=2,information=1.,rounds=20,correlated=False)
    raw=generate(s,['excluded-no-difficulty']);v=analyze(raw,s);close(v['J']['posterior'],v['S']['posterior'],tol=1e-12)
    s=s|dict(q=.5,independent_delib=True);raw=generate(s,['excluded-independent']);v=analyze(raw,s)
    assert abs(v['I']['posterior']-.25)<1e-12 and abs(v['S']['posterior']-.25)<1e-12
    s=s|dict(independent_delib=False);raw=generate(s,['excluded-enumeration'])
    close(analyze(raw,s),analyze(raw,s,True),tol=1e-12)
    return dict(passed=True,no_difficulty_sequential_equals_joint=True,independent_occurrence_redraw_prior=True,enumerated_likelihoods_match=True)


def design_check(s,identity):
    shares=[]
    for k in range(128):
        o=generate(s,[identity,'excluded',k]);causes=np.array(o['causes']);n=np.sum(causes>0)
        shares.append(bool(n and np.sum(causes==1)/n>=.25))
    return dict(passed=np.mean(shares)>=.25,case_share=float(np.mean(shares)),floor=.25,worlds=128,
        expected_signs={'J_minus_S':'nonnegative expected proper-score gain','I_minus_S':'open; hard iteration may lose',
            'Ic_minus_J_half_brier':'nonnegative in expectation; 0.02 bar not fixed'},scientific_draws=0)
