"""N6 finite-horizon Bayesian decision policy, including evidence acquisition.

The policy may stop observing permanently while retaining its present belief and
choosing uptake. Each optional observation costs .001. Technique pays +1 for
competence and -1 for failure; purpose uptake costs 1 for cold makers, including
purpose carried by coupled technique. Purpose has no separate positive reward
in the commissioned payoff, so a payoff maximizer never adopts it separately.
Terminal next-action logarithmic reward has weight stakes on low-low makers and
one on other makers. This is an exact finite constructed policy, not a claim
about physical procedure execution or human rationality.
"""
from functools import lru_cache
from itertools import product
import numpy as np
from .common import normalize

STATES=np.array(list(product((0,1),(0,1),range(16))))
WARM,COMP,LAW=STATES.T
REL=np.linspace(.6,.95,16)[LAW]
PW=np.where(WARM,REL,1-REL);PC=np.where(COMP,REL,1-REL)
OBS=np.stack([(1-PW)*(1-PC),(1-PW)*PC,PW*(1-PC),PW*PC],axis=1)


def beliefs(share,t,claims,procedures):
    prior=np.where((WARM==0)&(COMP==0),share,(1-share)/3)/16
    k=np.asarray(claims)[...,None];j=np.asarray(procedures)[...,None]
    log=np.log(prior)+k*np.log(PW)+(t-k)*np.log1p(-PW)+j*np.log(PC)+(t-j)*np.log1p(-PC)
    return normalize(log)


def utilities(post,stakes,coupled):
    technique=post@(2*COMP-1-coupled*(1-WARM))
    weights=np.where((WARM==0)&(COMP==0),stakes,1.)
    weighted=(post*weights)@OBS
    total=weighted.sum(-1,keepdims=True)
    # There is positive posterior mass on non-low-low states at every finite
    # count, even when stakes is zero. No artificial score floor is needed.
    forecast=weighted/total
    terminal=np.sum(weighted*np.log(forecast),axis=-1)
    return np.maximum(0,technique),forecast,terminal


@lru_cache(None)
def policy(share,stakes,horizon,coupled):
    values={};acquire={}
    for t in range(horizon,-1,-1):
        k,j=np.meshgrid(np.arange(t+1),np.arange(t+1),indexing='ij')
        post=beliefs(share,t,k,j);reward,_,terminal=utilities(post,stakes,coupled)
        stop=(horizon-t)*reward+terminal
        if t==horizon:values[t]=stop;acquire[t]=np.zeros_like(stop,dtype=bool);continue
        probability=post@OBS;continuation=np.full_like(stop,-.001)
        for observation in range(4):
            kk,jj=k+observation//2,j+observation%2
            after=beliefs(share,t+1,kk,jj)
            uptake,_,_=utilities(after,stakes,coupled)
            continuation+=probability[...,observation]*(uptake+values[t+1][kk,jj])
        acquire[t]=continuation>stop;values[t]=np.maximum(stop,continuation)
    return dict(values=values,acquire=acquire,expected_initial_payoff=float(values[0][0,0]),
                scope='optimal over permanent-stop versus observe, and purpose/technique uptake at each step')


def evaluate(v,p,truth=None):
    s=v['settings'];h=len(p['history']);table=policy(s['lowlow_share'],s['stakes'],h,s['coupled'])
    k=j=t=0;stopped=False;technique=0;path=[];post=beliefs(s['lowlow_share'],0,0,0)
    for row in p['history']:
        observe=not stopped and bool(table['acquire'][t][k,j])
        if observe:k+=row['claim'];j+=row['procedure'];t+=1
        else:stopped=True
        post=beliefs(s['lowlow_share'],t,k,j)
        expected_gain=float(post@(2*COMP-1-s['coupled']*(1-WARM)))
        adopt=expected_gain>0;technique+=adopt
        path.append(dict(observed=observe,technique_accepted=adopt,purpose_accepted=False,
                         observations=t,claims=k,procedures=j))
    _,forecast,_=utilities(post,s['stakes'],s['coupled'])
    detail=dict(path=path,updates=t,adopted_techniques=technique,
        adopted_purposes=technique*int(s['coupled']),expected_initial_payoff=table['expected_initial_payoff'],
        finite_horizon=h,exact_decision_policy=True)
    if truth is not None:
        detail['useful_technique']=technique*truth['competence']
        detail['failed_technique']=technique*(1-truth['competence'])
        detail['unwanted_uptake']=detail['adopted_purposes']*(1-truth['warmth'])
        detail['nonprediction_payoff']=detail['useful_technique']-detail['failed_technique']-detail['unwanted_uptake']-.001*t
    return forecast[None,:],detail


def verify_case(v,p,truth,pred,detail):
    """Independent count likelihoods and observed-action replay, no truth input to policy."""
    s=v['settings'];mass=np.array([(s['lowlow_share'] if w==c==0 else (1-s['lowlow_share'])/3)/16 for w,c,l in STATES])
    table=policy(s['lowlow_share'],s['stakes'],len(p['history']),s['coupled'])
    observed=0;k=j=0;stopped=False;adoptions=0
    for row,step in zip(p['history'],detail['path'],strict=True):
        act=not stopped and bool(table['acquire'][observed][k,j])
        if act:
            for i,(w,c,l) in enumerate(STATES):
                rel=.6+.35*l/15
                mass[i]*=(rel if row['claim']==w else 1-rel)*(rel if row['procedure']==c else 1-rel)
            mass/=mass.sum();observed+=1;k+=row['claim'];j+=row['procedure']
        else:stopped=True
        gain=sum(q*(2*c-1-s['coupled']*(1-w)) for q,(w,c,l) in zip(mass,STATES))
        adopt=gain>1e-12 if abs(gain)<1e-12 else gain>0
        # Exactly tied uptake can differ by harmless roundoff; reject only if the
        # chosen action has genuinely negative expected payoff, never score loss.
        if step['observed']!=act or (step['technique_accepted']!=adopt and abs(gain)>1e-12):
            raise ValueError('payoff policy path mismatch')
        adoptions+=step['technique_accepted']
    target=np.zeros(4)
    for q,(w,c,l) in zip(mass,STATES):
        rel=.6+.35*l/15;weight=s['stakes'] if w==c==0 else 1
        for a in range(4):target[a]+=q*weight*(rel if a//2==w else 1-rel)*(rel if a%2==c else 1-rel)
    target/=target.sum()
    if not np.allclose(pred,target[None,:],atol=1e-11,rtol=0) or detail['updates']!=observed or detail['adopted_techniques']!=adoptions:
        raise ValueError('payoff policy reconstruction mismatch')
    return float(np.max(np.abs(pred-target[None,:])))


@lru_cache(None)
def verify_policy(share,stakes,horizon,coupled):
    """Check every Bellman equation by independent probability and action enumeration."""
    table=policy(share,stakes,horizon,coupled);states=list(product((0,1),(0,1),range(16)))
    prior=np.array([(share if w==c==0 else (1-share)/3)/16 for w,c,l in states])
    claim=np.array([(.6+.35*l/15) if w else (1-(.6+.35*l/15)) for w,c,l in states])
    procedure=np.array([(.6+.35*l/15) if c else (1-(.6+.35*l/15)) for w,c,l in states])
    outcome=np.array([[(claim[i] if a//2 else 1-claim[i])*(procedure[i] if a%2 else 1-procedure[i]) for a in range(4)] for i in range(len(states))])
    rewards=np.array([[tech*(2*c-1)-(purpose or coupled and tech)*(1-w) for w,c,l in states]
                      for purpose,tech in product((0,1),(0,1))])
    weights=np.array([stakes if w==c==0 else 1. for w,c,l in states])
    def mass(t,k,j):
        values=prior*claim**k[...,None]*(1-claim)**(t-k[...,None])*procedure**j[...,None]*(1-procedure)**(t-j[...,None])
        return values/values.sum(-1,keepdims=True)
    largest=0.;states_checked=0
    for t in range(horizon,-1,-1):
        k,j=np.meshgrid(np.arange(t+1),np.arange(t+1),indexing='ij');post=mass(t,k,j)
        weighted=(post*weights)@outcome;forecast=weighted/weighted.sum(-1,keepdims=True)
        stop=(horizon-t)*np.max(post@rewards.T,axis=-1)+np.sum(weighted*np.log(forecast),axis=-1)
        best=stop
        if t<horizon:
            continuation=np.full_like(stop,-.001);likelihood=post@outcome
            for a in range(4):
                kk,jj=k+a//2,j+a%2;updated=mass(t+1,kk,jj)
                continuation+=likelihood[...,a]*(np.max(updated@rewards.T,axis=-1)+table['values'][t+1][kk,jj])
            best=np.maximum(stop,continuation)
            wrong=(table['acquire'][t]!=(continuation>stop)) & (np.abs(continuation-stop)>1e-10)
            if np.any(wrong):raise ValueError('Bellman acquisition action mismatch')
        largest=max(largest,float(np.max(np.abs(best-table['values'][t]))));states_checked+=best.size
    if largest>1e-10:raise ValueError('independent Bellman value mismatch')
    return dict(state='passed',states=states_checked,max_error=largest,all_actions_enumerated=True)
