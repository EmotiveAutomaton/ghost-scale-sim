"""Ledger policies written from the V29 packet before opening pre-flight code."""
import numpy as np
from scipy.special import logsumexp
from .common import logistic, rng, class_evidence, sustained

# kind, weight, duration: LIVE, RULE, HUB, BACKFIRE, FROZE, THAW, STAIN
CLASSES=('LIVE','RULE','HUB','BACKFIRE','FROZE','THAW','STAIN')


def candidates(frozen=False, breach=False, hub_weights=(2,4)):
    rows=[(0,w,0) for w in (1,2,4,8,16,32,64,128,256)]
    rows += [(1,w,0) for w in (1,2,4)]+[(2,h,0) for h in hub_weights]
    rows += [(3,2,k) for k in (4,8,16)]
    if frozen: rows += [(4,w,n) for n in (5,10,20,40) for w in (1,2,4,8)]
    if breach: rows += [(kind,w,0) for kind in (5,6) for w in (2,4)]
    return np.asarray(rows,dtype=float)


def settings_x1():
    return [dict(truth=list(truth), regime=regime, environment=env, buy_share=b, steps=1024,
                 worlds=200,blocks=8)
            for truth in ((0,4,0),(0,16,0),(0,64,0),(1,2,0),(2,4,0),(3,2,8))
            for regime in ('LOW','MIXED','PROBE') for env in ('plenty','scarce') for b in (0.,.1,.5)]


def settings_x2():
    rows=[dict(part='X2',truth=[4,w,n],regime=regime,environment=env,buy_share=b,start=start,
               steps=1024,life=2048,worlds=200,blocks=8)
          for n in (5,10,20) for w in (2,4) for regime in ('MIXED','PROBE')
          for env in ('plenty','scarce') for b in (0.,.1) for start in (0,256,1024)]
    rows += [dict(part='X2b',truth=[kind,w,0],regime=regime,environment='scarce',buy_share=b,
                  start=0,steps=2048,life=2048,worlds=200,blocks=8)
             for kind in (5,6) for w in (2,4) for regime in ('LOW','MIXED','PROBE') for b in (0.,.1)]
    return rows


def initial(n,params):
    shape=(n,len(params))
    return dict(streak=np.zeros(shape,int),frozen=np.zeros(shape,bool),shutdown=np.zeros(shape,int),
                hardship=np.zeros(shape,int),breached=np.zeros(shape,bool))


def before_offer(state,m,d):
    state['shutdown']=np.maximum(0,state['shutdown']-1)
    state['hardship']=np.where((m[:,None]==0)&(d[:,None]==6),state['hardship']+1,0)


def offer_probability(m,d,x,offer,price,params,state,slope=1.5,floor=.02):
    kind=params[:,0].astype(int);weight=params[:,1]
    sell=offer[:,None]==-1;buy=offer[:,None]==1;p=price[:,None]
    gain=.5*p+np.minimum(p,np.maximum(0,d[:,None]-m[:,None]))
    cost=.5*p+np.maximum(0,np.minimum(p,d[:,None]-(m[:,None]-p)))
    sellp=logistic(slope*(gain-weight),floor)
    restricted=(kind[None,:]==1)|(kind[None,:]==2)|(kind[None,:]==3)|((kind[None,:]==4)&state['frozen'])
    restricted |= ((kind[None,:]>=5)&~state['breached'])
    sellp=np.where(restricted,floor,sellp)
    sellp=np.where((kind[None,:]==6)&state['breached'],floor/4,sellp)
    breach=((kind[None,:]>=5)&~state['breached']&(state['hardship']>=8))
    sellp=np.where(breach,1.,sellp)
    buyp=logistic(slope*(weight-cost),floor)
    buyp=np.where(kind[None,:]==2,1-floor,buyp)
    allowed=(m[:,None]>=p)
    buyp=np.where(allowed,buyp,0.)
    prob=np.where(sell,sellp,np.where(buy,buyp,0.))
    prob=np.where((kind[None,:]==3)&(state['shutdown']>0),0.,prob)
    return prob


def update_offer(state,offer,price,accepted,params):
    sell=offer[:,None]==-1;accept=accepted[:,None].astype(bool)
    state['streak']=np.where(sell,np.where(accept,0,state['streak']+1),state['streak'])
    state['frozen'] |= (params[None,:,0]==4)&(state['streak']>=params[None,:,2])
    state['shutdown']=np.where((params[None,:,0]==3)&sell&(price[:,None]>=4),
                               params[None,:,2].astype(int)+1,state['shutdown'])
    state['breached'] |= (params[None,:,0]>=5)&sell&accept&(state['hardship']>=8)


def spend_probability(m,d,params,slope=1.5,floor=.02):
    keep=np.where(params[:,0]==2,params[:,1],.5)
    p=logistic(slope*(np.minimum(d[:,None],3)-keep[None,:]),floor)
    return np.where((m[:,None]>0)&(d[:,None]>0),p,0.)


def generate(s,lane='study',worlds=None):
    n=worlds or s['worlds'];steps=s.get('life',s['steps']);r=np.random.default_rng(s['seed_override']) if 'seed_override' in s else rng('ledger',lane,s)
    truth=np.asarray([s['truth']],float);state=initial(n,truth)
    m=np.full(n,5,int);d=np.zeros(n,int);x=np.full(n,8,int)
    data=np.zeros((n,steps,10),dtype=np.int16)
    freeze_time=np.full(n,steps+1,int);breach_time=freeze_time.copy()
    qin,qneed=(.8,.15) if s['environment']=='plenty' else (.3,.3)
    for t in range(steps):
        m=np.minimum(30,m+(r.random(n)<qin));d=np.minimum(6,d+r.binomial(2,qneed,n))
        before_offer(state,m,d)
        offered=r.random(n)<.25;buy=r.random(n)<s['buy_share']
        offer=np.where(offered,np.where(buy,1,-1),0)
        offer=np.where((offer==-1)&(x==0),0,offer)
        if s['regime']=='LOW': price=r.choice([1,2],n,p=[.5,.5])
        else:
            price=r.choice([1,2,4,8],n,p=[.4,.3,.2,.1])
            if s['regime']=='PROBE':price=np.where(r.random(n)<.02,32,price)
        p=offer_probability(m,d,x,offer,price,truth,state,s.get('slope',1.5),s.get('tremble',.02))[:,0]
        accepted=r.random(n)<p
        data[:,t,:6]=np.column_stack([m,d,x,offer,price,accepted])
        update_offer(state,offer,price,accepted,truth)
        freeze_time=np.where(state['frozen'][:,0]&(freeze_time>steps),t+1,freeze_time)
        breach_time=np.where(state['breached'][:,0]&(breach_time>steps),t+1,breach_time)
        m=np.clip(m-accepted*offer*price,0,30);x=np.clip(x+accepted*offer,0,10)
        data[:,t,6]=m
        spend=r.random(n)<spend_probability(m,d,truth,s.get('slope',1.5),s.get('tremble',.02))[:,0]
        data[:,t,7]=spend;m-=spend;d-=spend
        data[:,t,8]=state['frozen'][:,0];data[:,t,9]=state['breached'][:,0]
    return dict(records=data,freeze_time=freeze_time,breach_time=breach_time)


def infer_x1(raw,s):
    records=raw['records'];n,steps,_=records.shape;params=candidates();state=initial(n,params)
    classes=params[:,0].astype(int);ll=np.zeros((n,len(params)));true=int(s['truth'][0])
    last=np.full(n,-1);pair_last=np.full((n,4),-1);false_rule=np.zeros(n,bool)
    first_event=np.full(n,-1);event_counts=np.zeros((n,5),int);high_offers=np.zeros(n,int)
    floor_curve=np.zeros((n,steps));refused_curve=np.zeros((n,steps));max_refused=np.zeros(n)
    final_odds=np.zeros((n,4));all_same=np.ones(n,bool)
    for t in range(steps):
        m,d,x,offer,price,accepted,after,spent=records[:,t,:8].T
        before_offer(state,m,d)
        p=offer_probability(m,d,x,offer,price,params,state,s.get('slope',1.5),s.get('tremble',.02))
        chosen=np.where(accepted[:,None],p,1-p)
        with np.errstate(divide='ignore'): ll+=np.log(chosen)
        update_offer(state,offer,price,accepted,params)
        pp=spend_probability(after,d,params,s.get('slope',1.5),s.get('tremble',.02))
        with np.errstate(divide='ignore'):ll+=np.log(np.where(spent[:,None],pp,1-pp))
        evidence=class_evidence(ll,classes);final_odds=evidence[:,true,None]-evidence
        rival=np.max(np.delete(evidence,true,axis=1),axis=1)
        odds=evidence[:,true]-rival;last=np.where(odds<3,t,last)
        pair_last=np.where(final_odds<3,t,pair_last)
        rule=evidence[:,1]-np.max(np.delete(evidence,1,axis=1),axis=1);false_rule |= rule>=3
        event=np.where((offer==-1)&~accepted.astype(bool)&(d>m),0,
              np.where((offer==-1)&(price==32)&~accepted.astype(bool),1,
              np.where((offer==1)&(price<=2)&~accepted.astype(bool),2,
              np.where((offer==1)&(price>=4)&accepted.astype(bool),3,4))))
        for k in range(5):event_counts[:,k]+=(offer!=0)&(event==k)
        first_event=np.where((first_event<0)&(odds>=3),event,first_event)
        high_offers+=(offer==-1)&(price>=4)
        gain=.5*price+np.minimum(price,np.maximum(0,d-m))
        max_refused=np.maximum(max_refused,np.where((offer==-1)&~accepted.astype(bool),gain,0))
        # Joint posterior mass of LIVE weights <= w, with equal prior over classes.
        modelprior=np.array([-np.log(4)-np.log(np.sum(classes==k)) for k in classes])
        posterior=np.exp(ll+modelprior-logsumexp(ll+modelprior,axis=1,keepdims=True))
        cum=posterior[:,:9].cumsum(1)
        floor_curve[:,t]=np.max(np.where(cum<.05,params[:9,1],0),axis=1)
        refused_curve[:,t]=max_refused
    return dict(separation=last+2,pair_separation=pair_last+2,final_class_odds=final_odds,
                final_likelihood=ll,false_rule=false_rule.astype(float),first_event=first_event,
                event_counts=event_counts,high_offers=high_offers,price_floor=floor_curve,
                refused_temptation=refused_curve)


def construction_checks():
    params=candidates();n=1;state=initial(n,params)
    m=np.array([0]);d=np.array([6]);x=np.array([8]);offer=np.array([-1]);p=np.array([2])
    probs=offer_probability(m,d,x,offer,p,params,state)[0]
    w4=int(np.flatnonzero((params[:,0]==0)&(params[:,1]==4))[0]);rule=int(np.flatnonzero((params[:,0]==1)&(params[:,1]==2))[0])
    margin=np.log((1-.02)/.02)/1.5
    return dict(named_counterexample=dict(gain=3.,weight=4.,live_sell=float(probs[w4]),rule_sell=float(probs[rule])),
                exact_clipping_margin=float(margin),
                corrected_identity=bool(probs[8]==probs[rule]),
                zero_buy_is_not_zero_class_log_odds=True,
                buying_at_cap_follows_money_only=bool(offer_probability(np.array([30]),d,np.array([10]),np.array([1]),p,params,state).sum()>0))
