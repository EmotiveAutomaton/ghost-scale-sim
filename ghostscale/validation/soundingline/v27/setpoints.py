"""Exact sparse hidden-stock/precision/history filter, with explicit state cap.

DESIGN CHECK: known P0 recurrence is compared with V26 forward_scatter. Hidden
readers never receive realized consumption or stocks. Shock times and target
identities are observed interventions. Dynamics update before each choice from
the current stock and streak, then restoration and consumption occur together
with clipping, matching V26's transition order. No state pruning is permitted.
Exceeding the memory cap is a resource failure, not approximate evidence.
"""
import math,itertools,time
from collections import defaultdict
import numpy as np
from .common import rng,measure,posteriors,margin,sustained,scores,close

DYNAMICS=('P0','P+','P-','S','G')


def update(stock,edge,precision,streak,dynamic,width):
    edge=list(edge);precision=list(precision);streak=list(streak)
    for j in range(len(stock)):
        streak[j]=min(20,streak[j]+1) if stock[j]<edge[j] else 0
        if dynamic=='P+' and edge[j]<=stock[j] and (not width or stock[j]<=edge[j]+width):precision[j]=min(10.,precision[j]*1.03)
        if streak[j]>=20:
            if dynamic=='P-':precision[j]=max(.1,precision[j]*.97)
            if dynamic=='G':precision[j]=min(10.,precision[j]*1.03)
            if dynamic=='S':edge[j]=max(stock[j],edge[j]-.1)
    if dynamic in ('P0','P+'):streak=[0]*len(stock)
    return tuple(edge),tuple(precision),tuple(streak)


def probabilities(stock,edge,precision,lapse):
    urgency=np.minimum(5.,np.maximum(0,np.array(edge)-stock)*precision)
    u=np.r_[urgency,1.];p=np.exp(u-u.max());p=(1-lapse)*p/p.sum()+lapse/len(p)
    return p,urgency


def generate(s,identity):
    r=rng(identity);n=s['needs'];point=tuple(map(int,r.choice([8,11,14],n)));stock=tuple(map(float,r.integers(0,21,n)))
    edge=tuple(v-s['width']/2 for v in point);precision=(1.,)*n;streak=(0,)*n;records=[];first=None
    for t in range(s['records']):
        shock=int(r.integers(n)) if r.random()<s['probe'] else -1
        if shock>=0:stock=tuple(0. if j==shock else v for j,v in enumerate(stock))
        edge,precision,streak=update(stock,edge,precision,streak,s['dynamic'],s['width'])
        if max(streak)>=20 and first is None:first=t
        p,u=probabilities(stock,edge,precision,s['lapse']);action=int(r.choice(n+1,p=p));size=5+s['beta']*u[action] if action<n else 0.
        loss=r.choice([0.,1.,5.],n,p=[.45,.5,.05])*s['consumption']
        after=tuple(float(np.clip(stock[j]-loss[j]+(size if action==j else 0),0,20)) for j in range(n))
        records.append(dict(action=action,size=float(size),shock=shock,stock=list(stock),after=list(after),edge=list(edge),precision=list(precision),streak=list(streak)))
        stock=after
    return dict(point=list(point),dynamic=s['dynamic'],records=records,first_long_deficit=first)


def likelihood(raw,s,point,dynamic='P0',width=0,beta=None,visible=False,independent=False,max_states=200000):
    n=s['needs'];edge=tuple(v-width/2 for v in point);precision=(1.,)*n;streak=(0,)*n
    states={(tuple(map(float,x)),edge,precision,streak):1/21**n for x in itertools.product(range(21),repeat=n)}
    events=list(itertools.product((0.,1.,5.),repeat=n));weights=[math.prod(({0.:.45,1.:.5,5.:.05}[v] for v in e)) for e in events]
    logs=[];acc=0.;peak=len(states);known_edge=edge;known_precision=precision;known_streak=streak
    for t,row in enumerate(raw['records']):
        if visible:
            stock=tuple(row['stock']);known_edge,known_precision,known_streak=update(stock,known_edge,known_precision,known_streak,dynamic,width)
            states={(stock,known_edge,known_precision,known_streak):1.}
        nxt=defaultdict(float);a=row['action'];size=row['size'];shock=row['shock']
        ordered=reversed(list(states.items())) if independent else states.items()
        for (stock,ed,pi,st),mass in ordered:
            if not visible:
                if shock>=0:stock=tuple(0. if j==shock else v for j,v in enumerate(stock))
                ed,pi,st=update(stock,ed,pi,st,dynamic,width)
            p,u=probabilities(stock,ed,pi,s['lapse'])
            if beta is not None and a<n and abs(size-(5+beta*u[a]))>1e-10:continue
            value=mass*p[a]
            if visible:nxt[(stock,ed,pi,st)]+=value;continue
            iterable=reversed(list(zip(events,weights))) if independent else zip(events,weights)
            for ev,w in iterable:
                dest=tuple(float(min(20.,max(0.,stock[j]-s['consumption']*ev[j]+(size if a==j else 0)))) for j in range(n))
                nxt[(dest,ed,pi,st)]+=value*w
            if len(nxt)>max_states:raise MemoryError(f'exact hidden-state cap {max_states} exceeded at record {t+1}; no states pruned')
        total=math.fsum(nxt.values());peak=max(peak,len(nxt))
        if total==0:logs.extend([-math.inf]*(len(raw['records'])-t));break
        acc+=math.log(total);logs.append(acc);states={k:v/total for k,v in nxt.items()}
    return np.array(logs),peak


def analyze(raw,s,independent=False):
    points=list(itertools.product((8,11,14),repeat=s['needs']));truth=points.index(tuple(raw['point']));result={};cached={}
    libraries={'naive':[(p,'P0',0,None,False) for p in points],
        'aware':[(p,'P0',0,s['beta'],False) for p in points],
        'band':[(p,'P0',w,s['beta'],False) for p in points for w in (0,2,4)],
        'dynamic':[(p,d,s['width'],s['beta'],False) for p in points for d in DYNAMICS],
        'visible':[(p,s['dynamic'],s['width'],s['beta'],True) for p in points]}
    for reader,hs in libraries.items():
        paths=[];peaks=[]
        for h in hs:
            if h not in cached:cached[h]=likelihood(raw,s,*h,independent=independent)
            ll,peak=cached[h];paths.append(ll);peaks.append(peak)
        logs=np.array(paths).T;prior=np.ones(len(hs))/len(hs);p=posteriors(logs,prior,independent)
        yes=np.array([h[0]==tuple(raw['point']) for h in hs]);score=float(p[-1,yes].sum())
        result[reader]=dict(point_logscore=float(np.log(score)),point_records=sustained(margin(logs,prior,yes,~yes)),peak_states=max(peaks))
        if reader=='band':
            band=np.array([h[2]==s['width'] for h in hs]);zero=np.array([h[2]==0 for h in hs])
            result[reader]['band_records']=sustained(margin(logs,prior,band,zero)) if s['width'] else None
        if reader=='dynamic':
            yes=np.array([h[1]==s['dynamic'] for h in hs]);other=np.array([h[1] in ('S','P-','G') and h[1]!=s['dynamic'] for h in hs])
            begin=raw['first_long_deficit'];result[reader]['dynamic_records']=sustained(margin(logs,prior,yes,other)[begin:]) if begin is not None else None
    result['contrasts']=dict(overshoot_ratio=[result['naive']['point_records'],result['aware']['point_records']])
    return result


def block(s,identity,independent=False):
    raw=[];values={};out=[]
    for k in range(s['makers']):
        o=generate(s,[identity,k]);raw.append(o)
        for reader,stats in analyze(o,s,independent).items():
            for metric,value in stats.items():values.setdefault((reader,metric),[]).append(value)
    for (reader,key),v in values.items():measure(out,key,v,'median_ratio' if key=='overshoot_ratio' else 'median' if key.endswith('records') else 'mean',reader=reader)
    return dict(observations=raw,measures=out)


def controls():
    from ..v26 import day_velocity as old
    s=dict(needs=2,records=3,dynamic='P0',beta=0.,width=0,consumption=1.,probe=0.,lapse=.02)
    o=generate(s,['excluded-static-control']);menus=np.full(3,3);acts=np.array([x['action'] for x in o['records']]);fires=np.zeros(3,dtype=int)
    dest,fire,ch,rev,ptr=old.tables(1,5,.1,1.)
    expected=old.forward_scatter(menus,acts,fires,dest,fire,ch,old.EVENT_P)[2]
    for i,point in enumerate(old.POINTS):
        got,_=likelihood(o,s,tuple(point));close(got.tolist(),expected[:,i*3].tolist(),tol=1e-12)
    stock=(7.,11.);pi=(1.,1.);edge=(8.,11.);_,u=probabilities(stock,edge,pi,.02);size=5+u[0]
    assert stock[0]+(size-5)/pi[0]==edge[0]
    return dict(passed=True,V26_choices_only_exact=True,visible_uncapped_restore_edge_exact=True)


def design_check(s,identity):
    found=[generate(s,[identity,'excluded',k])['first_long_deficit'] is not None for k in range(128)]
    share=float(np.mean(found));applicable=s['beta']==0
    return dict(passed=share>=.25 or not applicable,case_share=share,floor=.25,case_applies=applicable,
        expected_signs=dict(naive='misspecified overshoot',aware='true static model on static worlds',band='lower-edge grid test',dynamic='exact hidden-state discrimination; finite cost unknown',visible='ceiling only'),scientific_draws=0)
