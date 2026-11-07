"""G5: observed-stock set-point, smooth-level and linear readers."""
import numpy as np
from .common import rng,close
from .day_common import choices,posteriors,class_prior,margin,sustained,measure

def hypotheses():
    grid=np.arange(25,76,5)/100
    return ([dict(type='SP',point=float(s),a=a,b=b) for s in grid for a in (5.,10.,20.) for b in (.25,.5,1.)]
        +[dict(type='LV',point=float(s),lam=l) for s in grid for l in (2.,4.,8.,16.)]
        +[dict(type='HO',h=h) for h in (.5,1.,2.,4.,8.)])

def value(stock,h):
    if h['type']=='SP':
        d=np.asarray(stock)-h['point'];return h['a']*np.minimum(d,0)+h['b']*np.maximum(d,0)
    if h['type']=='LV':return -2*np.expm1(-h['lam']*(np.asarray(stock)-h['point']))/h['lam']
    return h['h']*np.asarray(stock)

def policy(next_stocks,payments,hs,precision=3.,independent=False):
    stock=np.asarray(next_stocks);pay=np.asarray(payments)
    u=np.stack([pay+value(stock,h) for h in hs],axis=-2)
    return choices(u,precision,independent=independent)

def observations(s,r):
    typ=s['truth']
    if typ=='SP':h=dict(type=typ,point=float(r.uniform(.3,.7)) if s['offgrid'] else float(r.choice([.35,.45,.55,.65])),a=float(r.choice([5,20])) if s['offgrid'] else 10.,b=float(r.choice([.25,1])) if s['offgrid'] else .5)
    elif typ=='LV':h=dict(type=typ,point=float(r.choice([.35,.45,.55,.65])),lam=float(r.choice([4,8,16])))
    else:h=dict(type=typ,h=float(r.choice([.5,1,2,4,8])))
    stock=float(r.uniform(.1,.9));records=[]
    for _ in range(s['records']):
        d=r.uniform(.05,.25,2);price=np.exp(r.uniform(np.log(s['prices'][0]),np.log(s['prices'][1]),2))
        delta=np.array([0.,-d[0],d[1]]);pay=np.array([0.,price[0]*d[0],-price[1]*d[1]])
        nxt=np.clip(.95*stock+delta,0,1);p=policy(nxt,pay,[h],s['precision'])[0]
        chosen=int(r.choice(3,p=p));records.append(dict(stock=stock,delta=delta.tolist(),payment=pay.tolist(),choice=chosen));stock=float(nxt[chosen])
    return dict(hypothesis=h,records=records)

def analyze(raw,s,independent=False):
    hs=hypotheses();c=np.array([h['type'] for h in hs]);prior=class_prior(c);truth=raw['hypothesis']['type']
    stock=np.array([x['stock'] for x in raw['records']]);delta=np.array([x['delta'] for x in raw['records']]);pay=np.array([x['payment'] for x in raw['records']]);choice=np.array([x['choice'] for x in raw['records']]);n=len(stock)
    nxt=np.clip(.95*stock[:,None]+delta,0,1);prob=policy(nxt,pay,hs,s['precision'],independent)
    logs=np.cumsum(np.log(prob[np.arange(n),:,choice]),axis=0);p=posteriors(logs,prior,independent)
    path=np.stack([margin(logs,prior,c==truth,c==t) for t in np.unique(c) if t!=truth]).min(0)
    result=dict(records=sustained(path),true_class=float(p[-1,c==truth].sum()))
    if truth=='SP':
        sp=c=='SP';cond=posteriors(logs[:,sp],np.ones(sp.sum())/sp.sum(),independent);inside=np.array([abs(h['point']-raw['hypothesis']['point'])<=.05+1e-12 for h in hs if h['type']=='SP'])
        split=float(np.mean(stock<raw['hypothesis']['point']));passes=.25<=split<=.75;placed=float(cond[-1,inside].sum()>=.9)
        result.update(time_below=split,time_split=float(passes),placement=placed,conditional_placement=[placed*passes,float(passes)],placement_64=float(cond[min(63,n-1),inside].sum()>=.9),placement_128=float(cond[min(127,n-1),inside].sum()>=.9),sp_margin=float(path[-1]))
        keep=np.array([h['type']!='SP' or (h['a']==10 and h['b']==.5) for h in hs]);sub=[h for h,k in zip(hs,keep) if k];subc=c[keep];fixed=posteriors(logs[:,keep],class_prior(subc),independent);fixedcond=posteriors(logs[:,keep][:,subc=='SP'],np.ones(11)/11,independent)
        fixedinside=np.array([abs(h['point']-raw['hypothesis']['point'])<=.05+1e-12 for h in sub if h['type']=='SP'])
        fp=float(fixedcond[-1,fixedinside].sum()>=.9);result.update(fixed_placement=fp,fixed_conditional_placement=[fp*passes,float(passes)],fixed_sp_class=float(fixed[-1,subc=='SP'].sum()))
        true=policy(nxt,pay,[raw['hypothesis']],s['precision'])[:,0];lv=prob[:,c=='LV']
        kl=(true[:,None,:]*(np.log(true[:,None,:])-np.log(lv))).sum(2).sum(0);best=int(np.argmin(kl));rival=[h for h in hs if h['type']=='LV'][best]
        result.update(expected_evidence=float(kl[best]),large_tv_share=float(np.mean(np.abs(lv[:,best]-true).sum(1)/2>.2)),best_lv_c=rival['point'],best_lv_lambda=rival['lam'])
    else:result['time_below']=float(np.mean(stock<raw['hypothesis'].get('point',.5)))
    return result

def block(s,identity,independent=False):
    r=rng(*identity);raw=[];vals={};out=[]
    for _ in range(s['makers']):
        o=observations(s,r);raw.append(o)
        for k,v in analyze(o,s,independent).items():vals.setdefault(k,[]).append(v)
    for k,v in vals.items():measure(out,k,v,'median' if k=='records' else 'ratio_of_sums' if 'conditional_placement' in k else 'mean',truth=s['truth'],offgrid=s['offgrid'])
    return dict(observations=raw,measures=out)

def design_check(s,identity):
    r=rng('day-check-G5',identity);time=[];info=[];tv=[]
    check=s|dict(truth='SP',offgrid=False,records=max(256,s['records']))
    for _ in range(96):
        a=analyze(observations(check,r),check);time.append(a['time_split']);info.append(a['expected_evidence']>=3);tv.append(a['large_tv_share'])
    return dict(passed=bool(np.mean(time)>=.75 and np.mean(info)>=.25),checks=dict(time_split_share=float(np.mean(time)),information_share=float(np.mean(info)),large_tv_share=float(np.mean(tv))),worlds=96)

def controls():
    stock=np.array([.1,.3,.5,.9]);pay=np.array([0.,1.,-1.,2.]);sp=dict(type='SP',point=.5,a=2.,b=2.);ho=dict(type='HO',h=2.)
    close(policy(stock,pay,[sp])[0].tolist(),policy(stock,pay,[ho])[0].tolist(),tol=1e-12)
    h=dict(type='SP',point=.5,a=10.,b=.5)
    assert abs((value(.3,h)-value(.2,h))/.1-10)<1e-12 and abs((value(.8,h)-value(.7,h))/.1-.5)<1e-12
    close(policy(stock,pay,[dict(type='LV',point=.5,lam=1e-6)])[0].tolist(),policy(stock,pay,[ho])[0].tolist(),tol=1e-6)
    hs=hypotheses();prior=class_prior([h['type'] for h in hs]);assert len(hs)==148
    assert all(abs(prior[np.array([h['type'] for h in hs])==t].sum()-1/3)<1e-12 for t in ('SP','LV','HO'))
    assert policy(stock,pay,hs).min()>=.02/4
    return dict(passed=True,linear_alias=True,asymmetric_slopes=True,smooth_linear_limit=True,equal_class_priors=True,positive_floor=True)
