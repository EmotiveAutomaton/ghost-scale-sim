"""Trust/payoff ledger with a price-marginalizing conditional-policy partner."""
from pathlib import Path
import math
import numpy as np
from scipy.special import logsumexp
from numba import njit
from . import ledger
from .common import rng,class_evidence,mean_blocks,conditional_blocks,write,digest,audit_controls
from .ledger_reference import sigmoid,lse
PRICES=np.array([2,4,8,16,32])


def settings():
    return [dict(truth=[typ,w,0],tail=tail,loss=loss,environment=env,regime=regime,steps=1024,worlds=200,blocks=8) for typ,w in [(0,2),(0,4),(0,8),(0,16),(0,20),(1,2)] for tail in (0.,.02,.05,.10) for loss in (2,5,10) for env in ('plenty','scarce') for regime in ('LOW','MIXED','PROBE')]


def price_law(tail):return np.array([.45*(1-tail),.30*(1-tail),.15*(1-tail),.10*(1-tail),tail])


def model_prior(params):
    classes=params[:,0].astype(int)
    return np.array([-np.log(4)-np.log(np.sum(classes==k)) for k in classes])


def venture_prob(money,need,params,state,s,observed_after=None):
    n=len(money);out=np.zeros((n,len(params)));m=np.minimum(30,money+1)
    for price,weight in zip(PRICES,price_law(s['tail'])):
        compatible=np.ones(n) if observed_after is None else (np.minimum(30,m+price)==observed_after)
        out+=weight*compatible[:,None]*ledger.offer_probability(m,need,np.full(n,8),np.full(n,-1),np.full(n,price),params,state,s.get('slope',1.5),s.get('tremble',.02))
    return out


def generate(s,lane='study',worlds=None):
    n=worlds or s['worlds'];steps=s['steps'];r=rng('X3',lane,s);params=ledger.candidates();truth=np.array([s['truth']],float);state=ledger.initial(n,params);truthstate=ledger.initial(n,truth);ll=np.zeros((n,len(params)));prior=model_prior(params)
    m=np.full(n,5,int);d=np.zeros(n,int);x=np.full(n,8,int);records=np.zeros((n,steps,10),np.int16)
    public_after=np.empty((n,steps),np.int16);private_price=np.empty_like(public_after);entrusted=np.zeros((n,steps),np.int8);defects=np.zeros_like(entrusted);prediction=np.empty((n,steps));public_gain=np.empty((n,steps))
    qin,qneed=(.8,.15) if s['environment']=='plenty' else (.3,.3)
    for t in range(steps):
        m=np.minimum(30,m+(r.random(n)<qin));d=np.minimum(6,d+r.binomial(2,qneed,n));ledger.before_offer(state,m,d);ledger.before_offer(truthstate,m,d);old_shutdown=state['shutdown'].copy()
        offer=np.where((r.random(n)<.25)&(x>0),-1,0)
        if s['regime']=='LOW':price=r.choice([1,2],n,p=[.5,.5])
        else:
            price=r.choice([1,2,4,8],n,p=[.4,.3,.2,.1])
            if s['regime']=='PROBE':price=np.where(r.random(n)<.02,32,price)
        tp=ledger.offer_probability(m,d,x,offer,price,truth,truthstate,s.get('slope',1.5),s.get('tremble',.02))[:,0];accepted=r.random(n)<tp
        records[:,t,:6]=np.column_stack([m,d,x,offer,price,accepted]);before=class_evidence(ll,params[:,0].astype(int))
        p=ledger.offer_probability(m,d,x,offer,price,params,state,s.get('slope',1.5),s.get('tremble',.02))
        with np.errstate(divide='ignore'):ll+=np.log(np.where(accepted[:,None],p,1-p))
        after=class_evidence(ll,params[:,0].astype(int));public_gain[:,t]=(after[:,1]-after[:,0])-(before[:,1]-before[:,0])
        ledger.update_offer(state,offer,price,accepted,params);ledger.update_offer(truthstate,offer,price,accepted,truth)
        m=np.clip(m-accepted*offer*price,0,30);x=np.clip(x+accepted*offer,0,10);public_after[:,t]=m
        current=dict(state,shutdown=old_shutdown);p=venture_prob(m,d,params,current,s);posterior=np.exp(ll+prior-logsumexp(ll+prior,axis=1,keepdims=True));pred=np.sum(posterior*p,axis=1);prediction[:,t]=pred
        invite=pred<1/(1+s['loss']);entrusted[:,t]=invite;m=np.minimum(30,m+invite)
        temptation=r.choice(PRICES,n,p=price_law(s['tail']));private_price[:,t]=temptation
        actual=ledger.offer_probability(m,d,x,np.full(n,-1),temptation,truth,truthstate,s.get('slope',1.5),s.get('tremble',.02))[:,0]
        defect=invite&(r.random(n)<actual);defects[:,t]=defect;m=np.minimum(30,m+defect*temptation)
        joint=venture_prob(public_after[:,t],d,params,current,s,observed_after=m)
        with np.errstate(divide='ignore'):inc=np.log(np.where(defect[:,None],joint,1-p))
        ll+=np.where(invite[:,None],inc,0.)
        spent=r.random(n)<ledger.spend_probability(m,d,truth,s.get('slope',1.5),s.get('tremble',.02))[:,0]
        p=ledger.spend_probability(m,d,params,s.get('slope',1.5),s.get('tremble',.02))
        with np.errstate(divide='ignore'):ll+=np.log(np.where(spent[:,None],p,1-p))
        records[:,t,6]=m;records[:,t,7]=spent;m-=spent;d-=spent
    return dict(records=records,public_after=public_after,private_price=private_price,entrusted=entrusted,defects=defects,prediction=prediction,public_rule_gain=public_gain,partner_likelihood=ll)


@njit(cache=True)
def reference(raw,public_after,entrusted,defects,params,prior,prices,price_probs,loss,slope,floor):
    n,steps,_=raw.shape;m=len(params);ll=np.zeros((n,m));prediction=np.empty((n,steps));gain=np.empty((n,steps));decisions=np.empty((n,steps))
    for w in range(n):
        shutdown=np.zeros(m,np.int64)
        for t in range(steps):
            money,need,stock,offer,price,accepted,after,spent=raw[w,t,:8]
            before=lse(ll[w,params[:,0]==1])-math.log(np.sum(params[:,0]==1))-lse(ll[w,params[:,0]==0])+math.log(np.sum(params[:,0]==0))
            current=np.empty(m,np.int64)
            for j in range(m):
                kind,weight,duration=params[j];shutdown[j]=max(0,shutdown[j]-1);current[j]=shutdown[j]
                p=0.
                if offer==-1:
                    g=price/2+min(price,max(0,need-money));p=sigmoid(slope*(g-weight),floor) if kind==0 else floor
                    if kind==3 and current[j]>0:p=0.
                chosen=p if accepted else 1-p;ll[w,j]+=math.log(chosen) if chosen>0 else -math.inf
                if kind==3 and offer==-1 and price>=4:shutdown[j]=int(duration)+1
            after_public=lse(ll[w,params[:,0]==1])-math.log(np.sum(params[:,0]==1))-lse(ll[w,params[:,0]==0])+math.log(np.sum(params[:,0]==0));gain[w,t]=after_public-before
            vm=min(30,public_after[w,t]+1);prob=np.zeros(m);joint=np.zeros(m);norm=lse(ll[w]+prior);pred=0.
            for j in range(m):
                kind,weight,duration=params[j]
                for k in range(len(prices)):
                    price=prices[k];g=price/2+min(price,max(0,need-vm));p=sigmoid(slope*(g-weight),floor) if kind==0 else floor
                    if kind==3 and current[j]>0:p=0.
                    prob[j]+=price_probs[k]*p
                    if min(30,vm+price)==after:joint[j]+=price_probs[k]*p
                pred+=math.exp(ll[w,j]+prior[j]-norm)*prob[j]
            prediction[w,t]=pred;decisions[w,t]=1. if pred<1/(1+loss) else 0.
            for j in range(m):
                kind,weight,duration=params[j]
                if entrusted[w,t]:
                    chosen=joint[j] if defects[w,t] else 1-prob[j];ll[w,j]+=math.log(chosen) if chosen>0 else -math.inf
                p=sigmoid(slope*(min(need,3)-(weight if kind==2 else .5)),floor) if after>0 and need>0 else 0.
                chosen=p if spent else 1-p;ll[w,j]+=math.log(chosen) if chosen>0 else -math.inf
    return ll,prediction,gain,decisions


def infer(raw,s,reference_mode=False):
    if reference_mode:
        params=ledger.candidates();ll,pred,gain,invite=reference(raw['records'],raw['public_after'],raw['entrusted'],raw['defects'],params,model_prior(params),PRICES,price_law(s['tail']),s['loss'],s.get('slope',1.5),s.get('tremble',.02))
    else:ll=raw['partner_likelihood'];pred=raw['prediction'];gain=raw['public_rule_gain'];invite=raw['entrusted'].astype(float)
    rec=raw['records'];n,steps=invite.shape;end=rec[:,:,6]-rec[:,:,7];previous=np.concatenate([np.full((n,1),5),end[:,:-1]],axis=1)
    income_external=rec[:,:,0]-previous;income_public=raw['public_after']-rec[:,:,0];income_venture=rec[:,:,6]-raw['public_after'];unmet=rec[:,:,1]-rec[:,:,7]
    large=(raw['private_price']==32)&(raw['entrusted']>0);first=np.where(large.any(1),large.argmax(1),steps);time=np.arange(steps)[None,:];pre=time<first[:,None];post=(time>first[:,None])&(time<=first[:,None]+32)
    return dict(partner_likelihood=ll,prediction=pred,entrusted=invite,public_rule_gain=gain,trust_curve=invite.cumsum(1)/np.arange(1,steps+1),market_and_venture_income=(income_public+income_venture).sum(1),external_income=income_external.sum(1),mean_money=end.mean(1),unmet_need_steps=(unmet>0).sum(1),unmet_deficit_sum=unmet.sum(1),defections=raw['defects'].sum(1),private_large_event=(first<steps).astype(float),pre_large_trust=(invite*pre).sum(1)/np.maximum(1,pre.sum(1)),post_large_trust=(invite*post).sum(1)/np.maximum(1,post.sum(1)),pre_large_count=pre.sum(1),post_large_count=post.sum(1))


def run(destination,mode='original',original=None,pulse=lambda **kw:None,selected=None,lane='study'):
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True);roster=settings() if selected is None else selected;rows=[];files={}
    for i,s in enumerate(roster):
        name=f'cell-{i:03d}_points.npz'
        if mode=='reconstruct':
            with np.load(Path(original)/name,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        else:raw=generate(s,lane);np.savez_compressed(dest/name,**raw);files[name]=digest(dest/name)
        out=infer(raw,s,mode=='reconstruct');np.savez_compressed(dest/f'cell-{i:03d}_measures.npz',**out)
        for key in ('market_and_venture_income','external_income','mean_money','unmet_need_steps','unmet_deficit_sum','defections','private_large_event'):rows.append(dict(setting=s,measure=key,**mean_blocks(out[key],s['blocks'])))
        for k in sorted(set([x for x in (64,128,256,512,1024,2048) if x<=s['steps']]+[s['steps']])):rows.append(dict(setting=s,measure='entrusted_share',step=k,**mean_blocks(out['trust_curve'][:,k-1],s['blocks'])))
        for phase in ('pre','post'):rows.append(dict(setting=s,measure=phase+'_large_trust_conditional',**conditional_blocks(out[phase+'_large_trust'],(out['private_large_event']>0)&(out[phase+'_large_count']>0),s['blocks'])))
        rec=raw['records'];g=.5*rec[:,:,4]+np.minimum(rec[:,:,4],np.maximum(0,rec[:,:,1]-rec[:,:,0]));refused=(rec[:,:,3]==-1)&(rec[:,:,5]==0)
        for low,high in ((0,3),(3,8),(8,100)):
            mask=refused&(g>=low)&(g<high);rows.append(dict(setting=s,measure='public_RULE_over_LIVE_log_evidence_gain_per_refusal',gain_bin=[low,high],**conditional_blocks(out['public_rule_gain'],mask,s['blocks'])))
        pulse(family='X3',cell=i+1,cells=len(roster))
    result=dict(family='X3',settings=roster,rows=rows,no_registered_claim=True,reader_limit='Conditional-policy partner observes public choices, visible ledger states, venture invitations/outcomes and spending; private prices are marginalized over those consistent with the visible venture payoff; common exogenous transition factors cancel. LIVE20 truth is off the specified X1 parameter grid.')
    write(dest/'SUMMARY.json',result)
    if mode!='reconstruct':write(dest/'RAW_MANIFEST.json',files)
    return result


def controls(destination,pulse=lambda **kw:None):
    checks=[]
    for base in (settings()[0],settings()[120],settings()[300],settings()[-1]):
        s=base|dict(steps=64,worlds=8,blocks=8);raw=generate(s,'known-answer');a=infer(raw,s);b=infer(raw,s,True)
        checks.append(dict(name=str(s['truth'])+str(s['tail'])+' independent partner and accounting',passed=all(np.allclose(a[k],b[k],atol=1e-9,rtol=0,equal_nan=True) for k in a)))
        checks.append(dict(name=str(s['truth'])+' correct trust threshold',passed=bool(np.array_equal(raw['entrusted'],raw['prediction']<1/(1+s['loss'])))))
    params=np.array([[0,32,0],[1,2,0]],float);state=ledger.initial(8,params);m=np.arange(8);d=np.full(8,6)
    p=venture_prob(m,d,params,state,dict(tail=.1));checks.append(dict(name='sufficiently high individual weight aliases RULE after clipping margin',passed=bool(np.allclose(p[:,0],.02,atol=1e-14)) and bool(np.array_equal(p[:,0],p[:,1]))))
    s=settings()[-1]|dict(steps=32,worlds=8,blocks=8);raw=generate(s,'private-label');changed={k:v.copy() for k,v in raw.items()};changed['private_price'][:]=2
    a=infer(raw,s,True);b=infer(changed,s,True);checks.append(dict(name='partner cannot read evaluator private prices',passed=bool(np.array_equal(a['prediction'],b['prediction'])) and bool(np.array_equal(a['partner_likelihood'],b['partner_likelihood']))))
    checks.append(dict(name='zero tail has no 32-price private events',passed=price_law(0)[-1]==0 and abs(price_law(.1).sum()-1)<1e-14))
    result=audit_controls(checks);write(Path(destination)/'CONTROLS.json',result);return result