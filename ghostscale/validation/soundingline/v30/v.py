"""V visibility arms; STATE retains X3's reader roster and INSIDE knows the true type."""
from pathlib import Path
import numpy as np
from scipy.special import logsumexp,expit
from ..v29 import x3,ledger
from .common import rng,mean_blocks,conditional_blocks,audit_controls,write

def settings(arms=("STATE","INSIDE")):
    return [dict(arm=arm,truth=truth,tail=tail,loss=loss,environment=env,regime=reg,
                 steps=1024,worlds=200,blocks=8,particles=128)
      for arm in arms for truth in ([0,4,0],[0,8,0],[0,16,0],[0,20,0],[1,2,0],[7,8,0])
      for tail in (0.,.02,.10) for loss in (2,10) for env in ("plenty","scarce") for reg in ("LOW","MIXED")]

def truth_params(s):
    value=s["truth"].copy()
    if value[0]==7:value[0]=0
    return np.array([value],float)

def truth_offer(m,d,x,o,price,s,state,public=True):
    p=ledger.offer_probability(m,d,x,o,price,truth_params(s),state)[:,0]
    if s["truth"][0]==7 and public:p=np.where((o==-1)&(price<=4),0.,p)
    return p

def generate(s,lane="study",pulse=lambda **kw:None):
    if s["arm"] in ("BLIND","RECORD"):
        from .v_hidden import generate
        return generate(s,lane,pulse)
    n=s["worlds"];T=s["steps"];r=rng("V",lane,{k:v for k,v in s.items() if k not in ("arm","particles")})
    params=ledger.candidates();tp=truth_params(s)
    state=ledger.initial(n,params);ts=ledger.initial(n,tp);ll=np.zeros((n,len(params)));prior=x3.model_prior(params)
    m=np.full(n,5,int);d=np.zeros(n,int);x=np.full(n,8,int)
    rec=np.zeros((n,T,10),np.int16);pa=np.empty((n,T),np.int16);tempt=np.empty_like(pa)
    invites=np.empty((n,T),np.int8);defects=np.empty_like(invites)
    forecasts=np.empty((n,T));delta=np.zeros((n,T));preforecast=np.empty((n,T))
    qin,qn=(.8,.15) if s["environment"]=="plenty" else (.3,.3)
    for t in range(T):
        m=np.minimum(30,m+(r.random(n)<qin));d=np.minimum(6,d+r.binomial(2,qn,n))
        ledger.before_offer(state,m,d);ledger.before_offer(ts,m,d);old=state["shutdown"].copy()
        offer=np.where((r.random(n)<.25)&(x>0),-1,0)
        prices,probs=([1,2],[.5,.5]) if s["regime"]=="LOW" else ([1,2,4,8],[.4,.3,.2,.1])
        price=r.choice(prices,n,p=probs);actual=truth_offer(m,d,x,offer,price,s,ts)
        accept=r.random(n)<actual;rec[:,t,:6]=np.column_stack((m,d,x,offer,price,accept))
        belief=np.exp(ll+prior-logsumexp(ll+prior,axis=1,keepdims=True))
        before=(belief*x3.venture_prob(m,d,params,state,s)).sum(1)
        if s["arm"]=="STATE":
            p=ledger.offer_probability(m,d,x,offer,price,params,state)
            with np.errstate(divide="ignore"):ll+=np.log(np.where(accept[:,None],p,1-p))
        ledger.update_offer(state,offer,price,accept,params);ledger.update_offer(ts,offer,price,accept,tp)
        m=np.clip(m-accept*offer*price,0,30);x=np.clip(x+accept*offer,0,10);pa[:,t]=m
        current=dict(state,shutdown=old)
        if s["arm"]=="STATE":
            prob=x3.venture_prob(m,d,params,current,s)
            belief=np.exp(ll+prior-logsumexp(ll+prior,axis=1,keepdims=True));pred=(belief*prob).sum(1)
        else:
            # INSIDE means visible ledger plus the true type; MIMIC is LIVE8 privately.
            prob=x3.venture_prob(m,d,tp,ts,s);pred=prob[:,0];before=pred.copy()
        preforecast[:,t]=before;delta[:,t]=pred-before;forecasts[:,t]=pred
        invited=pred<1/(1+s["loss"]);invites[:,t]=invited;m=np.minimum(30,m+invited)
        price_private=r.choice(x3.PRICES,n,p=x3.price_law(s["tail"]));tempt[:,t]=price_private
        actual=truth_offer(m,d,x,np.full(n,-1),price_private,s,ts,False)
        defect=invited&(r.random(n)<actual);defects[:,t]=defect;m=np.minimum(30,m+defect*price_private)
        if s["arm"]=="STATE":
            joint=x3.venture_prob(pa[:,t],d,params,current,s,observed_after=m)
            with np.errstate(divide="ignore"):inc=np.log(np.where(defect[:,None],joint,1-prob))
            ll+=np.where(invited[:,None],inc,0.)
        spent=r.random(n)<ledger.spend_probability(m,d,tp)[:,0]
        if s["arm"]=="STATE":
            p=ledger.spend_probability(m,d,params)
            with np.errstate(divide="ignore"):ll+=np.log(np.where(spent[:,None],p,1-p))
        rec[:,t,6]=m;rec[:,t,7]=spent;m-=spent;d-=spent
        if t%128==0:pulse(step=t,steps=T)
    return dict(records=rec,public_after=pa,private_price=tempt,entrusted=invites,defects=defects,
                prediction=forecasts,before_prediction=preforecast,forecast_change=delta,partner_likelihood=ll)

def independent_visible(raw,s):
    # X3's independent scalar kernel reconstructs the unchanged STATE partner.
    params=ledger.candidates()
    ll,pred,gain,invite=x3.reference(raw["records"],raw["public_after"],raw["entrusted"],raw["defects"],
        params,x3.model_prior(params),x3.PRICES,x3.price_law(s["tail"]),s["loss"],1.5,.02)
    # Reconstruct the new before/after forecast diagnostic separately.
    rec=raw["records"];n,T,_=rec.shape;state=ledger.initial(n,params);lik=np.zeros((n,len(params)))
    prior=x3.model_prior(params);before=np.empty((n,T))
    for t in range(T):
        m,d,x,o,p,a,after,spent=rec[:,t,:8].T;ledger.before_offer(state,m,d)
        old=state["shutdown"].copy();belief=np.exp(lik+prior-logsumexp(lik+prior,axis=1,keepdims=True))
        before[:,t]=(belief*x3.venture_prob(m,d,params,state,s)).sum(1)
        prob=ledger.offer_probability(m,d,x,o,p,params,state)
        with np.errstate(divide="ignore"):lik+=np.log(np.where(a[:,None],prob,1-prob))
        ledger.update_offer(state,o,p,a,params);current=dict(state,shutdown=old)
        prob=x3.venture_prob(raw["public_after"][:,t],d,params,current,s)
        joint=x3.venture_prob(raw["public_after"][:,t],d,params,current,s,observed_after=after)
        with np.errstate(divide="ignore"):inc=np.log(np.where(raw["defects"][:,t,None]>0,joint,1-prob))
        lik+=np.where(raw["entrusted"][:,t,None]>0,inc,0.)
        prob=ledger.spend_probability(after,d,params)
        with np.errstate(divide="ignore"):lik+=np.log(np.where(spent[:,None]>0,prob,1-prob))
    return ll,pred,pred-before,invite

def infer(raw,s,reference=False,pulse=lambda **kw:None):
    if s["arm"] in ("BLIND","RECORD"):
        from .v_hidden import infer
        return infer(raw,s,reference,pulse)
    if reference and s["arm"]=="STATE":ll,pred,change,invite=independent_visible(raw,s)
    elif reference:
        rec=raw["records"];m=np.minimum(30,raw["public_after"]+1);d=rec[:,:,1]
        weight=s["truth"][1];prob=np.zeros(m.shape)
        for price,mass in zip(x3.PRICES,x3.price_law(s["tail"])):
            gain=.5*price+np.minimum(price,np.maximum(0,d-m))
            p=np.clip(1/(1+np.exp(-1.5*(gain-weight))),.02,.98) if s["truth"][0]!=1 else np.full(m.shape,.02)
            prob+=mass*p
        pred=prob;invite=(pred<1/(1+s["loss"])).astype(float);change=np.zeros_like(pred)
        ll=np.zeros_like(raw["partner_likelihood"])
    else:ll=raw["partner_likelihood"];pred=raw["prediction"];change=raw["forecast_change"];invite=raw["entrusted"].astype(float)
    return metrics(raw,s,ll,pred,change,invite)

def metrics(raw,s,ll,pred,change,invite):
    rec=raw["records"];end=rec[:,:,6]-rec[:,:,7]
    previous=np.concatenate((np.full((s["worlds"],1),5),end[:,:-1]),axis=1)
    public=raw["public_after"]-rec[:,:,0];venture=rec[:,:,6]-raw["public_after"]
    large=(raw["private_price"]==32)&(raw["entrusted"]>0)
    first=np.where(large.any(1),large.argmax(1),s["steps"]);t=np.arange(s["steps"])[None,:]
    pre=t<first[:,None];post=(t>first[:,None])&(t<=first[:,None]+32)
    return dict(entrusted_share=invite.mean(1),mean_money=end.mean(1),market_venture_income=(public+venture).sum(1),
        external_income=(rec[:,:,0]-previous).sum(1),defections=raw["defects"].sum(1),
        partner_payoff=(raw["entrusted"]*(1-raw["defects"])-s["loss"]*raw["defects"]).sum(1),
        _prediction=pred,_change=change,_likelihood=ll,_invite=invite,
        _refused=(rec[:,:,3]==-1)&(rec[:,:,5]==0),_price=rec[:,:,4],_need=rec[:,:,1],
        _large=large.any(1),_pre=(invite*pre).sum(1)/np.maximum(1,pre.sum(1)),
        _post=(invite*post).sum(1)/np.maximum(1,post.sum(1)),_post_count=post.sum(1),
        _first_large_loss=np.where(first<s["steps"],s["loss"]*raw["defects"][np.arange(s["worlds"]),np.minimum(first,s["steps"]-1)],0.))

def summarize(s,m):
    rows=[dict(measure=k,**mean_blocks(v,s["blocks"])) for k,v in m.items() if not k.startswith("_")]
    for price in (1,2,4,8):
        mask=m["_refused"]&(m["_price"]==price)
        rows.append(dict(measure="forecast_change_per_public_refusal",price=price,
            **conditional_blocks(m["_change"],mask,s["blocks"])))
        if s["arm"]=="STATE":
            for need in (0,1,2,3,4,5,6):
                rows.append(dict(measure="forecast_change_per_public_refusal",price=price,need=need,
                    **conditional_blocks(m["_change"],mask&(m["_need"]==need),s["blocks"])))
    for key in ("_pre","_post","_first_large_loss"):
        rows.append(dict(measure=key[1:]+"_large_event_conditional",
            **conditional_blocks(m[key],m["_large"]&(m["_post_count"]>0),s["blocks"])))
    return rows

def pilot_exclusion(s,m):
    # Payoff/forecast map: no declared competing-type separation criterion.
    return None

def controls(dest=None,pulse=lambda **kw:None):
    checks=[]
    for arm,truth in (("STATE",[0,20,0]),("STATE",[7,8,0]),("INSIDE",[7,8,0]),("INSIDE",[1,2,0])):
        s=dict(arm=arm,truth=truth,tail=.1,loss=10,environment="scarce",regime="MIXED",steps=32,worlds=8,blocks=8)
        raw=generate(s,"known-answer");a=infer(raw,s);b=infer(raw,s,True)
        checks.append(dict(name=arm+str(truth)+" independent visible reader",
            passed=all(np.allclose(a[k],b[k],atol=1e-9,rtol=1e-10,equal_nan=True) for k in a)))
        if arm=="INSIDE":checks.append(dict(name="known-type refusal changes no forecast",
            passed=bool(np.all(raw["forecast_change"]==0))))
        if truth[0]==7:
            rec=raw["records"];checks.append(dict(name="MIMIC refuses every cheap public offer",
                passed=not bool(np.any((rec[:,:,3]==-1)&(rec[:,:,4]<=4)&(rec[:,:,5]>0)))))
    result=audit_controls(checks)
    if dest is not None:write(Path(dest)/"CONTROLS.json",result)
    return result
