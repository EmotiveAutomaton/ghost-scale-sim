"""F rule ownership versus a prospectively specified drifting expected-fine weight."""
from pathlib import Path
import numpy as np
from scipy.special import expit
from .common import rng,mean_blocks,censored_blocks,sustained,audit_controls,write

def settings():
    return [dict(truth=truth,enforce_probability=e,fine=fine,drift=drift,duration=duration,
                 baseline=128,steps=128+duration+512,worlds=200,blocks=8,
                 environment="plenty") for e in (0.,.25,.75,1.) for fine in (4.,16.,32.)
            for drift in (.01,.05,.2) for duration in (16,64,256) for truth in (0,1)]
def weight_path(s,reference=False):
    e=np.zeros(s["steps"]);e[s["baseline"]:s["baseline"]+s["duration"]]=s["enforce_probability"]
    weights=np.full(s["steps"],64.);value=64.
    for t in range(s["baseline"],s["steps"]):
        value=value+s["drift"]*(e[t]*s["fine"]-value) if reference else (1-s["drift"])*value+s["drift"]*e[t]*s["fine"]
        weights[t]=value
    return weights,e
def choice(money,need,price,offer,weight,reference=False):
    gain=.5*price+np.minimum(price,np.maximum(0,need-money))
    z=1.5*(gain-weight)
    p=1/(1+np.exp(-z)) if reference else expit(z)
    return np.where(offer,np.clip(p,.02,.98),0.)
def generate(s,lane="study",pulse=lambda **kw:None):
    r=rng("F",lane,{k:v for k,v in s.items() if k!="truth"})
    n=s["worlds"];T=s["steps"];weights,enforce=weight_path(s)
    m=np.full(n,5);d=np.zeros(n,int);stock=np.full(n,8)
    rec=np.empty((n,T,8),float)
    qin,qn=(.8,.15) if s["environment"]=="plenty" else (.3,.3)
    for t in range(T):
        m=np.minimum(30,m+(r.random(n)<qin));d=np.minimum(6,d+r.binomial(2,qn,n))
        offer=(r.random(n)<.25)&(stock>0);price=r.choice([1,2,4,8],n,p=[.4,.3,.2,.1])
        p=np.where(offer,.02,0.) if s["truth"]==0 else choice(m,d,price,offer,weights[t])
        sell=r.random(n)<p;fine=sell&(r.random(n)<enforce[t])
        before=m.copy();m=np.clip(m+sell*price-fine*s["fine"],0,30).astype(int);stock-=sell
        sp=np.where((m>0)&(d>0),np.clip(expit(1.5*(np.minimum(d,3)-.5)),.02,.98),0.)
        spend=r.random(n)<sp;rec[:,t]=np.column_stack((before,d,offer,price,sell,fine,m,spend))
        m-=spend;d-=spend
    return dict(records=rec,weights=weights,enforcement=enforce)
def infer(raw,s,reference=False,pulse=lambda **kw:None):
    weights,enforce=weight_path(s,reference);rec=raw["records"];T=s["steps"]
    if not np.allclose(weights,raw["weights"],atol=1e-10):raise ArithmeticError("weight recursion differs")
    rule=np.where(rec[:,:,2]>0,.02,0.);priced=choice(rec[:,:,0],rec[:,:,1],rec[:,:,3],rec[:,:,2]>0,weights,reference)
    p=np.stack((rule,priced),axis=-1)
    chosen=np.where(rec[:,:,4,None]>0,p,1-p)
    ll=np.cumsum(np.log(np.maximum(chosen,1e-300)),axis=1)
    odds=ll[...,s["truth"]]-ll[...,1-s["truth"]]
    end=s["baseline"]+s["duration"]
    return dict(separation=sustained(odds),separation_after_departure=sustained(odds[:,end:]),
        final_odds=odds[:,-1],post_departure_sales=rec[:,end:,4].sum(1),
        post_departure_offers=rec[:,end:,2].sum(1),fines=rec[:,:,5].sum(1),
        mean_money=(rec[:,:,6]-rec[:,:,7]).mean(1),_likelihood=ll,
        _post_offer=rec[:,end:,2],_post_sell=rec[:,end:,4])
def summarize(s,m):
    rows=[]
    for k,v in m.items():
        if k.startswith("_"):continue
        horizon=512 if k=="separation_after_departure" else s["steps"]
        rows.append(dict(measure=k,**(censored_blocks(v,horizon,s["blocks"]) if k.startswith("separation") else mean_blocks(v,s["blocks"]))))
    return rows
def pilot_exclusion(s,m):
    if np.all(m["separation"]<=10):return "all forty separated within ten steps"
    return None
def controls(dest=None,pulse=lambda **kw:None):
    s=settings()[0];w,e=weight_path(s);wr,er=weight_path(s,True)
    checks=[dict(name="independent drift arithmetic",passed=bool(np.allclose(w,wr,atol=1e-10))),
        dict(name="pre-enforcement high weight aliases rule",passed=bool(np.all(
            choice(np.zeros(5),np.full(5,6),np.array([1,2,4,8,16]),np.ones(5,bool),64.)==.02))),
        dict(name="zero drift retains initial weight",passed=bool(np.all(weight_path(dict(s,drift=0.))[0]==64.))),
        dict(name="no offer supplies zero action evidence",passed=bool(choice(5,1,8,False,2.)==0.))]
    result=audit_controls(checks)
    if dest is not None:write(Path(dest)/"CONTROLS.json",result)
    return result
