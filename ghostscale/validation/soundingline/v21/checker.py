"""Independent scalar evidence grouping and sequential law/maker Bayes check."""
import math
import numpy as np

def key(row,context,tier,observed):
    out=list(map(int,row[11:14]))+[-1]*10
    if tier!='artifact':out[3:6]=context
    if tier in ('sparse','complete'):out[6]=int(row[2])
    if tier=='complete':
        out[7:10]=[int(row[4]),int(row[6]),int(row[7])]
        if observed:out[10]=int(row[5])
    return tuple(out)

def scalar_mixture(worlds,x,tier,contexts,history_tier=None,observed=False,history_observed=False,history_worlds=None):
    models=[]
    for wi,rows in enumerate(worlds):
        current={};history={};priors=np.zeros(32)
        for row in rows:
            m=int(row[0]);y=int(row[10]);weight=float(row[-1]);ctx=contexts[int(row[1])]
            ck=key(row,ctx,tier,observed);hk=key(row,ctx,history_tier or tier,history_observed)
            current.setdefault(ck,np.zeros((32,128)))[m,y]+=weight
            history.setdefault(hk,np.zeros(32))[m]+=weight;priors[m]+=weight
        if history_worlds is not None:
            history={}
            for row in history_worlds[wi]:
                hk=key(row,contexts[int(row[1])],history_tier or tier,history_observed)
                history.setdefault(hk,np.zeros(32))[int(row[0])]+=float(row[-1])
        models.append((current,history,priors))
    outputs=[]
    for row in x:
        # Normalize across every law and maker after each observed episode. This
        # differs from producer logaddexp integration while retaining law evidence.
        belief=np.array([m[2]/len(models) for m in models])
        for step in range(13,len(row),13):
            for j,(_,history,prior) in enumerate(models):belief[j]*=history[tuple(row[step:step+13])]/prior
            belief/=belief.sum()
        out=np.zeros(128)
        for j,(current,_,prior) in enumerate(models):out+=(current[tuple(row[:13])]*(belief[j]/prior)[:,None]).sum(0)
        out/=out.sum();outputs.append(out)
    return np.array(outputs)

def validate_design(d):
    if d['handler']=='retained-audit':return True
    splits=[set(d.get(k,[])) for k in ('lineages','training_lineages','validation_lineages')]
    if any(splits[i]&splits[j] for i in range(3) for j in range(i)):raise ValueError('whole-law split leakage')
    if d.get('split')!='confirmation' and any(71900000<=n<71900100 for s in splits for n in s):raise ValueError('confirmation worlds opened early')
    if d.get('history',0) not in (0,1,4,16,64):raise ValueError('unregistered history')
    if d.get('history_observed') and d.get('history_tier')!='complete':raise ValueError('inspection needs complete history projection')
    return True
