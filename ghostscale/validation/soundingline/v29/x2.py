"""Exact finite-state mixture for visible and late-entry frozen-sum readers.

Late-entry hidden streak prior: uniform over 0..n-1 and already frozen (n).
This is a declared reader prior, not the true conditional prehistory law.
"""
import numpy as np
from scipy.special import logsumexp
from . import ledger
from .common import class_evidence

def expand(start,breach=False):
    base=ledger.candidates(frozen=True,breach=breach);rows=[];initials=[];model=[];weight=[]
    for j,p in enumerate(base):
        states=range(int(p[2])+1) if start and int(p[0])==4 else (0,)
        for state in states:
            rows.append(p);initials.append(state);model.append(j);weight.append(-np.log(len(states)))
    return base,np.array(rows),np.array(initials),np.array(model),np.array(weight)

def infer(raw,s):
    base,params,initials,model,logprior=expand(s.get('start',0),s.get('part')=='X2b')
    n=raw['records'].shape[0];start=s.get('start',0);steps=s['steps'];state=ledger.initial(n,params)
    state['streak'][:]=initials
    state['frozen'][:]=(params[:,0]==4)&(initials>=params[:,2])
    ll=np.broadcast_to(logprior,(n,len(params))).copy()
    classes=base[:,0].astype(int);available=np.unique(classes);true=int(s['truth'][0])
    true_index=int(np.flatnonzero(available==true)[0])
    last=np.full((n,len(available)),-1);accepted_before=np.zeros(n);freeze_estimate=np.zeros(n)
    for t in range(steps):
        rec=raw['records'][:,start+t];m,d,x,offer,price,accepted,after,spent=rec[:,:8].T
        ledger.before_offer(state,m,d)
        p=ledger.offer_probability(m,d,x,offer,price,params,state,s.get('slope',1.5),s.get('tremble',.02))
        with np.errstate(divide='ignore'):ll+=np.log(np.where(accepted[:,None],p,1-p))
        ledger.update_offer(state,offer,price,accepted,params)
        p=ledger.spend_probability(after,d,params,s.get('slope',1.5),s.get('tremble',.02))
        with np.errstate(divide='ignore'):ll+=np.log(np.where(spent[:,None],p,1-p))
        collapsed=np.column_stack([logsumexp(ll[:,model==j],axis=1) for j in range(len(base))])
        evidence=class_evidence(collapsed,classes);odds=evidence[:,true_index,None]-evidence
        last=np.where(odds<3,t,last)
        accepted_before+=(offer==-1)&accepted.astype(bool)&((start+t+1)<raw['freeze_time'])
        # Dating is restricted to an observed transition; pre-entry freezes are left censored.
        frozen_model=params[:,0]==4
        normalized=np.exp(ll[:,frozen_model]-logsumexp(ll[:,frozen_model],axis=1,keepdims=True))
        probability=(normalized*state['frozen'][:,frozen_model]).sum(1)
        freeze_estimate=np.where((freeze_estimate==0)&(probability>=.5),start+t+1,freeze_estimate)
    return dict(pair_separation=last+2,final_class_odds=odds,final_likelihood=collapsed,
                froze=(raw['freeze_time']<=s.get('life',steps)).astype(float),
                froze_before_entry=(raw['freeze_time']<=start).astype(float),
                accepted_sales_before_freeze=accepted_before,
                estimated_freeze=freeze_estimate,truth_freeze=raw['freeze_time'],
                breach=(raw['breach_time']<=s.get('life',steps)).astype(float))
