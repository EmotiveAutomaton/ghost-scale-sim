"""Transport and numerical primitives for declared finite conditional worlds."""
import math
import numpy as np
from scipy.special import logsumexp
from . import ledger

RELIABILITY=np.linspace(.60,.95,16)

def normalize(logmass):
    return np.exp(logmass-logsumexp(logmass))

def noisy(rule):
    rule=np.asarray(rule,float)
    return RELIABILITY.reshape((16,)+(1,)*rule.ndim)*rule+(1-RELIABILITY).reshape((16,)+(1,)*rule.ndim)/rule.shape[-1]

def draw(prob,u):
    return int(np.searchsorted(np.cumsum(prob),u,side='right'))

def categorical(p):return np.stack([1-np.asarray(p),np.asarray(p)],axis=-1)

def emit(p,schema,identity,classes):
    e=ledger.Emission(schema,identity)
    for i,row in enumerate(p['observations']):e.decision('history',i,list(range(classes)),row['y'],observed=dict(context=row['context']))
    for i,context in enumerate(p['query']):e.decision('query',i,list(range(classes)),None,observed=dict(context=context))
    e.decision('design',0,[0],0,observed=p['design'])
    return e.finish()

def decode(bundle,schema):
    ledger.validate(bundle,schema);rows=bundle['rows']['Decision'];out=dict(observations=[],query=[])
    out['design']=dict(next(r for r in rows if r['context']['channel']=='design')['context']['observed'])
    for r in sorted(rows,key=lambda r:(r['context']['channel'],r['context']['index'])):
        channel=r['context']['channel']
        if channel=='history':
            y=next(o['position'] for o in r['options'] if o['option_id']==r['taken'])
            out['observations'].append(dict(context=r['context']['observed']['context'],y=y))
        elif channel=='query':
            if r['taken'] is not None:raise ValueError('held-out target in ledger')
            out['query'].append(r['context']['observed']['context'])
    return out

def features(p,max_history,width,classes,design_keys):
    x=np.zeros((max_history,width+classes+1))
    for i,row in enumerate(p['observations']):
        x[i,:width]=row['context'];x[i,width+row['y']]=1;x[i,-1]=1
    return np.r_[x.ravel(),np.asarray(p['query']).ravel(),[p['design'][k] for k in design_keys]]

def mechanism_gain(before,after,truth):
    if min(before[truth],after[truth])<=0:raise ValueError('empty posterior support')
    return math.log(after[truth])-math.log(before[truth])
