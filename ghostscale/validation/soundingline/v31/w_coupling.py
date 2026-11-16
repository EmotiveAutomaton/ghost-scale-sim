"""Owner-approved two-block intervention on within-schema temperature coupling."""
import numpy as np
from .common import rng,write,rows_for
from .w_settings import BASE,simulate,controls as base_controls

def variances(s,kind):
    ts=np.array([s['low'],s['high']]);coupled=(.35/ts)**2
    # The best constant Gaussian rival has the average variance over balanced blocks.
    free=np.repeat(coupled.mean(),2)
    return coupled if kind=='COUPLED' else free

def generate(s,lane,pulse=lambda **kw:None):
    rows=[]
    for w in range(s['worlds']):
        g=rng('W-coupling',s,lane,w);parts=[];order=[0,1] if w%2==0 else [1,0]
        for block in order:
            theta=BASE.copy();theta[1]=[s['low'],s['high']][block]
            z=g.random((s['steps']//2,11));z[:,10]=g.normal(size=len(z))
            path=simulate(z,s['volatility'],s['reliability'],s['actionable'],theta)
            action=g.normal(0,np.sqrt(variances(s,s['truth'])[block]),len(z))
            parts.append(np.column_stack((np.full(len(z),block),path[:,4],action)))
        rows.append(np.concatenate(parts));pulse(world=w+1,phase='generate')
    return dict(records=np.array(rows))

def infer(raw,s,reference=False,pulse=lambda **kw:None):
    out={k:[] for k in ['separation','wrongly_settled','low_action_variance','high_action_variance','low_switch_rate','high_switch_rate','alias']}
    truth=['COUPLED','FREE'].index(s['truth'])
    for w,path in enumerate(raw['records']):
        ll=[]
        for kind in ('COUPLED','FREE'):
            var=variances(s,kind)[path[:,0].astype(int)]
            terms=np.log(np.sqrt(1/var))-.5*np.square(path[:,2]/np.sqrt(var)) if reference else -.5*(np.log(var)+path[:,2]**2/var)
            ll.append(np.cumsum(terms))
        ll=np.array(ll);gap=ll[truth]-ll[1-truth];hit=np.flatnonzero(np.minimum.accumulate(gap[::-1])[::-1]>=3)
        out['separation'].append(int(hit[0]+1) if len(hit) else s['steps']+1);out['wrongly_settled'].append(float(gap[-1]<=-3));out['alias'].append(float(np.max(abs(gap))<1e-10))
        for block,name in enumerate(('low','high')):
            p=path[path[:,0]==block];out[name+'_action_variance'].append(float(np.var(p[:,2],ddof=1)));out[name+'_switch_rate'].append(float(np.mean(p[1:,1]!=p[:-1,1])))
        pulse(world=w+1,phase='infer')
    return {k:np.array(v) for k,v in out.items()}

def summarize(s,m):return rows_for(s,m)
def pilot_exclusion(s,m):
    if np.all(m['alias']==1):return 'single-setting coupling/free alias'
    if np.all(m['separation']<=10):return 'all forty worlds separate within ten steps'
    return None
def controls(dest,pulse=lambda **kw:None):
    checks=base_controls(dest/'base')['checks'];s=dict(low=.35,high=.35)
    checks['single_setting_alias']=bool(np.array_equal(variances(s,'COUPLED'),variances(s,'FREE')))
    checks['tighter_at_higher_temperature']=bool(variances(dict(low=.35,high=.7),'COUPLED')[1]<variances(dict(low=.35,high=.7),'COUPLED')[0])
    write(dest/'CONTROLS.json',checks);return dict(passed=all(checks.values()),checks=checks)
