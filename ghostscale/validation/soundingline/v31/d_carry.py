"""D.b: reactivity/carry correlation under independent latent settings."""
import numpy as np
from .common import rng,write
from .d_diary import summarize

CENTRES={'EXPOSED':(.1711,1.,.30),'REACTIVE':(.10,2.5684,.30),'STICKY':(.10,1.,.5741)}

def generate(s,lane,pulse=lambda **kw:None):
    values=[];truth=[];n=s['people'];t=s['steps'];p0,g0,rho0=CENTRES[s['kind']]
    for w in range(s['worlds']):
        g=rng('D-carry',s,lane,w);rates=np.clip(g0*np.exp(g.normal(0,.25,n)),.1,6)
        carry=np.clip(rho0+g.uniform(-.1,.1,n),.01,.95);exposure=np.clip(p0*np.exp(g.normal(0,.25,n)),.01,.5)
        level=np.zeros(n);rows=[]
        for j in range(t+40):
            gap=s['gap'];phi=carry**(gap/60)
            counts=g.poisson(-np.log1p(-exposure)*gap/60);impulse=np.zeros(n)
            # Conditional Poisson event times are uniform in the interval.
            for k in range(int(counts.max())):
                age=g.random(n)*gap;size=g.choice([.5,1.,1.5],n)
                impulse+=rates*size*carry**(age/60)*(counts>k)
            level=phi*level+impulse
            y=(g.random(n)<-np.expm1(-level)).astype(float) if s['view']=='SHOW' else level+g.normal(0,.2,n)
            if j>=40:rows.append(np.column_stack((counts>0,y)))
        values.append(np.swapaxes(rows,0,1));truth.append(np.column_stack((exposure,rates,carry)))
        pulse(world=w+1,phase='generate')
    return dict(records=np.array(values),truth=np.array(truth))

def person_slopes(records,reference=False):
    event=records[:,:,0];y=records[:,:,1];x=event[:,1:];previous=y[:,:-1];outcome=y[:,1:]
    result=np.full((len(y),2),np.nan)
    for p in range(len(y)):
        # Match ES2's event slope and ES3's within-person carry separately;
        # neither coefficient is presented as a true mechanistic parameter.
        for j,predictor in enumerate((x[p],previous[p])):
            if np.ptp(predictor)<1e-12:continue
            if reference:result[p,j]=np.linalg.lstsq(np.column_stack((np.ones(len(predictor)),predictor)),outcome[p],rcond=None)[0][1]
            else:
                a=predictor-predictor.mean();b=outcome[p]-outcome[p].mean();result[p,j]=a@b/(a@a)
    return result

def infer(raw,s,reference=False,pulse=lambda **kw:None):
    result={k:[] for k in ['estimated_reactivity_carry_correlation','true_reactivity_carry_correlation','reactivity_recovery','carry_recovery','available_people']}
    for w,(records,truth) in enumerate(zip(raw['records'],raw['truth'])):
        estimates=person_slopes(records,reference);ok=np.isfinite(estimates).all(1)
        def corr(a,b):return float(np.corrcoef(a,b)[0,1]) if len(a)>3 and np.std(a)>1e-12 and np.std(b)>1e-12 else np.nan
        result['estimated_reactivity_carry_correlation'].append(corr(estimates[ok,0],estimates[ok,1]))
        result['true_reactivity_carry_correlation'].append(corr(truth[:,1],truth[:,2]))
        result['reactivity_recovery'].append(corr(estimates[ok,0],truth[ok,1]))
        result['carry_recovery'].append(corr(estimates[ok,1],truth[ok,2]))
        result['available_people'].append(float(ok.mean()));pulse(world=w+1,phase='infer')
    return {k:np.array(v) for k,v in result.items()}

def controls(dest,pulse=lambda **kw:None):
    g=np.random.default_rng(3102);e=g.integers(0,2,(40,80));y=3*e+g.normal(size=e.shape);raw=np.stack((e,y),axis=2)
    a=person_slopes(raw);b=person_slopes(raw,True)
    checks={'independent_OLS':bool(np.allclose(a,b,rtol=1e-10,atol=1e-10)),
        'reactivity_known_near_three':abs(a[:,0].mean()-3)<.15,
        'independent_carry_near_zero':abs(a[:,1].mean())<.1}
    write(dest/'CONTROLS.json',checks);return dict(passed=all(checks.values()),checks=checks)

def pilot_exclusion(s,m):return None
