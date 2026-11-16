"""Independent text-first D implementation, prior to supplied script inspection.

Retain person-level sufficient statistics for both day halves. Reconstruction
uses an explicit normal-equation solve, independently of centered-sum slopes.
The third estimator is a correlated random-slope method of moments, with
within-person slope-error covariance removed before estimating correlation.
"""
import numpy as np
from scipy.special import expit
from ..v30.common import rows_for
from ..v29.common import audit_controls
from .common import rng

POPULATIONS = ('ONE','TWO80','TWO50','TWO0','RIVAL30','RIVAL60')

def target(pop):
    if pop.startswith('RIVAL'):
        sd=float(pop[5:])/100
        return float(np.sqrt(np.expm1(.25)/np.expm1(.25+sd*sd)))
    rho={'ONE':1.,'TWO80':.8,'TWO50':.5,'TWO0':0.}[pop]
    return float(np.expm1(.25*rho)/np.expm1(.25))

def population(g,n,pop):
    z=g.normal(size=(n,2)); a=.5*z[:,0]
    if pop.startswith('RIVAL'): b=a+float(pop[5:])/100*z[:,1]
    else:
        rho={'ONE':1.,'TWO80':.8,'TWO50':.5,'TWO0':0.}[pop]
        b=.5*(rho*z[:,0]+np.sqrt(1-rho*rho)*z[:,1])
    return np.exp(np.stack((a,b),axis=1))

def statistics(x,y):
    # n, sum x, sum x², sums y[2], sums xy[2], sums yy[3].
    return np.column_stack((np.full(len(x),x.shape[1]),x.sum(1),(x*x).sum(1),
        y.sum(1),(x[:,:,None]*y).sum(1),(y[:,:,0]**2).sum(1),
        (y[:,:,1]**2).sum(1),(y[:,:,0]*y[:,:,1]).sum(1),(x>0).sum(1)))

def generate(s,lane,pulse=lambda **kw:None):
    n=s['people'];days=s['days'];prompts=s['prompts'];t=days*prompts
    halves=np.arange(t)//prompts%2; records=[]; truths=[]
    for w in range(s['worlds']):
        g=rng('D',s,lane,w);react=population(g,n,s['population'])
        usual=g.normal(.6,.5,(n,1))
        if s['level_tied']:usual+=.4*np.log(react).mean(1,keepdims=True)/.5
        p=np.clip(g.beta(4,6,n)*s['event_rate']/.4,.05,.95) if prompts==1 else np.clip(g.beta(2,8,n)*s['event_rate']/.2,.02,.9)
        if s['rate_tied']:
            p=expit(np.log(p/(1-p))+.5*np.log(react[:,0])/.5)
        x=(g.random((n,t))<p[:,None]).astype(float)
        if s['graded']:x*=g.choice([.5,1.,1.5],(n,t))
        shared=g.normal(0,.5,(n,t,1))
        y=usual[:,None,:]+react[:,None,:]*x[:,:,None]+shared
        items=y[:,:,:,None]+g.normal(0,.8,(n,t,2,3))
        if s['floor']!='none':
            offset=0. if s['floor']=='mild' else s['heavy_offset']
            items=np.clip(np.rint(items-offset),0,4 if prompts==1 else 5)
        y=items.mean(3)
        records.append(np.stack([statistics(x[:,halves==h],y[:,halves==h]) for h in (0,1)]))
        truths.append(react)
        pulse(world=w+1,phase='generate')
    return dict(stats=np.asarray(records),truth=np.asarray(truths))

def slopes(a,reference=False):
    n,x,xx=a[:,0],a[:,1],a[:,2];yy=a[:,3:5];xy=a[:,5:7]
    den=xx-x*x/n;ok=(den>1e-12)&(n>2)
    if reference:
        b=np.full((len(a),2),np.nan);res=np.full((len(a),3),np.nan)
        gram=np.stack((n[ok],x[ok],x[ok],xx[ok]),axis=1).reshape(-1,2,2)
        rhs=np.stack((yy[ok],xy[ok]),axis=1)
        beta=np.linalg.solve(gram,rhs);b[ok]=beta[:,1,:]
        cross=np.stack((a[ok,7],a[ok,9],a[ok,9],a[ok,8]),axis=1).reshape(-1,2,2)-np.swapaxes(rhs,1,2)@beta
        v=cross/(n[ok]-2)[:,None,None]*np.linalg.inv(gram)[:,1,1,None,None]
        res[ok]=np.stack((v[:,0,0],v[:,1,1],v[:,0,1]),axis=1)
    else:
        b=np.divide(xy-x[:,None]*yy/n[:,None],den[:,None],out=np.full_like(xy,np.nan),where=ok[:,None])
        intercept=(yy-b*x[:,None])/n[:,None]
        resid=np.stack((a[:,7]-intercept[:,0]*yy[:,0]-b[:,0]*xy[:,0],
                        a[:,8]-intercept[:,1]*yy[:,1]-b[:,1]*xy[:,1],
                        a[:,9]-intercept[:,0]*yy[:,1]-b[:,0]*xy[:,1]),1)
        res=np.divide(resid,(n-2)[:,None]*den[:,None],out=np.full_like(resid,np.nan),where=ok[:,None])
    return b,res,ok

def estimates(a,reference=False,min_events=1):
    b0,_,ok0=slopes(a[0],reference);b1,_,ok1=slopes(a[1],reference)
    keep=ok0&ok1&np.all((a[:,:,10]>=min_events)&(a[:,:,0]-a[:,:,10]>=min_events),axis=0); out=np.full(3,np.nan)
    if keep.sum()>=12:
        z=np.concatenate((b0[keep],b1[keep]),axis=1)
        cov=np.corrcoef(z,rowvar=False)
        den=cov[0,2]*cov[1,3]
        if den>0 and cov[0,2]>0 and cov[1,3]>0:
            out[0]=(cov[0,3]+cov[1,2])/2/np.sqrt(den)
            out[1]=(cov[0,1]+cov[2,3])/2/np.sqrt(den)
    full=a.sum(0);b,e,ok=slopes(full,reference)
    if ok.sum()>=12:
        cov=np.cov(b[ok],rowvar=False,ddof=1);noise=e[ok].mean(0)
        v0=cov[0,0]-noise[0];v1=cov[1,1]-noise[1]
        if v0>0 and v1>0:out[2]=(cov[0,1]-noise[2])/np.sqrt(v0*v1)
    return out,float(keep.mean())

def infer(raw,s,reference=False,pulse=lambda **kw:None):
    values=[];valid=[];errors=[];one=[];multiple=[]
    for w,a in enumerate(raw['stats']):
        min_events=1 if s['prompts']==1 else 3
        est,share=estimates(a,reference,min_events)
        # Delete-group jackknife over persons, not prompts. Unavailable estimates
        # remain NaN with explicit availability; never turn instability into zero.
        jk=np.array([estimates(a[:,np.arange(s['people'])%8!=k],reference,min_events)[0] for k in range(8)])
        se=np.sqrt(7/8*np.sum((jk-jk.mean(0))**2,axis=0))
        values.append(est);valid.append(share);errors.append(est-target(s['population']))
        one.append((est-1.96*se>=.9).astype(float));multiple.append((est+1.96*se<=.85).astype(float))
        pulse(world=w+1,phase='infer')
    v=np.asarray(values);e=np.asarray(errors);o=np.asarray(one);m=np.asarray(multiple)
    result={'usable_people':np.asarray(valid)}
    for k,name in enumerate(('split','same','random_slope')):
        result.update({name:v[:,k],name+'_bias':e[:,k],name+'_available':np.isfinite(v[:,k]).astype(float),
            name+'_one':o[:,k],name+'_multiple':m[:,k]})
    return result

def summarize(s,m):
    # The common block estimator cannot consume NaNs. Report availability and
    # conditional estimates with complete block denominators explicitly.
    rows=[]
    for key,a in m.items():
        a=np.asarray(a);blocks=a.reshape(s['blocks'],-1);counts=np.isfinite(blocks).sum(1)
        means=np.array([b[np.isfinite(b)].mean() if np.isfinite(b).any() else np.nan for b in blocks])
        finite=means[np.isfinite(means)]
        mean=float(finite.mean()) if len(finite) else None
        se=float(finite.std(ddof=1)/np.sqrt(len(finite))) if len(finite)>1 else None
        rows.append(dict(measure=key,mean=mean,interval95=[mean-2.364624*se,mean+2.364624*se] if se is not None else None,
            available_worlds=int(counts.sum()),block_denominators=counts.tolist(),spread=float(np.nanstd(a,ddof=1)) if counts.sum()>1 else None))
    return rows

def controls(dest,pulse=lambda **kw:None):
    g=np.random.default_rng(3191);x=g.normal(size=(100,30));b=g.normal(size=(100,2));y=2+x[:,:,None]*b[:,None,:]
    a=statistics(x,y);p,_,_=slopes(a);q,_,_=slopes(a,True)
    checks={'slope_known':float(np.max(abs(p-b)))<1e-10,'independent_slope':float(np.max(abs(p-q)))<1e-10,
        'one_target':target('ONE')==1.,'zero_target':target('TWO0')==0.,
        'rival_order':0<target('RIVAL60')<target('RIVAL30')<1}
    from .common import write
    write(dest/'CONTROLS.json',checks)
    return dict(passed=all(checks.values()),checks=checks)

def pilot_exclusion(s,m):return None  # Estimator calibration, not a class-separation test.
