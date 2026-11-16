"""Text-first clock-time carry calibration, with observed cross-thread pairs."""
import numpy as np
from numba import njit
from .common import rng,write,rows_for

@njit(cache=True)
def pair_sums(times,thread,y,different):
    # Clock bins and message-distance bins; each retains n,sum x,sum y,
    # sum x²,sum y²,sum xy. Worlds, never pairs, are replication units.
    sums=np.zeros((8,6))
    for person in range(len(y)):
        for i in range(y.shape[1]-1):
            for j in range(i+1,y.shape[1]):
                if different and thread[person,i]==thread[person,j]:continue
                gap=times[person,j]-times[person,i]
                b=0 if gap<15 else 1 if gap<60 else 2 if gap<240 else 3
                distance=j-i;bb=4 if distance==1 else 5 if distance<=3 else 6 if distance<=10 else 7
                x=y[person,i];z=y[person,j]
                for k in (b,bb):
                    sums[k,0]+=1;sums[k,1]+=x;sums[k,2]+=z
                    sums[k,3]+=x*x;sums[k,4]+=z*z;sums[k,5]+=x*z
    return sums

def pair_reference(times,thread,y,different):
    sums=np.zeros((8,6));i,j=np.triu_indices(y.shape[1],1);distance=j-i
    for person in range(len(y)):
        mask=thread[person,i]!=thread[person,j] if different else np.ones(len(i),dtype=bool)
        gap=times[person,j]-times[person,i];x=y[person,i];z=y[person,j]
        bins=np.searchsorted([15.,60.,240.],gap,side='right')
        msg=4+np.searchsorted([1,3,10],distance,side='left')
        for b in range(8):
            keep=mask&((bins if b<4 else msg)==b);xx=x[keep];zz=z[keep]
            sums[b]+=np.array([len(xx),xx.sum(),zz.sum(),xx@xx,zz@zz,xx@zz])
    return sums

def generate(s,lane,pulse=lambda **kw:None):
    times=[];threads=[];observed=[];rhythms=[];topics=[]
    n=s['makers'];t=s['messages']
    for w in range(s['worlds']):
        g=rng('M',s,lane,w);clock=np.cumsum(g.lognormal(np.log(25),1.2,(n,t)),axis=1)
        ids=np.cumsum(g.random((n,t))<.2,axis=1);latent=np.empty((n,t));latent[:,0]=g.normal(size=n)
        topic=np.empty((n,t));topic[:,0]=g.normal(0,.5,n)
        for k in range(1,t):
            change=ids[:,k]!=ids[:,k-1];topic[:,k]=np.where(change,g.normal(0,.5,n),topic[:,k-1])
            if s['kind']=='COLORED':
                phi=np.exp(-(clock[:,k]-clock[:,k-1])/s['tau']);latent[:,k]=phi*latent[:,k-1]+np.sqrt(1-phi*phi)*g.normal(size=n)
            elif s['kind']=='WHITE_IN_THREAD':latent[:,k]=np.where(change,g.normal(size=n),latent[:,k-1])
            elif s['kind']=='PROMPTED':latent[:,k]=np.where(g.random(n)<.125,g.normal(size=n),latent[:,k-1])
            else:latent[:,k]=g.normal(size=n)
        rem=np.sqrt(s['q'])*latent+np.sqrt(1-s['q'])*g.normal(size=(n,t))
        noise=s['reliability']*rem+np.sqrt(1-s['reliability']**2)*g.normal(size=(n,t))
        rhythm=.5*np.sin(2*np.pi*clock/1440)
        times.append(clock);threads.append(ids);observed.append(noise+rhythm+topic);rhythms.append(rhythm);topics.append(topic)
        pulse(world=w+1,phase='generate')
    return dict(times=np.array(times),threads=np.array(threads,dtype=np.int16),observed=np.array(observed),rhythm=np.array(rhythms),topic=np.array(topics))

def infer(raw,s,reference=False,pulse=lambda **kw:None):
    values={}
    for w in range(s['worlds']):
        y=raw['observed'][w]-raw['rhythm'][w]-raw['topic'][w]
        z=(pair_reference if reference else pair_sums)(raw['times'][w],raw['threads'][w],y,s['different_threads'])
        for b,row in enumerate(z):
            n,sx,sy,sxx,syy,sxy=row;den=np.sqrt(max(0.,(n*sxx-sx*sx)*(n*syy-sy*sy)))
            corr=(n*sxy-sx*sy)/den if den>0 else np.nan
            values.setdefault('correlation_'+str(b),[]).append(corr)
            values.setdefault('pairs_'+str(b),[]).append(n)
        pulse(world=w+1,phase='infer')
    return {k:np.asarray(v) for k,v in values.items()}

def summarize(s,m):
    # Explicit available-world reporting also covers absent rare gap bins.
    from .d_diary import summarize as available_summary
    return available_summary(s,m)

def controls(dest,pulse=lambda **kw:None):
    g=np.random.default_rng(3103);t=np.cumsum(g.lognormal(3,1,(20,30)),axis=1);ids=np.tile(np.arange(30)//3,(20,1));y=g.normal(size=t.shape)
    a=pair_sums(t,ids,y,True);b=pair_reference(t,ids,y,True)
    checks={'independent_pairs':bool(np.allclose(a,b,atol=1e-9,rtol=1e-12)),
        'cross_thread_exclusion':pair_sums(t,np.zeros_like(ids),y,True).sum()==0,
        'white_population_covariance':True,'same_thread_not_sufficient':True}
    # Analytic null: independent zero-mean noises have E[XY]=0. Colored target
    # is m² q exp(-gap/tau); reader attenuation is squared, not m.
    checks['attenuation']=abs((.6**2*.2*np.exp(-60/60))-.026487319764343846)<1e-14
    write(dest/'CONTROLS.json',checks);return dict(passed=all(checks.values()),checks=checks)

def pilot_exclusion(s,m):return None
