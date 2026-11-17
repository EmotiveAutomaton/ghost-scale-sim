"""M R1: structural zero-variance handling for per-maker pair correlations."""
import numpy as np
from numba import njit
from .common import rng,write,rows_for

@njit(cache=True)
def kalman_odds(times,y,share,tau):
    out=np.zeros_like(y)
    for person in range(len(y)):
        mean=0.;variance=share;ll=0.
        for t in range(y.shape[1]):
            if t:
                phi=np.exp(-(times[person,t]-times[person,t-1])/tau)
                mean*=phi;variance=phi*phi*variance+share*(1-phi*phi)
            total=variance+1-share;error=y[person,t]-mean
            ll+=-.5*(np.log(total)+error*error/total-y[person,t]**2)
            mean+=variance/total*error;variance*=((1-share)/total);out[person,t]=ll
    return out

def precision_odds(times,y,share,tau):
    mean=np.zeros(len(y));variance=np.full(len(y),share);ll=np.zeros(len(y));out=np.zeros_like(y)
    for t in range(y.shape[1]):
        if t:
            phi=np.exp(-np.diff(times[:,t-1:t+1],axis=1)[:,0]/tau)
            mean=phi*mean;variance=phi**2*variance+share*(1-phi**2)
        total=variance+1-share;error=y[:,t]-mean
        ll-=.5*(np.log(total)+error**2/total-y[:,t]**2);out[:,t]=ll
        updated=1/(1/variance+1/(1-share))
        mean=updated*(mean/variance+y[:,t]/(1-share));variance=updated
    return out

@njit(cache=True)
def pair_sums(times,thread,y,different):
    # Clock bins and message-distance bins; each retains n,sum x,sum y,
    # sum x²,sum y²,sum xy. Worlds, never pairs, are replication units.
    sums=np.zeros((len(y),8,6))
    for person in range(len(y)):
        for i in range(y.shape[1]-1):
            for j in range(i+1,y.shape[1]):
                if different and thread[person,i]==thread[person,j]:continue
                gap=times[person,j]-times[person,i]
                b=0 if gap<15 else 1 if gap<60 else 2 if gap<240 else 3
                distance=j-i;bb=4 if distance==1 else 5 if distance<=3 else 6 if distance<=10 else 7
                x=y[person,i];z=y[person,j]
                for k in (b,bb):
                    sums[person,k,0]+=1;sums[person,k,1]+=x;sums[person,k,2]+=z
                    sums[person,k,3]+=x*x;sums[person,k,4]+=z*z;sums[person,k,5]+=x*z
    return sums

def pair_reference(times,thread,y,different):
    sums=np.zeros((len(y),8,6));i,j=np.triu_indices(y.shape[1],1);distance=j-i
    for person in range(len(y)):
        mask=thread[person,i]!=thread[person,j] if different else np.ones(len(i),dtype=bool)
        gap=times[person,j]-times[person,i];x=y[person,i];z=y[person,j]
        bins=np.searchsorted([15.,60.,240.],gap,side='right')
        msg=4+np.searchsorted([1,3,10],distance,side='left')
        for b in range(8):
            keep=mask&((bins if b<4 else msg)==b);xx=x[keep];zz=z[keep]
            sums[person,b]+=np.array([len(xx),xx.sum(),zz.sum(),xx@xx,zz@zz,xx@zz])
    return sums

def generate(s,lane,pulse=lambda **kw:None):
    times=[];threads=[];observed=[];rhythms=[];topics=[]
    n=s['makers'];t=s['messages']
    for w in range(s['worlds']):
        g=rng('M',s,lane,w);clock=np.cumsum(g.lognormal(np.log(25),1.5,(n,t)),axis=1)
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


@njit(cache=True)
def maker_short_stream(times,threads,y,different):
    result=np.full(len(y),np.nan)
    for person in range(len(y)):
        n=0;mx=0.;my=0.;xx=0.;yy=0.;xy=0.;first_x=0.;first_y=0.;vary_x=False;vary_y=False
        for i in range(y.shape[1]-1):
            for j in range(i+1,y.shape[1]):
                if times[person,j]-times[person,i]>=15:continue
                if different and threads[person,i]==threads[person,j]:continue
                x=y[person,i];z=y[person,j]
                if n==0:first_x=x;first_y=z
                else:vary_x=vary_x or x!=first_x;vary_y=vary_y or z!=first_y
                n+=1;dx=x-mx;dy=z-my;mx+=dx/n;my+=dy/n
                xx+=dx*(x-mx);yy+=dy*(z-my);xy+=dx*(z-my)
        if n>=2 and vary_x and vary_y and xx>0 and yy>0:result[person]=xy/np.sqrt(xx*yy)
    return result

def maker_short_reference(times,threads,y,different):
    result=np.full(len(y),np.nan);i,j=np.triu_indices(y.shape[1],1)
    for person in range(len(y)):
        keep=times[person,j]-times[person,i]<15
        if different:keep &= threads[person,i]!=threads[person,j]
        x=y[person,i[keep]];z=y[person,j[keep]]
        if len(x)<2 or np.all(x==x[0]) or np.all(z==z[0]):continue
        x=x-x.mean();z=z-z.mean();result[person]=(x@z)/np.sqrt((x@x)*(z@z))
    return result

def infer(raw,s,reference=False,pulse=lambda **kw:None):
    values={}
    for w in range(s['worlds']):
        y=raw['observed'][w]-raw['rhythm'][w]-raw['topic'][w]
        odds=(precision_odds if reference else kalman_odds)(raw['times'][w],y,s['q']*s['reliability']**2,s['tau'])
        values.setdefault('colored_evidence_maker_share',[]).append(float(np.mean(odds[:,-1]>=3)))
        values.setdefault('white_evidence_maker_share',[]).append(float(np.mean(odds[:,-1]<=-3)))
        values.setdefault('unresolved_maker_share',[]).append(float(np.mean(abs(odds[:,-1])<3)))
        # Exact two-law likelihood for each maker; the colored model has known
        # q/tau/m, so these are oracle-information requirements, not learned fits.
        color=np.minimum.accumulate(odds[:,::-1],axis=1)[:,::-1]>=3
        times=np.where(color.any(1),color.argmax(1)+1,s['messages']+1)
        values.setdefault('maker_messages_censored_mean',[]).append(float(times.mean()))
        individual=(pair_reference if reference else pair_sums)(raw['times'][w],raw['threads'][w],y,s['different_threads'])
        z=individual.sum(0)
        def correlations(z):
            n,sx,sy,sxx,syy,sxy=np.moveaxis(z,-1,0);den=np.sqrt(np.maximum(0.,(n*sxx-sx*sx)*(n*syy-sy*sy)))
            return np.divide(n*sxy-sx*sy,den,out=np.full_like(n,np.nan),where=den>0)
        jk=np.array([correlations(z-individual[np.arange(s['makers'])%10==k].sum(0)) for k in range(10)])
        errs=np.sqrt(.9*np.sum((jk-jk.mean(0))**2,axis=0));corrs=correlations(z)
        contrast=corrs[0]-corrs[3];jkdiff=jk[:,0]-jk[:,3];contrast_se=np.sqrt(.9*np.sum((jkdiff-jkdiff.mean())**2))
        values.setdefault('positive_short_gap',[]).append(float(corrs[0]-1.96*errs[0]>0))
        values.setdefault('fading_detected',[]).append(float(contrast-1.96*contrast_se>0))
        values.setdefault('short_minus_long',[]).append(float(contrast))
        values.setdefault('short_gap_standard_error',[]).append(float(errs[0]))
        values.setdefault('maker_short_correlation_mean',[]).append(float(np.nanmean((maker_short_reference if reference else maker_short_stream)(raw['times'][w],raw['threads'][w],y,s['different_threads']))))
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
    null_y=g.normal(size=(2000,30));null_t=np.tile(np.arange(30)*10.,(2000,1));null_ids=np.tile(np.arange(30),(2000,1))
    null=pair_sums(null_t,null_ids,null_y,True).sum(0);n,sx,sy,sxx,syy,sxy=null[1]
    corr=(n*sxy-sx*sy)/np.sqrt((n*sxx-sx*sx)*(n*syy-sy*sy))
    checks={'independent_pairs':bool(np.allclose(a,b,atol=1e-9,rtol=1e-12)),
        'cross_thread_exclusion':pair_sums(t,np.zeros_like(ids),y,True).sum()==0,
        'white_null_abs_correlation_below_03':abs(corr)<.03}
    # Analytic null: independent zero-mean noises have E[XY]=0. Colored target
    # is m² q exp(-gap/tau); reader attenuation is squared, not m.
    checks['attenuation']=abs((.6**2*.2*np.exp(-60/60))-.026487319764343846)<1e-14
    checks['independent_exact_gaussian_reader']=bool(np.allclose(kalman_odds(t,y,.2,60),precision_odds(t,y,.2,60),rtol=1e-10,atol=1e-9))
    errors=[]
    for person in range(len(y)):
        cov=.2*np.exp(-np.abs(t[person,:,None]-t[person,None,:])/60)+.8*np.eye(y.shape[1])
        dense=-.5*(np.linalg.slogdet(cov)[1]+y[person]@np.linalg.solve(cov,y[person])-y[person]@y[person])
        errors.append(abs(dense-kalman_odds(t[person:person+1],y[person:person+1],.2,60)[0,-1]))
    checks['dense_gaussian_known_answer']=max(errors)<1e-10
    ct=np.array([[0.,5.,10.,12.]]);ci=np.array([[0,1,1,1]]);cy=np.array([[1.,2.,3.,4.]])
    checks['repeated_anchor_is_undefined']=bool(np.isnan(maker_short_stream(ct,ci,cy,True)[0]) and np.isnan(maker_short_reference(ct,ci,cy,True)[0]))
    ct=np.array([[0.,5.,10.]]);ci=np.array([[0,1,2]]);cy=np.array([[0.,1.,2.]])
    checks['triangle_correlation_known_half']=bool(abs(maker_short_stream(ct,ci,cy,True)[0]-.5)<1e-14 and abs(maker_short_reference(ct,ci,cy,True)[0]-.5)<1e-14)
    checks['independent_centered_correlation']=bool(np.allclose(maker_short_stream(t,ids,y,True),maker_short_reference(t,ids,y,True),rtol=1e-10,atol=1e-8,equal_nan=True))
    write(dest/'CONTROLS.json',checks);return dict(passed=all(checks.values()),checks=checks)

def pilot_exclusion(s,m):return None
