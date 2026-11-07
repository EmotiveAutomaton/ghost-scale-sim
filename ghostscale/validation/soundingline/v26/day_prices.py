"""G4: adaptive finite trade ladder and time-dependent lock-state inference."""
import numpy as np
from scipy.special import logsumexp
from .common import rng,simplex,close
from .day_common import choices,draw_menu,posteriors,class_prior,margin,sustained,measure

def hypotheses(known_tau=None):
    grid=simplex(4);weights=list(grid)+[np.roll([.9,.05,.05],k) for k in range(3)]
    hs=[dict(type='A',w=w.tolist(),k=-1,tau=-1) for w in weights]
    for k in range(3):
        for w in grid:
            if w[k]>=w.max()-1e-12:
                for tau in ([known_tau] if known_tau is not None else [0,32,64,128,256]):hs.append(dict(type='C',w=w.tolist(),k=k,tau=tau))
    return hs

def policy(menu,t,hs,precision=3.,independent=False):
    menus=np.asarray(menu,float)
    if menus.ndim==2:menus=menus[None,:,:]
    times=np.broadcast_to(np.asarray(t),(len(menus),));u=np.einsum('tnk,hk->thn',menus,np.array([h['w'] for h in hs]));mask=np.ones_like(u,bool)
    for j,h in enumerate(hs):
        if h['type']!='C':continue
        service=menus[:,:,h['k']]==1;lock=(times>=h['tau'])&service.any(1);mask[:,j]=np.where(lock[:,None],service,True)
    return choices(u,precision,mask=mask,independent=independent)

def observations(s,r):
    typ=s['truth'];k=int(r.integers(3));tau=int(r.choice([0,32,64,128]))
    if typ=='C':
        grid=[w for w in simplex(4) if w[k]==.5 and w[k]>=w.max()];w=grid[int(r.integers(len(grid)))];h=dict(type='C',w=w.tolist(),k=k,tau=tau)
    else:
        w=np.zeros(3)
        if typ=='A75':w[k]=.75;w[int(r.choice([j for j in range(3) if j!=k]))]=.25
        elif typ=='A9':w[:]=.05;w[k]=.9
        else:w[k]=1.
        h=dict(type='A',w=w.tolist(),k=-1,tau=-1)
    records=[];past=[]
    for t in range(1,s['records']+1):
        offer=r.random()<s['rho'];target=None;m=None
        if offer:
            counts=np.sum(np.array(past[-16:])==1,axis=0) if past else np.zeros(3);target=int(r.choice(np.flatnonzero(counts==counts.max())));m=float(r.choice(s['ladder']))
            menu=np.zeros((2,3));menu[0,target]=1;menu[1]=m;menu[1,target]=0
        else:menu=draw_menu(r,4,s['specialists'])
        p=policy(menu,t,[h],s['precision'])[0,0];ch=int(r.choice(len(menu),p=p));past.append(menu[ch].tolist())
        records.append(dict(menu=menu.tolist(),choice=ch,offer=bool(offer),target=target,price=m))
    return dict(hypothesis=h,formal_type=typ,top=k,known_tau=tau if s['tau_known'] else None,records=records)

def analyze(raw,s,independent=False):
    hs=hypotheses(raw['known_tau']);classes=np.array([h['type'] for h in hs]);prior=class_prior(classes);records=raw['records'];n=len(records);incr=np.empty((n,len(hs)))
    for is_offer in (False,True):
        ids=[i for i,x in enumerate(records) if x['offer']==is_offer]
        if not ids:continue
        menus=np.array([records[i]['menu'] for i in ids]);ch=np.array([records[i]['choice'] for i in ids]);p=policy(menus,np.array(ids)+1,hs,s['precision'],independent);incr[ids]=np.log(p[np.arange(len(ids)),:,ch])
    offers=np.array([x['offer'] for x in records]);results={};truth=raw['formal_type'];taus=np.array([h['tau'] for h in hs]);c=classes=='C'
    true_locked=c[0]  # overwritten below; no evaluator truth enters a likelihood.
    for reader,mask in [('full',np.ones(n,bool)),('offers',offers),('ordinary',~offers)]:
        ll=np.cumsum(incr*mask[:,None],axis=0);p=posteriors(ll,prior,independent);ac=margin(ll,prior,~c,c)
        locked=(np.arange(1,n+1)[:,None]>=taus)&c[None,:];mass=np.sum(p*locked,axis=1)
        logweights=ll+np.log(prior)
        lockmargin=logsumexp(np.where(locked,logweights,-np.inf),axis=1)-logsumexp(np.where(~locked,logweights,-np.inf),axis=1)
        if truth=='C':
            start=max(1,raw['hypothesis']['tau']);delay=sustained(lockmargin[start-1:]);cross=None if delay is None else start+delay-1
        else:delay=None;cross=sustained(-lockmargin)
        classcross=sustained(-ac if truth=='C' else ac)
        a=dict(class_records=classcross,lock_records=cross,lock_delay=delay,offers_needed=int(offers[:classcross].sum()) if classcross is not None else None,final_A_margin=float(ac[-1]))
        for t in (48,96,192,384):
            if t<=n:
                true_locked=truth=='C' and t>=raw['hypothesis']['tau'];v=mass[t-1] if true_locked else 1-mass[t-1]
                a['lock_posterior_'+str(t)]=float(v)
                # Compute log probability directly to avoid clipping a tiny probability.
                logs=ll[t-1]+np.log(prior)
                keep=locked[t-1] if true_locked else ~locked[t-1];a['lock_logscore_'+str(t)]=float(logsumexp(logs[keep])-logsumexp(logs))
        results[reader]=a
    return results

def block(s,identity,independent=False):
    r=rng(*identity);raw=[];out=[];vals={}
    for _ in range(s['makers']):
        o=observations(s,r);raw.append(o);a=analyze(o,s,independent)
        for reader,items in a.items():
            for name,v in items.items():
                if name=='lock_delay' and s['truth']!='C':continue
                vals.setdefault((reader,name),[]).append(v)
        x,y=a['ordinary']['lock_records'],a['full']['lock_records'];vals.setdefault(('contrasts','records_saved'),[]).append([x,y]);vals.setdefault(('contrasts','nothing_saved'),[]).append(float((x is None and y is None) or (x is not None and y is not None and x<=y)))
        offers=[x for x in o['records'] if x['offer']];vals.setdefault(('design','off_target_share'),[]).append([sum(x['target']!=o['top'] for x in offers),len(offers)])
    for (reader,name),v in vals.items():
        kind='median_difference' if name=='records_saved' else 'ratio_of_sums' if name=='off_target_share' else 'median' if name.endswith(('records','delay','needed')) else 'mean'
        measure(out,name,v,kind,reader=reader,truth=s['truth'])
    return dict(observations=raw,measures=out)

def design_check(s,identity):
    r=rng('day-check-G4',identity);hs=hypotheses();a=[h for h in hs if h['type']=='A'];shares=[]
    for _ in range(96):
        k=int(r.integers(3));ws=[w for w in simplex(4) if w[k]==.5 and w[k]>=w.max()];h=dict(type='C',k=k,tau=0,w=ws[int(r.integers(len(ws)))].tolist())
        menus=np.stack([draw_menu(r,4,s['specialists']) for _ in range(512)]);p=policy(menus,1,[h],s['precision'])[:,0];q=policy(menus,1,a,s['precision']);tv=np.abs(q-p[:,None,:]).sum(2)/2;best=np.argmin(tv.mean(0));shares.append(float(np.mean(tv[:,best]>.2)))
    avg=float(np.mean(shares));worlds=float(np.mean(np.array(shares)>=.25));ladder=float(np.mean(np.array(s['ladder'])>3))
    return dict(passed=bool(avg>=.25 and worlds>=.75 and ladder>=.25),checks=dict(distinct_share=avg,worlds_above_floor=worlds,price_above_three=ladder),worlds=96)

def controls():
    for m in (1,2,4,8):
        menu=np.array([[1.,0,0],[0,m,m]])
        for w in ([.9,.05,.05],[.75,.25,0],[1.,0,0]):
            h=dict(type='A',w=w,k=-1,tau=-1);expected=.98/(1+np.exp(3*(w[0]-m*(1-w[0]))))+.01
            assert abs(policy(menu,1,[h])[0,0,1]-expected)<1e-12
        assert abs(policy(menu,1,[dict(type='C',w=[.5,.25,.25],k=0,tau=0)])[0,0,1]-.01)<1e-12
    menu=draw_menu(rng('day-fixture-G4'));a=dict(type='A',w=[.5,.25,.25],k=-1,tau=-1);c=dict(type='C',w=[.5,.25,.25],k=0,tau=100)
    close(policy(menu,4,[a]).tolist(),policy(menu,4,[c]).tolist(),tol=1e-12)
    hs=hypotheses();assert len(hs)==108;prior=class_prior([h['type'] for h in hs]);assert abs(prior[:18].sum()-.5)<1e-12
    assert policy(menu,5,hs).min()>=.005
    return dict(passed=True,closed_form_offer_acceptance=True,future_lock_alias=True,equal_class_priors=True,positive_floor=True)
