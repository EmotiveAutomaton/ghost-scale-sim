"""Control beliefs and deficit sensitivity, independently built before pre-flight."""
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from .common import rng, logistic, sustained, mean_blocks, censored_blocks, write, digest, audit_controls

PRIORS = ((1., 4.), (4., 1.))
KAPPAS = (.25, .5, 1., 2.)
PARAMS = np.array([(a,b,k) for a,b in PRIORS for k in KAPPAS])


def settings():
    main = [dict(part='Q2', truth=i, order=order, length=n, worlds=300, blocks=12)
            for i in range(8) for order in ('U','C','UC','CU','CUC','UCU') for n in (25,50,150)]
    # The U comparator is separately named and fresh, as required by Q2.b's 24 cells.
    return main + [dict(part='Q2b', truth=i, order=order, length=150, worlds=300, blocks=12)
                   for i in range(8) for order in ('WU','UW','U')]


def generate(s, lane='study', worlds=None):
    n = worlds or s['worlds']; tmax = s['length']*len(s['order']); r = np.random.default_rng(s['seed_override']) if 'seed_override' in s else rng('Q2', lane, s)
    d = np.full(n, 2, int); a = np.full(n, PARAMS[s['truth'],0]); b = np.full(n, PARAMS[s['truth'],1])
    kappa = PARAMS[s['truth'],2]; floor=s.get('tremble',.01); slope=s.get('slope',2.)
    deficits=np.empty((n,tmax),np.int8); acts=np.zeros((n,tmax),np.int8)
    successes=np.zeros_like(acts); arrivals=np.zeros_like(acts); watching=np.zeros(tmax,np.int8)
    for t in range(tmax):
        episode=s['order'][t//s['length']]
        inc=r.random(n)<.3; d=np.minimum(6,d+inc); arrivals[:,t]=inc; deficits[:,t]=d
        if episode in ('W','F'):
            # Watching is not an owned choice or success. The supplied W is successful.
            watching[t]=1; successes[:,t]=int(episode=='W'); a+=.5*int(episode=='W'); b+=.5*int(episode=='F')
        else:
            act=r.random(n)<logistic(slope*(a/(a+b)*kappa*d-1),floor)
            success=act & (r.random(n)<(.05 if episode=='U' else .7))
            acts[:,t]=act; successes[:,t]=success
            a+=success; b+=act & ~success; d=np.maximum(0,d-2*success)
    return dict(deficits=deficits,actions=acts,successes=successes,arrivals=arrivals,watching=watching)


def infer(raw,s):
    d=raw['deficits']; n,steps=d.shape
    a=np.broadcast_to(PARAMS[:,0],(n,8)).copy(); b=np.broadcast_to(PARAMS[:,1],(n,8)).copy()
    ll=np.zeros((n,8)); odds=np.zeros((n,steps)); prior_odds=np.zeros_like(odds); kappa_odds=np.zeros_like(odds)
    same_probability=np.ones((n,8),bool)
    posterior=np.empty((n,8)); success_delta=np.zeros(n); success_n=np.zeros(n)
    for t in range(steps):
        if raw['watching'][t]:
            a+=.5*raw['successes'][:,t,None]; b+=.5*(1-raw['successes'][:,t,None])
        else:
            p=logistic(s.get('slope',2.)*(a/(a+b)*PARAMS[:,2]*d[:,t,None]-1),s.get('tremble',.01))
            same_probability &= (p==p[:,s['truth'],None])
            act=raw['actions'][:,t,None]; success=raw['successes'][:,t,None]
            ll+=np.where(act,np.log(p),np.log1p(-p))
            a+=success; b+=act-success
            # Same-D response is a deterministic feature of the full public record,
            # not an extra independent likelihood factor.
            if t:
                matched=(raw['successes'][:,t-1]==1)&(d[:,t]==d[:,t-1])&(not bool(raw['watching'][t-1]))
                success_n+=matched; success_delta+=matched*(raw['actions'][:,t]-raw['actions'][:,t-1])
        post=np.exp(ll-logsumexp(ll,axis=1,keepdims=True)); posterior=post
        true=s['truth']; odds[:,t]=ll[:,true]-np.max(np.delete(ll,true,axis=1),axis=1)
        pe=logsumexp(ll.reshape(n,2,4),axis=2)-np.log(4)
        ke=logsumexp(ll.reshape(n,2,4),axis=1)-np.log(2)
        prior_odds[:,t]=pe[:,true//4]-pe[:,1-true//4]
        kappa_odds[:,t]=ke[:,true%4]-np.max(np.delete(ke,true%4,axis=1),axis=1)
    same_probability[:,s['truth']]=False
    return dict(same_probability=same_probability.astype(float),separation=sustained(odds),prior_separation=sustained(prior_odds),
                sensitivity_separation=sustained(kappa_odds), final_likelihood=ll,
                prior_read_right=(prior_odds[:,-1]>=3).astype(float),
                prior_read_wrong=(prior_odds[:,-1]<=-3).astype(float),
                prior_correct=(posterior.reshape(n,2,4).sum(2).argmax(1)==s['truth']//4).astype(float),
                sensitivity_correct=(posterior.reshape(n,2,4).sum(1).argmax(1)==s['truth']%4).astype(float),
                push_rate=raw['actions'].sum(1)/max(1,np.sum(raw['watching']==0)),
                success_response_sum=success_delta,success_response_count=success_n)


def summarize(result,s):
    h=len(s['order'])*s['length']; blocks=s['blocks']; rows=[]
    for name,value in result.items():
        if name in ('final_likelihood','same_probability'):continue
        stat=censored_blocks(value,h,blocks) if 'separation' in name else mean_blocks(value,blocks)
        rows.append(dict(setting=s,measure=name,**stat))
    return rows


def controls(destination,pulse=lambda **kw:None):
    from .q2_reference import infer as independent
    checks=[]
    def check(name,passed,**extra):checks.append(dict(name=name,passed=bool(passed),**extra));pulse(control=name)
    check('prior means',np.allclose(PARAMS[[0,4],0]/PARAMS[[0,4],:2].sum(1),[.2,.8]))
    check('zero deficit shared resting rule',len(set(logistic(2*(PARAMS[:,0]/PARAMS[:,:2].sum(1)*PARAMS[:,2]*0-1))))==1)
    s=dict(part='control',truth=6,order='UCW',length=20,worlds=40,blocks=8)
    raw=generate(s,'known-answer'); x=infer(raw,s); y=independent(raw,s)
    error=max(float(np.max(abs(x[k]-y[k]))) for k in x)
    check('independent full short paths',error<1e-9,maximum_error=error)
    neutral={k:v.copy() for k,v in raw.items()};neutral['watching'][:]=1
    v=infer(neutral,s)['final_likelihood']
    check('watching alone has zero type likelihood',np.array_equal(v,np.zeros_like(v)))
    check('upper lower clipping',np.array_equal(logistic(np.array([-1e4,1e4]),.01),[.01,.99]))
    check('sustained is not first crossing',sustained(np.array([[4.,2.,4.,4.],[4.,4.,4.,2.]])).tolist()==[3,5])
    result=audit_controls(checks);write(Path(destination)/'CONTROLS.json',result);return result


def run(destination,mode='original',original=None,pulse=lambda **kw:None,selected=None,lane='study'):
    from .q2_reference import infer as independent
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    rows=[];files={};roster=settings() if selected is None else selected
    for i,s in enumerate(roster):
        name=f'cell-{i:03d}_points.npz'
        if mode=='reconstruct':
            with np.load(Path(original)/name,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        else:
            raw=generate(s,lane);np.savez_compressed(destination/name,**raw);files[name]=digest(destination/name)
        result=(independent if mode=='reconstruct' else infer)(raw,s)
        np.savez_compressed(destination/f'cell-{i:03d}_measures.npz',**result)
        rows+=summarize(result,s);pulse(family='Q2',cell=i+1,cells=len(roster))
    output=dict(family='Q2',rows=rows,no_registered_claim=True,settings=roster)
    write(destination/'SUMMARY.json',output)
    if mode!='reconstruct':write(destination/'RAW_MANIFEST.json',files)
    return output
