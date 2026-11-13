"""Cross-context own-history reading on V28's small finite world."""
from pathlib import Path
import math
import numpy as np
from scipy.special import logsumexp
from .common import rng,mean_blocks,write,digest,audit_controls

READERS=('Pp-B','Pe-A','Pe-B','MIX')


def settings():
    return [dict(omega=o,K=k,rho=rho,overlap=overlap,worlds=200,blocks=8)
            for o in (1.,.75,.5,.25,0.) for k in (1,4,16)
            for rho in (.2,.3,.4) for overlap in ('low','high')]


def generate(s,lane='study',worlds=None):
    count=worlds or s['worlds'];blocks=s['blocks'];n=count//blocks
    if count%blocks:raise ValueError('unequal blocks')
    arrays={}
    for block in range(blocks):
        r=rng('G1b',lane,s['rho'],s['overlap'],block,n)
        h=r.dirichlet(np.full(12,.3),size=8);fresh=r.dirichlet(np.full(12,.3),size=8)
        hb=s['omega']*h+(1-s['omega'])*fresh
        goals=np.full((4,12),.1/12)
        remaining=list(r.permutation(12)) if s['overlap']=='low' else list(range(12))
        for g in range(4):
            pick=remaining[:3] if s['overlap']=='low' else sorted(remaining,key=lambda x:(-h[g,x],x))[:3]
            goals[g,pick]+=.9/3;remaining=[x for x in remaining if x not in pick]
        maker=r.integers(8,size=n);truth=r.integers(4,size=(n,17))
        a=np.zeros((n,17,12),np.int16);b=np.zeros_like(a)
        for w in range(n):
            for d in range(17):
                u=r.random(64)
                pa=(1-s['rho'])*h[maker[w]]+s['rho']*goals[truth[w,d]]
                pb=(1-s['rho'])*hb[maker[w]]+s['rho']*goals[truth[w,d]]
                a[w,d]=np.bincount(np.searchsorted(np.cumsum(pa),u),minlength=12)
                b[w,d]=np.bincount(np.searchsorted(np.cumsum(pb),u),minlength=12)
        for key,value in dict(hA=h,hB=hb,goals=goals,A=a,B=b,truth=truth,maker=maker).items():
            arrays.setdefault(key,[]).append(value)
    return {k:np.asarray(v) for k,v in arrays.items()}


def readers(raw,s):
    k=s['K'];rho=s['rho'];scores=[];probabilities=[]
    for block in range(len(raw['A'])):
        a=raw['A'][block];b=raw['B'][block];goals=raw['goals'][block];hb=raw['hB'][block]
        page=b[:,0];typical=np.broadcast_to(hb.mean(0),page.shape)
        total_a=page+a[:,1:k+1].sum(1);total_b=b[:,:k+1].sum(1)
        own_a=(total_a+.5)/(total_a.sum(1,keepdims=True)+6)
        own_b=(total_b+.5)/(total_b.sum(1,keepdims=True)+6)
        habits=(typical,own_a,own_b,k/(k+4)*own_a+4/(k+4)*typical)
        values=[];ps=[]
        for habit in habits:
            mix=(1-rho)*habit[:,None,:]+rho*goals[None,:,:]
            ll=np.einsum('wv,wgv->wg',page,np.log(mix));ll-=logsumexp(ll,axis=1,keepdims=True)
            values.append(ll[np.arange(len(page)),raw['truth'][block,:,0]])
            ps.append(np.exp(ll))
        scores.append(np.asarray(values).T);probabilities.append(np.asarray(ps).transpose(1,0,2))
    return np.asarray(scores),np.asarray(probabilities)


def independent(raw,s):
    blocks,worlds=raw['A'].shape[:2];out=np.empty((blocks,worlds,4));prob=np.empty((blocks,worlds,4,4))
    for b in range(blocks):
        for w in range(worlds):
            page=raw['B'][b,w,0].astype(float)
            typical=np.array([sum(raw['hB'][b,:,m])/8 for m in range(12)])
            ca=page+sum((raw['A'][b,w,d].astype(float) for d in range(1,s['K']+1)),np.zeros(12))
            cb=sum((raw['B'][b,w,d].astype(float) for d in range(s['K']+1)),np.zeros(12))
            ha=(ca+.5)/(ca.sum()+6);hb=(cb+.5)/(cb.sum()+6)
            for j,habit in enumerate((typical,ha,hb,(s['K']*ha+4*typical)/(s['K']+4))):
                ll=np.array([math.fsum(float(page[m])*math.log((1-s['rho'])*float(habit[m])+s['rho']*float(raw['goals'][b,g,m]))
                             for m in range(12)) for g in range(4)])
                peak=float(max(ll));normalizer=peak+math.log(math.fsum(math.exp(float(x)-peak) for x in ll))
                out[b,w,j]=ll[int(raw['truth'][b,w,0])]-normalizer;prob[b,w,j]=np.exp(ll-normalizer)
    return out,prob


def infer(raw,s,reference=False):
    score,p=(independent if reference else readers)(raw,s)
    n=score.shape[0]*score.shape[1]
    return dict(scores=score.reshape(n,4),probabilities=p.reshape(n,4,4),
                own_A_minus_typical=(score[:,:,1]-score[:,:,0]).ravel(),
                own_B_minus_own_A=(score[:,:,2]-score[:,:,1]).ravel(),
                MIX_minus_typical=(score[:,:,3]-score[:,:,0]).ravel(),
                MIX_minus_own_A=(score[:,:,3]-score[:,:,1]).ravel())


def run(destination,mode='original',original=None,pulse=lambda **kw:None,selected=None,lane='study'):
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True);rows=[];files={}
    roster=settings() if selected is None else selected
    for i,s in enumerate(roster):
        name=f'cell-{i:03d}_points.npz'
        if mode=='reconstruct':
            with np.load(Path(original)/name,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        else:
            raw=generate(s,lane);np.savez_compressed(dest/name,**raw);files[name]=digest(dest/name)
        result=infer(raw,s,mode=='reconstruct');np.savez_compressed(dest/f'cell-{i:03d}_measures.npz',**result)
        for j,label in enumerate(READERS):rows.append(dict(setting=s,measure='true_goal_log_score',reader=label,**mean_blocks(result['scores'][:,j],s['blocks'])))
        for key in ('own_A_minus_typical','own_B_minus_own_A','MIX_minus_typical','MIX_minus_own_A'):
            rows.append(dict(setting=s,measure=key,**mean_blocks(result[key],s['blocks'])))
        pulse(family='G1b',cell=i+1,cells=len(roster))
    result=dict(family='G1b',rows=rows,no_registered_claim=True,settings=roster)
    write(dest/'SUMMARY.json',result)
    if mode!='reconstruct':write(dest/'RAW_MANIFEST.json',files)
    return result


def controls(destination,pulse=lambda **kw:None):
    s=dict(omega=1.,K=4,rho=.3,overlap='high',worlds=40,blocks=8);raw=generate(s,'known-answer')
    a=infer(raw,s);b=infer(raw,s,True)
    error=max(float(np.max(abs(a[k]-b[k]))) for k in a)
    checks=[dict(name='independent full probability reconstruction',passed=error<1e-9,error=error),
            dict(name='omega one paired histories exactly equal',passed=bool(np.array_equal(raw['A'],raw['B']))),
            dict(name='omega one own readers exactly equal',passed=bool(np.array_equal(a['scores'][:,1],a['scores'][:,2])))]
    flat={k:v.copy() for k,v in raw.items()};flat['goals'][:]=1/12.
    score,p=readers(flat,s)
    checks.append(dict(name='identical goals uniform posterior',passed=bool(np.allclose(p,.25,atol=1e-12))))
    result=audit_controls(checks);write(Path(destination)/'CONTROLS.json',result);return result
