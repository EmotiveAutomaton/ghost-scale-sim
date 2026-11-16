"""G4: distinguish three defined sources of an identity-related bad feeling."""
import numpy as np
from .common import rng,write,rows_for
from .t_fluency import make_path

def generate(s,lane,pulse=lambda **kw:None):
    paths=[]
    for w in range(s['worlds']):
        g=rng('G4',s,lane,w);t=s['steps'];k=g.random(t);offer=np.where(g.random(t)<s['offer_rate'],128.,-1.)
        act=make_path(k,np.zeros(t),offer,g.random(t),s['attention'],64.,2.)
        cues=g.random((t,3))<s['cue_rate']
        if s['tied']:cues[:]=cues[:,:1]
        p=.05+.7*(1-act)*cues[:,s['truth']];show=g.random(t)<p
        paths.append(np.column_stack((k,offer,act,cues,show)));pulse(world=w+1,phase='generate')
    return dict(records=np.array(paths))

def infer(raw,s,reference=False,pulse=lambda **kw:None):
    out={k:[] for k in ['separation','wrongly_settled','opposite_share','telling_opportunities','alias']}
    for w,path in enumerate(raw['records']):
        cues=path[:,3:6] if s['view']=='CONTEXT' else np.full((len(path),3),s['cue_rate'])
        p=.05+.7*(1-path[:,2,None])*cues;show=path[:,-1,None]
        terms=show*np.log(p)+(1-show)*np.log1p(-p) if reference else np.log(np.where(show,p,1-p))
        ll=np.cumsum(terms,axis=0).T;gap=ll[s['truth']]-np.max(np.delete(ll,s['truth'],0),axis=0);hit=np.flatnonzero(np.minimum.accumulate(gap[::-1])[::-1]>=3)
        out['separation'].append(int(hit[0]+1) if len(hit) else s['steps']+1);final=ll[:,-1];order=np.sort(final)
        out['wrongly_settled'].append(float(np.argmax(final)!=s['truth'] and order[-1]-order[-2]>=3));out['alias'].append(float(np.max(abs(ll-ll[:1]))<1e-10))
        out['opposite_share'].append(float(np.mean(1-path[:,2])));out['telling_opportunities'].append(float(np.mean((1-path[:,2])*(np.ptp(cues,axis=1)>0))))
        pulse(world=w+1,phase='infer')
    return {k:np.array(v) for k,v in out.items()}
def summarize(s,m):return rows_for(s,m)
def pilot_exclusion(s,m):
    if np.all(m['alias']==1):return 'unseen or perfectly tied source cues give exact source alias'
    if np.all(m['separation']<=10):return 'all forty worlds separate within ten occasions'
    return None
def controls(dest,pulse=lambda **kw:None):
    s=dict(steps=100,worlds=2,truth=0,offer_rate=.2,attention=.9,cue_rate=.5,tied=False,view='SHOWS')
    raw=generate(s,'control');a=infer(raw,s);b=infer(raw,dict(s,view='CONTEXT'));ref=infer(raw,dict(s,view='CONTEXT'),True)
    checks={'unseen_sources_alias':bool(np.all(a['alias']==1)),'independent_likelihood':all(np.allclose(b[k],ref[k],atol=1e-10,rtol=0) for k in b),'opposite_has_support':bool(np.all(a['opposite_share']>0))}
    write(dest/'CONTROLS.json',checks);return dict(passed=all(checks.values()),checks=checks)
