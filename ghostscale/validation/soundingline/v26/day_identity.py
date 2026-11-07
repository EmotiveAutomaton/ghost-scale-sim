"""G2: frame-aware and frame-marginalized finite identity readers."""
import itertools
import numpy as np
from .common import rng, simplex, close
from .day_common import choices, draw_menu, posteriors, class_prior, margin, sustained, measure, scores

def hypotheses(pi=None):
    weights=simplex(4);hs=[dict(type='A',k=-1,w=w.tolist()) for w in weights]
    for typ in ('L','P')+ (('Ps',) if pi is not None else ()):
        hs += [dict(type=typ,k=k,w=w.tolist()) for k in range(3) for w in weights if w[k]>=w.max()-1e-12]
    return hs

def policy(menu,frame,hs,pi=None,independent=False):
    menus=np.asarray(menu,float)
    if menus.ndim==2:menus=menus[None,:,:]
    frames=np.broadcast_to(np.asarray(frame),(len(menus),))
    weights=np.array([h['w'] for h in hs]);u=np.einsum('tnk,hk->thn',menus,weights)
    mask=np.ones_like(u,dtype=bool)
    for j,h in enumerate(hs):
        if h['type']=='A':continue
        k=h['k'];cued=np.where(frames<0,k,frames)
        if h['type']=='Ps':u[:,j]+=float(pi)*menus[np.arange(len(menus)),:,cued];continue
        target=np.full(len(menus),k)
        if h['type']=='P':
            served=np.any(menus[np.arange(len(menus)),:,cued]==1,axis=1)
            target=np.where(served,cued,k)
        chosen=menus[np.arange(len(menus)),:,target]==1
        mask[:,j]=np.where(chosen.any(1)[:,None],chosen,True)
    return choices(u,mask=mask,independent=independent)

def blind_policy(menus,hs,cue,pi=None,independent=False):
    return sum(prob*policy(menus,frame,hs,pi,independent) for frame,prob in [(-1,1-3*cue),(0,cue),(1,cue),(2,cue)])

def observations(s,r,truth):
    hs=hypotheses(s.get('pi'));candidate=[h for h in hs if h['type']==truth]
    h=candidate[int(r.integers(len(candidate)))];n=s['records']
    menus=np.stack([draw_menu(r,s['options'],s['specialists']) for _ in range(n)])
    frames=r.choice([-1,0,1,2],size=n,p=[1-3*s['cue'],s['cue'],s['cue'],s['cue']])
    p=policy(menus,frames,[h],s.get('pi'))[:,0]
    chosen=np.array([r.choice(s['options'],p=v) for v in p],dtype=int)
    return dict(hypothesis=h,menus=menus.tolist(),frames=frames.tolist(),choices=chosen.tolist())

def analyze(raw,s,independent=False):
    hs=hypotheses(s.get('pi'));classes=np.array([h['type'] for h in hs]);prior=class_prior(classes)
    menus=np.asarray(raw['menus']);frames=np.asarray(raw['frames']);chosen=np.asarray(raw['choices']);h=raw['hypothesis'];truth=h['type'];n=len(menus)
    answer={};full={}
    for reader,prob in [('frame-aware',policy(menus,frames,hs,s.get('pi'),independent)),('frame-blind',blind_policy(menus,hs,s['cue'],s.get('pi'),independent))]:
        ll=np.cumsum(np.log(prob[np.arange(n),:,chosen]),axis=0);p=posteriors(ll,prior,independent)
        rivals=[t for t in np.unique(classes) if t!=truth]
        paths=np.stack([margin(ll,prior,classes==truth,classes==v) for v in rivals])
        out=dict(records=sustained(paths.min(0)),against_A=None if truth=='A' else sustained(margin(ll,prior,classes==truth,classes=='A')))
        if h['k']>=0:
            identity=np.array([sum(row[j] for j,x in enumerate(hs) if x['k']==h['k'])/sum(row[j] for j,x in enumerate(hs) if x['k']>=0) for row in p])
            out['identity_64']=float(identity[min(63,n-1)])
        cp={t:p[:,classes==t].sum(1) for t in np.unique(classes)}
        out['as_A_64']=float(cp['A'][min(63,n-1)]==max(x[min(63,n-1)] for x in cp.values()))
        out['L_over_P_64']=float(cp['L'][min(63,n-1)]>cp.get('P',cp['L'])[min(63,n-1)])
        out['true_class_64']=float(cp[truth][min(63,n-1)])
        full[reader]=out
    return full

def block(s,identity,independent=False):
    r=rng(*identity);raw=[];out=[];by={}
    for _ in range(s['makers']):
        obs=observations(s,r,s['truth']);a=analyze(obs,s,independent);raw.append(obs)
        for reader,stats in a.items():
            for name,v in stats.items():
                if name=='against_A' and s['truth']=='A':continue
                by.setdefault((reader,name),[]).append(v)
        by.setdefault(('contrasts','records_ratio'),[]).append([a['frame-blind']['records'],a['frame-aware']['records']])
    for (reader,name),values in by.items():
        measure(out,name,values,'median_ratio' if name=='records_ratio' else 'median' if name in ('records','against_A') else 'mean',reader=reader,truth=s['truth'])
    return dict(observations=raw,measures=out)

def design_check(s,identity):
    r=rng('day-check-G2',identity);hs=hypotheses(s.get('pi'));typ='Ps' if s.get('pi') is not None else 'P';makers=[h for h in hs if h['type']==typ];a=[h for h in hs if h['type']=='A'];diffs=[];conflicts=[]
    for i in range(96):
        h=makers[int(r.integers(len(makers)))];menus=np.stack([draw_menu(r,s['options'],s['specialists']) for _ in range(512)])
        p=blind_policy(menus,[h],s['cue'],s.get('pi'))[:,0];q=policy(menus,-1,a)  # A ignores frame.
        tv=np.abs(q-p[:,None,:]).sum(2)/2;best=int(np.argmin(tv.mean(0)));diffs.append(float(np.mean(tv[:,best]>.2)))
        k=h['k'];conflict=[]
        for other in range(3):
            if other!=k:conflict.append((menus[:,:,k]==1).any(1)&(menus[:,:,other]==1).any(1)&~((menus[:,:,k]==1)&(menus[:,:,other]==1)).any(1))
        # The competing cue picks one of the two rival values uniformly. A union
        # over either rival overstates the menu probability for that actual cue.
        conflicts.append(float(np.mean(conflict)))
    vals=dict(conflict_share=float(np.mean(conflicts)),competing_cue_share=2*s['cue'],distinct_share=float(np.mean(diffs)),worlds_above_floor=float(np.mean(np.array(diffs)>=.25)))
    return dict(passed=min(vals[k] for k in ('conflict_share','competing_cue_share','distinct_share'))>=.25,checks=vals,worlds=96,records=512)

def controls():
    hs=hypotheses();w=[.5,.25,.25];l=dict(type='L',k=0,w=w);p=dict(type='P',k=0,w=w);a=dict(type='A',k=-1,w=w)
    menus=np.array([[[1,0,0],[0,1,0],[0,0,1],[.5,.5,0]],[[.5,0,0],[0,.5,0],[0,0,.5],[.5,.5,.5]]]*2)
    close(policy(menus,-1,[l]).tolist(),policy(menus,-1,[p]).tolist(),tol=1e-12)
    close(policy(menus[1],1,[l,p,a]).tolist(),policy(menus[1],1,[a,a,a]).tolist(),tol=1e-12)
    probs=[.55,.15,.15,.15];frames=[-1,0,1,2];seq=[0,1,2,0]
    exact=sum(np.prod([probs[j]*policy(menus[t],frames[j],[p])[0,0,seq[t]] for t,j in enumerate(js)]) for js in itertools.product(range(4),repeat=4))
    blind=blind_policy(menus,[p],.15);assert abs(exact-np.prod(blind[np.arange(4),0,seq]))<1e-12
    assert len(hs)==51 and len(hypotheses(1))==69
    assert policy(menus,0,hs).min()>=.005
    return dict(passed=True,no_competing_cue_alias=True,enumerated_frame_sequence=True,no_full_service_alias=True,positive_floor=True)
