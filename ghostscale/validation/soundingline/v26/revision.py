"""H3: exact shared initial-goal inference, document-local settled goals."""
import itertools,math
import numpy as np
from .common import rng,posterior,scores,record,close
from .pauses import HABITS,weights,law,observe

def emission(h,w0,ws,draft_before,r,mode,final,revised,old,partial=0.):
    draft=law(h,w0 if draft_before else ws,.6,0).sum((1,2))
    target=h*((1-partial)*ws+partial*w0);target/=target.sum()
    chance=r*(ws<1) if draft_before else np.zeros(8)
    if mode=='final':return draft[final]*(1-chance[final])+(draft@chance)*target[final]
    if not revised:return draft[final]*(1-chance[final])
    if mode=='positions':return (draft@chance)*target[final]
    return draft[old]*chance[old]*target[final]

def hypotheses(n,fraction=.5):return list(itertools.product(range(6),range(6),(int(n*fraction),n),(.3,.6,.9),range(3)))

def context(w,n,partial=0.,fraction=.5):
    hs=hypotheses(n,fraction);g0,gs,point,rate,hi=map(np.array,zip(*hs));g0=g0.astype(int);gs=gs.astype(int);hi=hi.astype(int)
    draft=np.array([[law(h,g,.6,0).sum((1,2)) for h in HABITS] for g in w])
    before=draft[g0,hi];after=draft[gs,hi];chance=rate[:,None]*(w[gs]<1)
    target=HABITS[hi]*((1-partial)*w[gs]+partial*w[g0]);target/=target.sum(1,keepdims=True)
    return hs,point,before,after,chance,target

def document_likelihood(w,obs,n,mode,independent=False,partial=0.,ctx=None,fraction=.5):
    hs,point,before,after,chance,target=context(w,n,partial,fraction) if ctx is None else ctx
    values=np.zeros(len(hs))
    for i,(y,revised,old) in enumerate(obs['units']):
        pre=i<point
        if mode=='final':p=np.where(pre,before[:,y]*(1-chance[:,y])+(before*chance).sum(1)*target[:,y],after[:,y])
        elif not revised:p=np.where(pre,before[:,y]*(1-chance[:,y]),after[:,y])
        elif mode=='positions':p=np.where(pre,(before*chance).sum(1)*target[:,y],0.)
        else:p=np.where(pre,before[:,old]*chance[:,old]*target[:,y],0.)
        if independent:
            values+=np.array([math.log(float(v)) if v>0 else -np.inf for v in p])
        else:
            with np.errstate(divide='ignore'):values+=np.log(p)
    if mode=='discovery':values=np.where(point==obs['discovery'],values,-np.inf)
    return values

def draw(r,w,g0,hi,n,rate,partial=0.,fraction=.5):
    gs=g0 if r.random()<.5 else int(r.choice([i for i in range(6) if i!=g0]));point=int(r.choice([int(n*fraction),n]));units=[]
    for i in range(n):
        row,_=observe(r,HABITS[hi],w[g0 if i<point else gs],.6,0,());old=row[0]
        revised=bool(i<point and w[gs,old]<1 and r.random()<rate)
        target=HABITS[hi]*((1-partial)*w[gs]+partial*w[g0]);target/=target.sum()
        y=int(r.choice(8,p=target)) if revised else old;units.append([y,int(revised),old if revised else -1])
    return dict(settled=gs,discovery=point,units=units)

def block(setting,identity,independent=False):
    r=rng(*identity);w=weights(identity);n=setting.get('units',24);rate=setting['r'];partial=setting.get('partial',0.);fraction=setting.get('discovery_fraction',.5);ctx=context(w,n,partial,fraction);hs=ctx[0]
    prior=np.array([(.5 if a==b else .1)/6/2/3/3 for a,b,p,rt,h in hs]);gids=np.array([h[0]*9+h[4]*3+(.3,.6,.9).index(h[3]) for h in hs]);sids=np.array([h[1] for h in hs]);out=[];raw=[];primary_gain=0.;primary_events=0
    levels=setting.get('prefixes',[1,2,4,8,16]);modes=['final','positions','history']+([] if setting.get('omit_discovery') else ['discovery'])
    for maker in range(6):
        g0=maker;hi=maker%3;shared={k:np.zeros(54) for k in modes};docs=[];hits={k:None for k in modes}
        for t in range(max(levels)):
            obs=draw(r,w,g0,hi,n,rate,partial,fraction);docs.append(obs);posts={};gscores={}
            for mode in modes:
                ll=document_likelihood(w,obs,n,mode,independent,partial,ctx,fraction)
                # Previous documents inform only (g0, habit); settled goals, r,
                # and discovery point are marginalized independently per document.
                p=posterior(ll+shared[mode][gids],prior,independent);ps=np.bincount(sids,weights=p,minlength=6);p0=np.bincount(gids,weights=p,minlength=54).reshape(6,9).sum(1)
                scaled=np.exp(ll-np.max(ll));factor=np.bincount(gids,weights=scaled*prior*54,minlength=54)
                with np.errstate(divide='ignore'):shared[mode]+=np.log(factor)
                gscores[mode]=math.log(ps[obs['settled']]*6);posts[mode]=dict(initial=p0.tolist(),settled=ps.tolist())
                if math.log(p0[g0]*6)>=1 and hits[mode] is None:hits[mode]=t+1
                if t+1 in levels:
                    out.append(record(scores(p0,g0),maker=maker,reader=mode,target='initial',history=t+1))
                    out.append(record(scores(ps,obs['settled']),maker=maker,reader=mode,target='settled',history=t+1))
            revisions=sum(x[1] for x in obs['units'])
            if t==0:
                primary_gain+=gscores['history']-gscores['final'];primary_events+=revisions
                out.append(record(dict(total_gain=gscores['history']-gscores['final'],revision_count=revisions),maker=maker,reader='contrasts',history=1))
            obs['posteriors']=posts
        for mode in modes:out.append(record(dict(records_to_one_nat=hits[mode],not_within_16=float(hits[mode] is None)),maker=maker,reader=mode,target='initial',history='threshold'))
        raw.append(dict(initial=g0,habit=hi,documents=docs))
    out.append(record(dict(primary=primary_gain/primary_events if primary_events else None),reader='contrasts',history=1))
    return dict(public=dict(weights=w.tolist(),habit=HABITS.tolist(),r_prior=[.3,.6,.9],settled_prior='same .5, each other .1',n=n),truth_and_observations=raw,readouts=out)

def controls():
    h=HABITS[0];w=np.ones(8)
    for y in range(8):
        for rate in (0,.6):
            a=emission(h,w,w,True,rate,'final',y,0,-1);b=emission(h,w,w,True,rate,'history',y,0,-1);assert abs(a-b)<1e-12
    w0=np.array([.8,1.2]*4);ws=w0[::-1]
    for y in range(8):assert abs(emission(h,w0,ws,True,0,'final',y,0,-1)-emission(h,w0,ws,True,0,'history',y,0,-1))<1e-12
    assert abs(sum(emission(h,w0,ws,True,.6,'final',y,0,-1) for y in range(8))-1)<1e-12
    return dict(passed=True,equal_weights_no_revision=True,zero_revision_identity=True,emission_normalization=True)

def design_check(settings):
    uncertain=0;count=0
    for b in range(16):
        ident=['check-only-H3',b];r=rng(*ident);w=weights(ident);obs=draw(r,w,0,0,24,.6)
        # Check only changed-goal cases, conditioning the draw without selecting scores.
        while obs['settled']==0:obs=draw(r,w,0,0,24,.6)
        hs=hypotheses(24);prior=np.array([(.5 if a==s else .1)/108 for a,s,p,rt,h in hs]);p=posterior(document_likelihood(w,obs,24,'final'),prior)
        ps=np.bincount([x[1] for x in hs],weights=p,minlength=6);uncertain+=ps[obs['settled']]<.8;count+=1
    return dict(passed=uncertain/count>=.5,uncertain_share=uncertain/count,floor=.5,seed_domain='check-only-H3')
