"""H2: proposal/redraw and difficulty are distinct latent causes."""
import math
import numpy as np
from .common import rng,posterior,scores,record,close

HABITS=np.array([[.125]*8,[.2,.2,.15,.15,.1,.1,.05,.05],[.05,.1,.15,.2,.2,.15,.1,.05]])

def weights(identity,low=.5,high=2.,n=6):
    r=rng('goal-weights',*identity);w=[]
    for _ in range(n):
        for trial in range(1000):
            x=np.exp(r.uniform(math.log(low),math.log(high),8))
            if sum(x<1)>=2:w.append(x);break
        else:raise ValueError('goal redraw exhausted')
    return np.array(w)

def law(h,w,d,q,hard=(0,1),independent=False):
    # outcome axes: final type, pause flag, replaced proposal (8 = not redrawn).
    out=np.zeros((8,2,9));target=h*w;target/=target.sum()
    for old in range(8):
        deliberation=d if w[old]<1 else 0.
        for y in range(8):out[y,1,old]+=h[old]*deliberation*target[y]
        difficulty=q if old in hard else 0.
        out[old,0,8]+=h[old]*(1-deliberation)*(1-difficulty)
        out[old,1,8]+=h[old]*(1-deliberation)*difficulty
    return out

def observe(r,h,w,d,q,hard):
    old=int(r.choice(8,p=h));deliberated=w[old]<1 and r.random()<d
    y=int(r.choice(8,p=h*w/(h@w))) if deliberated else old
    pause=bool(deliberated or (y in hard and r.random()<q))
    return [y,int(pause),old if deliberated else 8],int(deliberated)

def block(setting,identity,independent=False):
    w=weights(identity,setting.get('low',.5),setting.get('high',2.));habits=HABITS.copy()
    if setting.get('dominant'):habits=np.array([[.65]+[.05]*7,[.05,.65]+[.05]*6,[.05]*7+[.65]])
    d,q=setting['d'],setting['q'];hard=tuple(range(setting.get('hard',2)));n=setting.get('units',32)
    laws=np.array([law(h,g,d,q,hard,independent) for g in w for h in habits]);wrong=np.array([law(h,g,d,0,hard,independent) for g in w for h in habits]);r=rng('data',*identity);out=[];raw=[]
    gains=[];mis=[];rev=[];nd=0;npause=0
    for maker in range(6):
        gi=maker;hi=maker%3;obs=[];delibs=0
        for _ in range(n):
            row,flag=observe(r,habits[hi],w[gi],d,q,hard);obs.append(row);delibs+=flag
        ll={k:np.zeros(18) for k in ('text','pauses','replaced','misread') if k!='replaced' or not setting.get('omit_replacements')}
        for y,p,old in obs:
            for name,probs in [('text',laws[:,y].sum((1,2))),('pauses',laws[:,y,p].sum(1)),('replaced',laws[:,y,p,old]),('misread',wrong[:,y,p].sum(1))]:
                if name not in ll:continue
                with np.errstate(divide='ignore'):ll[name]+=np.log(probs)
        posts={k:posterior(v,independent=independent).reshape(6,3).sum(1) for k,v in ll.items()}
        for name,p in posts.items():out.append(record(scores(p,gi),maker=maker,reader=name,history=n))
        gain=math.log(posts['pauses'][gi]/posts['text'][gi]);bad=math.log(posts['misread'][gi]/posts['text'][gi]);extra=math.log(posts['replaced'][gi]/posts['pauses'][gi]) if 'replaced' in posts else None
        gains.append(gain);mis.append(bad);rev.append(extra);nd+=delibs;npause+=sum(x[1] for x in obs)
        raw.append(dict(goal=gi,habit=hi,observations=obs,deliberations=delibs,posteriors={k:v.tolist() for k,v in posts.items()}))
    out.append(record(dict(primary=math.fsum(gains)/nd if nd else None,misread_per_pause=math.fsum(mis)/npause if npause else None,replacement_gain=math.fsum(rev)/6 if all(v is not None for v in rev) else None,total_gain=math.fsum(gains)/6,deliberation_count=nd,pause_count=npause),reader='contrasts',history=n))
    return dict(public=dict(weights=w.tolist(),habits=habits.tolist(),hard=hard,choice_model='habit proposal; conditional one redraw; independent difficulty pauses'),truth_and_observations=raw,readouts=out)

def controls():
    h=HABITS[0];w=np.array([.8,.8,1.2,1.1,.9,1.3,.7,1.])
    a=law(h,w,0,0);assert abs(a.sum()-1)<1e-12 and a[:,1].sum()==0
    close(a.sum((1,2)).tolist(),h.tolist(),tol=1e-12)
    one=law(h,np.ones(8),.6,.2);assert np.all(one[:,:,0:8]==0)
    # Enumerate proposed unit and branch independently to check the text marginal.
    d=.6;expected=h*(1-d*(w<1))+(h@(d*(w<1)))*h*w/(h@w)
    close(law(h,w,d,.5).sum((1,2)).tolist(),expected.tolist(),tol=1e-12)
    return dict(passed=True,no_pause_identity=True,equal_weights_prior=True,enumerated_branch_marginal=True)

def design_check(settings):
    low,high=.5,2.;count=0
    for b in range(16):
        ident=['check-only-H2',b];w=weights(ident,low,high);r=rng(*ident);gi=int(r.integers(6));hi=int(r.integers(3));laws=np.array([law(h,g,.6,.2).sum((1,2)) for g in w for h in HABITS]);ll=np.zeros(18)
        for _ in range(32):
            obs,_=observe(r,HABITS[hi],w[gi],.6,.2,(0,1));ll+=np.log(laws[:,obs[0]])
        p=posterior(ll).reshape(6,3).sum(1);count+=p[gi]<.8
    return dict(passed=count/16>=.5,uncertain_share=count/16,floor=.5,seed_domain='check-only-H2',declared_range=[low,high])
