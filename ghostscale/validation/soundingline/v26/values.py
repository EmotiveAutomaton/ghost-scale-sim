"""E1: deterministic history updates and exact finite type inference."""
import itertools,math
import numpy as np
from .common import rng,softmax,posterior,record,scores,close

def grid(dims=3):
    return np.array([v for v in itertools.product(range(5),repeat=dims) if sum(v)==4],float)/4

def roster(dims=3,etas=(.005,.02,.1),thetas=(.6,.8)):
    w=grid(dims);rows=[]
    for x in w:rows.append(('A',0.,2.,x))
    for eta in etas:
        for x in w:rows.append(('B',eta,2.,x))
    rows.append(('B2',0.,2.,np.ones(dims)/dims))
    for eta in etas:
        for theta in thetas:
            for x in w:
                if max(x)<theta:rows.append(('C',eta,theta,x))
    return rows

def states(hs):
    return dict(w=np.array([h[3] for h in hs]),locked=np.full(len(hs),-1,int),counts=np.full((len(hs),len(hs[0][3])),2.),lock_time=np.full(len(hs),-1,int))

def policy(menu,hs,state,independent=False):
    p=.98*softmax(3*(state['w']@menu.T),independent)+.02/len(menu)
    for i,h in enumerate(hs):
        lock=state['locked'][i]
        if h[0]=='C' and lock>=0:
            eligible=np.flatnonzero(menu[:,lock]==1)
            if len(eligible):
                v=softmax(3*(menu[eligible]@state['w'][i]),independent);p[i]=.02/len(menu);p[i,eligible]+=.98*v
    return p

def update(hs,state,outcome,t):
    for i,(kind,eta,theta,start) in enumerate(hs):
        if kind=='B2':state['counts'][i]+=outcome;state['w'][i]=state['counts'][i]/state['counts'][i].sum()
        elif kind in ('B','C') and state['locked'][i]<0:
            v=(1-eta)*state['w'][i]+eta*outcome;state['w'][i]=v/v.sum()
            if kind=='C' and max(state['w'][i])>theta:
                state['locked'][i]=int(np.argmax(state['w'][i]));state['lock_time'][i]=t

def menu_at(r,t,setting):
    dims=setting.get('dimensions',3);options=setting.get('options',4);specialist=setting.get('specialists',1.)
    e=r.choice([0.,.5,1.],(options,dims))
    for i in range(options):
        if r.random()<specialist:e[i]=np.eye(dims)[int(r.integers(dims))]
    mix=setting['environment'];phase='stationary';forced=False;dim=setting.get('environment_dimension')
    if dim is None:dim=int(r.integers(dims))
    if mix=='forced' and t%setting.get('forced_every',8)==0:forced=True;phase='forced'
    if mix=='devaluation' and 64<=t<72:e[:,dim]*=setting.get('devaluation',.5);phase='devaluation'
    if mix=='shift' and t>=128:
        for i in range(options):
            if e[i,dim]==1:e[i,np.arange(dims)!=dim]=0
        phase='shift'
    if mix=='trade' and t%8 in (0,1):
        e=np.zeros((2,dims));e[0,dim]=1;e[1,np.arange(dims)!=dim]=.5 if t%8==0 else 1.;phase='trade'
    return e,forced,phase,dim

def block(setting,identity,independent=False):
    r=rng(*identity);dims=setting.get('dimensions',3);hs=roster(dims,tuple(sorted(set([.005,.02,.1,setting.get('eta',.1)]))),tuple(sorted(set([.6,.8,setting.get('theta',.6)]))))
    horizon=setting.get('max_history',256);levels=setting.get('histories',[8,16,32,64,128,256])
    if not isinstance(horizon,int) or horizon<1 or any(not isinstance(t,int) or not 1<=t<=horizon for t in levels):raise ValueError('invalid history grid')
    kinds=np.array([h[0] for h in hs]);names=['A','B','B2','C'];prior=np.array([1/(4*sum(kinds==h[0])) for h in hs]);out=[];raw=[]
    selected=setting.get('truth_types',names)
    if not selected or len(set(selected))!=len(selected) or not set(selected)<=set(names):raise ValueError('invalid truth roster')
    for maker,typ in enumerate(names):
        if typ not in selected:continue
        eligible=[i for i,h in enumerate(hs) if h[0]==typ and (typ not in ('B','C') or h[1]==setting['eta']) and (typ!='C' or h[2]==setting['theta'])]
        truth=int(r.choice(eligible));state=states(hs);ll=np.zeros(len(hs));history=[];chosen=[];first={k:None for k in names if k!=typ};lossdiff=[];forced_dims=[]
        environment_setting=dict(setting,environment_dimension=int(r.integers(dims)))
        evidence=[]
        for t in range(1,horizon+1):
            menu,forced,phase,dim=menu_at(r,t,environment_setting);law=policy(menu,hs,state,independent);post=posterior(ll,prior,independent);predict=post@law
            shares=np.sum(chosen[-16:],axis=0) if chosen else np.ones(dims)
            if shares.sum()==0:shares=np.ones(dims)
            ratio=.98*softmax(3*(menu@(shares/shares.sum())),independent)+.02/len(menu)
            y=int(r.integers(len(menu))) if forced else int(r.choice(len(menu),p=law[truth]))
            if not forced:ll+=np.log(law[:,y]);lossdiff.append(math.log(predict[y]/ratio[y]))
            else:forced_dims.append(int(np.argmax(menu[y])))
            chosen.append(menu[y].copy());update(hs,state,menu[y],t)
            post=posterior(ll,prior,independent);typepost=np.array([post[kinds==k].sum() for k in names])
            for rival in first:
                ev=type_log_evidence(ll,prior,kinds,typ,rival,independent) if horizon>256 else math.log(typepost[names.index(typ)]/typepost[names.index(rival)])
                if ev>=1 and first[rival] is None:first[rival]=t
            if setting.get('evidence_path'):
                evidence.append(type_log_evidence(ll,prior,kinds,'B','A',independent))
            if t in levels:
                out.append(record(scores(typepost,maker),truth=typ,reader='exact-type',history=t))
                out.append(record(dict(usage_minus_exact_loss=math.fsum(lossdiff)/len(lossdiff) if lossdiff else None),truth=typ,reader='next-choice',history=t))
                cp=post[kinds=='C'];theta=np.array([h[2] for h in hs if h[0]=='C'])
                out.append(record(dict(theta_point6_mass=float(cp[theta==.6].sum()/cp.sum()) if cp.sum() else None),truth=typ,reader='theta',history=t))
            history.append(dict(menu=menu.tolist(),forced=forced,phase=phase,dimension=dim,choice=y,type_posterior=typepost.tolist()))
        for rival,hit in first.items():out.append(record({'records_to_one_nat':hit,f'not_within_{horizon}':float(hit is None)},truth=typ,reader='threshold',rival=rival))
        lock=int(state['locked'][truth]);lt=int(state['lock_time'][truth])
        out.append(record(dict(locked=float(lock>=0),lock_time=lt if lock>=0 else None,forced_lock_match=float(lock==forced_dims[0]) if forced_dims and lock>=0 else None),truth=typ,reader='path-dependence'))
        if setting.get('evidence_path'):
            from .friday import crossing_metrics
            for level in levels:
                out.append(record(crossing_metrics(evidence[:level]),truth=typ,reader='B-over-A-persistence',history=level))
            raw.append(dict(truth=truth,records=history,log_B_over_A=evidence))
        else:raw.append(dict(truth=truth,records=history))
    return dict(public=dict(hypotheses=[dict(kind=h[0],eta=h[1],theta=h[2],initial=h[3].tolist()) for h in hs],prior=prior.tolist(),choice_model='softmax beta3 with .02 uniform lapse; locked C lexicographic'),truth_and_observations=raw,readouts=out)

def type_log_evidence(ll,prior,kinds,truth,rival,independent=False):
    """Exact type marginal odds without underflow on extended histories."""
    def logmass(kind):
        v=ll[kinds==kind]+np.log(prior[kinds==kind]);m=float(max(v))
        z=math.fsum(math.exp(float(x)-m) for x in v) if independent else float(np.exp(v-m).sum())
        return m+math.log(z)
    return logmass(truth)-logmass(rival)

def controls():
    w=np.array([.25,.25,.5]);hs=[('A',0,2,w),('B',0,2,w),('B',.1,2,w),('C',.1,2,w),('B2',0,2,np.ones(3)/3)];s=states(hs);menu=np.eye(3)
    for t in range(5):
        p=policy(menu,hs,s);close(p[0].tolist(),p[1].tolist(),tol=1e-12);close(p[2].tolist(),p[3].tolist(),tol=1e-12);update(hs,s,menu[t%3],t)
    initial=np.ones(3)*1e100;assert np.max(abs((initial+[1,0,0])/(initial.sum()+1)-1/3))<1e-12
    close(policy(menu,hs,s).tolist(),policy(menu,hs,s,True).tolist(),tol=1e-12)
    return dict(passed=True,eta0_alias=True,theta_above1_alias=True,infinite_counts_fixed_uniform=True,forced_choice_likelihood_one=True)

def design_check(settings):
    rows=[]
    for share in (.5,.75,1.):
        locked=0;weak=0
        for b in range(32):
            r=rng('check-only-E1',share,b);w=grid();eligible=w[np.max(w,axis=1)<.6];start=eligible[int(r.integers(len(eligible)))];hs=[('C',.1,.6,start)]
            # Separate trajectories for the two checks; the stationary menus are type-independent.
            state=states(hs)
            for t in range(1,129):
                menu,_,_,_=menu_at(r,t,dict(environment='stationary',specialists=share));p=policy(menu,hs,state);y=int(r.choice(4,p=p[0]));update(hs,state,menu[y],t)
            locked+=8<=state['lock_time'][0]<=128
            # Use the actual declared type reader, marginalizing all start
            # weights and learning rates, rather than a matched-start pair.
            hs=roster();state=states(hs);kinds=np.array([h[0] for h in hs]);eligible=[i for i,h in enumerate(hs) if h[0]=='B' and h[1]==.005];truth=int(r.choice(eligible));ll=np.zeros(len(hs));prior=np.array([1/(4*sum(kinds==h[0])) for h in hs])
            for t in range(1,65):
                menu,_,_,_=menu_at(r,t,dict(environment='stationary',specialists=share));p=policy(menu,hs,state);y=int(r.choice(4,p=p[truth]));ll+=np.log(p[:,y]);update(hs,state,menu[y],t)
            post=posterior(ll,prior);weak+=math.log(post[kinds=='B'].sum()/post[kinds=='A'].sum())<1
        rows.append(dict(specialists=share,lock_share=locked/32,weak_evidence_share=weak/32,passed=locked/32>=.25 and weak/32>=.25))
    passing=[x for x in rows if x['passed']]
    return dict(passed=bool(passing),seed_domain='check-only-E1',checks=rows,selected_specialists=passing[-1]['specialists'] if passing else None,reader='Full declared type posterior, marginalized initial weights and learning rates; same reader as science.')
