"""Rank versus level, with explicit observation boundaries and checked particles."""
import math,zlib,json
import numpy as np
from scipy.special import expit,logsumexp
from .common import rng,write,rows_for

WEIGHTS=np.linspace(0,1,5)

def generate(s,lane,pulse=lambda **kw:None):
    allrows=[]
    for w in range(s['worlds']):
        g=rng('Q',s,lane,w);standing=np.zeros(64);members=np.arange(64);pos=0;rows=[]
        for t in range(s['steps']):
            oldmean=standing[members[pos//8*8:pos//8*8+8]].mean()
            move=g.random()<s['move_rate']
            if move:
                targets=np.flatnonzero(np.arange(64)//8!=pos//8);dest=g.choice(targets)
                members[pos],members[dest]=members[dest],members[pos];pos=dest
            personal=g.choice([-1.,0.,1.],64,p=[.05,.9,.05])
            group=g.choice([-1.,0.,1.],8,p=[s['group_rate']/2,1-s['group_rate'],s['group_rate']/2])
            increment=personal.copy()
            for k in range(8):increment[members[8*k:8*k+8]]+=group[k]
            standing+=increment;neighbours=members[pos//8*8:pos//8*8+8]
            meanchange=standing[neighbours].mean()-oldmean
            change=increment[0]-s['weight']*meanchange
            show=g.random()<expit(-2.-2.*change)
            neighbour_events=np.sort(increment[neighbours[neighbours!=0]])
            # Only V3 may read meanchange and movement. V2 receives anonymous
            # neighbours' events; V1 receives only the focal event and show.
            rows.append(np.r_[increment[0],show,meanchange,move,neighbour_events])
        allrows.append(rows);pulse(world=w+1,phase='generate')
    return dict(records=np.array(allrows))

def probability(delta,reference=False):
    if not reference:return expit(-2.-2.*delta)
    z=-2.-2.*np.asarray(delta)
    return np.exp(-np.logaddexp(0,-z))

def log_evidence(p,show,reference=False):
    if reference:
        return np.log(p) if show else np.log1p(-p)
    return np.log(np.maximum(p if show else 1-p,1e-300))

def exact_nomove(path,s,weight,reference=False):
    prior=np.array([.05,.9,.05]);sums=np.array([1.])
    for _ in range(7):sums=np.convolve(sums,prior)
    support=np.arange(-7,8);pg=np.array([s['group_rate']/2,1-s['group_rate'],s['group_rate']/2]);values=[]
    for row in path:
        if s['view']=='V2':
            delta=row[0]-weight*(row[0]+row[4:].sum())/8
            z=-2-2*delta;signed=z if row[1] else -z;values.append(float(-np.logaddexp(0,-signed)));continue
        mixture=[];mass=[]
        for j,group in enumerate((-1,0,1)):
            own=row[0]-group;prob=pg[j]*(prior[int(own+1)] if own in (-1,0,1) else 0)
            if not prob:continue
            delta=row[0]-weight*(row[0]+7*group+support)/8
            z=-2-2*delta;sign=z if row[1] else -z
            likelihood=np.exp(-np.logaddexp(0,-sign)) if reference else expit(sign)
            mixture.append(prob*float(sums@likelihood));mass.append(prob)
        values.append(float(np.log(math.fsum(mixture)/math.fsum(mass))))
    return np.cumsum(values)

def particle_path(path,s,weight,seed,reference=False):
    n=s['particles'];g=np.random.default_rng(seed);standing=np.zeros((n,64));members=np.tile(np.arange(64),(n,1));pos=np.zeros(n,dtype=int)
    result=[];total=0.;base=np.arange(n);own=np.array([0])
    pr=np.array([.05,.9,.05]);pg=np.array([s['group_rate']/2,1-s['group_rate'],s['group_rate']/2]);support=np.array([-1.,0.,1.])
    for row in path:
        idx=pos[:,None]//8*8+np.arange(8);groupids=np.take_along_axis(members,idx,axis=1)
        oldmean=np.take_along_axis(standing,groupids,axis=1).mean(1)
        move=g.random(n)<s['move_rate'];other=(pos//8+g.integers(1,8,n))%8;dest=other*8+g.integers(0,8,n)
        a=members[base,pos].copy();b=members[base,dest].copy()
        members[base[move],pos[move]]=b[move];members[base[move],dest[move]]=a[move];pos=np.where(move,dest,pos)
        personal=g.choice(support,(n,64),p=pr);global_event=g.choice(support,(n,8),p=pg)
        current=pos//8
        observed=np.r_[row[0],row[4:]] if s['view']=='V2' else np.array([row[0]])
        posterior=[]
        for value,p0 in zip(support,pg):
            pp=p0
            for val in observed:
                delta=val-value;pp*=pr[int(delta+1)] if delta in support else 0.
            posterior.append(pp)
        posterior=np.array(posterior);posterior/=posterior.sum()
        global_event[base,current]=g.choice(support,n,p=posterior)
        increments=personal.copy()
        for k in range(8):
            ids=members[:,k*8:k*8+8];increments[base[:,None],ids]+=global_event[:,k,None]
        increments[:,0]=row[0]
        if s['view']=='V2':
            ids=np.take_along_axis(members,current[:,None]*8+np.arange(8),axis=1)
            others=ids[ids!=0].reshape(n,7)
            permutations=np.argsort(g.random((n,7)),axis=1)
            increments[base[:,None],others]=row[4:][permutations]
        standing+=increments
        ids=np.take_along_axis(members,current[:,None]*8+np.arange(8),axis=1)
        meanchange=np.take_along_axis(standing,ids,axis=1).mean(1)-oldmean
        z=-2.-2.*(row[0]-weight*meanchange)
        signed=z if row[1] else -z
        likelihood=np.exp(-np.logaddexp(0,-signed)) if reference else expit(signed)
        norm=math.fsum(likelihood.tolist())/n if reference else float(likelihood.mean())
        if not np.isfinite(norm) or norm<=0:raise ArithmeticError('hidden-rank filter lost support')
        total+=np.log(norm);result.append(total)
        weights=likelihood/likelihood.sum();cdf=np.cumsum(weights);cdf[-1]=1
        positions=(g.random()+np.arange(n))/n;take=np.searchsorted(cdf,positions)
        standing=standing[take];members=members[take];pos=pos[take]
    return np.array(result)

def infer(raw,s,reference=False,pulse=lambda **kw:None):
    result={k:[] for k in ('separation','wrongly_settled','weight_error','shows_without_own_event','own_event_explains','alias')}
    truth=0 if s['weight']==0 else 2 if s['weight']==1 else 1
    for w,path in enumerate(raw['records']):
        if s['view']=='V3':
            delta=path[:,0,None]-path[:,2,None]*WEIGHTS
            z=-2.-2.*delta;signed=np.where(path[:,1,None]>0,z,-z)
            terms=-np.logaddexp(0,-signed) if not reference else -np.maximum(0,-signed)-np.log1p(np.exp(-np.abs(signed)))
            ll=np.cumsum(terms,axis=0).T
        elif s['move_rate']==0:
            ll=np.array([exact_nomove(path,s,weight,reference) for weight in WEIGHTS])
        else:
            ll=np.array([particle_path(path,s,weight,zlib.crc32(f'Q-filter-{w}-{k}'.encode()),reference) for k,weight in enumerate(WEIGHTS)])
        classes=np.stack((ll[0],logsumexp(ll[1:4],axis=0)-np.log(3),ll[4]))
        gap=classes[truth]-np.max(np.delete(classes,truth,0),axis=0);hit=np.flatnonzero(np.minimum.accumulate(gap[::-1])[::-1]>=3);final=classes[:,-1];order=np.sort(final)
        result['separation'].append(int(hit[0]+1) if len(hit) else s['steps']+1)
        result.setdefault('shows_to_separation_censored',[]).append(float(path[:hit[0]+1,1].sum()) if len(hit) else float(path[:,1].sum()+1))
        result['wrongly_settled'].append(float(np.argmax(final)!=truth and order[-1]-order[-2]>=3))
        fit=ll[:,-1];post=np.exp(fit-fit.max());post/=post.sum();result['weight_error'].append(float(abs(post@WEIGHTS-s['weight'])))
        shows=path[:,1]>0;count=shows.sum();result['shows_without_own_event'].append(float(np.sum(shows&(path[:,0]==0))/count) if count else np.nan)
        result['own_event_explains'].append(float(np.sum(shows&(path[:,0]<0))/count) if count else np.nan)
        result.setdefault('shows_without_true_rank_drop',[]).append(float(np.sum(shows&(path[:,0]-s['weight']*path[:,2]>=0))/count) if count else np.nan)
        result['alias'].append(float(np.max(abs(gap))<1e-10));pulse(world=w+1,phase='infer')
    return {k:np.array(v) for k,v in result.items()}

def summarize(s,m):return rows_for(s,m)
def pilot_exclusion(s,m):
    if np.all(m['separation']<=s['steps']) and np.all(m['shows_to_separation_censored']<=10):return 'all forty worlds separate within ten shows'
    if np.all(m['alias']==1):return 'exact observed-probability alias'
    return None

def controls(dest,pulse=lambda **kw:None):
    checks={'rank_ignores_group_rise':float(probability(1.-1.))==float(probability(0.)),
        'level_ignores_neighbour_rise':float(probability(0.-0.*1.))==float(probability(0.)),
        'rank_neighbour_drop':float(probability(-1/8))>float(probability(0.)),
        'independent_emission':bool(np.allclose(probability(np.linspace(-10,10,500)),probability(np.linspace(-10,10,500),True),rtol=1e-14,atol=1e-14))}
    setting=dict(weight=1.,worlds=2,steps=128,move_rate=0.,group_rate=.1,view='V2')
    paths=generate(setting,'exact-control')['records'];errors=[]
    for path in paths:
        for weight in WEIGHTS:
            delta=path[:,0]-weight*path[:,2];z=-2-2*delta;exact=np.cumsum(-np.logaddexp(0,-np.where(path[:,1],z,-z)))
            errors.append(float(np.max(abs(exact-exact_nomove(path,setting,weight)))))
            first=exact_nomove(path,dict(setting,view='V1'),weight)
            second=exact_nomove(path,dict(setting,view='V1'),weight,True)
            errors.append(float(np.max(abs(first-second))))
    checks['exact_observed_neighbours']=max(errors)<1e-10
    checks['level_hidden_equals_observed']=all(np.allclose(exact_nomove(p,dict(setting,view='V1'),0.),exact_nomove(p,setting,0.),atol=1e-10,rtol=0) for p in paths)
    write(dest/'CONTROLS.json',checks);return dict(passed=all(checks.values()),checks=checks)

def validate(s,dest,pulse=lambda **kw:None):
    # Prospective adjacent-resolution gate, on excluded paths. Accuracy against
    # exact full-view restrictions is checked independently of replica agreement.
    fixture=dict(s,worlds=2,blocks=2,steps=128);raw=generate(fixture,'hidden-validation',pulse);rows=[]
    for w,path in enumerate(raw['records']):
        for weight in WEIGHTS:
            a=particle_path(path,dict(s,particles=256),weight,3106+w)
            b=particle_path(path,dict(s,particles=1024),weight,4106+w)
            ref=particle_path(path,dict(s,particles=256),weight,3106+w,True)
            rows.append(dict(world=w,weight=float(weight),resolution_error=float(np.max(abs(a-b))),reconstruction_error=float(np.max(abs(a-ref)))))
            pulse(world=w+1,phase='hidden-validation')
    return dict(passed=all(x['resolution_error']<=.25 and x['reconstruction_error']<=1e-8 for x in rows),rows=rows,
        scope='Short-path adjacent particle resolution only; full-record calibration still required before production')
