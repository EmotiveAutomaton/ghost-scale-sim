"""G1: exact hidden-stock filtering, with separately ordered scatter/gather recurrences."""
import math,itertools
from functools import lru_cache
import numpy as np
from numba import njit
from .common import rng,close
from .day_common import posteriors,margin,sustained,measure

POINTS=np.array(list(itertools.product((8,11,14),repeat=2)),dtype=np.int64)
GAINS=(8.,16.,32.)
EVENTS=np.array(list(itertools.product((0,1,5),repeat=2)),dtype=np.int64)
EVENT_P=np.array([a*b for a,b in itertools.product((.45,.5,.05),repeat=2)])

@njit
def triggers(before,after,point,law,floor=.1,power=1.):
    d0=np.maximum(0.,point-before);d1=np.maximum(0.,point-after);fall=np.maximum(0.,before-after);x=np.empty(2)
    for i in range(2):
        if law==0:x[i]=d1[i]/20
        elif law==1:x[i]=fall[i]/20
        elif law==2:x[i]=max(0.,d1[i]-d0[i])/20
        elif law==3:
            u0=(floor+d1[0])**power;u1=(floor+d1[1])**power
            x[i]=fall[i]/20*2*(u0 if i==0 else u1)/(u0+u1)
        else:
            old=((d0[0]/20)**4+(d0[1]/20)**4)**(1/3)
            other=1-i;new=((d1[i]/20)**4+(d0[other]/20)**4)**(1/3)
            x[i]=max(0.,new-old)
    return x

@njit
def choice_probability(stock,point,menu,restore):
    u=np.array([min(float(restore),max(0.,point[0]-stock[0])),min(float(restore),max(0.,point[1]-stock[1])),1.])
    available=np.array([bool(menu&1),bool(menu&2),True]);p=np.zeros(3);top=-1e99
    for a in range(3):
        if available[a]:top=max(top,u[a])
    total=0.;count=0
    for a in range(3):
        if available[a]:p[a]=math.exp(u[a]-top);total+=p[a];count+=1
    for a in range(3):
        if available[a]:p[a]=.98*p[a]/total+.02/count
    return p

def hypotheses(laws):return [(law,pi,g) for law in range(len(laws)) for pi in range(9) for g in GAINS]

@njit
def build_tables(nlaws,restore,floor,power,points,events,event_p):
    nh=nlaws*27;dest=np.empty((3,441,9),np.int64);fire=np.empty((3,4,nh,441,9));choice=np.empty((4,3,9,441))
    for state in range(441):
        before=np.array([state//21,state%21],dtype=np.float64)
        for pi in range(9):
            for menu in range(4):choice[menu,:,pi,state]=choice_probability(before,points[pi],menu,restore)
        for action in range(3):
            for e in range(9):
                after=np.maximum(0.,np.minimum(20.,before-events[e]+np.array([restore if action==0 else 0,restore if action==1 else 0])))
                dest[action,state,e]=int(after[0])*21+int(after[1])
                for law in range(nlaws):
                    for pi in range(9):
                        x=triggers(before,after,points[pi],law,floor,power)
                        for gi in range(3):
                            h=law*27+pi*3+gi;gain=8.*2**gi;p0=.02+.96*(-math.expm1(-gain*x[0]));p1=.02+.96*(-math.expm1(-gain*x[1]))
                            for bits in range(4):fire[action,bits,h,state,e]=(p0 if bits&1 else 1-p0)*(p1 if bits&2 else 1-p1)*event_p[e]
    return dest,fire,choice

@lru_cache(maxsize=4)
def tables(nlaws,restore,floor,power):
    d,f,c=build_tables(nlaws,restore,floor,power,POINTS,EVENTS,EVENT_P)
    reverse=[];ptr=[]
    for action in range(3):
        flat=d[action].ravel();order=np.argsort(flat,kind='stable');reverse.append(order);ptr.append(np.searchsorted(flat[order],np.arange(442)))
    return d,f,c,np.array(reverse),np.array(ptr)

@njit
def forward_scatter(menus,actions,fires,dest,fire,choice,event_p,known_start=-1):
    nh=fire.shape[2];q=np.full((2,nh,441),1/441.);qc=np.full((9,441),1/441.);logs=np.zeros((3,len(menus),nh));acc=np.zeros((3,nh));ac=np.zeros(9)
    if known_start>=0:
        q[:]=0.;qc[:]=0.;q[:,:,known_start]=1.;qc[:,known_start]=1.
    for t in range(len(menus)):
        menu=menus[t];a=actions[t];bits=fires[t];nxt=np.zeros_like(q);nc=np.zeros_like(qc)
        for h in range(nh):
            pi=(h//3)%9
            for state in range(441):
                cp=choice[menu,a,pi,state]
                for e in range(9):
                    target=dest[a,state,e];w=fire[a,bits,h,state,e]
                    nxt[0,h,target]+=q[0,h,state]*w*cp;nxt[1,h,target]+=q[1,h,state]*w
            for reader in range(2):
                total=nxt[reader,h].sum();acc[reader,h]+=math.log(total);nxt[reader,h]/=total;logs[reader,t,h]=acc[reader,h]
        for pi in range(9):
            for state in range(441):
                for e in range(9):nc[pi,dest[a,state,e]]+=qc[pi,state]*choice[menu,a,pi,state]*event_p[e]
            total=nc[pi].sum();ac[pi]+=math.log(total);nc[pi]/=total
        for h in range(nh):logs[2,t,h]=ac[(h//3)%9]
        q=nxt;qc=nc
    return logs

@njit
def forward_gather(menus,actions,fires,dest,fire,choice,event_p,reverse,ptr):
    """Review groups incoming edges by destination, independent of producer scatter."""
    nh=fire.shape[2];q=np.full((2,nh,441),1/441.);qc=np.full((9,441),1/441.);logs=np.zeros((3,len(menus),nh));acc=np.zeros((2,nh));ac=np.zeros(9)
    for t in range(len(menus)):
        menu=menus[t];a=actions[t];bits=fires[t];nxt=np.zeros_like(q);nc=np.zeros_like(qc)
        for target in range(441):
            for edge in range(ptr[a,target],ptr[a,target+1]):
                code=reverse[a,edge];state=code//9;e=code%9
                for pi in range(9):nc[pi,target]+=qc[pi,state]*event_p[e]*choice[menu,a,pi,state]
                for h in range(nh):
                    w=fire[a,bits,h,state,e];nxt[0,h,target]+=q[0,h,state]*choice[menu,a,(h//3)%9,state]*w;nxt[1,h,target]+=q[1,h,state]*w
        for pi in range(9):
            total=nc[pi].sum();nc[pi]/=total;ac[pi]+=math.log(total)
        for h in range(nh):
            for reader in range(2):
                total=nxt[reader,h].sum();nxt[reader,h]/=total;acc[reader,h]+=math.log(total);logs[reader,t,h]=acc[reader,h]
            logs[2,t,h]=ac[(h//3)%9]
        q=nxt;qc=nc
    return logs

def observations(s,r):
    point=POINTS[int(r.integers(9))];gain=float(r.choice(GAINS));stock=r.integers(0,21,2).astype(float);records=[];law=s['laws'].index(s['truth'])
    for _ in range(s['records']):
        menu=int(r.random()<s['menu'])+2*int(r.random()<s['menu']);p=choice_probability(stock,point,menu,s['restore']);action=int(r.choice(3,p=p));event=int(r.choice(9,p=EVENT_P));after=np.clip(stock-EVENTS[event]+np.array([s['restore'] if action==0 else 0,s['restore'] if action==1 else 0]),0,20)
        x=triggers(stock,after,point,law,s['urgency_floor'],s['urgency_power']);prob=.02+.96*(-np.expm1(-gain*x));fires=(r.random(2)<prob);bits=int(fires[0])+2*int(fires[1]);records.append([menu,action,bits,stock.tolist(),after.tolist(),event]);stock=after
    return dict(point=point.tolist(),gain=gain,law=s['truth'],records=records)

def analyze(raw,s,independent=False):
    hs=hypotheses(s['laws']);nh=len(hs);prior=np.ones(nh)/nh;dest,fire,ch,rev,ptr=tables(len(s['laws']),s['restore'],s['urgency_floor'],s['urgency_power']);records=raw['records'];menus=np.array([v[0] for v in records]);actions=np.array([v[1] for v in records]);fires=np.array([v[2] for v in records]);logs=forward_gather(menus,actions,fires,dest,fire,ch,EVENT_P,rev,ptr) if independent else forward_scatter(menus,actions,fires,dest,fire,ch,EVENT_P)
    ceiling=np.zeros((len(records),nh));current=np.zeros(nh)
    for t,record in enumerate(records):
        menu,action,bits,before,after,event=record;state=int(before[0])*21+int(before[1])
        for h,(law,pi,gain) in enumerate(hs):current[h]+=math.log(ch[menu,action,pi,state])+math.log(fire[action,bits,h,state,event]/EVENT_P[event])
        ceiling[t]=current
    lawids=np.array([h[0] for h in hs]);truth=s['laws'].index(raw['law']);result={}
    for reader,ll in list(zip(('both','firings','choices'),logs))+[('ceiling',ceiling)]:
        p=posteriors(ll,prior,independent);out={}
        for other in range(len(s['laws'])):
            if other!=truth and not (reader=='firings' and s.get('omit_firings_matrix')):out['against_'+s['laws'][other]]=sustained(margin(ll,prior,lawids==truth,lawids==other))
        for t in (128,256,512):
            if t>len(records):continue
            out['law_posterior_'+str(t)]=float(p[t-1,lawids==truth].sum())
            for need in (0,1):out['setpoint_'+str(need)+'_'+str(t)]=float(sum(p[t-1,h] for h,(_,pi,_) in enumerate(hs) if POINTS[pi,need]==raw['point'][need]))
        result[reader]=out
    if len(records)>=256:result['contrasts']=dict(arm=float(np.mean([result['firings']['setpoint_'+str(k)+'_256']-result['choices']['setpoint_'+str(k)+'_256'] for k in (0,1)])))
    return result

def block(s,identity,independent=False):
    r=rng(*identity);raw=[];out=[];vals={}
    for _ in range(s['makers']):
        o=observations(s,r);raw.append(o)
        for reader,stats in analyze(o,s,independent).items():
            for name,v in stats.items():vals.setdefault((reader,name),[]).append(v)
        vals.setdefault(('design','firing_rate'),[]).append(float(np.mean([(int(v[2]&1>0)+int(v[2]&2>0))/2 for v in o['records']])))
        vals.setdefault(('design','restoring_rate'),[]).append(float(np.mean([v[1]<2 for v in o['records']])))
    for (reader,name),v in vals.items():measure(out,name,v,'median' if name.startswith('against_') else 'mean',reader=reader,truth=s['truth'])
    return dict(observations=raw,measures=out)

def design_check(s,identity):
    r=rng('day-check-G1',identity);numer=np.zeros((len(s['laws']),len(s['laws'])));denom=numer.copy();split=[]
    for i in range(240):
        o=observations(s|dict(truth=s['laws'][i%len(s['laws'])],records=512),r);before=np.array([v[3] for v in o['records']]);after=np.array([v[4] for v in o['records']]);point=np.array(o['point']);below=(before<point).mean(0);split.append(bool(np.all((below>=.25)&(below<=.75))))
        x=np.array([[triggers(b,a,point,l,s['urgency_floor'],s['urgency_power']) for b,a in zip(before,after)] for l in range(len(s['laws']))])
        for a in range(len(s['laws'])):
            p=.02+.96*(-np.expm1(-o['gain']*x[a]))
            for b in range(len(s['laws'])):
                if a==b:continue
                eligible=(x[a]>0)|(x[b]>0);q=.02+.96*(-np.expm1(-np.array(GAINS)[:,None,None]*x[b]));best=np.argmin(np.mean((q-p)**2,axis=(1,2)));numer[a,b]+=np.sum((np.abs(p-q[best])>.2)&eligible);denom[a,b]+=eligible.sum()
    pairs={s['laws'][a]+'-'+s['laws'][b]:float(numer[a,b]/denom[a,b]) if denom[a,b] else 0. for a in range(len(s['laws'])) for b in range(len(s['laws'])) if a!=b and not (a==4 and b==0 or a==0 and b==4)}
    return dict(passed=bool(min(pairs.values())>=.25 and np.mean(split)>=.5),checks=dict(pair_shares=pairs,time_split_share=float(np.mean(split))),worlds=240)

@njit
def brute_path(depth,state,menus,actions,bits,point,law,gain,restore):
    if depth==len(menus):return 1.
    before=np.array([state//21,state%21],dtype=np.float64);cp=choice_probability(before,point,menus[depth],restore)[actions[depth]];total=0.
    for e in range(9):
        a=actions[depth];after=np.clip(before-EVENTS[e]+np.array([restore if a==0 else 0,restore if a==1 else 0]),0,20);x=triggers(before,after,point,law);p=.02+.96*(-np.expm1(-gain*x));f=(p[0] if bits[depth]&1 else 1-p[0])*(p[1] if bits[depth]&2 else 1-p[1]);target=int(after[0])*21+int(after[1]);total+=EVENT_P[e]*cp*f*brute_path(depth+1,target,menus,actions,bits,point,law,gain,restore)
    return total

def controls():
    point=np.array([10,10]);b=np.array([12.,8.]);a=np.array([8.,7.]);close(triggers(b,a,point,2).tolist(),[.1,.05],tol=1e-12)
    assert triggers(np.array([15.,15.]),np.array([12.,13.]),point,2).sum()==0
    close(triggers(np.array([8.,8.]),np.array([7.,7.]),point,3).tolist(),triggers(np.array([8.,8.]),np.array([7.,7.]),point,1).tolist(),tol=1e-12)
    unequal=triggers(np.array([12.,8.]),np.array([11.,7.]),point,3);assert unequal[1]>unequal[0]
    menus=np.array([3,1,2,3]);actions=np.array([2,0,1,2]);bits=np.array([0,1,2,0]);dest,fire,ch,rev,ptr=tables(4,5,.1,1.)
    ll=forward_scatter(menus,actions,bits,dest,fire,ch,EVENT_P);review=forward_gather(menus,actions,bits,dest,fire,ch,EVENT_P,rev,ptr);close(ll.tolist(),review.tolist(),tol=1e-9)
    brute=sum(brute_path(0,state,menus,actions,bits,POINTS[0],2,8.,5) for state in range(441))/441
    assert abs(math.exp(ll[0,-1,54])-brute)<1e-9
    # Under raw slope, all nine set-point alternatives have identical firings-only likelihoods.
    raw=ll[1,:,27:54].reshape(4,9,3);assert np.max(abs(raw-raw[:,:1]))<1e-12
    # Deterministic known transition reduces filtering to the observed-state product.
    q=np.zeros(441);q[220]=1.;expected=1.
    for t in range(4):
        state=int(np.argmax(q));cp=ch[menus[t],actions[t],0,state];f=fire[actions[t],bits[t],54,state,1]/EVENT_P[1];expected*=cp*f;q=np.eye(1,441,int(dest[actions[t],state,1]))[0]
    deterministic=np.zeros_like(fire);deterministic[:,:,:,:,1]=fire[:,:,:,:,1]/EVENT_P[1];event=np.zeros(9);event[1]=1.
    actual=forward_scatter(menus,actions,bits,dest,deterministic,ch,event,220)
    assert abs(math.exp(actual[0,-1,54])-expected)<1e-12 and fire.min()>0
    return dict(passed=True,clipped_deficit_fixture=True,above_setpoint_zero=True,balanced_raw_alias=True,unequal_urgency=True,brute_four_record_initial_and_outcome_enumeration=True,scatter_gather_equal=True,raw_slope_setpoint_prior=True,deterministic_known_path=True,positive_floors=True)
