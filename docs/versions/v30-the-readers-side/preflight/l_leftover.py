"""V30 pre-flight, family L: exposure, reaction or carry-over. One channel.
World: each step an event arrives with probability p_e and size x in {0.5,1.0,1.5} (equal odds).
Activation A_t = g*x_t*e_t + rho*A_{t-1}. The channel shows with probability 1-exp(-A_t).
Three makers share one overall show rate and differ in one setting each.
Reader: exact forward filter over a grid of A, with events seen, part seen, or unseen."""
import numpy as np, zlib, json, sys
XS=np.array([0.5,1.0,1.5]); GRID=np.linspace(0,6,121)
def rate(p,g,rho,n=400000,seed=1):
    r=np.random.default_rng(seed); e=r.random(n)<p; x=XS[r.integers(0,3,n)]; a=0.0; s=0
    drive=g*x*e; out=np.empty(n)
    for t in range(n):
        a=drive[t]+rho*a; out[t]=a
    return float((1-np.exp(-out)).mean())
def tune(which,target,base):
    lo,hi={'p':(0.01,0.9),'g':(0.1,12.0),'rho':(0.0,0.97)}[which]
    for _ in range(26):
        mid=(lo+hi)/2; pr=dict(base); pr[which]=mid
        if rate(pr['p'],pr['g'],pr['rho'])<target: lo=mid
        else: hi=mid
    pr=dict(base); pr[which]=round((lo+hi)/2,4); return pr
BASE={'p':0.10,'g':1.0,'rho':0.30}; TARGET=0.16
TYPES={'EXPOSED':tune('p',TARGET,BASE),'REACTIVE':tune('g',TARGET,BASE),'STICKY':tune('rho',TARGET,BASE)}
def simulate(pr,T,rng):
    e=rng.random(T)<pr['p']; xi=rng.integers(0,3,T); a=0.0; d=np.zeros(T,bool)
    for t in range(T):
        a=pr['g']*XS[xi[t]]*e[t]+pr['rho']*a; d[t]=rng.random()<1-np.exp(-a)
    return e,xi,d
def loglik_path(pr,eo,xo,do,mode,he,fe,hd,fd):
    """eo: observed event flag (after detection noise), xo: size index when sizes are seen, do: observed show flag."""
    w=np.zeros(len(GRID)); w[0]=1.0; ll=np.zeros(len(do)); pshow=1-np.exp(-GRID)
    # precompute landing bins for each (prior bin, option): option 0 = no event, 1..3 = event of size XS
    land=[np.clip(np.rint((pr['rho']*GRID)/(GRID[1]-GRID[0])).astype(int),0,len(GRID)-1)]
    for x in XS: land.append(np.clip(np.rint((pr['g']*x+pr['rho']*GRID)/(GRID[1]-GRID[0])).astype(int),0,len(GRID)-1))
    prior=np.array([1-pr['p']]+[pr['p']/3]*3)
    for t in range(len(do)):
        new=np.zeros(len(GRID))
        for k in range(4):
            if mode=='unseen': pe=1.0
            elif mode=='exact': pe=1.0 if ((k==0 and not eo[t]) or (k>0 and eo[t] and xo[t]==k-1)) else 0.0
            else:  # 'noisy': event flag seen through a detector, size never seen
                pe=(fe if eo[t] else 1-fe) if k==0 else (he if eo[t] else 1-he)
            if pe==0.0: continue
            np.add.at(new,land[k],w*prior[k]*pe)
        ps=pshow*hd+(1-pshow)*fd                      # chance the reader records a show
        new*=ps if do[t] else 1-ps
        s=new.sum(); ll[t]=np.log(s); w=new/s
    return np.cumsum(ll)
def run(mode,T=1500,worlds=60,he=1.0,fe=0.0,hd=1.0,fd=0.0):
    out={}
    for name,pr in TYPES.items():
        rng=np.random.default_rng(zlib.crc32(f'{mode}|{name}|{he}|{hd}'.encode())); sep=[]; fin=[]
        for _ in range(worlds):
            e,xi,d=simulate(pr,T,rng)
            eo=np.where(e,rng.random(T)<he,rng.random(T)<fe); do=np.where(d,rng.random(T)<hd,rng.random(T)<fd)
            L={n:loglik_path(p2,eo,xi,do,mode,he,fe,hd,fd) for n,p2 in TYPES.items()}
            rival=np.max([L[n] for n in TYPES if n!=name],axis=0); lo=L[name]-rival
            ok=lo>=3; first=None
            if ok[-1]:
                bad=np.where(~ok)[0]; first=int(bad[-1]+2) if len(bad) else 1
            sep.append(first); fin.append(float(lo[-1]))
        got=[s for s in sep if s is not None]
        out[name]={'separated':round(len(got)/worlds,3),'median_steps':(int(np.median(got)) if got else None),'median_final_log_odds':round(float(np.median(fin)),2)}
    return out
if __name__=='__main__':
    print('types',json.dumps(TYPES)); print('show rates',{n:round(rate(**{k:v for k,v in p.items()}),4) for n,p in TYPES.items()})
    res={}
    for label,kw in [('shows only',dict(mode='unseen')),('shows and events, exact',dict(mode='exact')),('shows and events through a detector (hit .7, false alarm .05), sizes unseen',dict(mode='noisy',he=0.7,fe=0.05)),
                     ('as above, and shows through a reader (hit .6, false alarm .05)',dict(mode='noisy',he=0.7,fe=0.05,hd=0.6,fd=0.05)),('shows only, through a reader (hit .6, false alarm .05)',dict(mode='unseen',hd=0.6,fd=0.05))]:
        res[label]=run(**kw); print(label,json.dumps(res[label]))
    json.dump({'types':TYPES,'results':res},open('l_leftover_out.json','w'),indent=1)
