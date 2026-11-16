"""G7: a finite action graph measures reachable futures, in N's need world."""
import itertools
import numpy as np
from numba import njit
from .common import rng,write,rows_for

# Each action quiets its need and leaves this next-step action menu. There is
# no value in the labels; the menu is rotated randomly on every occasion.
MENUS=np.array([[1,1,1],[0,1,1],[0,0,1]],dtype=np.int8)

def reachable(gaps,menus,reference=False,opened=None):
    result=[]
    for a in range(3):
        post=np.array(gaps).copy();post[a]=max(0,post[a]-1) if opened is None or opened[a] else post[a]
        if reference:
            futures=set()
            for b in range(3):
                if menus[a,b] and post[b]>0:
                    end=post.copy();end[b]-=1;futures.add(tuple(end))
            result.append(len(futures))
        else:result.append(np.sum(menus[a]*(post>0)))
    return np.array(result)

def choice(gaps,opened,menus,weight,temp,reference=False):
    options=reachable(gaps,menus,reference,opened);utility=np.asarray(gaps)*opened+weight*options
    z=utility/temp;z-=z.max();p=np.exp(z);return p/p.sum()

def generate(s,lane,pulse=lambda **kw:None):
    paths=[]
    for w in range(s['worlds']):
        g=rng('G7',s,lane,w);gaps=np.zeros(3,dtype=int);permit=np.ones(3,dtype=int);rows=[]
        for t in range(s['steps']):
            gaps=np.minimum(6,gaps+(g.random(3)<[.15,.15,.05]));opened=(g.random(3)<s['opening'])*permit
            shift=int(g.integers(3));menus=np.roll(np.roll(MENUS,shift,0),shift,1)
            p=choice(gaps,opened,menus,s['weight'],s['temperature']);a=g.choice(3,p=p)
            rows.append(np.r_[gaps,opened,shift,a]);gaps[a]-=bool(opened[a] and gaps[a]>0);permit=menus[a].copy()
        paths.append(rows);pulse(world=w+1,phase='generate')
    return dict(records=np.array(paths))

def infer(raw,s,reference=False,pulse=lambda **kw:None):
    out={k:[] for k in ['separation','wrongly_settled','reachable_futures','unmet_need','no_progress_share']};truth=int(s['weight']>0)
    for w,path in enumerate(raw['records']):
        terms=[];counts=[];progress=[]
        for row in path:
            gaps=row[:3];opened=row[3:6];shift=int(row[6]);a=int(row[7]);menus=np.roll(np.roll(MENUS,shift,0),shift,1)
            terms.append([np.log(choice(gaps,opened,menus,weight,s['temperature'],reference)[a]) for weight in (0.,s['contrast_weight'])])
            counts.append(reachable(gaps,menus,reference,opened)[a]);progress.append(bool(opened[a] and gaps[a]>0))
        ll=np.cumsum(terms,axis=0);gap=ll[:,truth]-ll[:,1-truth];hit=np.flatnonzero(np.minimum.accumulate(gap[::-1])[::-1]>=3)
        out['separation'].append(int(hit[0]+1) if len(hit) else s['steps']+1);out['wrongly_settled'].append(float(gap[-1]<=-3));out['reachable_futures'].append(float(np.mean(counts)));out['unmet_need'].append(float(path[:,:3].sum(1).mean()));out['no_progress_share'].append(1-float(np.mean(progress)));pulse(world=w+1,phase='infer')
    return {k:np.array(v) for k,v in out.items()}
def summarize(s,m):return rows_for(s,m)
def pilot_exclusion(s,m):
    if np.all(m['separation']<=10):return 'all forty worlds separate within ten steps'
    return None
def controls(dest,pulse=lambda **kw:None):
    checks={'reachable_count_exact':all(np.array_equal(reachable(g,MENUS),reachable(g,MENUS,True)) for g in itertools.product(range(3),repeat=3)),
        'zero_gaps_zero_futures':bool(np.all(reachable([0,0,0],MENUS)==0)),
        'equal_menus_alias':bool(np.allclose(choice([2,2,2],[1,1,1],np.ones((3,3)),0.,1.),choice([2,2,2],[1,1,1],np.ones((3,3)),2.,1.)))}
    write(dest/'CONTROLS.json',checks);return dict(passed=all(checks.values()),checks=checks)
