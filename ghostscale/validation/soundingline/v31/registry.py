"""Finite V31 prospective grids. Builders are not admissions."""
import itertools
import numpy as np
from scipy.special import ndtr,expit
from scipy.optimize import brentq
from .d_diary import POPULATIONS,population

def heavy_offset(s):
    # Excluded, deterministic integration panel; exact Gaussian rating noise is
    # integrated out. The floor target is set before scientific generation.
    g=np.random.default_rng(310511);n=32768;r=population(g,n,s['population'])
    usual=g.normal(.6,.5,(n,1))
    if s['level_tied']:usual+=.4*np.log(r).mean(1,keepdims=True)/.5
    p=np.clip(g.beta(4,6,n)*s['event_rate']/.4,.05,.95) if s['prompts']==1 else np.clip(g.beta(2,8,n)*s['event_rate']/.2,.02,.9)
    if s['rate_tied']:p=expit(np.log(p/(1-p))+.5*np.log(r[:,0])/.5)
    sizes=[.5,1.,1.5] if s['graded'] else [1.]
    def floor(z):
        absent=ndtr((.5+z-usual)/np.sqrt(.5**2+.8**2))
        present=np.mean([ndtr((.5+z-usual-r*x)/np.sqrt(.5**2+.8**2)) for x in sizes],axis=0)
        return np.mean((1-p[:,None])*absent+p[:,None]*present)-2/3
    return float(brentq(floor,-5,8))

def d_settings():
    allrows=[]
    for floor,level,rate_tied,graded,n,(days,pp),rate,pop in itertools.product(
            ('mild','none','heavy'),(False,True),(False,True),(False,True),(2272,1200,300),
            ((10,6),(14,6),(8,1),(16,1)),(.2,.4,.08),POPULATIONS):
        allrows.append(dict(population=pop,people=n,days=days,prompts=pp,event_rate=rate,floor=floor,
            level_tied=level,rate_tied=rate_tied,graded=graded,worlds=200,blocks=8,steps=days*pp))
    return allrows

def s_settings():
    sets=((.75,),(.9,.75),(.75,.5),(.75,.25),(.25,.5,.75,.9))
    out=[]
    for temp,lift,odds,costs,rewards,truth,rival in itertools.product((.5,.25,1.),(4.,2.,8.),sets,((1.,),(1.,2.)),((20.,),(20.,40.)),('COST','SHIFT','SHARPEN','SLOW'),('COST','SHIFT','SHARPEN','SLOW')):
        if truth==rival:continue
        out.append(dict(temperature=temp,lift=lift,conditions=[list(x) for x in itertools.product(odds,costs,rewards)],
            truth=truth,rival=rival,law='geometric',worlds=200,blocks=8,steps=400,reported_prefixes=[25,50,100,200,400]))
    return out
