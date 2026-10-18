"""N18 exact binary-channel information; weak-situation advantage is constructed."""
import math
import numpy as np
from scipy.optimize import brentq
from .common import rng,interval
FAMILY='N18';READERS=['ignore-frames','detector','oracle-frames']

def variants():return [dict(id='N18-information',family=FAMILY)]
def mutual(prob):
    p=np.asarray(prob,float);p=np.stack([1-p,p],axis=-1);mix=p.mean(0)
    return float(np.sum(np.where(p>0,.5*p*np.log(p/mix),0)))
def channel(s,a,norm,amplitude,reader):
    p=.5+amplitude*(s*norm+(1-s)*np.array([-1.,1.]))
    if s==.2:
        effective={'ignore-frames':.5,'detector':a,'oracle-frames':1.}[reader]
        p=effective*p+(1-effective)*(1-p)
    return p
def case(v,phase,world,case,fixed,design):
    random=rng('N18-law',phase,world);norm=int(random.choice([-1,1]));amplitude=float(random.uniform(.15,.45))
    public=dict(norm=norm,amplitude=amplitude,rows=[]);values={}
    for s in (.2,.5,.8):
        for a in (.6,.8,.95,1.):
            for reader in READERS:
                p=channel(s,a,norm,amplitude,reader);key=f's{s}|a{a}|{reader}'
                public['rows'].append(dict(key=key,strength=s,accuracy=a,reader=reader,probabilities=p.tolist()))
                values[key]=mutual(p)
    target=mutual(channel(.8,1,norm,amplitude,'detector'))
    threshold=float(brentq(lambda a:mutual(channel(.2,a,norm,amplitude,'detector'))-target,.5,1,xtol=1e-13))
    values['primary']=threshold
    return public,{},dict(threshold=threshold,information=values),values
def check(v,public,truth,evidence,fixed,design):
    def entropy(p):return -sum(x*math.log(x) for x in (p,1-p) if x>0)
    values={}
    for row in public['rows']:
        p=channel(row['strength'],row['accuracy'],public['norm'],public['amplitude'],row['reader'])
        if not np.allclose(p,row['probabilities'],atol=1e-14,rtol=0):raise ValueError('N18 channel differs')
        values[row['key']]=entropy(float(p.mean()))-(entropy(float(p[0]))+entropy(float(p[1])))/2
    low,high=.5,1.
    target=values['s0.8|a1.0|detector']
    for _ in range(50):
        mid=(low+high)/2;p=channel(.2,mid,public['norm'],public['amplitude'],'detector')
        info=entropy(float(p.mean()))-sum(entropy(float(x)) for x in p)/2
        if info<target:low=mid
        else:high=mid
    values['primary']=(low+high)/2
    if abs(values['primary']-evidence['threshold'])>1e-10:raise ValueError('N18 root differs')
    return values
def study_readout(items):
    return dict(break_even_accuracy=items[0]['metrics']['primary'],inference_laws=16,
        repeated_enumerations_are_not_independent=True,weak_situation_advantage_built_in=True,
        route='Weight play only above the law-dependent symmetric-detector threshold; human calibration required.')
def controls():
    assert abs(mutual(channel(1,1,1,.3,'detector')))<1e-14
    assert abs(mutual(channel(.2,.5,1,.3,'detector')))<1e-14
    assert np.array_equal(channel(.2,1,1,.3,'detector'),channel(.2,.6,1,.3,'oracle-frames'))
    vals=[mutual(channel(.2,a,1,.3,'detector')) for a in (.5,.6,.8,.95,1)]
    assert all(b>=a for a,b in zip(vals,vals[1:]))
    v=variants()[0];p,t,e,m=case(v,'controls',0,0,{},{});rebuilt=check(v,p,t,e,{},{} )
    assert all(abs(m[k]-rebuilt[k])<1e-10 for k in m)
    return dict(passed=True,entropy_joint_identity=True,full_constraint_zero=True,chance_detector_zero=True,
        perfect_detector_equals_oracle=True,monotone_channel=True,independent_bisection=True)
def run(root,plan,pulse):
    from .directed_common import run as execute
    return execute(root,plan,pulse)
def reconstruct(root,plan):
    from .directed_common import reconstruct as verify
    return verify(root,plan)
