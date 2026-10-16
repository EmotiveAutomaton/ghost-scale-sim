"""Independent finite-state reconstruction for the single-maker N families.

Likelihoods and state enumeration are rebuilt here, without production posterior
calls. The supplied trained RT weights are a frozen learned artifact, not truth.
"""
from itertools import product
import math
import numpy as np


def softmax(x):
    x=np.asarray(x,float);e=np.exp(x-x.max(axis=-1,keepdims=True))
    return e/e.sum(axis=-1,keepdims=True)


def check(actual,expected):
    error=0.
    for name,value in expected.items():
        if value is None:
            if actual[name] is not None: raise ValueError('empty hard support was hidden')
        else:
            error=max(error,float(np.max(np.abs(np.asarray(value)-actual[name]))))
    if error>1e-11: raise ValueError('independent N-family posterior mismatch')
    return error


def n8(v,p,law=None,blind=False):
    s=v['settings'];mass=np.ones((16,2))
    if law is not None: mass[np.arange(16)!=law]=0
    for row in p['history']:
        for ell in range(16):
            reliability=.6+.35*ell/15
            for preference in (0,1):
                if blind: probability=reliability if row['choice']==preference else 1-reliability
                else:
                    probability=0.
                    share=0. if row['threat'] else s['play_share']
                    for playing in (0,1):
                        prior=share if playing else 1-share
                        signal=s['signal_reliability'] if row['signal']==playing else 1-s['signal_reliability']
                        choice=reliability if row['choice']==(preference^playing) else 1-reliability
                        probability+=prior*signal*choice
                mass[ell,preference]*=probability
        mass/=mass.sum()
    return mass.sum(0)[None,:]


def verify_n8(v,p,truth,pred,detail):
    expected={'law-marginal':n8(v,p),'mode-blind':n8(v,p,blind=True),'always-closed':[[.5,.5]]}
    if 'oracle-law' in pred: expected['oracle-law']=n8(v,p,law=truth['law'])
    return check(pred,expected)


def n6(v,p,withdraw=False,law=None):
    states=list(product((0,1),(0,1),range(16)));s=v['settings']
    mass=np.array([(s['lowlow_share'] if w==c==0 else (1-s['lowlow_share'])/3)/16 for w,c,l in states])
    if law is not None:
        mass*=np.array([l==law for w,c,l in states]);mass/=mass.sum()
    stopped=False;path=[]
    for row in p['history']:
        stopped|=withdraw and math.fsum(m for m,(w,c,l) in zip(mass,states) if w==c==0)>=s['threshold']
        if not stopped:
            for i,(w,c,l) in enumerate(states):
                rel=.6+.35*l/15
                mass[i]*=(rel if row['claim']==w else 1-rel)*(rel if row['procedure']==c else 1-rel)
            mass/=mass.sum()
        path.append((stopped,math.fsum(m for m,(w,c,l) in zip(mass,states) if w),math.fsum(m for m,(w,c,l) in zip(mass,states) if c)))
    answer=np.zeros(4)
    for m,(w,c,l) in zip(mass,states):
        rel=.6+.35*l/15
        for a in range(4): answer[a]+=m*(rel if a//2==w else 1-rel)*(rel if a%2==c else 1-rel)
    return answer[None,:],path


def verify_n6(v,p,truth,pred,detail):
    model,mp=n6(v,p);withdraw,wp=n6(v,p,True)
    expected={'law-marginal':model,'model-and-gate':model,'absorb-everything':model,'withdraw':withdraw}
    if 'oracle-law' in pred: expected['oracle-law']=n6(v,p,law=truth['law'])[0]
    for path,key in ((mp,'model'),(wp,'withdraw')):
        for (stopped,w,c),row in zip(path,detail[key]['path'],strict=True):
            if stopped!=row['stopped'] or max(abs(w-row['warmth']),abs(c-row['competence']))>1e-11:
                raise ValueError('independent uptake path mismatch')
    return check(pred,expected)


def n3_scalar(v,p,reader):
    d=v['settings']['dimensions'];points=np.array([x for x in product((-1.,0.,1.),repeat=d) if any(x)])
    points/=np.linalg.norm(points,axis=1,keepdims=True)
    states=list(product((v['settings']['beta'],),(v['settings']['local_share'],),range(len(points))))
    mass=np.ones(len(states))
    for row in p['history']:
        if reader=='half-space' and row['picked']:
            for i,(_,_,k) in enumerate(states):
                if np.dot(points[k],row['edits'][row['picked']])<=0:mass[i]=0
        else:
            for i,(beta,share,k) in enumerate(states):
                if reader in ('law-marginal','corrections-only'):
                    preference=softmax(beta*(np.asarray(row['edits'])@points[k]-row['costs']))[row['picked']]
                    local=softmax(beta*(np.asarray(row['edits'])@row['context']-row['costs']))[row['picked']]
                    mass[i]*=(1-share)*preference+share*local
                elif reader=='soft-constraint-supplement' and row['picked']:
                    mass[i]*=.9 if np.dot(points[k],row['edits'][row['picked']])>0 else .1
        if reader!='corrections-only':
            for i,(beta,share,k) in enumerate(states):
                mass[i]*=softmax(beta*(np.asarray(row['menu'])@points[k]))[row['choice']]
        if mass.sum()==0: return None,None
        mass/=mass.sum()
    answer=np.zeros(4)
    for m,(beta,share,k) in zip(mass,states):answer+=m*softmax(beta*(np.asarray(p['query'])@points[k]))
    return answer[None,:],mass.reshape(1,1,len(points)).sum((0,1))


def n3(v,p,reader):
    s=v['settings'];points=np.array([x for x in product((-1.,0.,1.),repeat=s['dimensions']) if any(x)])
    points/=np.linalg.norm(points,axis=1,keepdims=True)
    mass=np.ones(len(points))/len(points);beta=s['beta'];share=s['local_share']
    for row in p['history']:
        edits=np.asarray(row['edits']);picked=row['picked']
        if reader=='half-space' and picked:
            mass*=np.einsum('ij,j->i',points,edits[picked])>0
        elif reader in ('law-marginal','corrections-only'):
            preference=softmax(beta*(np.einsum('ij,kj->ik',points,edits)-np.array(row['costs'])) )[:,picked]
            local=softmax(beta*(edits@np.asarray(row['context'])-row['costs']))[picked]
            mass*=(1-share)*preference+share*local
        elif reader=='soft-constraint-supplement' and picked:
            mass*=np.where(np.einsum('ij,j->i',points,edits[picked])>0,.9,.1)
        if reader!='corrections-only':
            mass*=softmax(beta*np.einsum('ij,kj->ik',points,np.array(row['menu'])))[:,row['choice']]
        if mass.sum()==0:return None,None
        mass/=mass.sum()
    probability=mass@softmax(beta*np.einsum('ij,kj->ik',points,np.asarray(p['query'])))
    return probability[None,:],mass


def verify_n3(v,p,truth,pred,detail):
    expected={}
    for reader in ('law-marginal','corrections-only','half-space','soft-constraint-supplement','final-choice-only'):
        expected[reader],post=n3(v,p,reader)
        if post is not None and np.max(np.abs(post-detail[reader]['direction_posterior']))>1e-11:
            raise ValueError('independent correction posterior mismatch')
    expected['corrections-plus-choices']=expected['law-marginal']
    summed=np.sum([row['edits'][row['picked']] for row in p['history']],axis=0)
    length=np.linalg.norm(summed);estimate=summed/length if length else summed
    expected['coactive']=softmax(v['settings']['beta']*(np.asarray(p['query'])@estimate))[None,:]
    return check(pred,expected)


def n2(v,p,weights,average,mode):
    s=v['settings'];classes=('preference','near-tie','mixed-strategy','spontaneous','prime','context')
    prior=np.zeros(6);prior[0]=1-s['nondiagnostic'];prior[classes.index(s['mark_class'])]+=s['nondiagnostic']
    mass=np.ones((16,2));plain=np.ones(2)
    for row in p['history']:
        weight=weights[row['latency_bin']]
        for ell in range(16):
            opponent=.15+.7*ell/15
            for preference in (0,1):
                sign=2*preference-1;logistic=lambda z:1/(1+math.exp(-z))
                if row['opponent'] is not None and not math.isclose(row['opponent'],opponent,abs_tol=1e-12):
                    mass[ell,preference]=0;continue
                likelihood=0.
                for kind in range(6):
                    probs=[]
                    for target in (-1,1):
                        probs.append((logistic(s['beta']*target),logistic(.05*s['beta']*target),opponent,.5,
                          .6*row['previous']+.4*logistic(s['beta']*target),logistic(s['beta']*(target+2*row['context'])))[kind])
                    choice=probs[preference] if row['choice'] else 1-probs[preference]
                    cue=.7 if row['class_cue']==kind else .3/5
                    rt=softmax(-s['rt_slope']*abs(math.log(probs[1])-math.log(probs[0]))*np.arange(5))[row['latency_bin']]
                    likelihood+=prior[kind]*choice*cue*(1 if mode=='class-aware-no-rt' else rt)
                mass[ell,preference]*=likelihood**(weight if mode=='exact-plus-weights' else 1.)
        for preference in (0,1):
            prob=1/(1+math.exp(-s['beta']*(2*preference-1)))
            choice=prob if row['choice'] else 1-prob
            exponent=weight if mode=='response-time-weighted' else average if mode=='matched-tempering' else 1
            plain[preference]*=choice**exponent
        plain/=plain.sum();mass/=mass.sum()
    return (mass.sum(0) if mode in ('law-marginal','class-aware-no-rt','exact-plus-weights') else plain)[None,:]


def verify_n2(v,p,truth,pred,detail):
    weights=np.array(detail['rt_weights']);average=float(np.average(weights,weights=detail['training_bin_counts']))
    # Independently scale probability masses after each mark. Production sums a
    # complete log-likelihood tensor; no production probability functions enter.
    s=v['settings'];classes=('preference','near-tie','mixed-strategy','spontaneous','prime','context')
    prior=np.zeros(6);prior[0]=1-s['nondiagnostic'];prior[classes.index(s['mark_class'])]+=s['nondiagnostic']
    masses={name:np.ones((16,2))/32 for name in ('law-marginal','class-aware-no-rt','exact-plus-weights')}
    plains={name:np.ones(2)/2 for name in ('everything-is-preference','response-time-weighted','matched-tempering')}
    signs=np.array([-1.,1.]); logistic=lambda z:1/(1+np.exp(-z)); laws=np.linspace(.15,.85,16)
    for row in p['history']:
        prob=np.empty((16,6,2));prob[:,0]=logistic(s['beta']*signs);prob[:,1]=logistic(.05*s['beta']*signs)
        prob[:,2]=laws[:,None];prob[:,3]=.5;prob[:,4]=.6*row['previous']+.4*logistic(s['beta']*signs)
        prob[:,5]=logistic(s['beta']*(signs+2*row['context']))
        separation=np.abs(np.log(prob[:,:,1])-np.log(prob[:,:,0]))
        rt=softmax(-s['rt_slope']*separation[:,:,None]*np.arange(5))[:,:,row['latency_bin']]
        cue=np.array([.7 if k==row['class_cue'] else .3/5 for k in range(6)])
        latent=prior[None,:,None]*cue[None,:,None]*(prob if row['choice'] else 1-prob)
        possible=np.ones(16,bool) if row['opponent'] is None else np.isclose(laws,row['opponent'],atol=1e-12)
        full=(latent*rt[:,:,None]).sum(1);plain=latent.sum(1)
        for name,mass in masses.items():
            likelihood=plain if name=='class-aware-no-rt' else full
            exponent=weights[row['latency_bin']] if name=='exact-plus-weights' else 1.
            mass*=likelihood**exponent*possible[:,None];mass/=mass.sum()
        naive=logistic(s['beta']*signs);naive=naive if row['choice'] else 1-naive
        for name,mass in plains.items():
            exponent=weights[row['latency_bin']] if name=='response-time-weighted' else average if name=='matched-tempering' else 1.
            mass*=naive**exponent;mass/=mass.sum()
    expected={name:mass.sum(0)[None,:] for name,mass in masses.items()}
    expected.update({name:mass[None,:] for name,mass in plains.items()})
    return check(pred,expected)
