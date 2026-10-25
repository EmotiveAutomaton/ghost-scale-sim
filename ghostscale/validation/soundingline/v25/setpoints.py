"""W2: hidden controller, noisy observed transitions, prospective disturbances."""
import math,itertools
import numpy as np
from numpy.polynomial.hermite import hermgauss
from .common import rng,posterior,softmax,scores,record,close

HYPOTHESES=list(itertools.product(range(3),(-2,-1,0,1,2),(.5,.9)))
PREFIXES=(2,4,8,16,32)
PERCEPTIONS=np.array([[1.,0.],[0.,1.],[1.,1.]])

def means(x,decay=.9,independent=False):
    if independent:
        answer=[]
        for pi,reference,gain in HYPOTHESES:
            v=PERCEPTIONS[pi];error=reference-math.fsum(float(v[j]*x[j]) for j in range(2))
            answer.append([decay*x[j]+gain*error*v[j]/sum(v) for j in range(2)])
        return np.array(answer)
    v=PERCEPTIONS[[h[0] for h in HYPOTHESES]];g=np.array([h[2] for h in HYPOTHESES]);ref=np.array([h[1] for h in HYPOTHESES])
    return decay*np.array(x)+g[:,None]*(ref-v@x)[:,None]*v/v.sum(1)[:,None]

def likelihood(x,y,sigma,independent=False):
    m=means(x,independent=independent);variance=np.asarray(sigma)**2+.01
    if independent:return np.array([math.fsum(-.5*(float(y[j]-row[j])**2/variance[j]+math.log(2*math.pi*variance[j])) for j in range(2)) for row in m])
    return -.5*np.sum((np.asarray(y)-m)**2/variance+np.log(2*np.pi*variance),1)

def expected_information(p,x,sigma,axis):
    # A pulse changes the observed state before the controller responds.
    # Five-point product Gauss-Hermite integrates the Gaussian predictive mixture.
    disturbed=np.asarray(x)+2*np.eye(2)[axis];m=means(disturbed);var=np.array(sigma)**2+.01
    nodes,weights=hermgauss(5);nodes=np.array(list(itertools.product(nodes,nodes)));w=np.outer(weights,weights).ravel()/math.pi
    ys=m[:,None,:]+np.sqrt(2*var)*nodes[None,:,:]
    loglaw=-.5*np.sum((ys[:,:,None,:]-m[None,None,:,:])**2/var+np.log(2*np.pi*var),-1)
    from scipy.special import logsumexp
    with np.errstate(divide='ignore'):mix=logsumexp(loglaw+np.log(p),axis=-1)
    own=np.stack([loglaw[h,:,h] for h in range(len(p))])
    return float(np.sum(p[:,None]*w[None,:]*(own-mix)))

def marginals(p):return p.reshape(3,5,2).sum(-1).ravel()

def block(setting,identity,independent=False):
    r=rng(*identity);sigma=setting['sigma'];decay=setting['decay'];std=np.sqrt(np.array(sigma)**2+.01)
    outputs=[];raw=[]
    for truth,(perception,ref,gain) in enumerate(HYPOTHESES):
        x=np.zeros(2);ll=np.zeros(30);states=[x.tolist()];lls={};state_at={}
        for t in range(32):
            y=means(x,decay,independent)[truth]+r.normal(0,std);ll+=likelihood(x,y,sigma,independent);x=y;states.append(x.tolist())
            if t+1 in PREFIXES:
                p=posterior(ll,independent=independent);mp=marginals(p)
                outputs.append(record(scores(mp,perception*5+ref+2),reader='exact',truth=truth,history=t+1))
                lls[t+1]=ll.copy();state_at[t+1]=x.copy()
                arr=np.array(states);ratios=np.var(arr@PERCEPTIONS.T,axis=0)/((PERCEPTIONS**2)@(np.array(sigma)**2+.01)/(1-.9**2))
                outputs.append(record(dict(controlled_perception_correct=float(np.argmin(ratios)==perception),minimum_variance_ratio=float(min(ratios))),reader='variance',truth=truth,history=t+1))
        branches=[]
        for prefix in (4,8):
            paired_noise=r.normal(0,std,(4,2));results={}
            for kind in ('natural','designed'):
                x=state_at[prefix].copy();ll=lls[prefix].copy();trajectory=[]
                for t in range(4):
                    axis=None
                    if kind=='designed':
                        p=posterior(ll,independent=independent);ig=[expected_information(p,x,sigma,j) for j in range(2)]
                        axis=int(ig[1]>ig[0]+1e-12);x=x+2*np.eye(2)[axis]
                    y=means(x,decay,independent)[truth]+paired_noise[t]
                    ll+=likelihood(x,y,sigma,independent);trajectory.append(dict(state_before_response=x.tolist(),observed=y.tolist(),axis=axis));x=y
                results[kind]=marginals(posterior(ll,independent=independent))[perception*5+ref+2]
                branches.append(dict(prefix=prefix,kind=kind,trajectory=trajectory))
            value=math.log(results['designed'])-math.log(results['natural'])
            outputs.append(record(dict(primary=value,above_point_one=float(value>.1)),reader='designed-minus-natural',truth=truth,history=prefix))
        raw.append(dict(truth=truth,states=states,branches=branches))
    return dict(public=dict(sigma=sigma,reader_decay=.9,pulse_timing='observed-state impulse before response',quadrature_order=5),truth_decay=decay,truth_and_observations=raw,readouts=outputs)

def controls():
    for pi,r,g in HYPOTHESES:
        if r!=0:continue
        h=HYPOTHESES.index((pi,r,g));x=np.array([2.,0.]);base=np.zeros(2)
        for k in range(8):
            if pi==0:assert abs(x[0]-2*(.9-g)**k)<1e-12
            if pi==1:assert abs(x[0]-2*.9**k)<1e-12
            if pi==2:
                assert abs(sum(x)-2*(.9-g)**k)<1e-12
                assert abs(x[0]-(.9**k+(.9-g)**k))<1e-12
            x=means(x)[h]
    close(means([.3,-.4]).tolist(),means([.3,-.4],independent=True).tolist(),tol=1e-12)
    close(likelihood([.3,-.4],[1.,-.2],[.25,1]).tolist(),likelihood([.3,-.4],[1.,-.2],[.25,1],True).tolist(),tol=1e-12)
    return dict(passed=True,noise_free_decay_identity=True,sum_perception_identity=True,independent_gaussian_likelihood=True)
