"""N17 finite menu study with an honest held-out e-value for false taste claims."""
from itertools import product
import math
import numpy as np
from scipy.special import softmax,logsumexp
from .common import rng,interval
FAMILY='N17';READERS=['lean','sensitivity','taste','oracle']
C=np.array([-2.,-1.,0.,1.,2.]);D=np.array([.25,.75,1.5,2.5]);W=np.array([-1.5,-.75,0.,.75,1.5])
GRIDS={'lean':np.array(list(product(C,[0.],[0.]))),'sensitivity':np.array(list(product(C,D,[0.]))),
    'taste':np.array(list(product(C,D,W)))}
TARGETS={'within_agreement':.4217687074829932,'cross_agreement':.3645,'chance_agreement':.41,
    'within_kappa':.0890826931766833,'cross_kappa':.0271148623215494,
    'acceptance_icc':.09333793798050197,'acceptance_split_half':.7802622575370917}

def variants():return [dict(id=f'N17-{n}-{taste}-{mechanism}',family=FAMILY,history=n,taste=taste,mechanism=mechanism)
    for n in (20,50,100,250) for taste in (False,True) for mechanism in (0,1)]

def probabilities(parameters,menus,temperature,intervention=None,mechanism=None):
    parameters=np.atleast_2d(parameters);menus=np.asarray(menus);c=parameters[:,0].copy()
    if intervention=='stakes' and mechanism==0:c-=math.log(2)
    if intervention=='base-rate' and mechanism==1:c[:]=0
    logits=(c[:,None,None]-math.log(5)+parameters[:,1,None,None]*menus[None,:,:,0]+parameters[:,2,None,None]*menus[None,:,:,1])/temperature
    return softmax(np.concatenate([np.zeros((*logits.shape[:2],1)),logits],axis=-1),axis=-1)

def loglikes(probs,y):return np.log(np.take_along_axis(probs,np.asarray(y)[None,:,None],axis=2)[:,:,0]).sum(1)
def posterior(grid,menus,y,temperature):return softmax(loglikes(probabilities(grid,menus,temperature),y))
def forecast(grid,mass,menus,temperature):return np.einsum('p,pny->ny',mass,probabilities(grid,menus,temperature))

def calibration_moments(config,phase):
    random=rng('N17-calibration',phase);count=96;menus=random.normal(size=(256,5,2))
    weights=softmax(-.5*((C-config['location'])/config['spread'])**2)
    params=np.c_[random.choice(C,count,p=weights),random.choice(D,count),random.choice(W,count)]
    probabilities_all=probabilities(params,menus,config['temperature'])
    within=float(np.mean(np.sum(probabilities_all**2,axis=-1)))
    # Pair different writers on the identical menus, avoiding self-pairs.
    summed=probabilities_all.sum(0)
    cross=float(np.mean((np.sum(summed**2,axis=-1)-np.sum(probabilities_all**2,axis=(0,2)))/(count*(count-1))))
    marginal=probabilities_all.mean((0,1));chance=float(marginal@ marginal)
    accept=1-probabilities_all[:,:,0];maker_mean=accept.mean(1);overall=float(accept.mean())
    icc=float(np.var(maker_mean,ddof=1)/(overall*(1-overall)))
    u=random.random((2,count,256));accepted=u<accept[None]
    split=float(np.corrcoef(accepted[0].mean(1),accepted[1].mean(1))[0,1])
    return dict(within_agreement=within,cross_agreement=cross,chance_agreement=chance,
        within_kappa=(within-chance)/(1-chance),cross_kappa=(cross-chance)/(1-chance),
        acceptance_icc=icc,acceptance_split_half=split)

def fit(plan,pulse):
    candidates=[]
    for location,spread,temp in product((-2.,-1.,0.),(.5,1.,2.),(.5,1.,1.5)):
        config=dict(location=location,spread=spread,temperature=temp)
        moments=calibration_moments(config,'fit')
        error=sum(((moments[k]-v)/(.05 if k!='acceptance_split_half' else .15))**2 for k,v in TARGETS.items())
        candidates.append(dict(config=config,moments=moments,error=error));pulse(phase='N17-calibration-grid',candidate=len(candidates))
    best=min(candidates,key=lambda x:x['error']);validation=calibration_moments(best['config'],'validation')
    return dict(family=FAMILY,config=best['config'],calibration_candidates=candidates,targets=TARGETS,
        validation=validation,validation_errors={k:validation[k]-v for k,v in TARGETS.items()},
        reader_grids={k:v.tolist() for k,v in GRIDS.items()},e_value_threshold=20.,
        source_targets='Sounding Line Stage 15 V5, V5.d cutoff .8, V3, D12-rank-canonical; aggregate pins in protocol companion',
        populations_and_channels_not_identical=True)

def law(phase,world,fixed):
    random=rng('N17-law',phase,world);config=fixed['config']
    base=softmax(-.5*((C-config['location'])/config['spread'])**2)
    weights=random.dirichlet(1+20*base)
    return dict(criterion_weights=weights,temperature=config['temperature']*float(random.uniform(.85,1.15)),
        scales=random.uniform(.7,1.3,size=2))

def evaluate(public,truth,independent=False):
    train=np.asarray(public['train']);test=np.asarray(public['test']);ys=np.asarray(public['train_y']);temp=public['temperature'];actual=np.asarray(truth['heldout_y'])
    fitted={name:posterior(grid,train,ys,temp) for name,grid in GRIDS.items()}
    forecasts={name:forecast(GRIDS[name],mass,test,temp) for name,mass in fitted.items()}
    forecasts['oracle']=probabilities(truth['parameters'],test,temp)[0]
    losses={name:float(-np.mean(np.log(p[np.arange(len(actual)),actual]))) for name,p in forecasts.items()}
    null_ll=loglikes(probabilities(GRIDS['sensitivity'],test,temp),actual)
    taste_ll=-losses['taste']*len(actual);log_e=taste_ll-float(np.max(null_ll))
    estimate=fitted['taste']@GRIDS['taste'];parameters=np.asarray(truth['parameters'])
    values=dict(primary=losses['sensitivity']-losses['taste'],sensitivity_gain=losses['lean']-losses['sensitivity'],
        taste_detected=float(log_e>=math.log(20)),log_e_value=log_e,
        criterion_error=float((estimate[0]-parameters[0])**2),sensitivity_error=float((estimate[1]-parameters[1])**2),
        criterion_error_improvement=float((C.mean()-parameters[0])**2-(estimate[0]-parameters[0])**2),
        sensitivity_error_improvement=float((D.mean()-parameters[1])**2-(estimate[1]-parameters[1])**2),
        taste_direction_correct=float(np.sign(estimate[2])==np.sign(parameters[2])))
    values.update({name+'_log_loss':loss for name,loss in losses.items()})
    mass=np.tile(fitted['taste'][None],(2,1))/2;mechanism_post=[]
    for row in public['probes']:
        for mechanism in (0,1):
            probability=probabilities(GRIDS['taste'],[row['menu']],temp,row['kind'],mechanism)[:,0,row['outcome']]
            mass[mechanism]*=probability
        mass/=mass.sum();mechanism_post.append(mass.sum(1).tolist())
    values['mechanism_gain']=math.log(float(mass[truth['mechanism']].sum()))+math.log(2)
    evidence=dict(probabilities={k:v.tolist() for k,v in forecasts.items()},fitted={k:v.tolist() for k,v in fitted.items()},
        parameter_estimate=estimate.tolist(),mechanism_posterior=mechanism_post,null_max_log_likelihood=float(null_ll.max()))
    return evidence,values

def case(v,phase,world,case,fixed,design):
    l=law(phase,world,fixed);random=rng('N17-maker',phase,v['id'],world,case);n=v['history']
    parameters=np.array([random.choice(C,p=l['criterion_weights']),random.choice(D),random.choice(W[W!=0]) if v['taste'] else 0.])
    menus=random.normal(size=(n,5,2))*l['scales'];prob=probabilities(parameters,menus,l['temperature'])[0]
    y=(random.random(n)[:,None]>np.cumsum(prob,axis=-1)).sum(-1);half=n//2
    public=dict(train=menus[:half].tolist(),train_y=y[:half].tolist(),test=menus[half:].tolist(),temperature=l['temperature'],probes=[])
    for kind in ('stakes','base-rate'):
        menu=random.normal(size=(5,2))*l['scales'];p=probabilities(parameters,[menu],l['temperature'],kind,v['mechanism'])[0,0]
        public['probes'].append(dict(kind=kind,menu=menu.tolist(),outcome=int(random.choice(6,p=p))))
    truth=dict(parameters=parameters.tolist(),heldout_y=y[half:].tolist(),mechanism=v['mechanism'],law=world)
    evidence,values=evaluate(public,truth);return public,truth,evidence,values

def check(v,p,t,evidence,fixed,design):
    # Independently accumulate log likelihoods per observation, rather than the
    # vectorized gather used for fitting/scoring in the producer.
    temp=p['temperature'];actual=t['heldout_y'];predictions={};fitted={}
    for name,grid in GRIDS.items():
        probs=probabilities(grid,p['train'],temp)
        ll=np.array([math.fsum(math.log(float(probs[i,j,y])) for j,y in enumerate(p['train_y'])) for i in range(len(grid))])
        mass=np.exp(ll-logsumexp(ll));fitted[name]=mass
        if not np.allclose(mass,evidence['fitted'][name],atol=1e-11,rtol=0):raise ValueError('N17 fitted posterior')
        predictions[name]=np.sum(mass[:,None,None]*probabilities(grid,p['test'],temp),axis=0)
    predictions['oracle']=probabilities(t['parameters'],p['test'],temp)[0]
    for name,prob in predictions.items():
        if not np.allclose(prob,evidence['probabilities'][name],atol=1e-11,rtol=0):raise ValueError('N17 forecast')
    rebuilt,values=evaluate(p,t)
    for name,prob in predictions.items():
        independent=-math.fsum(math.log(float(prob[j,y])) for j,y in enumerate(actual))/len(actual)
        if abs(independent-values[name+'_log_loss'])>1e-11:raise ValueError('N17 log score')
    null=probabilities(GRIDS['sensitivity'],p['test'],temp)
    maximum=max(math.fsum(math.log(float(null[i,j,y])) for j,y in enumerate(actual)) for i in range(len(null)))
    if abs(maximum-evidence['null_max_log_likelihood'])>1e-10:raise ValueError('N17 null denominator')
    if not np.allclose(rebuilt['mechanism_posterior'],evidence['mechanism_posterior'],atol=1e-11,rtol=0):raise ValueError('N17 mechanism posterior')
    return values

def study_readout(items):
    def combine(subset,metric):
        x=np.mean([s['metrics'][metric]['world_values'] for s in subset],axis=0)
        return dict(world_values=x.tolist(),interval=interval(x,['N17','combined',[s['variant']['id'] for s in subset],metric]))
    sizes=sorted(set(s['variant']['history'] for s in items));nulls={}
    for n in sizes:
        sub=[s for s in items if s['variant']['history']==n and not s['variant']['taste']]
        nulls[str(n)]=combine(sub,'taste_detected')
    gate=all(x['interval']['mean'] is not None and x['interval']['mean']<=.05 for x in nulls.values())
    primary=combine([s for s in items if s['variant']['history']==100 and s['variant']['taste']],'primary')
    separation={k:combine([s for s in items if s['variant']['history']==100],k) for k in ('sensitivity_gain','criterion_error_improvement','sensitivity_error_improvement')}
    identified=gate and all(x['interval']['lower'] is not None and x['interval']['lower']>0 for x in separation.values())
    return dict(primary=primary,no_taste_rates=nulls,no_taste_gate=gate,criterion_sensitivity=separation,
        SDT1_conditional_route=identified,mechanism_probe_gain=combine(items,'mechanism_gain'),
        finite_null_e_value_bound=.05,quality_coordinate_known_in_simulation=True,
        empirical_rates_do_not_prove_uniform_control_outside_registered_nulls=True)

def controls():
    menus=rng('N17-controls').normal(size=(6,5,2));changed=menus.copy();changed[:,:,1]*=-1
    assert np.array_equal(probabilities([0,1,0],menus,1),probabilities([0,1,0],changed,1))
    assert np.array_equal(probabilities([1,1,.5],menus,1,mechanism=0),probabilities([1,1,.5],menus,1,mechanism=1))
    assert not np.allclose(probabilities([1,1,.5],menus,1,'stakes',0),probabilities([1,1,.5],menus,1,'stakes',1))
    # Exact short-sequence e-value check for every finite null, independent of menu code.
    nulls=(.2,.5,.8);num=(.3,.7);expect=[]
    for true in nulls:
        total=0.
        for y in product((0,1),repeat=2):
            probability=lambda p:math.prod(p if x else 1-p for x in y)
            numerator=math.prod(num[i] if x else 1-num[i] for i,x in enumerate(y))
            total+=probability(true)*numerator/max(probability(p) for p in nulls)
        expect.append(total);assert total<=1+1e-12
    return dict(passed=True,no_taste_feature_invariance=True,ordinary_mechanisms_alias=True,
        interventions_change_laws=True,finite_null_e_value_expectation=expect,heldout_not_used_for_fitting=True)
def run(root,plan,pulse):
    from .directed_common import run as execute
    return execute(root,plan,pulse)
def reconstruct(root,plan):
    from .directed_common import reconstruct as verify
    return verify(root,plan)
