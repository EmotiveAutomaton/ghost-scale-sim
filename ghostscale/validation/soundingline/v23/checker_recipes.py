"""Independent finite recipe enumeration and probability-mass reconstruction."""
from itertools import combinations, product
from functools import lru_cache
import numpy as np
from .checker_new import check, softmax


@lru_cache(None)
def dictionary(drives=4, exactly=None, seeking='dimension'):
    dimensions=range(drives if seeking=='dimension' else drives-1)
    recipes=[(d,) for d in dimensions]+list(combinations(dimensions,2)); weights=[]
    for count in ([exactly] if exactly else (1,2)):
        for selected in combinations(range(len(recipes)),count):
            for values in product((1.,2.),repeat=count):
                row=np.zeros(len(recipes))
                for i,value in zip(selected,values):row[i]=value
                weights.append(row)
    return recipes,np.array(weights)


def affordance(options,recipes,meaning='conjunctive',need=1.):
    options=np.array(options,float);out=[]
    for recipe in recipes:
        selected=options[...,list(recipe)]
        value=np.min(selected,axis=-1) if meaning=='conjunctive' else np.sum(selected,axis=-1)/len(recipe)
        out.append(value*(need if 0 in recipe else 1.))
    return np.stack(out,axis=-1)


def n7_scalar(v,p,mode,shift_estimate):
    s=v['settings']; recipes,weights=dictionary()
    direction=np.array([int(1 in r)-int(2 in r) for r in recipes])
    states=list(product((-1.,1.) if s['shift_kind']=='individual' else (1.,),
        (False,True) if s['aspiration'] else (False,),np.linspace(.75,3.,16)))
    prior=np.array([s['aspiration'] if aspiring else 1-s['aspiration'] if s['aspiration'] else 1.
                    for sign,aspiring,beta in states])[:,None]*np.ones((1,len(weights)))
    log_statement=np.zeros_like(prior)
    for i,(sign,aspiring,beta) in enumerate(states):
        shift=sign*s['shift']*direction if mode=='law-marginal' else np.zeros(len(recipes))
        if mode=='pooled-correction':shift=np.asarray(shift_estimate)
        if mode not in ('choices-only','training-choices'):
            log_statement[i]=-.5*np.sum(((np.array(p['statement'])-weights-shift)/s['statement_sd'])**2,axis=1)
    mass=prior*np.exp(log_statement-log_statement.max());mass/=mass.sum()
    if mode!='statements-only':
        for step,row in enumerate(p['history']):
            sat=affordance(row['options'],recipes)
            for i,(sign,aspiring,beta) in enumerate(states):
                personal=weights+aspiring*step/max(1,s['records'])*sign*s['shift']*direction
                mass[i]*=softmax(beta*(personal@sat.T))[:,row['choice']]
            mass/=mass.sum()
    answer=np.zeros(3); sat=affordance(p['query'],recipes)
    for i,(sign,aspiring,beta) in enumerate(states):
        personal=weights+aspiring*sign*s['shift']*direction
        answer+=mass[i]@softmax(beta*(personal@sat.T))
    marginal=mass.sum(0)
    return answer[None,:],dict(recipe_presence=(marginal@(weights>0)).tolist(),expected_weights=(marginal@weights).tolist())


def n7(v,p,mode,shift_estimate):
    s=v['settings'];recipes,weights=dictionary();direction=np.array([int(1 in r)-int(2 in r) for r in recipes])
    states=list(product((-1.,1.) if s['shift_kind']=='individual' else (1.,),
        (False,True) if s['aspiration'] else (False,),np.linspace(.75,3.,16)))
    sign=np.array([a for a,b,c in states]);aspiring=np.array([b for a,b,c in states]);beta=np.array([c for a,b,c in states])
    prior=np.where(aspiring,s['aspiration'],1-s['aspiration'])[:,None]*np.ones((1,len(weights)))
    shift=sign[:,None]*s['shift']*direction if mode=='law-marginal' else np.zeros((len(states),len(recipes)))
    if mode=='pooled-correction':shift+=np.asarray(shift_estimate)
    statement=np.zeros_like(prior)
    if mode not in ('choices-only','training-choices'):
        statement=-.5*np.sum(((np.asarray(p['statement'])[None,None,:]-weights[None,:,:]-shift[:,None,:])/s['statement_sd'])**2,axis=2)
    mass=prior*np.exp(statement-statement.max());mass/=mass.sum()
    drift=aspiring[:,None]*sign[:,None]*s['shift']*direction
    if mode!='statements-only':
        for step,row in enumerate(p['history']):
            personal=weights[None,:,:]+step/max(1,s['records'])*drift[:,None,:]
            logits=beta[:,None,None]*(personal@affordance(row['options'],recipes).T)
            probability=np.exp(logits-logits.max(axis=2,keepdims=True));probability/=probability.sum(axis=2,keepdims=True)
            mass*=probability[:,:,row['choice']];mass/=mass.sum()
    logits=beta[:,None,None]*((weights[None,:,:]+drift[:,None,:])@affordance(p['query'],recipes).T)
    probability=np.exp(logits-logits.max(axis=2,keepdims=True));probability/=probability.sum(axis=2,keepdims=True)
    answer=np.sum(mass[:,:,None]*probability,axis=(0,1));marginal=mass.sum(0)
    return answer[None,:],dict(recipe_presence=(marginal@(weights>0)).tolist(),expected_weights=(marginal@weights).tolist())


@lru_cache(None)
def verify_shift_training(identity,sd,shift,kind,aspiration):
    from .statements import shift_training
    expected,rows=shift_training(identity,sd,shift,kind,aspiration)
    v=dict(id=identity,settings=dict(statement_sd=sd,shift=shift,shift_kind=kind,aspiration=aspiration,records=128))
    residuals=[]
    for row in rows:
        _,detail=n7(v,row['public'],'training-choices',None)
        if not np.allclose(detail['expected_weights'],row['choice_only_expected_weights'],atol=1e-11,rtol=0):
            raise ValueError('independent shift-training reconstruction mismatch')
        residuals.append(np.array(row['public']['statement'])-detail['expected_weights'])
    result=np.mean(residuals,axis=0)
    if not np.allclose(result,expected,atol=1e-11,rtol=0):raise ValueError('learned shift mismatch')
    return result


def verify_n7(v,p,truth,pred,detail):
    expected={}
    for name in ('law-marginal','choices-only','statements-only','both-naive','pooled-correction'):
        expected[name],rebuilt=n7(v,p,name,detail['training']['shift'])
        for key in rebuilt:
            if not np.allclose(rebuilt[key],detail[name][key],atol=1e-11,rtol=0):
                raise ValueError('independent statement recipe posterior mismatch')
    return check(pred,expected)


def n4(v,p,linear=False,seeking=None,law=None,prior=None,retain_types=False):
    s=v['settings'];mode=seeking or s['seeking'];recipes,weights=dictionary(s['drives'],s['recipes'],mode)
    meaning='additive' if linear else s['meaning']
    states=list(product(range(16) if law is None else (law,),range(3) if mode=='vigor' else (1,)))
    mass=np.ones((len(states),len(weights)));mass/=mass.sum()
    if prior is not None:mass*=np.array(prior)[None,:];mass/=mass.sum()
    action=np.array([1/(1+np.exp(.5-(.5,1.,2.)[level])) if mode=='vigor' else .75 for _,level in states])
    betas=np.array([(.75+2.25*ell/15)*((.5,1.,2.)[level] if mode=='vigor' else 1.) for ell,level in states])
    for row in p['history']:
        mass*=action[:,None] if row['acted'] else 1-action[:,None]
        if row['acted']:
            utility=weights@affordance(row['options'],recipes,meaning,row['need']).T
            for i,beta in enumerate(betas):mass[i]*=softmax(beta*utility)[:,row['choice']]
        mass/=mass.sum()
    utility=weights@affordance(p['query']['options'],recipes,meaning,p['query']['need']).T
    answer=sum(mass[i]@softmax(beta*utility) for i,beta in enumerate(betas))
    marginal=mass.sum(0);vigor=np.zeros(3)
    for i,(_,level) in enumerate(states):vigor[level]+=mass[i].sum()
    has_seeking=np.array([any(weight>0 and s['drives']-1 in recipe for weight,recipe in zip(row,recipes)) for row in weights])
    details=dict(recipe_presence=(marginal@(weights>0)).tolist(),act_probability=float(mass.sum(1)@action),
        vigor_posterior=vigor.tolist(),seeking_profile_mass=float(marginal@has_seeking))
    if retain_types:details['type_posterior']=marginal.tolist()
    return answer[None,:],details


@lru_cache(None)
def verify_pooling_training(drives,recipes,meaning,seeking):
    from .values import pooling_training
    prior,rows=pooling_training('shared',drives,recipes,meaning,seeking)
    v=dict(family='N4',settings=dict(drives=drives,recipes=recipes,meaning=meaning,seeking=seeking,records=8,menus='random',extension=False))
    total=np.full(len(prior),.5)
    for row in rows:
        _,detail=n4(v,row['public'],retain_types=True)
        if not np.allclose(detail['type_posterior'],row['type_posterior'],atol=1e-11,rtol=0):raise ValueError('recipe-pooling training posterior mismatch')
        total+=detail['type_posterior']
    if not np.allclose(total/total.sum(),prior,atol=1e-11,rtol=0):raise ValueError('pooled recipe prior mismatch')
    return True


def verify_n4(v,p,truth,pred,detail):
    if v['settings']['extension']:
        probabilities=np.linspace(.6,.95,16);weights=np.ones(16)/16
        for row in p['history']:
            weights*=probabilities if row[0] else 1-probabilities;weights/=weights.sum()
        future=[1-probabilities,np.full(16,.5),probabilities] if p['changed'] else [probabilities]*3
        expected={name:np.array([[1-f@weights,f@weights]]) for name,f in zip(('transition','habit','intrinsic'),future)}
        expected['law-marginal']=sum(expected.values())/3
        if not np.allclose(detail['mechanism_posterior'],[1/3]*3,atol=1e-12):raise ValueError('aliased history was treated as identifying')
        return check(pred,expected)
    expected={}
    modes=[('law-marginal',{}),('drive-weights-only',{'linear':True}),
           ('seeking-dimension',{'seeking':'dimension'}),('seeking-vigor',{'seeking':'vigor'})]
    if 'oracle-law' in pred:modes.append(('oracle-law',{'law':truth['law']}))
    if 'pooled-recipe' in pred:
        from .values import pooling_training
        s=v['settings'];prior,_=pooling_training(v['id'],s['drives'],s['recipes'],s['meaning'],s['seeking'])
        modes.append(('pooled-recipe',{'prior':prior}))
    for name,kwargs in modes:
        expected[name],rebuilt=n4(v,p,**kwargs)
        for key in rebuilt:
            if not np.allclose(rebuilt[key],detail[name][key],atol=1e-11,rtol=0):raise ValueError('independent value posterior mismatch')
    return check(pred,expected)
