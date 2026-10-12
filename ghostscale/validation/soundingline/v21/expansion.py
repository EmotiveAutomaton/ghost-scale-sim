"""C06 finite model expansion with explicit full-library planning privilege.

Selection records precede independent query outcomes. The expected-gain planner
evaluates the whole registered library; that work is counted, not free search.
"""
from itertools import combinations
import math
import numpy as np
from . import world as W
from .studies import retain
from ..v18_3.io import write

PRIOR=np.array([.5,1/6,1/6,1/6])
READERS=('base','confidence','surprise','expected-gain','fixed','full-library')
PRICE=.01


def outputs():
    x=np.arange(4);base=x//2
    return np.array([base,base^(x%2),base^(x==3),base^np.isin(x,[1,2])],int)


def subset_posterior(likelihood,selected):
    # Candidate four is an alias of base, not a fifth independent hypothesis.
    ids=sorted({0}|{j for j in selected if j<4})
    mass=likelihood[:,ids]*PRIOR[ids];mass/=mass.sum(1,keepdims=True)
    out=np.zeros_like(likelihood);out[:,ids]=mass
    return out


def analyze(history_x,history_y,reliability,cue_noise,budget):
    response=outputs();a=.5+(reliability-.5)*(1-2*cue_noise)
    match=response[:,history_x].transpose(1,0,2)==history_y[:,None,:]
    likelihood=np.prod(np.where(match,a,1-a),axis=2)
    predictive=np.where(response==1,reliability,1-reliability)
    full=subset_posterior(likelihood,(1,2,3));ideal=full@predictive
    n=history_x.shape[1];hits=match[:,0].sum(1)
    tail=np.array([sum(math.comb(n,j)*a**j*(1-a)**(n-j) for j in range(int(h)+1)) for h in hits])
    surprise=tail<=.05;confidence=np.full(len(hits),max(reliability,1-reliability)<.75)
    candidates=[s for count in range(budget+1) for s in combinations((1,2,3,4),count)]
    posteriors=[];risk=[]
    for selected in candidates:
        post=subset_posterior(likelihood,selected);pred=post@predictive
        risk.append(np.mean(-ideal*np.log(pred)-(1-ideal)*np.log1p(-pred),axis=1)+PRICE*len(selected));posteriors.append(post)
    costs=np.array(risk).T;choice=np.argmax(costs<=costs.min(1,keepdims=True)+1e-12,axis=1)
    result={}
    fixed=tuple(range(1,budget+1));fixed_post=subset_posterior(likelihood,fixed);base_post=subset_posterior(likelihood,())
    for reader in READERS:
        selected=np.zeros((len(hits),4),int)
        if reader=='full-library':selected[:,:3]=1;post=full;evals=np.full(len(hits),4);search=np.zeros(len(hits),int)
        elif reader=='expected-gain':
            post=np.array([posteriors[j][i] for i,j in enumerate(choice)])
            for i,j in enumerate(choice):
                for item in candidates[j]:selected[i,item-1]=1
            evals=np.full(len(hits),4);search=np.full(len(hits),len(candidates))
        else:
            trigger=np.ones(len(hits),bool) if reader=='fixed' else surprise if reader=='surprise' else confidence if reader=='confidence' else np.zeros(len(hits),bool)
            selected[trigger,:budget]=1;post=np.where(trigger[:,None],fixed_post,base_post)
            evals=1+np.minimum(selected.sum(1),3);search=np.zeros(len(hits),int)
        result[reader]=dict(posterior=post,probability_one=post@predictive,selected=selected,added=selected.sum(1),likelihood_evaluations=evals,subset_evaluations=search)
    return result,dict(likelihood=likelihood,full_posterior=full,base_tail=tail,surprise_trigger=surprise,confidence_trigger=confidence)


def controls():
    x=np.tile([1],(2,16));y=np.array([[1]*16,[0]*16]);r,aux=analyze(x,y,.9,.05,4)
    uninformative,nu=analyze(x,y,.9,.5,4)
    alias=subset_posterior(aux['likelihood'],(4,));base=subset_posterior(aux['likelihood'],())
    return dict(live_confident_misspecification=bool(aux['surprise_trigger'][0] and not aux['confidence_trigger'][0]),
        placebo_correct_model=bool(not aux['surprise_trigger'][1]),
        placebo_uninformative=bool(np.array_equal(nu['full_posterior'][0],nu['full_posterior'][1])),
        positive_alias_removal=bool(np.array_equal(alias,base)),
        live_expansion_gain=bool(r['expected-gain']['probability_one'][0,1]>.8))


def run(root,d,pulse):
    from .expansion_check import reference
    population=[];maximum=0.;casebook=[]
    for world in d['lineages']:
        reliability=float(np.random.default_rng(W.seed('v21-c06-law',world)).uniform(.82,.94))
        rng=np.random.default_rng(W.seed('v21-c06-cases',world));n=d['test_cases'];h=d['history']
        kind=rng.integers(1,4,size=n);hx=rng.integers(4,size=(n,h));hu=rng.random((n,h));qu=rng.random(n);qx=rng.integers(4,size=n)
        er=root/'evaluator';er.mkdir(exist_ok=True);rr=root/'reader';rr.mkdir(exist_ok=True)
        np.savez_compressed(er/f'{world}-draws_points.npz',kind=kind,history_context=hx,history_uniform=hu,query_uniform=qu,query_context=qx,reliability=np.array(reliability))
        for missing in (False,True):
            true_kind=kind if missing else np.zeros(n,int);target=(outputs()[true_kind,qx]+(qu>=reliability))%2
            for strength,noise in (('weak',.45),('strong',.05)):
                rate=.5+(reliability-.5)*(1-2*noise);hy=(outputs()[true_kind[:,None],hx]+(hu>=rate))%2
                key=f'{world}-m{int(missing)}-{strength}'
                np.savez_compressed(rr/f'{key}_points.npz',history_context=hx,history_observation=hy,query_context=qx)
                np.savez_compressed(er/f'{key}_points.npz',truth=target,model=true_kind)
                for budget in (1,2,4):
                    result,aux=analyze(hx,hy,reliability,noise,budget)
                    independent=[reference(xx,yy,reliability,noise,budget) for xx,yy in zip(hx,hy)]
                    for reader in READERS:
                        row=result[reader]
                        for i,ref in enumerate(independent):
                            error=float(abs(row['probability_one'][i]-np.array(ref[reader]['probability_one'])).max());maximum=max(maximum,error);assert error<1e-12
                            assert np.array_equal(row['selected'][i],ref[reader]['selected'])
                        one=row['probability_one'][np.arange(n),qx];lp=np.full((n,128),-np.inf);lp[:,0]=np.log1p(-one);lp[:,1]=np.log(one)
                        metrics=retain(root,f'{key}-b{budget}-{reader}',lp,target)
                        np.savez_compressed(er/f'{key}-b{budget}-{reader}-decisions_points.npz',**row)
                        population.append(dict(lineage=world,omission=missing,cue=strength,budget=budget,reader=reader,metrics=metrics,expansion_fraction=float(np.mean(row['added']>0)),mean_added=float(row['added'].mean()),likelihood_evaluations=float(row['likelihood_evaluations'].mean()),subset_evaluations=float(row['subset_evaluations'].mean()),priced_loss=float(metrics['log_loss']+PRICE*row['added'].mean())))
                    np.savez_compressed(er/f'{key}-b{budget}-checks_points.npz',**aux)
                    pulse(phase='finite-expansion',lineage=world,omission=missing,cue=strength,budget=budget)
                if world==d['lineages'][0]:casebook.append(dict(omission=missing,cue=strength,history_context=hx[0].tolist(),history_observation=hy[0].tolist(),query_context=int(qx[0]),truth=int(target[0]),model=int(true_kind[0]),role='evaluator only'))
    write(root/'CASEBOOK_EVALUATOR.json',casebook)
    return dict(card='G21-C06',population=population,controls=controls(),independent_max_probability_error=maximum,
        scope='Finite supplied-law model-missingness ruler; held-out query outcomes, no learned mechanism invention.',
        limitations=['Four known binary response laws instantiate omitted belief, constraint and order switches; their semantics are stipulated, not measured psychological variables.',
        'The expected-gain planner evaluates every model and every allowed subset; its full-library posterior is privileged and its likelihood/subset counts are reported. It does not demonstrate cheap bounded search.',
        'The .01 price per added hypothesis is a declared utility fee, not CPU seconds. Actual CPU, likelihood and subset evaluations remain separate.',
        'The base always exceeds the fixed .75 confidence threshold by construction; this known-answer failure is not an empirical claim about every confidence policy.',
        'Cue strength changes the independent historical observation channel, not the query mechanism. Query truth is never used for expansion decisions.',
        'Candidate four duplicates the base law and is removed from probabilistic mass; fixed selection still pays its search fee.',
        'Exact priors, reliability and observation rates are supplied; no fitting, optimizer convergence or label-matched B/C superiority claim.',
        'Eight reused development worlds; conditions and cases nested, no fresh confirmation. Purpose-built miniature — architecture untested.',
        'No feature-product arm or native-to-execution-shift contrast; weak-to-strong cue differences and redundant-successor controls are separate.'])
