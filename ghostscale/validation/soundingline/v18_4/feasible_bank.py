"""Finite-law convex projection of retained learned response banks.

The state/targets are evaluator-only. Law access and a uniform 24-state prior
are supplied assumptions; neither projection nor a good score identifies roles.
"""
import math
from functools import lru_cache
import numpy as np
import scipy
from scipy.optimize import minimize
from ..v18_3 import world as W
from . import neural_data as D
from .bank_data import linear_map, repair

DIMENSIONS=('family','cell','support')
METRICS=('expected_loss','brier','bank_residual','projection_gap','raw_invalid')
QUERIES=D.NEW_QUERIES+D.FAR_QUERIES


def certificate(atoms,target,weights):
    projected=weights@atoms
    gradient=atoms@(projected-target)
    return dict(gap=float(weights@gradient-gradient.min()),
                residual=float(np.sum((projected-target)**2)),
                sum_error=float(abs(weights.sum()-1)),minimum_weight=float(weights.min()))


def project(atoms,target):
    atoms=np.asarray(atoms,float);target=np.asarray(target,float)
    if atoms.ndim!=2 or target.shape!=(atoms.shape[1],) or not np.isfinite(atoms).all() or not np.isfinite(target).all():
        raise ValueError('invalid projection input')
    n=len(atoms)
    def objective(w):
        r=w@atoms-target
        return .5*float(r@r),atoms@r
    result=minimize(objective,np.full(n,1/n),jac=True,method='SLSQP',bounds=[(0,1)]*n,
        constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(n)},
        options={'ftol':1e-15,'maxiter':1000})
    # Only roundoff normalization; an explicit independent gap decides validity.
    weights=np.maximum(result.x,0);weights/=weights.sum()
    # Solve the currently active affine face directly; this avoids confusing
    # SLSQP's objective-change stop with a small first-order optimality gap.
    active=np.flatnonzero(weights>1e-9);polished=False
    if len(active)>1:
        differences=(atoms[active[:-1]]-atoms[active[-1]]).T
        partial=np.linalg.lstsq(differences,target-atoms[active[-1]],rcond=1e-12)[0]
        face=np.r_[partial,1-partial.sum()]
        if face.min()>=-1e-12:
            candidate=np.zeros(n);candidate[active]=np.maximum(face,0);candidate/=candidate.sum()
            before=certificate(atoms,target,weights);after=certificate(atoms,target,candidate)
            if after['residual']<=before['residual']+1e-12 and after['gap']<before['gap']:
                weights=candidate;polished=True
    refinements=0
    for _ in range(2048):
        residual=weights@atoms-target;gradient=atoms@residual
        vertex=int(np.argmin(gradient));gap=float(weights@gradient-gradient[vertex])
        if gap<=1e-9:break
        direction=atoms[vertex]-weights@atoms;denominator=float(direction@direction)
        if denominator==0:break
        step=float(np.clip(gap/denominator,0,1));weights*=1-step;weights[vertex]+=step
        refinements+=1
    check=certificate(atoms,target,weights)
    return weights,dict(check,solver_success=bool(result.success),iterations=int(result.nit),
        affine_face_polished=polished,frank_wolfe_refinements=refinements,
        valid=check['gap']<=1e-7 and check['gap']>=-1e-10 and check['sum_error']<1e-10 and check['minimum_weight']>=0)


def infer(world,history,banks):
    atoms=np.concatenate([W.artifact_matrix(world,c) for c in D.TRAIN_QUERIES],axis=1)
    target=np.concatenate([W.artifact_matrix(world,c) for c in QUERIES],axis=1)
    augmented=np.concatenate([np.ones((len(W.STATES),1)),atoms],axis=1)
    mapping,diagnostic=linear_map(augmented,target,1e-3)
    posterior=W.posterior(W.packet(world,history));prior=np.full(len(W.STATES),1/len(W.STATES))
    forecasts={};checks={}
    for reader,bank in banks.items():
        y=np.asarray(bank,float).reshape(-1)
        if y.shape!=(80,) or np.any(y<0) or not np.allclose(y.reshape(5,16).sum(1),1,atol=1e-6,rtol=0):raise ValueError('invalid retained bank')
        weights,check=project(atoms,y)
        raw=np.r_[1.,y]@mapping;raw=raw.reshape(len(QUERIES),16)
        forecasts[reader+'|hull']=(weights@target).reshape(len(QUERIES),16)
        forecasts[reader+'|truncated']=repair(raw)
        checks[reader]=dict(check,weights=weights.tolist(),raw=raw.tolist(),
            raw_invalid=float(np.mean(np.any(raw< -1e-8,axis=1)|np.any(raw>1+1e-8,axis=1)|(abs(raw.sum(1)-1)>1e-6))))
    forecasts['prior']=(prior@target).reshape(len(QUERIES),16)
    forecasts['history-posterior']=(posterior@target).reshape(len(QUERIES),16)
    return forecasts,checks,dict(diagnostic,affine_rank=int(np.linalg.matrix_rank(atoms-atoms[0],tol=1e-10)))


def score(truth,probabilities):
    p=np.maximum(probabilities,1e-12);p/=p.sum(axis=-1,keepdims=True)
    return float(-np.mean(np.sum(truth*np.log(p),axis=1))),float(np.mean(np.sum((truth-p)**2,axis=1)))


def unit(index,cell,support,payload,scipy_version):
    if scipy.__version__!=scipy_version:raise ValueError('frozen scipy version differs')
    forecasts,checks,diagnostic=infer(payload['world'],payload['history'],payload['banks'])
    truth=np.stack([W.artifact_matrix(payload['world'],c)[payload['state']] for c in QUERIES])
    grouped={}
    for name,p in forecasts.items():
        if '|' in name:
            reader,mode=name.split('|');kind=reader.rsplit('-seed',1)[0];check=checks[reader]
        else:kind=name;mode='reference';check=dict(residual=0.,gap=0.,raw_invalid=0.)
        for label,chosen in (('composition',slice(0,3)),('farther',slice(3,5))):
            loss,brier=score(truth[chosen],p[chosen]);method=kind+'|'+mode+'|'+label
            grouped.setdefault(method,[]).append(dict(expected_loss=loss,brier=brier,bank_residual=check['residual'],
                projection_gap=check['gap'],raw_invalid=check['raw_invalid'] if mode=='truncated' else 0.))
    rows=[dict(method=name,**{key:float(np.mean([r[key] for r in values])) for key in METRICS}) for name,values in grouped.items()]
    return dict(family='L2c',index=index,cell=cell,support=support,payload=payload,scipy_version=scipy_version,
        forecasts={k:p.tolist() for k,p in forecasts.items()},checks=checks,diagnostic=diagnostic,rows=rows,
        gates=dict(live=all(c['valid'] for c in checks.values()),positive_and_placebo=all(controls().values())),
        scope='paired retained learned banks; supplied law and uniform 24-state prior; no training or role identification')


def verify(unit):
    if not all(unit['gates'].values()):raise ValueError('feasible-bank instrument failed')
    payload=unit['payload'];w=payload['world']
    # Scalar mixtures independently reconstruct every hull output and certificate.
    atoms=np.concatenate([W.artifact_matrix(w,c) for c in D.TRAIN_QUERIES],axis=1)
    for reader,c in unit['checks'].items():
        weights=c['weights'];y=np.asarray(payload['banks'][reader],float).reshape(-1)
        projected=np.array([math.fsum(weights[s]*atoms[s,j] for s in range(len(weights))) for j in range(atoms.shape[1])])
        dots=[math.fsum(float(atoms[s,j])*(float(projected[j])-float(y[j])) for j in range(len(y))) for s in range(len(weights))]
        gap=math.fsum(weights[s]*dots[s] for s in range(len(weights)))-min(dots)
        if abs(gap-c['gap'])>1e-10 or gap>1e-7:raise ValueError('projection certificate differs')
        if abs(math.fsum((float(a)-float(b))**2 for a,b in zip(projected,y))-c['residual'])>1e-10:raise ValueError('projection residual differs')
        if min(weights)<0 or abs(math.fsum(weights)-1)>1e-10:raise ValueError('invalid simplex')
        for q,context in enumerate(QUERIES):
            matrix=W.artifact_matrix(w,context)
            forecast=[math.fsum(weights[s]*float(matrix[s,j]) for s in range(len(weights))) for j in range(16)]
            if not np.allclose(forecast,unit['forecasts'][reader+'|hull'][q],atol=1e-12,rtol=0):raise ValueError('hull forecast differs')
        target=np.concatenate([W.artifact_matrix(w,q) for q in QUERIES],axis=1)
        augmented=np.column_stack((np.ones(len(atoms)),atoms))
        raw=(np.r_[1.,y]@np.linalg.pinv(augmented,rcond=1e-3)@target).reshape(len(QUERIES),16)
        if not np.allclose(raw,c['raw'],atol=1e-8,rtol=1e-10):raise ValueError('raw inverse differs')
        if not np.allclose(repair(raw),unit['forecasts'][reader+'|truncated'],atol=1e-8,rtol=0):raise ValueError('inverse repair differs')
    # Independent batch likelihood multiplication, with explicit root deduplication.
    likelihood=np.ones(len(W.STATES));seen=set()
    for obs in payload['history']:
        if obs['source'] in seen:continue
        seen.add(obs['source']);j=W.PROGRAMS.index(tuple(obs['program']))
        likelihood*=W.matrix(w,obs['context'])[:,j]
    likelihood/=math.fsum(likelihood)
    for q,context in enumerate(QUERIES):
        matrix=W.artifact_matrix(w,context)
        prior=[math.fsum(matrix[:,j])/len(W.STATES) for j in range(16)]
        post=[math.fsum(float(likelihood[s])*float(matrix[s,j]) for s in range(len(W.STATES))) for j in range(16)]
        if not np.allclose(prior,unit['forecasts']['prior'][q],atol=1e-12,rtol=0) or not np.allclose(post,unit['forecasts']['history-posterior'][q],atol=1e-12,rtol=0):raise ValueError('reference reconstruction differs')
    truth=np.stack([W.artifact_matrix(w,c)[payload['state']] for c in QUERIES])
    groups={}
    for name,prob in unit['forecasts'].items():
        p=np.asarray(prob,float)
        if np.any(p<0) or not np.allclose(p.sum(1),1,atol=1e-6,rtol=0):raise ValueError('forecast invalid')
        if '|' in name:
            reader,mode=name.split('|');kind=reader.rsplit('-seed',1)[0]
        else:kind=name;mode='reference'
        for label,indices in (('composition',range(3)),('farther',range(3,5))):
            losses=[];briers=[]
            for q in indices:
                row=[max(float(v),1e-12) for v in p[q]];total=math.fsum(row);row=[v/total for v in row]
                losses.append(-math.fsum(float(t)*math.log(v) for t,v in zip(truth[q],row)))
                briers.append(math.fsum((float(t)-v)**2 for t,v in zip(truth[q],row)))
            groups.setdefault(kind+'|'+mode+'|'+label,[]).append((math.fsum(losses)/len(losses),math.fsum(briers)/len(briers)))
    for row in unit['rows']:
        values=groups[row['method']]
        for j,key in enumerate(('expected_loss','brier')):
            if abs(math.fsum(v[j] for v in values)/len(values)-row[key])>1e-10:raise ValueError('independent score differs')
    return True


@lru_cache(maxsize=1)
def controls():
    atoms=np.eye(3);truth=np.array([.2,.3,.5]);w,c=project(atoms,truth)
    outside,oc=project(atoms,np.array([2.,0.,0.]))
    uniform,uc=project(np.full((3,4),.25),np.full(4,.25))
    permutation=np.array([2,0,1]);pw,pc=project(atoms[permutation],truth)
    return dict(live_known_mixture=bool(c['valid'] and np.max(abs(w-truth))<1e-6),
        positive_boundary=bool(oc['valid'] and np.max(abs(outside-[1,0,0]))<1e-6),
        placebo_alias=bool(uc['valid'] and np.max(abs(uniform@np.full((3,4),.25)-.25))<1e-12),
        state_order_invariance=bool(pc['valid'] and np.max(abs(pw@atoms[permutation]-truth))<1e-6))
