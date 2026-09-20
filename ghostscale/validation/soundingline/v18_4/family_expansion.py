"""Sequential finite-family selection, averaging and paid catalog expansion.

Reader entry points consume public histories only. Prices are declared sensitivity
assumptions, not CPU timings or a learned value of computation.
"""
import math
import numpy as np
from ..v18_3 import world as W
from ..v16.world import execute

DIMENSIONS=('family','cell','kind','order','copy_span','length')
METRICS=('expected_loss','final_loss','expected_match','expanded','purchase_step',
         'likelihood_evaluations','net_match_low','net_match_high')
KINDS=('in-family','initial-alternative','new-rule','outside-menu')
ORDERS=('old-first','composed-first','interleaved')
METHODS=('fixed','select','mixture','paid-select-1.5','paid-mixture-1.5',
         'paid-select-3','paid-mixture-3','expanded-select','expanded-mixture')


def update(weights,scores,likelihoods):
    evidence=np.einsum('ij,ij->i',weights,likelihoods)
    if np.any(evidence<=0):raise ValueError('empty candidate support')
    return weights*likelihoods/evidence[:,None],scores+np.log(evidence)


def mixture(scores):
    p=np.exp(scores-np.max(scores));return p/p.sum()


def catalog(w):
    return [dict(w),dict(w,coupled=not w['coupled']),
            dict(w,rule='satisficing' if w['rule']=='softmax' else 'softmax')]


def make_case(index,cell,kind,order,copy_span,length=32):
    if kind not in KINDS or order not in ORDERS or copy_span not in (1,3):raise ValueError('undeclared case')
    if length<8 or length%2:raise ValueError('balanced even stream required')
    w=W.make_world(cell,18047000+index);truth=dict(w)
    if kind=='initial-alternative':truth=catalog(w)[1]
    elif kind=='new-rule':truth=catalog(w)[2]
    elif kind=='outside-menu':truth=dict(w,rule='lexicographic',coupled=not w['coupled'])
    # State/draw positions shared across orders and copy counts, independent of kind.
    state=int(W.rng('v18.4-family-expansion-state',index,cell).integers(len(W.STATES)))
    base=[];half=length//2
    for j in range(length):
        c=W.QUERIES[j%5] if j<half else W.QUERIES[5+(j-half)%3]
        obs=W.observe(truth,state,c,W.rng('v18.4-family-expansion',index,cell,j),f'root-{j}')
        base.append(obs)
    permutation=(list(range(length)) if order=='old-first' else
                 list(range(half,length))+list(range(half)) if order=='composed-first' else
                 [j for i in range(half) for j in (i,i+half)])
    history=[dict(base[j]) for j in permutation for _ in range(copy_span)]
    queries=[W.FUTURES[j%4] for j in permutation]
    return dict(world=w,history=history,queries=queries),dict(world=truth,state=state,permutation=permutation)


def read_stream(public,method,allow_expansion=True,force_at=None):
    if method not in METHODS:raise ValueError('undeclared reader')
    worlds=catalog(public['world']);n=len(W.STATES)
    count=1 if method=='fixed' else (3 if method.startswith('expanded-') else 2)
    weights=np.ones((count,n))/n;scores=np.zeros(count);seen={};unique=[]
    evaluations=0;purchase=0 if count==3 else None;surprises=[];trace=[]
    paid=method.startswith('paid-');threshold=float(method.rsplit('-',1)[1]) if paid else None
    select=method=='fixed' or 'select' in method
    for obs in public['history']:
        if obs['source'] in seen:
            if obs!=seen[obs['source']]:raise ValueError('conflicting copied source')
            continue
        t=len(unique)
        buy=(paid and count==2 and allow_expansion and
             ((force_at is not None and t>=force_at) or
              (force_at is None and t>=8 and math.fsum(surprises[-4:])/4>threshold)))
        if buy:
            added=np.ones((1,n))/n;score=np.zeros(1)
            for past in unique:
                added,score=update(added,score,W.likelihood(worlds[2],past)[None,:]);evaluations+=n
            weights=np.concatenate((weights,added));scores=np.concatenate((scores,score));count=3;purchase=t
        q=public['queries'][t]
        predictions=np.array([weights[j]@W.artifact_matrix(worlds[j],q) for j in range(count)])
        p=predictions[int(np.argmax(scores))] if select else mixture(scores)@predictions
        trace.append(dict(step=t,source=obs['source'],query=q,prediction=p.tolist(),
                          expanded=count==3,model_weights=mixture(scores).tolist()))
        lik=np.array([W.likelihood(worlds[j],obs) for j in range(count)])
        # The purchase sensor always uses the initial two models and preceding data.
        if paid:
            evidence=np.einsum('ij,ij->i',weights[:2],lik[:2])
            surprises.append(-math.log(float(mixture(scores[:2])@evidence)))
        weights,scores=update(weights,scores,lik);evaluations+=count*n
        seen[obs['source']]=obs;unique.append(obs)
    final=[]
    for q in W.FUTURES:
        predictions=np.array([weights[j]@W.artifact_matrix(worlds[j],q) for j in range(count)])
        p=predictions[int(np.argmax(scores))] if select else mixture(scores)@predictions
        final.append(dict(query=q,prediction=p.tolist()))
    return dict(method=method,trace=trace,final=final,expanded=float(count==3),
                purchase_step=len(unique)+1 if purchase is None else purchase,
                likelihood_evaluations=evaluations,final_state_weights=weights.tolist(),
                final_log_evidence=scores.tolist(),sensor_surprises=surprises)


def score(reader,truth):
    losses=[];matches=[]
    for item in reader['trace']:
        p=np.array(item['prediction']);q=W.artifact_matrix(truth['world'],item['query'])[truth['state']]
        # Canonical native program implements the chosen artifact; no acquisition gain.
        legal=sorted(set(map(int,W.ARTIFACTS)))
        chosen=min(a for a in legal if p[a]>=max(p[legal])-1e-10)
        program=min((p for p in W.PROGRAMS if execute(p).artifact==chosen),key=lambda p:(len(p),p))
        item.update(expected_loss=W.cross_entropy(q,p),expected_match=float(q[chosen]),
                    program=list(program),constructed_artifact=execute(program).artifact)
        losses.append(item['expected_loss']);matches.append(item['expected_match'])
    final=[]
    for item in reader['final']:
        q=W.artifact_matrix(truth['world'],item['query'])[truth['state']]
        item['expected_loss']=W.cross_entropy(q,item['prediction']);final.append(item['expected_loss'])
    length=len(losses);match=math.fsum(matches)/length;cost=reader['likelihood_evaluations']
    reader.update(expected_loss=math.fsum(losses)/length,final_loss=math.fsum(final)/len(final),expected_match=match,
                  net_match_low=match-(reader['expanded']+1e-5*cost)/length,
                  net_match_high=match-(4*reader['expanded']+1e-4*cost)/length)
    return reader


def unit(index,cell=0,kind='new-rule',order='old-first',copy_span=1,length=32,split='test'):
    public,truth=make_case(index,cell,kind,order,copy_span,length)
    rows=[score(read_stream(public,m),truth) for m in METHODS]
    return dict(family='R1',index=index,cell=cell,kind=kind,order=order,copy_span=copy_span,length=length,
                public=public,evaluator=truth,rows=rows,
                scope='stationary finite law; paid supplied catalog access, not law invention; expected native artifact matching, not uptake')


def verify(unit):
    public=unit['public'];truth=unit['evaluator'];n=len(W.STATES)
    history=list({o['source']:o for o in public['history']}.values())
    likelihood_bank=[np.array([W.likelihood(w,o) for o in history]) for w in catalog(public['world'])]
    if len(history)!=unit['length'] or len(public['history'])!=unit['length']*unit['copy_span']:raise ValueError('trace roster differs')
    for source in history:
        copies=[o for o in public['history'] if o['source']==source['source']]
        if len(copies)!=unit['copy_span'] or any(o!=source for o in copies):raise ValueError('copy identity differs')
    for row in unit['rows']:
        losses=[];matches=[]
        if len(row['trace'])!=unit['length'] or len(row['final'])!=4:raise ValueError('forecast roster differs')
        for j,item in enumerate(row['trace']+row['final']):
            p=item['prediction'];q=W.artifact_matrix(truth['world'],item['query'])[truth['state']]
            if any(v<0 or not math.isfinite(v) for v in p) or abs(math.fsum(p)-1)>1e-10:raise ValueError('invalid probabilities')
            loss=math.fsum(-float(a)*math.log(float(b)) for a,b in zip(q,p) if a>0)
            if abs(loss-item['expected_loss'])>1e-10:raise ValueError('scalar score differs')
            if j<unit['length']:
                if item['source']!=history[j]['source'] or item['query']!=public['queries'][j]:raise ValueError('query/source alignment differs')
                artifact=execute(item['program']).artifact
                if artifact!=item['constructed_artifact'] or abs(q[artifact]-item['expected_match'])>1e-10:raise ValueError('native match differs')
                legal=sorted(set(map(int,W.ARTIFACTS)))
                if artifact!=min(a for a in legal if p[a]>=max(p[a] for a in legal)-1e-10):raise ValueError('modal choice differs')
                losses.append(loss);matches.append(float(q[artifact]))
        if abs(math.fsum(losses)/len(losses)-row['expected_loss'])>1e-10:raise ValueError('mean loss differs')
        if abs(math.fsum(matches)/len(matches)-row['expected_match'])>1e-10:raise ValueError('mean match differs')
        if abs(math.fsum(x['expected_loss'] for x in row['final'])/4-row['final_loss'])>1e-10:raise ValueError('final mean differs')
        # Independent batch likelihood product reconstructs the final joint law.
        for k in range(len(row['final_state_weights'])):
            logs=[-math.log(n)+math.fsum(math.log(float(v)) for v in likelihood_bank[k][:,s]) for s in range(n)]
            top=max(logs);total=math.fsum(math.exp(x-top) for x in logs)
            expected=[math.exp(x-top)/total for x in logs]
            if np.max(abs(np.array(expected)-row['final_state_weights'][k]))>1e-10:raise ValueError('batch posterior differs')
            if abs(top+math.log(total)-row['final_log_evidence'][k])>1e-10:raise ValueError('batch evidence differs')
        count=1 if row['method']=='fixed' else 2+int(row['expanded'])
        if row['likelihood_evaluations']!=count*n*unit['length']:raise ValueError('candidate work differs')
        for name,fee,rate in (('net_match_low',1,1e-5),('net_match_high',4,1e-4)):
            value=row['expected_match']-(fee*row['expanded']+rate*row['likelihood_evaluations'])/unit['length']
            if abs(value-row[name])>1e-12:raise ValueError('priced score differs')
    return True


def controls():
    p=np.array([[.5,.5]]);scores=np.zeros(1)
    posterior,_=update(p,scores,np.array([[0.,1.]]))
    uniform,evidence=update(p,scores,np.array([[.25,.25]]))
    return {'live:binary_recovery':bool(np.array_equal(posterior,[[0.,1.]])),
            'placebo:uniform_observation':bool(np.array_equal(uniform,p) and np.allclose(evidence,[math.log(.25)]))}
