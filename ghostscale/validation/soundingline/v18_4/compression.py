"""Prospective selectors: held-out questions never enter code selection."""
from itertools import combinations
import numpy as np
from ..v18_3 import world as W, compression as C

TOL=1e-10
TRAIN=(W.context(),W.context(goal=0),W.context(signal=0))
FUTURE=(W.context(goal=1),W.context(signal=1),W.context(goal=0,signal=1),
        W.context(goal=1,signal=0),W.context(goal=1,signal=1,budget=3,price_scale=1.7))


def select(histories,ph,training,index):
    codes,masks,cardinality=C.codebooks()
    losses=np.array([C.code_losses(masks,ph,p)[0] for p in training]).T
    _,entropy=C.code_losses(masks,ph,training[0])
    full=losses[np.flatnonzero(cardinality==8)[0]]
    rows=[]
    for k in (1,2,4,8):
        candidates=np.flatnonzero(cardinality==k)
        tied=candidates[losses[candidates,0]<=losses[candidates,0].min()+TOL]
        product_ids=[]
        for axes in combinations(range(3),int(np.log2(k))):
            labels=[sum(h[a]<<j for j,a in enumerate(axes)) for h in histories]
            lookup={};labels=[lookup.setdefault(v,len(lookup)) for v in labels]
            product_ids.append(int(np.flatnonzero(np.all(codes==labels,axis=1))[0]))
        choices={
            'old-canonical':int(tied[0]),
            'old-random-tie':int(W.rng('v18.4-prospective-tie',index,k).choice(tied)),
            'old-max-entropy':int(tied[np.argmax(entropy[tied])]),
            'observation-product':min(product_ids,key=lambda i:(losses[i,0],i)),
            'purpose-portfolio':int(candidates[np.argmin(losses[candidates].mean(1))]),
            'purpose-minimax-regret':int(candidates[np.argmin((losses[candidates]-full).max(1))])}
        for method,code_id in choices.items():
            rows.append(dict(method=method,cardinality=k,code_id=code_id,code=codes[code_id].tolist(),
                old_optimum_ties=len(tied),training_losses=losses[code_id].tolist(),
                storage_bits=float(np.log2(k)),entropy_nats=float(entropy[code_id]),
                training_purpose_access=3 if method.startswith('purpose-') else 1))
    return rows


def unit(index,cell=0,rule=None,tilt=0.,split='test'):
    w=W.make_world(cell,18040000+index)
    if rule is not None:
        if rule!='lexicographic':raise ValueError('undeclared decision rule')
        w['rule']=rule
    histories,ph,post,_,_=C.finite_histories(w)
    states=[i for i,s in enumerate(W.STATES) if s[0]>0 and s[3]==0]
    training=[post@W.artifact_matrix(w,c)[states] for c in TRAIN]
    selected=select(histories,ph,training,index)
    future=[post@W.artifact_matrix(w,c)[states] for c in FUTURE]
    deployment=ph*np.exp(tilt*(np.sum(histories,axis=1)-1.5));deployment/=deployment.sum()
    codes,masks,_=C.codebooks()
    future_losses=np.array([C.code_losses(masks,deployment,p)[0] for p in future]).T
    old_losses=C.code_losses(masks,deployment,training[0])[0]
    rows=[]
    for row in selected:
        i=row['code_id']
        scalar=[C.reference_loss(row['code'],deployment,p) for p in future]
        if not np.allclose(scalar,future_losses[i],atol=1e-12,rtol=0):raise ValueError('independent compression score differs')
        rows.append(dict(row,old_loss=float(old_losses[i]),new_loss=float(np.mean(future_losses[i])),
                         future_losses=future_losses[i].tolist()))
    return dict(family='P',index=index,cell=cell,rule=rule,tilt=tilt,world=w,histories=histories,
        history_probabilities=ph.tolist(),deployment_probabilities=deployment.tolist(),
        training_predictions=[x.tolist() for x in training],future_predictions=[x.tolist() for x in future],rows=rows)


def verify(unit):
    ph=np.asarray(unit['history_probabilities']);deployment=np.asarray(unit['deployment_probabilities'])
    if abs(ph.sum()-1)>1e-12 or abs(deployment.sum()-1)>1e-12:raise ValueError('invalid history law')
    selected=select(unit['histories'],ph,[np.asarray(x) for x in unit['training_predictions']],unit['index'])
    if [x['code'] for x in selected]!=[x['code'] for x in unit['rows']]:raise ValueError('selector depends on undeclared data')
    for row in unit['rows']:
        scores=[C.reference_loss(row['code'],deployment,np.asarray(p)) for p in unit['future_predictions']]
        if not np.allclose(scores,row['future_losses'],atol=1e-12,rtol=0):raise ValueError('retained future score mismatch')
    return True
