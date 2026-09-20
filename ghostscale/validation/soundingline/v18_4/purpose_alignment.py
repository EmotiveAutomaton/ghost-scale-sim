"""Matched supplemental targets with changed history alignment; no future selection."""
import numpy as np
from ..v18_3 import world as W, compression as C
from . import compression as P

DIMENSIONS=('family','cell','rule')
METRICS=('old_loss','new_loss','entropy_nats','old_optimum_ties','training_loss')


def portfolios(training):
    """Keep the passive task fixed and rotate both supplemental vectors together."""
    training=[np.asarray(x,float) for x in training]
    if len(training)!=3 or any(x.shape!=training[0].shape for x in training):
        raise ValueError('three matched target banks required')
    if training[0].shape[0]!=8 or any(np.any(x<0) or not np.allclose(x.sum(1),1,atol=1e-12,rtol=0) for x in training):
        raise ValueError('eight normalized target rows required')
    banks={'aligned':training}
    for shift in range(1,8):
        banks[f'shift{shift}']=[training[0]]+[np.roll(x,shift,axis=0) for x in training[1:]]
    banks['marginal']=[training[0]]+[np.repeat(x.mean(0,keepdims=True),8,axis=0) for x in training[1:]]
    return banks


def select(histories,training,index):
    ph=np.ones(8)/8;rows=[]
    for name,bank in portfolios(training).items():
        choices=P.select(histories,ph,bank,index)
        for row in choices:
            if row['method']=='purpose-portfolio':
                rows.append(dict(row,method='portfolio-'+name,portfolio=name,
                                 training_loss=float(np.mean(row['training_losses']))))
            elif name=='aligned' and not row['method'].startswith('purpose-'):
                rows.append(dict(row,portfolio='old-only',training_loss=row['training_losses'][0]))
    return rows


def unit(index,cell=0,rule=None,split='test'):
    w=W.make_world(cell,18042000+index)
    if rule is not None:
        if rule!='lexicographic':raise ValueError('undeclared decision rule')
        w['rule']=rule
    histories,natural,post,_,_=C.finite_histories(w)
    states=[i for i,s in enumerate(W.STATES) if s[0]>0 and s[3]==0]
    training=[post@W.artifact_matrix(w,c)[states] for c in P.TRAIN]
    selected=select(histories,training,index)
    # All targets and codebooks are frozen before evaluator-only future losses.
    future=[post@W.artifact_matrix(w,c)[states] for c in P.FUTURE]
    ph=np.ones(8)/8;_,masks,_=C.codebooks()
    losses=np.array([C.code_losses(masks,ph,p)[0] for p in future]).T
    old=C.code_losses(masks,ph,training[0])[0];rows=[]
    for row in selected:
        i=row['code_id']
        rows.append(dict(row,old_loss=float(old[i]),new_loss=float(losses[i].mean()),
                         future_losses=losses[i].tolist()))
    result=dict(family='P2',index=index,cell=cell,rule=rule,world=w,histories=histories,
                natural_history_probabilities=natural.tolist(),history_probabilities=ph.tolist(),
                training_predictions=[x.tolist() for x in training],
                future_predictions=[x.tolist() for x in future],rows=rows,
                scope='uniform-history allocation; exact supplied-law decoding; paired alignment controls')
    return result


def verify(data):
    ph=np.asarray(data['history_probabilities'])
    if not np.array_equal(ph,np.ones(8)/8):raise ValueError('uniform allocation changed')
    training=[np.asarray(x) for x in data['training_predictions']]
    expected=select(data['histories'],training,data['index'])
    if len(expected)!=len(data['rows']):raise ValueError('selector roster changed')
    banks=portfolios(training)
    for selected,row in zip(expected,data['rows']):
        if any(row[k]!=v for k,v in selected.items()):raise ValueError('selected code or training record changed')
        bank=training if row['portfolio']=='old-only' else banks[row['portfolio']]
        actual=[C.reference_loss(row['code'],ph,p) for p in bank]
        if not np.allclose(actual,row['training_losses'],atol=1e-12,rtol=0):raise ValueError('training score differs')
        old=C.reference_loss(row['code'],ph,training[0])
        future=[C.reference_loss(row['code'],ph,np.asarray(p)) for p in data['future_predictions']]
        if not np.allclose(future,row['future_losses'],atol=1e-12,rtol=0):raise ValueError('future score differs')
        if abs(old-row['old_loss'])>1e-12 or abs(np.mean(future)-row['new_loss'])>1e-12:raise ValueError('mean loss differs')
    return True
