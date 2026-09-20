"""Diagnose retained purpose codes without reselecting them from future answers."""
import math
import numpy as np
from ..v18_3.compression import reference_loss

DIMENSIONS=('family','cell','rule')
METRICS=('same_partition','pair_disagreement','same_forecasts','forecast_max_difference',
         'aligned_loss','comparison_loss','loss_difference')
TOL=1e-12


def canonical(code):
    labels={}
    return tuple(labels.setdefault(x,len(labels)) for x in code)


def decode(code,targets):
    code=np.asarray(code);targets=np.asarray(targets,float)
    if code.shape!=(8,) or targets.ndim!=2 or targets.shape[0]!=8:
        raise ValueError('eight histories required')
    if not np.isfinite(targets).all() or np.any(targets<0) or not np.allclose(targets.sum(1),1,atol=TOL,rtol=0):
        raise ValueError('normalized target probabilities required')
    return np.stack([targets[code==label].mean(0) for label in code])


def compare(aligned,other,targets):
    a=canonical(aligned);b=canonical(other)
    disagreements=sum((a[i]==a[j])!=(b[i]==b[j]) for i in range(8) for j in range(i))
    pa=[decode(a,t) for t in targets];pb=[decode(b,t) for t in targets]
    maxdiff=max(float(np.max(abs(x-y))) for x,y in zip(pa,pb))
    ph=np.ones(8)/8
    la=math.fsum(reference_loss(a,ph,np.asarray(t)) for t in targets)/len(targets)
    lb=math.fsum(reference_loss(b,ph,np.asarray(t)) for t in targets)/len(targets)
    return dict(same_partition=int(a==b),pair_disagreement=disagreements/28,
                same_forecasts=int(maxdiff<=TOL),forecast_max_difference=maxdiff,
                aligned_loss=la,comparison_loss=lb,loss_difference=lb-la)


def unit(index,cell,rule,payload,split='test'):
    if payload['index']!=index or payload['cell']!=cell or payload['rule']!=rule:
        raise ValueError('parent identity differs')
    if payload['history_probabilities']!=[.125]*8:raise ValueError('uniform allocation required')
    selected=payload['rows'];targets=payload['future_predictions'];rows=[]
    for k in (1,2,4,8):
        a=next(r for r in selected if r['cardinality']==k and r['method']=='portfolio-aligned')
        for method in [f'portfolio-shift{i}' for i in range(1,8)]+['portfolio-marginal']:
            b=next(r for r in selected if r['cardinality']==k and r['method']==method)
            metrics=compare(a['code'],b['code'],targets)
            if abs(metrics['aligned_loss']-a['new_loss'])>TOL or abs(metrics['comparison_loss']-b['new_loss'])>TOL:
                raise ValueError('parent number not reproduced')
            rows.append(dict(method=method,cardinality=k,aligned_code=a['code'],comparison_code=b['code'],**metrics))
    invariant=all(not r['same_partition'] or (r['same_forecasts'] and abs(r['loss_difference'])<=TOL) for r in rows)
    return dict(family='P3',index=index,cell=cell,rule=rule,payload=payload,rows=rows,
                gates=dict(live_parent_scores_reconstructed=True,positive_partition_implies_prediction=invariant),
                scope='retained paired purpose codes; no new independent worlds or selected codes')


def verify(data):
    targets=data['payload']['future_predictions']
    expected=unit(data['index'],data['cell'],data['rule'],data['payload'])
    if expected['rows']!=data['rows']:raise ValueError('retained diagnostic differs')
    if not all(data['gates'].values()):raise ValueError('failed diagnostic gate')
    # Independent membership sets and scalar decoder, independent of canonical/decode.
    for row in data['rows']:
        a=row['aligned_code'];b=row['comparison_code']
        partitions=[{frozenset(i for i,x in enumerate(code) if x==label) for label in code} for code in (a,b)]
        if int(partitions[0]==partitions[1])!=row['same_partition']:raise ValueError('partition identity differs')
        scalar=[];scores=[]
        for code in (a,b):
            forecasts=[];losses=[]
            for target in targets:
                pred=[[math.fsum(target[j][out] for j in range(8) if code[j]==code[i])/sum(code[j]==code[i] for j in range(8))
                       for out in range(len(target[0]))] for i in range(8)]
                forecasts.extend(x for r in pred for x in r)
                losses.append(-math.fsum(target[i][out]*math.log(max(pred[i][out],1e-300))
                    for i in range(8) for out in range(len(target[0])))/8)
            scalar.append(forecasts);scores.append(math.fsum(losses)/len(losses))
        diff=max(abs(x-y) for x,y in zip(*scalar))
        if abs(diff-row['forecast_max_difference'])>TOL:raise ValueError('scalar forecasts differ')
        if max(abs(scores[0]-row['aligned_loss']),abs(scores[1]-row['comparison_loss']))>TOL:
            raise ValueError('scalar proper score differs')
        if row['same_partition'] and (diff>TOL or abs(row['loss_difference'])>TOL):
            raise ValueError('partition invariance failed')
    return True
