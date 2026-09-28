"""Independent scalar transition/probability and proper-score checker.

Does not import the producer's executor, projection, posterior or scoring code.
"""
from fractions import Fraction as F
import math
import numpy as np

def independent_record(row,w,maker,context):
    goal,skill,pref,review=maker;initial,request,available=context
    _,_,route,proposal,selected,inspection,revised,order,precision,*_=map(int,row[:-1])
    claim=proposal if selected else initial
    replacement=(goal%2) if precision else initial
    if order==0:
        if revised:claim=replacement
        evidence=claim
    else:
        evidence=claim
        if revised:claim=replacement
    if w['shift']=='changed-tool':evidence=1-evidence
    presentation=goal//2 if revised else 0
    if revised and w['shift']=='presentation-shift':presentation=1-presentation
    p=F(w['goal'] if goal%2==request else 10-w['goal'],160)*F(1,8)*F(1,4)
    if available:p*=F(w['route'] if route==pref else 10-w['route'],10)
    elif route:raise ValueError('tool route without availability')
    ps=[w['select'] if proposal==goal%2 else 10-w['select'],w['revise'] if selected else 10-w['revise'],
        w['inspect'] if review else 10-w['inspect'],w['skill'] if skill else 10-w['skill']]
    for bit,v in zip([selected,revised,inspection,precision],ps):p*=F(v if bit else 10-v,10)
    label=int(''.join(map(str,[route,selected,revised,order,inspection,goal%2,goal//2])),2)
    return (claim,evidence,presentation),label,p

def verify_world(rows,w,makers,contexts):
    total=F(0);weights=[]
    for row in rows:
        endpoint,y,p=independent_record(row,w,makers[int(row[0])],contexts[int(row[1])])
        if tuple(row[11:14].astype(int))!=endpoint or int(row[10])!=y:raise ValueError('execution or label disagreement')
        weights.append(p);total+=p
    expected=np.array([float(p/total) for p in weights])
    if not np.allclose(expected,rows[:,-1],atol=2e-16,rtol=1e-12):raise ValueError('policy disagreement')
    return {'trajectories':len(rows),'exact_mass_before_normalization':str(total),'independent_transitions':len(rows)}

def probabilities(p):
    p=np.asarray(p,float)
    if p.ndim!=2 or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-10):raise ValueError('invalid forecast')
    return p

def verify_posterior(rows,evidence,tier,contexts,forecast,observed=False):
    """Separate scalar grouping and sequentially normalized Bayes calculation."""
    grouped={};maker_total=np.zeros(32)
    for row in rows:
        key=list(map(int,row[11:14]))+[-1]*10
        if tier!='artifact':key[3:6]=contexts[int(row[1])]
        if tier in ('sparse','complete'):key[6]=int(row[2])
        if tier=='complete':
            key[7:10]=[int(row[4]),int(row[6]),int(row[7])]
            if observed:key[10]=int(row[5])
        key=tuple(key)
        if key not in grouped:grouped[key]=np.zeros((32,128))
        mi=int(row[0]);grouped[key][mi,int(row[10])]+=row[-1];maker_total[mi]+=row[-1]
    likelihood={k:a.sum(1)/maker_total for k,a in grouped.items()}
    for i,x in enumerate(evidence):
        belief=np.full(32,1/32)
        for start in range(13,len(x),13):
            belief*=likelihood[tuple(x[start:start+13])];belief/=belief.sum()
        expected=np.sum(grouped[tuple(x[:13])]*belief[:,None],axis=0);expected/=expected.sum()
        if not np.allclose(expected,forecast[i],atol=2e-12,rtol=2e-10):raise ValueError('independent posterior differs')
    return True

def reaggregate(predictions,truth,stored):
    p=probabilities(predictions);y=np.asarray(truth,int)
    losses=np.array([-math.log(max(1e-12,float(row[t]))) for row,t in zip(p,y)])
    brier=np.array([math.fsum(float(v)**2 for v in row)-2*float(row[t])+1 for row,t in zip(p,y)])
    correct=np.array([int(int(np.argmax(row))==t) for row,t in zip(p,y)])
    for name,arr in [('log_loss',losses),('brier',brier),('correct',correct)]:
        if not np.allclose(arr,stored[name],rtol=2e-10,atol=2e-10):raise ValueError('corrupt score '+name)
    # Build bit targets independently of the producer's LABELS and scoring code.
    bits=np.array([[(label>>shift)&1 for shift in range(6,-1,-1)] for label in range(p.shape[1])])
    marginal=p@bits;target=bits[y];confidence=p.max(1)
    extras=dict(tool_brier=(marginal[:,0]-target[:,0])**2,goal_brier=((marginal[:,-2:]-target[:,-2:])**2).mean(1),
        order_brier=(marginal[:,3]-target[:,3])**2,inspection_brier=(marginal[:,4]-target[:,4])**2,
        unsupported_attribution=((confidence>=.9)&(correct==0)).astype(float),confidence=confidence,
        zero_truth_probability=(p[np.arange(len(y)),y]==0).astype(float))
    for name,arr in extras.items():
        if name in stored and not np.allclose(arr,stored[name],rtol=2e-10,atol=2e-10):raise ValueError('corrupt secondary score '+name)
    return {'rows':len(y),'log_loss':float(losses.mean()),'brier':float(brier.mean()),'accuracy':float(correct.mean())}

def exact_witnesses():
    """Symbolic finite examples, independent of floating-point collision screens."""
    joint0={ (0,0):F(1,2),(1,1):F(1,2)}
    joint1={ (0,1):F(1,2),(1,0):F(1,2)}
    def marginal(d,i):return [sum(p for k,p in d.items() if k[i]==v) for v in [0,1]]
    assert all(marginal(joint0,i)==marginal(joint1,i) for i in [0,1])
    # Inspection is recorded but is not artifact content. Both routes can retain
    # the same proposal. Order is still known from a ledger after unchanged edits.
    return dict(same_marginals_different_joints=True,factorized_impossible_mass='1/2',
        cooccurring_goal_sets={'none':'1/4','A':'1/4','B':'1/4','A+B':'1/4'},
        maker_bank_order_witness='Order has uniform independent policy and unchanged edits; exact maker likelihoods are equal for every maker, while complete event orders differ.',
        inspection_witness='Selected unchanged proposal has the same endpoint whether inspection occurred or not; recorded scope resolves it only when present.',
        history_inert_control='Under independent fresh episodes, prior order is independent of next order; a ledger cannot improve its true predictive distribution.')
