"""Purpose-sensitive paid inquiry with explicit response/access uncertainty.

NULL: no-information menus yield zero expected information; all rivals see the
same realized attempts. ALTERNATIVE: disjoint diagnostic queries change ranking
with target. The scientific outcome is held-out behavior/native enactment.
"""
import numpy as np
from . import world as W

METHODS = ('task', 'information', 'fixed', 'random', 'none', 'task-naive', 'task-robust',
           'task-three-attempts', 'information-three-attempts')
PURPOSES = ('prediction', 'historical-state', 'enactment')
ACCESS = ('uniform', 'query-dependent', 'hidden-state')
ATTEMPTS = 3
FEES = np.array([.015,.025,.025,.03,.03,.04,.04,.05])
WORK_PRICE = 1e-8


def access(q, profile, state=None, hidden=False):
    rates = (.8, (.95 if q < 3 else .25), (.25 if q < 3 else .95))
    value = rates[profile]
    if hidden:
        value *= .2 if W.STATES[state][1] != q % 2 else 1.
    return float(value)


def bank(w, naive=False, uninformative=False):
    tables=[]
    for j, ctx in enumerate(W.QUERIES):
        ctx = dict(ctx, uninformative=True) if uninformative else ctx
        base = W.matrix(w, ctx)
        # Flatten state-major, profile-minor; access is a latent nuisance role.
        table=np.zeros((len(W.STATES)*3,len(W.PROGRAMS)+1))
        for s in range(len(W.STATES)):
            for a in range(3):
                rate=1. if naive else access(j,a)
                table[s*3+a,:-1]=rate*base[s]
                table[s*3+a,-1]=1-rate
        tables.append(table)
    return tables


def marginal(weights):
    return weights.reshape(len(W.STATES),3).sum(axis=1)


def update(weights, table, outcome, ignore_missing=False):
    if ignore_missing and outcome==len(W.PROGRAMS):
        return weights.copy()
    result=weights*table[:,outcome]
    if result.sum()<=0:
        raise ValueError('observation outside declared response support')
    return result/result.sum()


def expected_gains(weights, tables, w, purpose, target, current_success=0.):
    # Future outcome bank includes contexts never purchased as the identical query.
    future=np.concatenate([W.artifact_matrix(w,c) for c in W.FUTURES],axis=1)
    future=np.repeat(future,3,axis=0)
    initial=marginal(weights)
    before=np.mean([W.entropy(initial @ W.artifact_matrix(w,c)) for c in W.FUTURES])
    gains=[]
    success=np.array([W.enact(p,target)['success'] for p in W.PROGRAMS]+[False],float)
    for table in tables:
        joint=weights[:,None]*table
        prob=joint.sum(axis=0)
        valid=prob>0
        post=joint[:,valid].T/prob[valid,None]
        if purpose=='enactment':
            value=float(prob@np.maximum(success,current_success))-current_success
        elif purpose=='historical-state':
            states=post.reshape(-1,len(W.STATES),3).sum(axis=2)
            value=W.entropy(initial)-sum(float(p)*W.entropy(s) for p,s in zip(prob[valid],states))
        else:
            predictions=(post@future).reshape(-1,len(W.FUTURES),16)
            value=before-sum(float(p)*np.mean([W.entropy(v) for v in ps]) for p,ps in zip(prob[valid],predictions))
        gains.append(float(value))
    return np.array(gains)


def reconstruct(initial, tables, attempts, naive=False):
    weights=initial.copy()
    for item in attempts:
        weights=update(weights,tables[item['query']],item['outcome'],naive)
    return weights


def unit(index, cell=0, mode='uniform', control='ordinary', split='test',rule=None):
    random=W.rng('active',split,index,mode,control)
    w=W.make_world(cell,index)
    if rule is not None:
        if rule!='lexicographic':raise ValueError('undeclared holdout rule')
        w['rule']=rule
    state=int(random.integers(len(W.STATES)))
    history=W.initial_history(w,state,random)
    prior=W.posterior(W.packet(w,history))
    if control=='resolved':
        # Explicit privileged control, excluded from ordinary comparisons.
        prior=np.eye(len(W.STATES))[state]
    target=sum(1<<x for x in (w['groups'][index%2]+w['groups'][1-index%2][:1]))
    profile=0 if mode=='uniform' else index%2+1
    true_outcomes=[]
    true_contexts=[]
    for step in range(ATTEMPTS):
        step_out=[]
        for q,c in enumerate(W.QUERIES):
            c=dict(c,uninformative=True) if control=='uninformative' else c
            # Shared potential outcomes by attempted-query step and query identity.
            r=W.rng('potential',split,index,mode,control,step,q)
            if r.random()>=access(q,profile,state,mode=='hidden-state'):
                outcome=len(W.PROGRAMS)
            else:
                outcome=int(r.choice(len(W.PROGRAMS),p=W.matrix(w,c)[state]))
            step_out.append(outcome)
        true_outcomes.append(step_out)
    rows=[]
    for purpose in PURPOSES:
        for method in METHODS:
            naive=method=='task-naive'
            tables=bank(w,naive,control=='uninformative')
            initial=np.repeat(prior/3,3);weights=initial.copy()
            attempts=[];selected_program=();selector_work=0;fees=0.;primitive_work=0;learning_work=0
            for step in range(ATTEMPTS):
                if method=='none':break
                if method=='fixed':q=3+step%2
                elif method=='random':q=int(W.rng('random-selector',split,index,mode,control,step).integers(len(tables)))
                else:
                    objective='historical-state' if method.startswith('information') else purpose
                    gains=expected_gains(weights,tables,w,objective,target,float(W.enact(selected_program,target)['success']))
                    work=len(tables)*len(weights)*(len(W.PROGRAMS)+1)
                    decision_work=work
                    if method=='task-robust':
                        variants=[]
                        for a in range(3):
                            perturbed=weights.reshape(-1,3).copy()
                            perturbed[:,a]*=3;perturbed=perturbed.ravel();perturbed/=perturbed.sum()
                            variants.append(expected_gains(perturbed,tables,w,objective,target,float(W.enact(selected_program,target)['success'])))
                        gains=np.min(variants,axis=0);decision_work+=3*work
                    selector_work+=decision_work
                    # Decision fee and expected native observation work are explicit.
                    lengths=np.array([len(p) for p in W.PROGRAMS]+[0])
                    expected_actions=np.array([float((weights@t)@lengths) for t in tables])
                    if objective=='enactment':
                        learning=np.array([sum(W.enact(p,target)[k] for k in ('practice_actions','definition_actions','planning_evaluations')) for p in W.PROGRAMS]+[0])
                        expected_actions+=np.array([float((weights@t)@learning) for t in tables])
                    values=gains-FEES-.002*expected_actions-decision_work*WORK_PRICE
                    q=int(np.argmax(values))
                    if values[q]<=0 and not method.endswith('three-attempts'):break
                outcome=true_outcomes[step][q]
                weights=update(weights,tables[q],outcome,naive)
                fees+=float(FEES[q]);primitive_work+=1 # apparatus provisioning per attempted observation
                if outcome<len(W.PROGRAMS):
                    demonstration=W.PROGRAMS[outcome];primitive_work+=len(demonstration)
                    candidate=W.enact(demonstration,target)
                    learning_work+=sum(candidate[k] for k in ('practice_actions','definition_actions','planning_evaluations'))
                    if candidate['success'] and not W.enact(selected_program,target)['success']:
                        selected_program=demonstration
                attempts.append(dict(query=q,outcome=outcome))
            direct=reconstruct(initial,tables,attempts,naive)
            if not np.allclose(weights,direct,atol=1e-13,rtol=0):raise ValueError('same-evidence direct mismatch')
            acquisition=W.enact(selected_program,target)
            score=W.scoring(w,marginal(weights),state)
            rows.append(dict(purpose=purpose,method=method,attempts=attempts,posterior=marginal(weights).tolist(),
                scores=score,enactment=acquisition,same_evidence_direct_error=float(np.max(abs(weights-direct))),
                costs=dict(attempted_queries=len(attempts),responses=sum(a['outcome']<len(W.PROGRAMS) for a in attempts),
                           query_fees=fees,selector_operations=selector_work,observation_actions=primitive_work,
                           reader_work=learning_work+acquisition['execution_actions'])))
    return dict(family='A',index=index,cell=cell,mode=mode,control=control,**({'rule':rule} if rule else {}),public=json_payload(w,history),
                evaluator=dict(state=state,profile=profile,target=target,potential_outcomes=true_outcomes),rows=rows,
                work_unit='selector likelihood-table entries considered; a declared logical cost proxy, not hardware instructions')


def json_payload(w,history):
    import json
    return json.loads(W.packet(w,history))
