"""Independent finite executor, probability reference, and retained-row checks."""
from itertools import product
import math
import numpy as np
from . import world as W


def execute(program):
    cells=set()
    for action in program:
        if type(action) is not int or not 0<=action<4:raise ValueError('invalid placement')
        cells.add(action)
    return sum(2**x for x in cells)


def code_cost(program,library):
    tokens=[(i,) for i in range(4)]+list(library)
    for length in range(len(program)+1):
        for seq in product(tokens,repeat=length):
            if tuple(x for part in seq for x in part)==tuple(program):return length
    raise ValueError('unencodable program')


def reference_policy(w,state,c):
    skill,goal,belief,tradeoff=W.STATES[state]
    # Independent acquisition expectation for the explicit repeated curricula.
    lib=[] if skill==0 else [tuple(w['groups'][(2-skill) if w['coupled'] else (skill-1)])]
    g=goal if c['goal'] is None else c['goal'];b=belief if c['signal'] is None else c['signal']
    target=set(w['groups'][g^b]);allowed=[];values=[];errors=[];costs=[];imbalances=[]
    for p in W.PROGRAMS:
        cost=code_cost(p,lib);costs.append(cost)
        good=cost<=c['budget'] and (c['offered'] is None or set(p)<=set(c['offered']))
        if w['endogenous']:good=good and w['groups'][1-g][tradeoff] not in p
        allowed.append(good);error=len(set(p)^target);errors.append(error)
        imbalance=len(set(p)&set(w['groups'][1]))-len(set(p)&set(w['groups'][0]));imbalances.append(imbalance)
        values.append(-1.6*error+.5*(2*tradeoff-1)*imbalance-w['price']*c['price_scale']*cost)
    if c['uninformative']:base=[float(i==0) for i in range(len(W.PROGRAMS))]
    elif w['rule']=='softmax':
        maximum=max(v for v,a in zip(values,allowed) if a)
        base=[math.exp((v-maximum)/w['temperature']) if a else 0. for v,a in zip(values,allowed)]
        total=sum(base);base=[x/total for x in base]
    else:
        legal=[i for i,a in enumerate(allowed) if a]
        acceptable=[i for i in legal if errors[i]<=1]
        if w['rule']=='satisficing':chosen=acceptable[0] if acceptable else max(legal,key=lambda j:values[j])
        else:chosen=min(legal,key=lambda j:(errors[j],costs[j],-imbalances[j]*(2*tradeoff-1),j))
        base=[float(i==chosen) for i in range(len(W.PROGRAMS))]
    physical=[c['offered'] is None or set(p)<=set(c['offered']) for p in W.PROGRAMS]
    return np.array([(1-w['noise'])*p+w['noise']*int(a)/sum(physical) for p,a in zip(base,physical)])


def distribution(p):
    p=np.asarray(p,float)
    if np.any(~np.isfinite(p)) or np.any(p<0) or not np.isclose(p.sum(),1.,atol=1e-10,rtol=0):
        raise ValueError('invalid probability distribution')


def check_unit(unit,reference=False):
    executions=0;distributions=0
    if unit['family'] in ('A','B','D'):
        public=unit['public'];W.parse(W.canonical(public))
        for obs in public['history']:
            if execute(obs['program'])!=obs['artifact']:raise ValueError('independent board mismatch')
            executions+=1
        if reference:
            w=public['world']
            for state in (0,7,16,23):
                for c in (W.context(),W.context(goal=1,budget=1),W.context(signal=0,offered=[0,2,3])):
                    if not np.allclose(W.matrix(w,c)[state],reference_policy(w,state,c),atol=1e-12,rtol=0):
                        raise ValueError('independent policy mismatch')
    if unit['family']=='A':
        for row in unit['rows']:
            distribution(row['posterior']);distributions+=1
            if row['same_evidence_direct_error']>1e-12:raise ValueError('unequal evidence reader')
            e=row['enactment'];target=unit['evaluator']['target']
            if execute(e['program'])!=e['artifact'] or (e['artifact']==target)!=e['success']:
                raise ValueError('enactment mismatch')
            if code_cost(tuple(e['program']),list(map(tuple,e['library'])))!=e['code_cost']:
                raise ValueError('encoding mismatch')
            if row['costs']['attempted_queries']!=len(row['attempts']):raise ValueError('lost failed access')
            executions+=1
    elif unit['family']=='B':
        for row in unit['rows']:
            for point in row['trace']:distribution(point['posterior']);distributions+=1
            e=row['shared_enactment']
            if (execute(e['program'])==e['target'])!=e['success']:raise ValueError('shared enactment mismatch')
            executions+=1
    elif unit['family']=='C':
        for row in unit['rows']:
            if row['instrument']!='valid':continue
            distribution(row['posterior']);distributions+=1
            truth=unit['evaluator']['truth']
            if abs(row['initial_loss']+math.log(row['posterior'][truth]))>1e-12:raise ValueError('source loss mismatch')
            for p in row['after_correction']:distribution(p);distributions+=1
    elif unit['family']=='D':
        for row in unit['rows']:
            for f in row['forecasts']:
                distribution(f['probabilities']);distribution(f['truth']);distributions+=2
                terms=[-q*math.log(p) for p,q in zip(f['probabilities'],f['truth']) if q>0]
                if abs(sum(terms)-f['expected_loss']['value'])>1e-10:raise ValueError('held-future loss mismatch')
    return dict(executions=executions,distributions=distributions,reference=reference)


def metrics(unit):
    """Unit averages precede group intervals; no probe or fit-seed pseudoreplication."""
    results=[]
    for row in unit['rows']:
        tags={k:unit[k] for k in ('family','cell','mode','control','condition','roots','copies','kind','order') if k in unit}
        tags['method']=row['method']
        if unit['family']=='A':
            tags['purpose']=row['purpose'];s=row['scores'];c=row['costs']
            values=dict(expected_loss=s['expected_loss']['value'],infinite_loss=s['expected_loss']['infinite'],
                brier=s['brier'],state_mass=s['state_mass'],state_entropy=s['entropy'],
                enactment=float(row['enactment']['success']),**c)
        elif unit['family']=='B':
            trace=row['trace'];change=unit['evaluator']['change']
            values=dict(expected_loss=float(np.mean([x['expected_loss']['value'] for x in trace])),
                post_change_loss=float(np.mean([x['expected_loss']['value'] for x in trace[change:]])),
                retained_skill_mass=float(np.mean([x['true_skill_mass'] for x in trace[change:]])),
                false_resets=row['false_resets'],detected=float(row['detection_delay'] is not None),
                enactment=float(row['shared_enactment']['success']))
        elif unit['family']=='C':
            values={'valid':float(row['instrument']=='valid')}
            if row['instrument']=='valid':values.update({k:float(row[k]) for k in ('initial_loss','final_loss','initial_brier','final_brier','false_confidence','corrected_false_confidence','credible_coverage')})
        else:
            values=dict(expected_loss=float(np.mean([f['expected_loss']['value'] for f in row['forecasts']])),
                abstained=float(np.mean([f['abstained'] for f in row['forecasts']])),candidate_evaluations=row['candidate_evaluations'])
        results.append((tags,values))
    return results
