"""Independent board replay and direct likelihood enumeration; no maker imports."""
import itertools
import math
import numpy as np


def replay(program):
    occupied=set()
    if len(program)>3: raise ValueError('overlong board program')
    for action in program:
        if type(action) is not int or not 0<=action<8: raise ValueError('bad primitive')
        if action<4: occupied.add(action)
        else: occupied.discard(action-4)
    return sum(2**a for a in occupied)


def reference(w, state, ctx):
    programs=[p for n in range(4) for p in itertools.permutations(range(4),n)]
    skill,pref,goal,belief=state
    motif=None if skill==0 else tuple(w['groups'][skill-1])
    costs=[]
    for program in programs:
        encodings=[len(program)]
        if motif:
            for i in range(len(program)-1):
                if program[i:i+2]==motif: encodings.append(len(program)-1)
        costs.append(min(encodings))
    result=np.zeros(len(programs))
    possible=[(belief,1.)] if ctx['signal'] is None else [(ctx['signal'],ctx['noticed']),(belief,1-ctx['noticed'])]
    for b,mass in possible:
        task=goal if ctx['goal'] is None else ctx['goal']
        target=set(range(4)) if task==2 else set(w['groups'][task^b]); values=[]
        for program,cost in zip(programs,costs):
            made=set(program)
            contrast=len(made.intersection(w['groups'][1]))-len(made.intersection(w['groups'][0]))
            u=-1.6*len(made.symmetric_difference(target))+.65*(pref-1)*contrast-w['price']*cost
            allowed=cost<=w['max_code'] and not (w['family']=='restricted' and 3 in program)
            values.append(math.exp(u/w['temperature']) if allowed else 0.)
        result+=mass*np.array(values)/sum(values)
    return programs,result


def posterior(public):
    states=list(itertools.product(range(3),range(3),range(2),range(2)))
    joint=np.ones(len(states))/len(states)
    for obs in public['history']:
        for i,s in enumerate(states):
            programs,probs=reference(public['world'],s,obs['context'])
            indices=[j for j,p in enumerate(programs) if list(p)==obs['program']] if obs['program'] is not None else [j for j,p in enumerate(programs) if replay(p)==obs['artifact']]
            joint[i]*=sum(probs[j] for j in indices)
    if joint.sum()==0: return None
    return joint/joint.sum()


def check_case(case):
    errors=[]; executions=0
    for obs in case['history']+[p['observed'] for p in case['probes']]:
        if replay(obs['program'])!=obs['artifact']: errors.append('execution')
        executions+=1
    return dict(passed=not errors, executions=executions, errors=errors)


def interval(values):
    # Descriptive paired-unit normal interval; each entry is a lineage mean.
    a=np.array(values,float)
    mean=float(a.mean())
    half=1.96*float(a.std(ddof=1))/math.sqrt(len(a)) if len(a)>1 else 0.
    return dict(mean=mean,lower=mean-half,upper=mean+half,clusters=len(a),
                method='descriptive normal interval over independent lineage means')
