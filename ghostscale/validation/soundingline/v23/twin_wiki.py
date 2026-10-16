"""T3 streaming multi-maker article revisions with finite planted purposes."""
import numpy as np
from .common import rng,normalize

PURPOSES=('inform','maintain','promote-view')
INTENTS=('copyedit','add','remove','revert','promote')
BASE=np.array([[.25,.55,.1,.05,.05],[.45,.08,.2,.25,.02],[.05,.2,.12,.03,.6]])


def alternatives(state,previous):
    # Four abstract article coordinates: amount, polish, stance, sourced support.
    return np.array([[0,.2,0,0],[1,0,0,1],[-min(1,state[0]),0,0,-min(1,state[3])],
                     np.asarray(previous)-state,[.25,0,1,0]])


def policy(state):
    adjustment=np.array([0,-.03*state[0],.03*state[0],0,-.03*abs(state[2])])
    return normalize(np.log(BASE)+adjustment)


def world(v,phase,index,*,makers=None,records=None):
    """Yield revisions in global chronology, preserving shared article state."""
    s=v['settings'];count=s['makers'] if makers is None else makers
    if not 1<=count<=s['makers']:raise ValueError('invalid maker prefix')
    r=rng('T3-world',v['id'],phase,index);purposes=r.integers(3,size=count)
    histories=np.minimum(s['max_records'],np.maximum(100,np.floor(100*(1+r.pareto(1.5,count))))).astype(int)
    if records is not None:
        if not 1<=records<=s['max_records']:raise ValueError('invalid edit prefix')
        histories=np.full(count,records)
    articles=max(2,(count+4)//5);states=np.zeros((articles,4));previous=states.copy()
    for time in range(int(histories.max())):
        for maker in range(count):
            if time>=histories[maker]:continue
            article=int(r.integers(articles));state=states[article].copy();before=previous[article].copy()
            choices=alternatives(state,before);action=int(r.choice(5,p=policy(state)[purposes[maker]]))
            sd=.5 if s['dirty'] else .25;diff=choices[action]+r.normal(0,sd,4)
            honesty=s['summary_honesty'];cue=int(purposes[maker]) if r.random()<honesty else int(r.choice([g for g in range(3) if g!=purposes[maker]]))
            missing=s['dirty'] and r.random()<.05
            reverted=bool(r.random()<(.7 if action==4 else .1))
            public=dict(maker=maker,article=article,time=time*count+maker,state=state.tolist(),
                previous_state=before.tolist(),alternatives=choices.tolist(),diff=diff.tolist(),
                summary=None if missing else cue,reverted=reverted,record_id=[maker,time],duplicate_of=None)
            truth=dict(purpose=int(purposes[maker]),intent=action,steering=action==4,summary_truthful=cue==purposes[maker])
            previous[article]=state;states[article]=state+diff
            yield public,truth
            if s['dirty'] and r.random()<.02:yield dict(public,record_id=[maker,time,'dup'],duplicate_of=[maker,time]),dict(truth,duplicate=True)


def likelihood(v,row,mode):
    p=policy(np.array(row['state']));sd=.5 if v['settings']['dirty'] else .25
    delta=np.array(row['alternatives'])-row['diff']
    log_density=-.5*np.sum(delta**2,axis=1)/sd**2
    # Remove one shared factor before exponentiation; no state-specific flooring.
    density=np.exp(log_density-log_density.max())
    reversion=np.where(np.arange(5)==4,.7,.1)
    reversion=reversion if row['reverted'] else 1-reversion
    action_evidence=density*reversion if mode!='summary-only' else np.ones(5)
    result=p@action_evidence
    if mode!='diff-only' and row['summary'] is not None:
        honesty=v['settings']['summary_honesty']
        result*=np.where(np.arange(3)==row['summary'],honesty,(1-honesty)/2)
    return result


def evaluate(v,history,maker):
    results={};details={}
    for mode in ('diff-only','summary-only','joint'):
        lp=np.zeros(3);seen=set()
        for row in history:
            if row['maker']!=maker:continue
            key=tuple(row['duplicate_of'] or row['record_id'])
            if key in seen:continue
            seen.add(key);lp+=np.log(likelihood(v,row,mode))
        post=normalize(lp);results[mode]=post[None,:];details[mode]=dict(purpose_posterior=post.tolist(),unique_edits=len(seen))
    return results,details


def verify(v,history,maker,details):
    import math
    for mode in ('diff-only','summary-only','joint'):
        mass=np.ones(3);seen=set()
        for row in history:
            if row['maker']!=maker:continue
            key=tuple(row['duplicate_of'] or row['record_id'])
            if key in seen:continue
            seen.add(key);state=row['state'];adjustment=[0,-.03*state[0],.03*state[0],0,-.03*abs(state[2])]
            sd=.5 if v['settings']['dirty'] else .25
            exponents=[-.5*sum((a-b)**2 for a,b in zip(option,row['diff']))/sd**2 for option in row['alternatives']]
            for goal in range(3):
                weights=[BASE[goal,j]*math.exp(adjustment[j]) for j in range(5)];total=sum(weights)
                q=1. if mode=='summary-only' else sum(weights[j]/total*math.exp(exponents[j]-max(exponents))*
                    ((.7 if j==4 else .1) if row['reverted'] else (.3 if j==4 else .9)) for j in range(5))
                if mode!='diff-only' and row['summary'] is not None:
                    h=v['settings']['summary_honesty'];q*=h if goal==row['summary'] else (1-h)/2
                mass[goal]*=q
            mass/=mass.sum()
        if not np.allclose(mass,details[mode]['purpose_posterior'],atol=1e-11,rtol=0):raise ValueError('article purpose posterior mismatch')
    return dict(independent_scalar_posterior=True)
