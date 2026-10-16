"""T2 constructed game trees and streaming players; no chess-human claim.

The complete minimax tree is solved once. A player's type comprises precision,
style and a habitual root move. Observed clocks alter precision. Hidden mixture
labels distinguish value play, style, opening habit and random blunders.
"""
from functools import lru_cache
from itertools import product
import numpy as np
from scipy.special import logsumexp
from .common import rng,normalize


@lru_cache(maxsize=2)
def solve(depth,branching,identity):
    if not 1<=depth<=10 or not 2<=branching<=5:raise ValueError('tree outside bounded support')
    r=rng('T2-tree',identity,depth,branching)
    levels=[None]*(depth+1);levels[-1]=r.uniform(-1,1,branching**depth)
    for ply in reversed(range(depth)):
        children=levels[ply+1].reshape(-1,branching)
        levels[ply]=children.max(1) if ply%2==0 else children.min(1)
    return tuple(levels)


def verify_tree(levels,branching):
    """Reconstruct each parent through scalar comparisons, independently."""
    error=0.
    for ply in range(len(levels)-1):
        if len(levels[ply+1])!=len(levels[ply])*branching:raise ValueError('incomplete game tree')
        children=np.asarray(levels[ply+1]).reshape(-1,branching)
        # Columnwise pair comparisons differ from production's row reduction.
        rebuilt=children[:,0].copy()
        for i in range(1,branching):
            rebuilt=np.where(children[:,i]>rebuilt,children[:,i],rebuilt) if ply%2==0 else np.where(children[:,i]<rebuilt,children[:,i],rebuilt)
        error=max(error,float(np.max(np.abs(rebuilt-levels[ply]))))
    if error>1e-12:raise ValueError('minimax reconstruction mismatch')
    return error


def types(branching):return tuple(product((.5,1.,2.,4.),(-1.,1.),range(branching)))


def player_type(v,phase,world,maker):
    return int(rng('T2-player-type',v['id'],phase,world,maker).integers(len(types(v['settings']['branching']))))


def components(row,states):
    values=np.array(row['values']);features=np.array(row['style_features'])
    precision=np.array([s[0] for s in states]);style=np.array([s[1] for s in states]);habit=np.array([s[2] for s in states])
    pressure=.25+.75*min(1.,row['clock_s']/60.)
    value=normalize(precision[:,None]*pressure*values)
    styled=normalize(precision[:,None]*pressure*values+.8*style[:,None]*features)
    habitual=np.eye(len(values))[habit]
    noise=np.full_like(value,1/len(values))
    shares=np.array([.55,.25,.15 if row['ply']==0 else 0.,.05 if row['ply']==0 else .2])
    return np.array([value,styled,habitual,noise]),shares


def probabilities(row,states):
    parts,shares=components(row,states)
    return np.einsum('c,cta->ta',shares,parts)


def public_position(levels,branching,ply,node,clock,identity):
    values=levels[ply+1][node*branching:(node+1)*branching]*(-1 if ply%2 else 1)
    style=rng('T2-style-features',identity,ply,node).uniform(-1,1,branching)
    return dict(ply=ply,node=node,values=values.tolist(),style_features=style.tolist(),clock_s=float(clock))


def player(v,phase,world,maker,*,games=None,clock_quantiles=None):
    from .twin_contract import empirical_quantiles
    s=v['settings'];count=s['games'] if games is None else games
    if count<1 or count>s['games']:raise ValueError('invalid bounded game prefix')
    levels=solve(s['depth'],s['branching'],str((v['id'],phase,world)))
    r=rng('T2-player',v['id'],phase,world,maker);states=types(s['branching']);hidden=player_type(v,phase,world,maker)
    if clock_quantiles is None:raise ValueError('declared clock distribution required; no invented real timing')
    if 'values' in clock_quantiles and 'seconds' in clock_quantiles and clock_quantiles['values']!=clock_quantiles['seconds']:
        raise ValueError('conflicting clock quantile values and seconds aliases')
    values=clock_quantiles.get('values',clock_quantiles.get('seconds'))
    if values is None:raise ValueError('clock quantile values required')
    clocks=empirical_quantiles(clock_quantiles['probabilities'],values,r,(count+1)*s['depth'])
    history=[];truth=[]
    for game in range(count):
        node=0;opponent=(maker+game+1)%s['players']
        if opponent==maker:opponent=(maker+1)%s['players']
        for ply in range(s['depth']):
            row=public_position(levels,s['branching'],ply,node,clocks[game*s['depth']+ply],(v['id'],world))
            actor=maker if ply%2==0 else opponent;actor_type=hidden if actor==maker else player_type(v,phase,world,actor)
            parts,shares=components(row,[states[actor_type]]);kind=int(r.choice(4,p=shares));chosen=int(r.choice(s['branching'],p=parts[kind,0]))
            observed=dict(row,actor=actor,game=game,taken=chosen,record_id=[maker,game,ply],duplicate_of=None)
            noisy=s['dirty'] and r.random()<.02
            if noisy:observed['taken']=int(r.integers(s['branching']))
            if not s['dirty'] or r.random()>=.05:
                history.append(observed);truth.append(dict(kind=kind,original_choice=chosen,label_noise=bool(noisy),missing=False))
                if s['dirty'] and r.random()<.02:
                    duplicate=dict(observed,record_id=[maker,game,ply,'duplicate'],duplicate_of=observed['record_id'])
                    history.append(duplicate);truth.append(dict(truth[-1],duplicate=True))
            # Hidden dropped records still affect the publicly offered later position.
            node=node*s['branching']+chosen
    query=public_position(levels,s['branching'],0,0,clocks[-1],(v['id'],world))
    prediction=probabilities(query,[states[hidden]])[0];y=int(r.choice(s['branching'],p=prediction))
    return dict(maker=maker,history=history,query=query),dict(type=hidden,skill=states[hidden][0],style=states[hidden][1],
        opening=states[hidden][2],marks=truth,y=[y],complete_games=count,law=world%16)


def evaluate(v,public):
    states=types(v['settings']['branching']);lp=np.zeros(len(states));seen=set();classification=[]
    for row in public['history']:
        if row['actor']!=public['maker']:continue
        identity=tuple(row['duplicate_of'] or row['record_id'])
        if identity in seen:continue
        seen.add(identity);parts,shares=components(row,states)
        likelihood=np.einsum('c,cta->ta',shares,parts)[:,row['taken']]
        # Dirty labels have a declared uniform contamination; no hidden class used.
        if v['settings']['dirty']:likelihood=.98*likelihood+.02/v['settings']['branching']
        before=normalize(lp);class_probability=parts[:,:,row['taken']]
        if v['settings']['dirty']:class_probability=.98*class_probability+.02/v['settings']['branching']
        class_mass=shares*(class_probability@before)
        classification.append((class_mass/class_mass.sum()).tolist())
        lp+=np.log(likelihood)
    post=normalize(lp);query=probabilities(public['query'],states)
    return {'population':query.mean(0)[None,:],'individual':(post@query)[None,:]},dict(type_posterior=post.tolist(),
        deviation_class_posterior=classification,unique_observed_moves=len(seen),duplicate_handling='identity deduplication',
        exact_ceiling_eligible=not v['settings']['dirty'],
        exact_scope='clean finite player model; dirty reader conditions on retained decisions and does not use path-implied missing moves')


def verify_posterior(v,public,prediction,detail):
    states=types(v['settings']['branching']);mass=np.ones(len(states))/len(states);seen=set()
    for row in public['history']:
        if row['actor']!=public['maker']:continue
        key=tuple(row['duplicate_of'] or row['record_id'])
        if key in seen:continue
        seen.add(key);weights=[]
        for beta,style,opening in states:
            pressure=.25+.75*min(1.,row['clock_s']/60.);value=np.array(row['values']);feature=np.array(row['style_features'])
            raw=np.exp(beta*pressure*value);raw/=raw.sum()
            styled=np.exp(beta*pressure*value+.8*style*feature);styled/=styled.sum()
            q=.55*raw[row['taken']]+.25*styled[row['taken']]+(.15*(opening==row['taken'])+.05/len(value) if row['ply']==0 else .2/len(value))
            weights.append(.98*q+.02/len(value) if v['settings']['dirty'] else q)
        mass*=weights;mass/=mass.sum()
    if not np.allclose(mass,detail['type_posterior'],atol=1e-11,rtol=0):raise ValueError('independent player posterior mismatch')
    return dict(independent_type_posterior=True,unique_observed_moves=len(seen))


def world(v,phase,index,clock_quantiles):
    for maker in range(v['settings']['players']):yield player(v,phase,index,maker,clock_quantiles=clock_quantiles)
