"""Executed assembly choice under separately declared menus and bounded search."""
from itertools import product
import math
from .assembly import World,step,execute
from .dependency_monitor import learned

PROFILES=["stable-low","stable-high","changing-up","changing-down","absent"]

def considered(context,old_state):
    targets=list(product([-1,0,1],repeat=3))
    if context["consideration"]=="familiar":
        targets=[target for target in targets if target[0] in [-1,old_state[0]]]
    elif context["consideration"]!="all":
        raise ValueError("unknown consideration rule")
    initial=tuple(context["initial"])
    if initial not in targets:
        targets.append(initial)
    return sorted(targets)

def search(config,context,routine,old_state):
    """One shared bounded search, charging every tested primitive, including failures.

    Consideration is fixed before the search. Intermediate states may be outside
    the contemplated terminal menu. Found plans need not be globally shortest.
    """
    world=World(tuple(config["parents"]),tuple(config["defaults"]))
    targets=set(considered(context,old_state))
    vocabulary=([tuple(routine)] if routine else [])+[(action,) for action in range(10)]
    initial=tuple(context["initial"])
    queue=[(initial,())];seen={initial};cursor=0;attempts=[];options={}
    exhausted=False
    while cursor<len(queue) and not exhausted:
        state,program=queue[cursor];cursor+=1
        for fragment in vocabulary:
            if len(program)+len(fragment)>context["max_steps"]:
                continue
            after=state;stopped=False;legal=True
            for action in fragment:
                if len(attempts)>=context["search_budget"]:
                    exhausted=True;break
                before=after
                after,stopped,legal=step(world,after,stopped,action)
                attempts.append([list(before),action,list(after),legal,stopped])
                if not legal:
                    break
            if exhausted:
                break
            if not legal:
                continue
            proposal=program+fragment
            if stopped:
                if after in targets and after not in options:
                    options[after]={"state":list(after),"program":list(proposal)}
            elif after not in seen:
                seen.add(after);queue.append((after,proposal))
    if not options:
        raise ValueError("bounded search failed even the considered stop option")
    return {"options":[options[state] for state in sorted(options)],"attempts":attempts,
            "successor_evaluations":len(attempts),"search_exhausted":exhausted,
            "considered_targets":[list(target) for target in sorted(targets)],
            "frontier_states":len(queue)}

def features(state,context):
    present=[part for part,value in enumerate(state) if value>=0]
    return {"task":float(state[context["required_part"]]>=0),
            "coverage":len(present)/3,
            "quality":sum(state[part]==context["orientation_standard"][part] for part in present)/len(present) if present else 0.0}

def alpha(profile,date,law):
    if profile=="absent":
        return None
    high=profile=="stable-high" or (profile=="changing-up" and date>=4) or (profile=="changing-down" and date<4)
    return law["high"] if high else law["low"]

def distribution(options,context,law,profile):
    preference=alpha(profile,context["date"],law)
    utilities=[]
    for option in options:
        f=features(option["state"],context)
        extra=0 if preference is None else preference*f["coverage"]+(1-preference)*f["quality"]
        utilities.append(2*f["task"]+extra-context["price"]*len(option["program"]))
    peak=max(utilities)
    weights=[math.exp(law["beta"]*(value-peak)) for value in utilities]
    total=sum(weights)
    return [weight/total for weight in weights]

def selected_distribution(probabilities,options,context):
    if context["retention"]=="random":
        return list(probabilities)
    preferred=[features(option["state"],context)["coverage"]>=features(option["state"],context)["quality"] for option in options]
    mass=sum(p for p,keep in zip(probabilities,preferred) if keep)
    n=context["produced_count"]
    return [p*(sum((1-mass)**rank for rank in range(n)) if keep else (1-mass)**(n-1))
            for p,keep in zip(probabilities,preferred)]

def draw(probabilities,rng):
    threshold=rng.random();cumulative=0.0
    for index,probability in enumerate(probabilities):
        cumulative+=probability
        if threshold<cumulative:
            return index
    return len(probabilities)-1

def batch(config,context,law,profile,routine,old_state,rng):
    menu=search(config,context,routine,old_state)
    probabilities=distribution(menu["options"],context,law,profile)
    events=[]
    for _ in range(context["produced_count"]):
        choice=draw(probabilities,rng)
        option=menu["options"][choice]
        result=execute(World(tuple(config["parents"]),tuple(config["defaults"])),option["program"],initial=tuple(context["initial"]))
        events.append({"choice":choice,"execution":result})
    if context["retention"]=="random":
        retained=rng.randrange(len(events))
    else:
        retained=next((position for position,event in enumerate(events)
            if features(event["execution"]["state"],context)["coverage"]>=features(event["execution"]["state"],context)["quality"]),0)
    return {"menu":menu,"candidates":events,"retained_index":retained,
            "probabilities":probabilities}

def public_training(training):
    return [{key:trial[key] for key in ["date","program","target","feedback"]} for trial in training]
