"""Independent bounded assembly menu and finite-profile likelihood audit anchor."""
from collections import deque,Counter
from itertools import product
import math
from .assembly_reference import interpret

PROFILES=("stable-low","stable-high","changing-up","changing-down","absent")

def acquired(training):
    counts=Counter(tuple(trial["program"]) for trial in training if trial["feedback"])
    candidates=[program for program,count in counts.items() if count>=3]
    return list(sorted(candidates,key=lambda program:(-counts[program],program))[0][:-1]) if candidates else []

def menu(config,context,routine,old_state):
    candidates=set(product((-1,0,1),repeat=3))
    if context["consideration"]=="familiar":
        candidates={state for state in candidates if state[0]==-1 or state[0]==old_state[0]}
    candidates.add(tuple(context["initial"]))
    tokens=([list(routine)] if routine else [])+[[value] for value in range(10)]
    pending=deque([(list(context["initial"]),[])])
    visited={tuple(context["initial"])}
    found={};attempts=[];exhausted=False
    while pending and not exhausted:
        initial,prefix=pending.popleft()
        for token in tokens:
            if len(prefix)+len(token)>context["max_steps"]:
                continue
            state=list(initial);halted=False;valid=True
            for command in token:
                if len(attempts)==context["search_budget"]:
                    exhausted=True;break
                result=interpret(config,[command],initial=state)
                valid=result["legal"] and not halted
                after=result["state"] if valid else state
                halted=result["stopped"] if valid else halted
                attempts.append([state,command,after,valid,halted])
                state=after
                if not valid:
                    break
            if exhausted:
                break
            if valid and halted:
                if tuple(state) in candidates and tuple(state) not in found:
                    found[tuple(state)]={"state":state,"program":prefix+token}
            elif valid and tuple(state) not in visited:
                visited.add(tuple(state));pending.append((state,prefix+token))
    return {"options":[found[key] for key in sorted(found)],"attempts":attempts,
        "successor_evaluations":len(attempts),"search_exhausted":exhausted,
        "considered_targets":[list(state) for state in sorted(candidates)],"frontier_states":len(visited)}

def raw_law(options,context,law,profile):
    if profile=="absent":
        coefficients=(0,0)
    else:
        high=profile=="stable-high" or (profile=="changing-up" and context["date"]>=4) or (profile=="changing-down" and context["date"]<4)
        a=law["high" if high else "low"]
        coefficients=(a,1-a)
    values=[]
    for option in options:
        state=option["state"]
        occupied=[i for i in range(3) if state[i]!=-1]
        quality=sum(state[i]==context["orientation_standard"][i] for i in occupied)/len(occupied) if occupied else 0
        reward=2*(state[context["required_part"]]!=-1)+coefficients[0]*len(occupied)/3+coefficients[1]*quality-context["price"]*len(option["program"])
        values.append(math.exp(law["beta"]*reward))
    return [value/sum(values) for value in values]

def retained_law(probabilities,options,context):
    if context["retention"]=="random":
        return list(probabilities)
    preferred=[]
    for option in options:
        occupied=[i for i in range(3) if option["state"][i]>=0]
        quality=sum(option["state"][i]==context["orientation_standard"][i] for i in occupied)/len(occupied) if occupied else 0
        preferred.append(len(occupied)/3>=quality)
    selected=[0.0]*len(options)
    for sequence in product(range(len(options)),repeat=context["produced_count"]):
        weight=math.prod(probabilities[index] for index in sequence)
        retained=next((index for index in sequence if preferred[index]),sequence[0])
        selected[retained]+=weight
    return selected

def prediction(public,policy):
    routine=acquired(public["training"])
    old=interpret(public["world"],routine)["state"]
    profiles=list(PROFILES[:2])+["absent"] if policy=="stable" else ["absent"] if policy=="task-cost" else list(PROFILES)
    weights={profile:1/len(profiles) for profile in profiles}
    searches=0;terms=0
    if policy in {"chronological","direct-chronological","stable"}:
        for event in public["history"]:
            found=menu(public["world"],event["context"],routine,old)
            searches+=found["successor_evaluations"]
            position=next((i for i,option in enumerate(found["options"]) if option["state"]==event["state"]),None)
            if position is None:
                return {"model_mismatch":True,"reason":"observed state outside public bounded support"}
            for profile in profiles:
                probabilities=retained_law(raw_law(found["options"],event["context"],public["law"],profile),found["options"],event["context"])
                weights[profile]*=probabilities[position]
                terms+=len(found["options"])
    evidence=sum(weights.values())
    weights={key:weight/evidence for key,weight in weights.items()}
    future=menu(public["world"],public["future_context"],routine,old)
    searches+=future["successor_evaluations"]
    probabilities=[0.0]*len(future["options"])
    for profile,weight in weights.items():
        probabilities=[value+weight*other for value,other in zip(probabilities,raw_law(future["options"],public["future_context"],public["law"],profile))]
    if policy=="surface-habit" and public["history"]:
        weights2=[math.exp(-2*sum(a!=b for a,b in zip(option["state"],public["history"][-1]["state"]))) for option in future["options"]]
        probabilities=[value/sum(weights2) for value in weights2]
    return {"model_mismatch":False,"states":[item["state"] for item in future["options"]],"probabilities":probabilities,
        "profile_posterior":weights,"expected_revision_primitives":sum(p*len(option["program"]) for p,option in zip(probabilities,future["options"])),
        "costs":{"training_feedback_checks":len(public["training"]),
            "training_successful_program_tokens":sum(len(trial["program"]) for trial in public["training"] if trial["feedback"]),
            "routine_replay_primitives":len(routine),"search_primitives":searches,"likelihood_terms":terms,
            "prediction_terms":len(profiles)*len(future["options"]),
            "surface_comparisons":3*len(future["options"]) if policy=="surface-habit" and public["history"] else 0}}
