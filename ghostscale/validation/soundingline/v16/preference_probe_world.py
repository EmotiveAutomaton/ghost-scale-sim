"""V03 actual interventions on three observationally colliding assembly planners."""
import math
import random
from .assembly import World,execute
from .dependency_monitor import learned
from .tradeoffs_world import search,features,draw

CAUSES=["changed-profile","tight-capability","audience-incentive"]
QUERIES=["open","time","neutral","all"]

def actor_parameters(cause,date,law,query):
    high=cause=="changed-profile" and date>=4
    intrinsic=law["high"] if high else law["low"]
    audience=law["high"]-law["low"] if cause=="audience-incentive" and query not in ["neutral","all","future"] else 0.0
    steps=2 if cause=="tight-capability" and query not in ["time","all","future"] else 6
    return intrinsic,audience,steps

def opportunity(base,cause,law,query):
    intrinsic,audience,steps=actor_parameters(cause,base["date"],law,query)
    return {**base,"max_steps":1 if query=="history" else steps,"produced_count":1,"retention":"random"},intrinsic,audience

def model(config,base,law,cause,query,routine,old):
    context,intrinsic,audience=opportunity(base,cause,law,query)
    menu=search(config,context,routine,old)
    values=[]
    for option in menu["options"]:
        f=features(option["state"],context)
        values.append(2*f["task"]+intrinsic*f["coverage"]+(1-intrinsic)*f["quality"]
            +audience*(f["coverage"]-f["quality"])-context["price"]*len(option["program"]))
    peak=max(values);weights=[math.exp(law["beta"]*(value-peak)) for value in values];total=sum(weights)
    probabilities=[value/total for value in weights]
    return {"context":context,"intrinsic":intrinsic,"audience_coefficient":audience,
        "menu":menu,"utilities":values,"probabilities":probabilities}

def produce(config,base,law,cause,query,routine,old,rng):
    decision=model(config,base,law,cause,query,routine,old)
    chosen=draw(decision["probabilities"],rng)
    program=decision["menu"]["options"][chosen]["program"]
    result=execute(World(tuple(config["parents"]),tuple(config["defaults"])),program,initial=tuple(base["initial"]))
    return {**decision,"choice":chosen,"execution":result}

def aligned_models(public,query):
    routine=learned(public["training"])
    old=execute(World(tuple(public["world"]["parents"]),tuple(public["world"]["defaults"])),routine)["state"]
    models=[model(public["world"],public["task_context"],public["law"],cause,query,routine,old) for cause in CAUSES]
    states=sorted({tuple(option["state"]) for entry in models for option in entry["menu"]["options"]})
    distributions=[]
    for entry in models:
        lookup={tuple(option["state"]):p for option,p in zip(entry["menu"]["options"],entry["probabilities"])}
        distributions.append([lookup.get(state,0.0) for state in states])
    return states,distributions,{"search_primitives":sum(entry["menu"]["successor_evaluations"] for entry in models),
        "utility_terms":sum(len(entry["menu"]["options"]) for entry in models),
        "model_training_feedback_checks":len(public["training"]),
        "model_training_successful_tokens":sum(len(trial["program"]) for trial in public["training"] if trial["feedback"]),
        "model_old_replay_primitives":len(routine)}
