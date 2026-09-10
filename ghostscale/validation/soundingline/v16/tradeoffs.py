"""Finite profile reading, distinct from task, physics, consideration and publication."""
import json
import math
from .assembly import execute,World
from .dependency_monitor import learned
from .tradeoffs_world import PROFILES,search,distribution,selected_distribution

POLICIES=["chronological","stable","context-mixture","task-cost","surface-habit","direct-chronological"]
EXTENSION="ghostscale.validation.soundingline.v16.tradeoffs:public_reader"

def design(card):
    if card=="V01":
        conditions=[{"id":f"{profile}-steps-{steps}-{menu}-{retention}","profile":profile,
            "steps":steps,"consideration":menu,"retention":retention,"kind":"initially-useful","purpose":"varied","price_regime":"varied"}
            for profile in PROFILES for steps in [4,6] for menu in ["all","familiar"] for retention in ["random","selected"]]
    elif card=="V02":
        conditions=[{"id":f"{profile}-{kind}-{purpose}-{price}","profile":profile,"steps":None,
            "consideration":"varied","retention":"varied","kind":kind,"purpose":purpose,"price_regime":price}
            for profile in PROFILES for kind in ["aligned","subtle","initially-useful"]
            for purpose in ["stable","redirected"] for price in ["ordinary","costly"]]
    else:
        raise ValueError("unknown tradeoff card")
    return {"card_id":card,"conditions":conditions,"arms":POLICIES,
        "primary":[("chronological","context-mixture","future_choice_log_score","nats_per_event",0.02),
                   ("chronological","stable","future_choice_log_score","nats_per_event",0.02),
                   ("chronological","direct-chronological","future_choice_log_score","nats_per_event",0.02)],
        "sample_rule":{"scout":64,"constructors":8,"expansion":256,"expansion_constructors":20,"final_expansion":1024},
        "dependencies":["tradeoff-physical","tradeoff-null","tradeoff-selection","tradeoff-reference"],
        "eligibility":"validated P03, O01-O04 and M01-M04 separation instruments; scientific gains are not admission gates",
        "target":"held-out unselected executed choice and revision cost after eight dated decisions",
        "profile_dimensions":"constructor-defined coverage versus orientation quality, beyond ordinary required-part task and paid primitive cost",
        "scope":"conditional known public task, acquired training, physical/search constraints, considered menu and publication law; no human-value claim",
        "trajectory":"V01 independent episodes from an acquired old state; V02 eight sequential retained revisions with purpose and price crossed against profile dynamics",
        "null":"absent additional coverage/quality profile retains ordinary task and execution-cost preferences",
        "support":"five explicit profile schedules over actually found legal stopped plans; no continuous unrestricted preference recovery",
        "fairness":"all readers receive identical public bytes; ordinary stable mixture, task-cost, public surface habit and direct joint rivals",
        "costs":"all actual search attempts, training, all rejected production, reader reconstruction/search and inference; batch search is shared within one public context",
        "adversaries":["X01","X02","X03","X04","X05","X06","X07","X08"],"repair_budget":1,
        "continuation":"one named capability, nuisance-separation or changing-profile boundary; never tune a failed profile into a positive result"}

def validate(public):
    if set(public)!={"schema_version","task_id","world","law","training","history","future_context"} or public["schema_version"]!="v16.tradeoffs.1":
        raise ValueError("tradeoff public schema violation")
    if set(public["world"])!={"parents","defaults"} or set(public["law"])!={"low","high","beta"}:
        raise ValueError("undeclared public model")
    if not 0<public["law"]["low"]<public["law"]["high"]<1 or public["law"]["beta"]<=0:
        raise ValueError("invalid tradeoff production law")
    fields={"date","initial","required_part","orientation_standard","price","max_steps","search_budget","consideration","retention","produced_count"}
    for event in public["history"]:
        if set(event)!={"context","state"}:
            raise ValueError("hidden event fields")
    contexts=[event["context"] for event in public["history"]]+[public["future_context"]]
    for context in contexts:
        if set(context)!=fields or context["retention"] not in ["random","selected"] or context["consideration"] not in ["all","familiar"]:
            raise ValueError("undeclared opportunity or selection context")
        if context["max_steps"]<1 or context["search_budget"]<16 or context["produced_count"] not in [1,2]:
            raise ValueError("invalid bounded opportunity")
    for trial in public["training"]:
        if set(trial)!={"date","program","target","feedback"}:
            raise ValueError("undeclared training field")

def public_reader(payload:bytes,policy):
    public=json.loads(payload);validate(public)
    if policy not in POLICIES:
        raise ValueError("unknown tradeoff policy")
    routine=learned(public["training"])
    world=World(tuple(public["world"]["parents"]),tuple(public["world"]["defaults"]))
    old_state=execute(world,routine)["state"]
    hypotheses=PROFILES if policy not in ["stable","task-cost"] else (PROFILES[:2]+["absent"] if policy=="stable" else ["absent"])
    weights=[1/len(hypotheses)]*len(hypotheses)
    logs=[0.0]*len(hypotheses)
    search_cost=0;likelihood_terms=0
    use_history=policy in ["chronological","stable","direct-chronological"]
    if use_history:
        for event in public["history"]:
            menu=search(public["world"],event["context"],routine,old_state)
            search_cost+=menu["successor_evaluations"]
            position=next((i for i,option in enumerate(menu["options"]) if option["state"]==event["state"]),None)
            if position is None:
                return {"model_mismatch":True,"reason":"observed state outside public bounded support"}
            for h,profile in enumerate(hypotheses):
                probabilities=selected_distribution(distribution(menu["options"],event["context"],public["law"],profile),menu["options"],event["context"])
                likelihood_terms+=len(menu["options"])
                if policy=="direct-chronological":
                    weights[h]*=probabilities[position]
                else:
                    logs[h]+=math.log(probabilities[position])
        if policy!="direct-chronological":
            peak=max(logs)
            weights=[math.exp(value-peak) for value in logs]
    total=sum(weights);weights=[weight/total for weight in weights]
    future=search(public["world"],public["future_context"],routine,old_state)
    search_cost+=future["successor_evaluations"]
    choices=[0.0]*len(future["options"])
    for weight,profile in zip(weights,hypotheses):
        probabilities=distribution(future["options"],public["future_context"],public["law"],profile)
        choices=[a+weight*b for a,b in zip(choices,probabilities)]
    if policy=="surface-habit" and public["history"]:
        previous=public["history"][-1]["state"]
        scores=[math.exp(-2*sum(a!=b for a,b in zip(option["state"],previous))) for option in future["options"]]
        total=sum(scores);choices=[value/total for value in scores]
    return {"model_mismatch":False,"states":[item["state"] for item in future["options"]],
        "probabilities":choices,"profile_posterior":{profile:weight for profile,weight in zip(hypotheses,weights)},
        "expected_revision_primitives":sum(probability*len(option["program"]) for probability,option in zip(choices,future["options"])),
        "costs":{"training_feedback_checks":len(public["training"]),
            "training_successful_program_tokens":sum(len(trial["program"]) for trial in public["training"] if trial["feedback"]),
            "routine_replay_primitives":len(routine),"search_primitives":search_cost,"likelihood_terms":likelihood_terms,
            "prediction_terms":len(hypotheses)*len(future["options"]),
            "surface_comparisons":3*len(future["options"]) if policy=="surface-habit" and public["history"] else 0}}
