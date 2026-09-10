"""V03 paid probes and target-specific information gain on explicit rival planners."""
import json
import math
from .dependency_monitor import learned
from .preference_probe_world import CAUSES,QUERIES,aligned_models

POLICIES=["none","open","time","neutral","all","eig-cause","eig-future","eig-artifact","eig-history","direct-all"]
EXTENSION="ghostscale.validation.soundingline.v16.preference_probe:public_reader"
DESIGN={"card_id":"V03","conditions":[{"id":f"{cause}-{kind}-search-{budget}-fee-{fee}",
    "cause":cause,"kind":kind,"search_budget":budget,"query_fee":fee}
    for cause in CAUSES for kind in ["aligned","subtle","initially-useful"] for budget in [64,256] for fee in [0.01,0.10]],
    "arms":POLICIES,"primary":[("all","none","future_choice_log_score","nats_per_event",0.02),
        ("eig-future","eig-cause","future_choice_log_score","nats_per_event",0.02),
        ("all","direct-all","future_choice_log_score","nats_per_event",0.02)],
    "sample_rule":{"scout":64,"constructors":8,"expansion":256,"expansion_constructors":20,"final_expansion":1024},
    "dependencies":["probe-physical","probe-collisions","probe-information","probe-reference"],
    "target":"new neutral-audience full-time maker choice; rival-cause uncertainty and known historical programs remain separate",
    "rivals":"actual low-to-high profile change; stable low profile with two-step capability; stable low profile with external coverage-minus-quality incentive",
    "collision":"eight forced stop-only dated episodes have identical public histories for all three rivals, constructed before any unknown draw",
    "queries":"open alternatives, optionally grant six steps, optionally remove the audience incentive; each query causes a real new execution",
    "target_comparison":"exact expected information for future choice and rival cause; artifact and identifiable-history targets are already determined here and rationally decline inquiry",
    "target_comparison_limit":"deliberate zero-entropy artifact/history boundary, not a general ranking of inquiry targets",
    "fairness":"all arms have the same public law and possible interventions; direct-all receives exactly the all-policy evidence",
    "costs":"query fees in nats-valued decision cost, physical production/search, and separately counted model inquiry/prediction work",
    "scope":"finite known-rival planners in constructed assembly; no inference of human values, hidden motives or arbitrary unknown planners",
    "adversaries":["X01","X02","X04","X05","X06","X07","X08"],"repair_budget":1,
    "continuation":"expand a named intervention-cost or identifiability boundary; indistinguishable rivals remain ambiguous"}

def entropy(probabilities):
    return -sum(value*math.log(value) for value in probabilities if value>0)

def validate(public):
    if set(public)!={"schema_version","task_id","phase","world","law","training","history","task_context","query_fees","answer"} or public["schema_version"]!="v16.preference-probe.1":
        raise ValueError("preference probe schema violation")
    if set(public["world"])!={"parents","defaults"} or set(public["law"])!={"low","high","beta"} or set(public["query_fees"])!=set(QUERIES):
        raise ValueError("undeclared probe world or price")
    if public["phase"] not in [1,2]:
        raise ValueError("unknown probe phase")
    if public["phase"]==1 and public["answer"] is not None:
        raise ValueError("probe answer supplied before committed request")
    if public["answer"] is not None and set(public["answer"])!={"query","state"}:
        raise ValueError("hidden probe answer fields")
    fields={"date","initial","required_part","orientation_standard","price","max_steps","search_budget","consideration","retention","produced_count"}
    for context in [public["task_context"]]+[event["context"] for event in public["history"]]:
        if set(context)!=fields or context["consideration"]!="all" or context["retention"]!="random" or context["produced_count"]!=1:
            raise ValueError("undeclared probe context")
    if len(public["history"])!=8 or not 0<public["law"]["low"]<public["law"]["high"]<1 or public["law"]["beta"]<=0:
        raise ValueError("invalid collision history or model law")
    if any(fee<0 or not math.isfinite(fee) for fee in public["query_fees"].values()):
        raise ValueError("invalid inquiry fee")
    for trial in public["training"]:
        if set(trial)!={"date","program","target","feedback"}:
            raise ValueError("hidden training fields")
    for event in public["history"]:
        if set(event)!={"context","state"} or event["context"]["max_steps"]!=1 or event["context"]["initial"]!=event["state"]:
            raise ValueError("history is not the declared stop-only collision")

def decision(public,policy):
    costs={"search_primitives":0,"utility_terms":0,"mixture_terms":0,"posterior_hypotheses":0,"entropy_terms":0,
        "model_training_feedback_checks":0,"model_training_successful_tokens":0,"model_old_replay_primitives":0}
    if policy in ["none","eig-artifact","eig-history"]:
        return {"query":None,"information_gain":0.0,"net_information":0.0,"costs":costs}
    if policy in QUERIES or policy=="direct-all":
        query="all" if policy=="direct-all" else policy
        return {"query":query,"information_gain":None,"net_information":None,"costs":costs}
    if policy=="eig-future":
        future_states,future,work=aligned_models(public,"future")
        for key,value in work.items():
            costs[key]+=value
        marginal=[sum(row[i] for row in future)/3 for i in range(len(future_states))]
        before=entropy(marginal)
        costs["mixture_terms"]+=3*len(future_states)
        costs["entropy_terms"]+=len(future_states)
    else:
        before=math.log(3)
    candidates=[]
    for query in QUERIES:
        states,rows,work=aligned_models(public,query)
        for key,value in work.items():
            costs[key]+=value
        after=0.0
        for outcome in range(len(states)):
            mass=sum(row[outcome] for row in rows)/3
            if mass==0:
                continue
            posterior=[row[outcome]/(3*mass) for row in rows]
            costs["posterior_hypotheses"]+=3
            if policy=="eig-cause":
                after+=mass*entropy(posterior)
                costs["entropy_terms"]+=3
            else:
                conditional=[sum(posterior[h]*future[h][i] for h in range(3)) for i in range(len(future_states))]
                costs["mixture_terms"]+=3*len(future_states)
                costs["entropy_terms"]+=len(future_states)
                after+=mass*entropy(conditional)
        gain=max(0.0,before-after)
        candidates.append((gain-public["query_fees"][query],query,gain))
    best=max(candidates,key=lambda item:(item[0],-QUERIES.index(item[1])))
    return {"query":best[1] if best[0]>1e-12 else None,"information_gain":best[2] if best[0]>1e-12 else 0.0,
        "net_information":best[0] if best[0]>1e-12 else 0.0,"costs":costs}

def public_reader(payload:bytes,policy):
    public=json.loads(payload);validate(public)
    if policy not in POLICIES:
        raise ValueError("unknown probe policy")
    choice=decision(public,policy)
    if public["phase"]==1:
        return choice
    answer=public["answer"]
    if (answer is None)!=(choice["query"] is None) or (answer is not None and answer["query"]!=choice["query"]):
        raise ValueError("probe supplied unpurchased answer")
    posterior=[1/3]*3
    costs=dict(choice["costs"])
    if answer is not None:
        states,models,work=aligned_models(public,answer["query"])
        for key,value in work.items():
            costs[key]+=value
        if tuple(answer["state"]) not in states:
            return {"model_mismatch":True}
        index=states.index(tuple(answer["state"]))
        evidence=sum(row[index] for row in models)
        if evidence<=0:
            return {"model_mismatch":True}
        posterior=[row[index]/evidence for row in models]
        costs["posterior_hypotheses"]+=3
    states,models,work=aligned_models(public,"future")
    for key,value in work.items():
        costs[key]+=value
    probabilities=[sum(posterior[h]*models[h][i] for h in range(3)) for i in range(len(states))]
    costs["mixture_terms"]+=3*len(states)
    costs["entropy_terms"]+=3
    routine=learned(public["training"])
    costs["training_feedback_checks"]=len(public["training"])
    costs["training_successful_program_tokens"]=sum(len(trial["program"]) for trial in public["training"] if trial["feedback"])
    costs["reproduction_program_tokens"]=len(routine)+1
    return {"model_mismatch":False,"states":[list(state) for state in states],"probabilities":probabilities,
        "cause_posterior":posterior,"cause_entropy":entropy(posterior),
        "reproduction_program":routine+[9],"historical_programs":[[9] for event in public["history"]],
        "costs":costs}
