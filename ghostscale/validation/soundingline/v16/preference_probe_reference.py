"""Independent V03 planner laws and direct mutual-information enumeration."""
import math
from .assembly_reference import interpret
from .tradeoffs_reference import acquired,menu

CAUSES=("changed-profile","tight-capability","audience-incentive")
QUERIES=("open","time","neutral","all")

def model(config,base,law,cause,query,routine,old):
    intrinsic=law["high"] if cause=="changed-profile" and base["date"]>=4 else law["low"]
    incentive=(law["high"]-law["low"]) if cause=="audience-incentive" and query in {"history","open","time"} else 0.0
    limit=2 if cause=="tight-capability" and query in {"history","open","neutral"} else 6
    context={**base,"max_steps":1 if query=="history" else limit,"retention":"random","produced_count":1}
    found=menu(config,context,routine,old)
    utilities=[]
    for option in found["options"]:
        state=option["state"];parts=[i for i in range(3) if state[i]>=0]
        quality=sum(state[i]==context["orientation_standard"][i] for i in parts)/len(parts) if parts else 0
        coverage=len(parts)/3
        utility=2*(state[context["required_part"]]>=0)+intrinsic*coverage+(1-intrinsic)*quality+incentive*(coverage-quality)-context["price"]*len(option["program"])
        utilities.append(utility)
    weights=[math.exp(law["beta"]*value) for value in utilities]
    return {"context":context,"intrinsic":intrinsic,"audience_coefficient":incentive,"menu":found,
        "utilities":utilities,"probabilities":[value/sum(weights) for value in weights]}

def aligned(public,query):
    routine=acquired(public["training"]);old=interpret(public["world"],routine)["state"]
    hypotheses=[model(public["world"],public["task_context"],public["law"],cause,query,routine,old) for cause in CAUSES]
    states=sorted({tuple(option["state"]) for hypothesis in hypotheses for option in hypothesis["menu"]["options"]})
    rows=[]
    for hypothesis in hypotheses:
        lookup={tuple(option["state"]):p for option,p in zip(hypothesis["menu"]["options"],hypothesis["probabilities"])}
        rows.append([lookup.get(state,0.0) for state in states])
    return states,rows,{"search_primitives":sum(hypothesis["menu"]["successor_evaluations"] for hypothesis in hypotheses),
        "utility_terms":sum(len(hypothesis["menu"]["options"]) for hypothesis in hypotheses),
        "model_training_feedback_checks":len(public["training"]),
        "model_training_successful_tokens":sum(len(trial["program"]) for trial in public["training"] if trial["feedback"]),
        "model_old_replay_primitives":len(routine)}

def information(rows,future=None):
    py=[sum(row[i] for row in rows)/3 for i in range(len(rows[0]))]
    if future is None:
        return sum((rows[h][y]/3)*math.log(rows[h][y]/py[y])
            for h in range(3) for y in range(len(py)) if rows[h][y]>0)
    pz=[sum(row[i] for row in future)/3 for i in range(len(future[0]))]
    total=0.0
    for y in range(len(py)):
        for z in range(len(pz)):
            joint=sum(rows[h][y]*future[h][z] for h in range(3))/3
            if joint>0:
                total+=joint*math.log(joint/(py[y]*pz[z]))
    return max(0.0,total)

def choice(public,policy):
    costs={"search_primitives":0,"utility_terms":0,"mixture_terms":0,"posterior_hypotheses":0,"entropy_terms":0,
        "model_training_feedback_checks":0,"model_training_successful_tokens":0,"model_old_replay_primitives":0}
    if policy in {"none","eig-artifact","eig-history"}:
        return {"query":None,"information_gain":0.0,"net_information":0.0,"costs":costs}
    if policy in QUERIES or policy=="direct-all":
        return {"query":"all" if policy=="direct-all" else policy,"information_gain":None,"net_information":None,"costs":costs}
    future=None
    if policy=="eig-future":
        states,future,work=aligned(public,"future")
        for key,value in work.items():
            costs[key]+=value
        costs["mixture_terms"]+=3*len(states)
        costs["entropy_terms"]+=len(states)
    candidates=[]
    for query in QUERIES:
        states,rows,work=aligned(public,query)
        for key,value in work.items():
            costs[key]+=value
        supported=sum(any(row[i]>0 for row in rows) for i in range(len(states)))
        costs["posterior_hypotheses"]+=3*supported
        if future is None:
            costs["entropy_terms"]+=3*supported
        else:
            costs["mixture_terms"]+=3*len(future[0])*supported
            costs["entropy_terms"]+=len(future[0])*supported
        gain=information(rows,future)
        candidates.append((gain-public["query_fees"][query],query,gain))
    best=sorted(candidates,key=lambda item:(-item[0],QUERIES.index(item[1])))[0]
    return {"query":best[1] if best[0]>1e-12 else None,"information_gain":best[2] if best[0]>1e-12 else 0.0,
        "net_information":best[0] if best[0]>1e-12 else 0.0,"costs":costs}

def prediction(public,policy):
    selected=choice(public,policy)
    costs=dict(selected["costs"])
    posterior=[1/3]*3
    if public["answer"] is not None:
        states,rows,work=aligned(public,selected["query"])
        for key,value in work.items():
            costs[key]+=value
        index=states.index(tuple(public["answer"]["state"]))
        evidence=sum(row[index] for row in rows)
        posterior=[row[index]/evidence for row in rows]
        costs["posterior_hypotheses"]+=3
    states,rows,work=aligned(public,"future")
    for key,value in work.items():
        costs[key]+=value
    probabilities=[sum(weight*rows[h][i] for h,weight in enumerate(posterior)) for i in range(len(states))]
    costs["mixture_terms"]+=3*len(states);costs["entropy_terms"]+=3
    routine=acquired(public["training"])
    costs["training_feedback_checks"]=len(public["training"])
    costs["training_successful_program_tokens"]=sum(len(trial["program"]) for trial in public["training"] if trial["feedback"])
    costs["reproduction_program_tokens"]=len(routine)+1
    return {"model_mismatch":False,"states":[list(state) for state in states],"probabilities":probabilities,"cause_posterior":posterior,
        "cause_entropy":-sum(value*math.log(value) for value in posterior if value>0),
        "reproduction_program":routine+[9],"historical_programs":[[9] for _ in public["history"]],"costs":costs}
