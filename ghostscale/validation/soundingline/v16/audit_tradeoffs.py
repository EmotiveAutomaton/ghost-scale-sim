"""Independent V01/V02 acquisition, dated production, prediction and reduction."""
import math
import random
import json
import hashlib
from .assembly_reference import interpret
from .audit_graphic_maker import seed
from .tradeoffs_reference import acquired,menu,raw_law,prediction
from .audit_statistics import verify

def load(path,expected):
    payload=path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected:
        raise ValueError("tradeoff commitment hash differs")
    return json.loads(payload)

def preparation(namespace,condition,index,constructors):
    identity=f"constructor-{index%constructors:03d}"
    rng=random.Random(seed(namespace,"assembly-constructor",identity))
    world={"parents":list(rng.choice([(-1,0,0),(-1,0,1),(-1,-1,1)])),
           "defaults":[rng.randrange(2) for _ in range(3)]}
    reliability=random.Random(seed(namespace,"routine-reliability",identity)).uniform(0.85,1.0)
    recipe={"aligned":[0,1],"subtle":[0,6],"initially-useful":[0,6,1]}[condition["kind"]]
    old_goal=interpret(world,recipe)["state"]
    training=[]
    for date in range(8):
        program=list(recipe)
        rng=random.Random(seed(namespace,condition["kind"],index,"old-training",date))
        if rng.random()>reliability:
            program[rng.randrange(len(program))]=rng.randrange(9)
        program.append(9)
        result=interpret(world,program)
        training.append({"date":date,"program":program,"target":old_goal,"execution":result,
            "feedback":result["legal"] and result["successfully_stopped"] and result["state"]==old_goal})
    routine=acquired(training);current=interpret(world,routine)
    goal=[world["defaults"][0],world["defaults"][1],-1]
    return {"world":world,"reliability":reliability,"training":training,"routine":routine,"current_execution":current,
        "goal":goal,"constructor_id":identity,
        "old_goal_error_change":sum(a!=b for a,b in zip([-1,-1,-1],goal))-sum(a!=b for a,b in zip(current["state"],goal))}

def context(namespace,index,date,condition,world,initial,future=False):
    rng=random.Random(seed(namespace,"tradeoff-context",index,date))
    required=rng.randrange(3);standard=[rng.randrange(2) for _ in range(3)]
    if condition["purpose"]!="varied":
        changed=condition["purpose"]=="redirected" and date>=4
        required=2 if changed else 1
        standard=[1-value if changed else value for value in world["defaults"]]
    price=rng.uniform(0.03,0.15)*(3 if condition["price_regime"]=="costly" else 1)
    steps=rng.choice([4,6]) if condition["steps"] is None else condition["steps"]
    budget=rng.choice([64,256])
    considered=rng.choice(["all","familiar"]) if condition["consideration"]=="varied" else condition["consideration"]
    retention=rng.choice(["random","selected"]) if condition["retention"]=="varied" else condition["retention"]
    return {"date":date,"initial":list(initial),"required_part":required,"orientation_standard":standard,"price":price,
        "max_steps":steps,"search_budget":budget,"consideration":considered,
        "retention":"random" if future else retention,"produced_count":1 if future else 2}

def check_batch(actual,config,ctx,law,profile,routine,old,rng):
    expected_menu=menu(config,ctx,routine,old)
    if actual["menu"]!=expected_menu:
        raise ValueError("independent considered/bounded menu differs")
    probabilities=raw_law(expected_menu["options"],ctx,law,profile)
    if len(probabilities)!=len(actual["probabilities"]) or any(abs(a-b)>1e-12 for a,b in zip(probabilities,actual["probabilities"])):
        raise ValueError("actual utility probabilities differ independently")
    candidates=[]
    for _ in range(ctx["produced_count"]):
        point=rng.random();total=0;choice=len(probabilities)-1
        for i,probability in enumerate(probabilities):
            total+=probability
            if point<total:
                choice=i;break
        result=interpret(config,expected_menu["options"][choice]["program"],initial=ctx["initial"])
        candidates.append({"choice":choice,"execution":result})
    if ctx["retention"]=="random":
        retained=rng.randrange(len(candidates))
    else:
        matches=[]
        for i,event in enumerate(candidates):
            state=event["execution"]["state"];parts=[j for j in range(3) if state[j]>=0]
            quality=sum(state[j]==ctx["orientation_standard"][j] for j in parts)/len(parts) if parts else 0
            if len(parts)/3>=quality:
                matches.append(i)
        retained=matches[0] if matches else 0
    if candidates!=actual["candidates"] or retained!=actual["retained_index"]:
        raise ValueError("actual rejected work, physical draw or retention differs")
    return candidates[retained]["execution"]

def audit_unit(root,row):
    uid=row["unit_id"]
    public=load(root/"public"/f"{uid}.json",row["public_hash"])
    private=load(root/"private"/f"{uid}.json",row["private_hash"])
    original=load(root/"private"/f"{uid}-original.json",private["original_hash"])
    final=load(root/"predictions"/f"{uid}.json",row["prediction_hash"])
    if final["public_hash"]!=row["public_hash"] or final["submitted_at"]!=row["prediction_submitted_at"] or final["submitted_at"]>row["scored_at"]:
        raise ValueError("tradeoff predictions were not committed against this observation")
    namespace,index=row["lineage"],row["seed_components"]["index"]
    condition=row["condition_spec"]
    prep=preparation(namespace,condition,index,row["seed_components"]["constructors"])
    if prep!=original["acquired"] or prep["constructor_id"]!=row["constructor_id"] or original["profile"]!=condition["profile"]:
        raise ValueError("acquired training or profile assignment differs")
    rng=random.Random(seed(namespace,"tradeoff-law",prep["constructor_id"]))
    law={"low":rng.uniform(0.15,0.30),"high":rng.uniform(0.70,0.85),"beta":rng.uniform(4.0,10.0)}
    if law!=original["law"] or len(original["history"])!=8:
        raise ValueError("constructor law or eight dated episodes differs")
    world=prep["world"];old=prep["current_execution"]["state"];initial=list(old)
    visible_history=[]
    search_cost=0;production_cost=0
    for date,event in enumerate(original["history"]):
        ctx=context(namespace,index,date,condition,world,initial)
        if ctx!=event["context"]:
            raise ValueError("chronological actual constraints or purpose differs")
        execution=check_batch(event["batch"],world,ctx,law,condition["profile"],prep["routine"],old,
            random.Random(seed(namespace,"tradeoff-choice",condition["id"],index,date)))
        visible_history.append({"context":ctx,"state":execution["state"]})
        search_cost+=event["batch"]["menu"]["successor_evaluations"]
        production_cost+=sum(item["execution"]["primitive_cost"] for item in event["batch"]["candidates"])
        initial=list(execution["state"]) if row["card_id"]=="V02" else list(old)
    future_context=context(namespace,index,8,condition,world,initial,True)
    expected_public={"schema_version":"v16.tradeoffs.1","task_id":public["task_id"],"world":world,"law":law,
        "training":[{key:trial[key] for key in ["date","program","target","feedback"]} for trial in prep["training"]],
        "history":visible_history,"future_context":future_context}
    if public!=expected_public:
        raise ValueError("public model or dated evidence contains undeclared information")
    execution=check_batch(private["future"],world,future_context,law,condition["profile"],prep["routine"],old,
        random.Random(seed(namespace,"tradeoff-future",condition["id"],index)))
    for policy,arm in row["arms"].items():
        expected=prediction(public,policy)
        actual=final["arms"][policy]
        if expected["model_mismatch"] or actual["model_mismatch"] or actual["states"]!=expected["states"] or actual["costs"]!=expected["costs"]:
            raise ValueError("independent prediction support or computation differs")
        if any(abs(a-b)>1e-11 for a,b in zip(actual["probabilities"],expected["probabilities"])) or any(
                abs(actual["profile_posterior"][key]-value)>1e-11 for key,value in expected["profile_posterior"].items()):
            raise ValueError("independent tradeoff posterior probabilities differ")
        if abs(actual["expected_revision_primitives"]-expected["expected_revision_primitives"])>1e-11:
            raise ValueError("independent revision-cost prediction differs")
        position=expected["states"].index(execution["state"])
        outcomes={"future_choice_log_score":math.log(expected["probabilities"][position]),
            "future_choice_brier":sum((p-float(i==position))**2 for i,p in enumerate(expected["probabilities"])),
            "revision_cost_accuracy":-abs(expected["expected_revision_primitives"]-execution["primitive_cost"]),
            "future_legal":float(execution["legal"] and execution["successfully_stopped"]),
            "future_revision_primitives":float(execution["primitive_cost"]),
            "original_search_primitives":float(search_cost),"original_production_primitives":float(production_cost),
            "future_search_primitives":float(private["future"]["menu"]["successor_evaluations"]),
            "training_primitives":float(sum(trial["execution"]["primitive_cost"] for trial in prep["training"])),
            "old_execution_primitives":float(prep["current_execution"]["primitive_cost"]),"routine_definition_primitives":float(len(prep["routine"])),
            "future_considered_targets":float(len(private["future"]["menu"]["considered_targets"])),
            "future_found_options":float(len(private["future"]["menu"]["options"])),
            "future_search_exhausted":float(private["future"]["menu"]["search_exhausted"])}
        outcomes.update({f"reader_{key}":float(value) for key,value in expected["costs"].items()})
        if set(outcomes)!=set(arm["outcomes"]) or any(abs(arm["outcomes"][key]-value)>1e-10 for key,value in outcomes.items()):
            raise ValueError("independent tradeoff outcomes differ")
    if any(abs(a-b)>1e-12 for a,b in zip(final["arms"]["chronological"]["probabilities"],final["arms"]["direct-chronological"]["probabilities"])):
        raise ValueError("same-information direct joint identity failed")

def audit(root,summary):
    rows=[json.loads(path.read_bytes()) for path in sorted((root/"units").glob("*_points.json"))]
    for row in rows:
        audit_unit(root,row)
    count=verify(rows,summary)
    return {"execution_state":"completed","instrument_state":"valid","n_raw_units":len(rows),
        "reproduced_contrasts":count,"all_reported_aggregates_reproduced":True,"full_rollout_replay":False}
