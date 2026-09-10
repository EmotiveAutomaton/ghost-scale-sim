"""Independent M03 paid-access joins, physical continuation and historical scores."""
import hashlib
from itertools import product
import json
import math
import random
from .audit_graphic_maker import seed,verify_maker,verify_production
from .audit_selection import style_of,style_probability,selected_probability,order_probability
from .graphic_reference import interpret
from .audit_statistics import verify

def load(path,expected):
    payload=path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected:
        raise ValueError("audience commitment hash differs")
    return json.loads(payload)

def independent_prediction(public):
    world=public["world"]
    audiences=[0,1] if public["audience"] is None else [public["audience"]]
    entries=[]
    for core,style,audience in product([0,1],[0,1],audiences):
        weight=1/(4*len(audiences))
        for batch in public["history"]:
            event=batch["released"][0]
            decoration=style_of(world,event)
            weight*=selected_probability(world,decoration,style,audience,public["produced_count"]) if public["retention"]=="selected" else style_probability(world,decoration,style)
        for event in public["rehearsals"]:
            weight*=style_probability(world,style_of(world,event),style)*order_probability(world,event,core)
        entries.append((core,style,audience,weight))
    evidence=sum(entry[3] for entry in entries)
    values={key:[0.0,0.0] for key in ["acquired_core","acquired_style","audience","future_core","future_style"]}
    values["historical_vector"]=[0.0]*8
    for core,style,audience,weight in entries:
        weight/=evidence
        for key,bit in [("acquired_core",core),("acquired_style",style),("audience",audience)]:
            values[key][bit]+=weight
        for bit in [0,1]:
            values["future_core"][bit]+=weight*(world["core_reuse"] if bit==core else 1-world["core_reuse"])
            values["future_style"][bit]+=weight*style_probability(world,bit,style)
        for index,trajectory in enumerate(product([0,1],repeat=3)):
            values["historical_vector"][index]+=weight*math.prod(world["core_reuse"] if bit==core else 1-world["core_reuse"] for bit in trajectory)
    values["historical_core"]=list(values["future_core"])
    choice=0 if values["audience"][0]>=values["audience"][1]-1e-12 else 1
    p=world["permutation"]
    values["new_composition"]=p[:2]+[p[3]]+p[4+2*choice:6+2*choice]
    values["audience_choice"]=choice
    values["costs"]={"likelihood_terms":len(entries)*(3+2*len(public["rehearsals"])),
                     "hypotheses":len(entries),"prediction_terms":28*len(entries)}
    return values

def audit_unit(root,row):
    uid=row["unit_id"]
    public=load(root/"public"/f"{uid}.json",row["public_hash"])
    private=load(root/"private"/f"{uid}.json",row["private_hash"])
    original=load(root/"private"/f"{uid}-original.json",private["original_hash"])
    final=load(root/"predictions"/f"{uid}.json",row["prediction_hash"])
    first=load(root/"predictions"/f"{uid}-phase-1.json",final["phase_one_hash"])
    if first["public_hash"]!=row["public_hash"] or first["submitted_at"]>final["submitted_at"]:
        raise ValueError("query commitment does not precede prediction submission")
    world,maker,audience=original["world"],original["maker"],original["audience"]
    namespace,index=row["lineage"],row["seed_components"]["index"]
    condition=row["condition_spec"]
    if public["world"]!={key:world[key] for key in ["permutation","style_reuse","core_reuse"]} or public["audience"] is not None or public["rehearsals"]:
        raise ValueError("private answer appeared before query commitment")
    if public["retention"]!=condition["retention"] or public["produced_count"]!=condition["count"] or public["target_topic"]!=1:
        raise ValueError("public production and reconstruction contract differs")
    if audience!=random.Random(seed(namespace,"audience",index)).randrange(2):
        raise ValueError("audience was not sampled independently")
    training,definitions=verify_maker(world,maker,namespace,index)
    original_cost=0
    old_orders=[]
    if len(original["batches"])!=3 or len(public["history"])!=3:
        raise ValueError("original history is incomplete")
    for date,(batch,visible) in enumerate(zip(original["batches"],public["history"])):
        if len(batch["candidates"])!=condition["count"]:
            raise ValueError("original rejected work was discarded")
        for position,event in enumerate(batch["candidates"]):
            original_cost+=verify_production(world,event,maker,random.Random(seed(namespace,"batch",index,date,position)),date%2)
        if condition["retention"]=="random":
            position=random.Random(seed(namespace,"batch-selection",index,date)).randrange(condition["count"])
        else:
            position=next((i for i,event in enumerate(batch["candidates"]) if event["decoration"]==audience),0)
        if batch["retained_indices"]!=[position]:
            raise ValueError("original publication did not execute its actual retention rule")
        event=batch["candidates"][position]
        if visible!={"released":[{"artifact":event["execution"]["artifact"],"first_action":None}],"full_candidates":None}:
            raise ValueError("original public artifact differs")
        old_orders.append(event["core_order"])
    rehearsal_cost=0
    if len(private["rehearsals"])!=2:
        raise ValueError("rehearsals are not actual complete new work")
    for date,event in enumerate(private["rehearsals"]):
        rehearsal_cost+=verify_production(world,event,maker,random.Random(seed(namespace,"audience-rehearsal",index,date)),1)
    future=private["future"]
    future_cost=verify_production(world,future,maker,random.Random(seed(namespace,"audience-future",index)),1)
    history_index=sum(order*(2**(2-date)) for date,order in enumerate(old_orders))
    p=world["permutation"]
    targets=[sum(2**cell for cell in p[:2]+[p[3]]+p[4+2*style:6+2*style]) for style in [0,1]]
    for policy,arm in row["arms"].items():
        expected_queries={"audience":policy in {"audience-only","both","direct-both"},
                          "rehearsals":2 if policy in {"rehearsal-only","both","direct-both"} else 0}
        if first["arms"][policy]!={"queries":expected_queries}:
            raise ValueError("paid request differs from the registered policy")
        expected_public={**public,"phase":2,"audience":audience if expected_queries["audience"] else None,
            "rehearsals":[{"artifact":event["execution"]["artifact"],"first_action":event["program"][0]} for event in private["rehearsals"]] if expected_queries["rehearsals"] else []}
        visible=final["public_inputs"][policy]
        if visible!=expected_public:
            raise ValueError("reader received unpurchased or incorrect evidence")
        prediction=final["arms"][policy]
        independent=independent_prediction(visible)
        for key in ["acquired_core","acquired_style","audience","future_core","future_style","historical_core","historical_vector"]:
            if any(abs(a-b)>1e-12 for a,b in zip(prediction[key],independent[key])):
                raise ValueError("audience historical/predictive probabilities differ independently")
        if any(prediction[key]!=independent[key] for key in ["new_composition","audience_choice","costs"]):
            raise ValueError("audience construction decision or work counts differ")
        execution=interpret(prediction["new_composition"])
        if arm["construction_execution"]!=execution:
            raise ValueError("new reader composition was not actually executed")
        legal=execution["legal"] and execution["stopped"]
        expected={
            "audience_success":float(legal and execution["artifact"]==targets[audience]),
            "brief_success":float(legal and execution["artifact"] in targets),"legal":float(legal),
            "historical_core_log_score":math.log(prediction["historical_core"][old_orders[0]]),
            "historical_vector_log_score":math.log(prediction["historical_vector"][history_index]),
            "acquired_core_log_score":math.log(prediction["acquired_core"][maker["core"]["choice"]]),
            "future_core_log_score":math.log(prediction["future_core"][future["core_order"]]),
            "future_style_log_score":math.log(prediction["future_style"][future["decoration"]]),
            "audience_log_score":math.log(prediction["audience"][audience]),
            "audience_queries":float(expected_queries["audience"]),"rehearsal_observations":float(expected_queries["rehearsals"]),
            "rehearsal_production_primitives":float(rehearsal_cost if expected_queries["rehearsals"] else 0),
            "training_primitives":float(training),"library_definition_cost":float(definitions),
            "original_production_primitives":float(original_cost),"future_production_primitives":float(future_cost),
            "reader_construction_primitives":float(execution["primitive_cost"]),
            "likelihood_terms":float(prediction["costs"]["likelihood_terms"]),
            "prediction_terms":float(prediction["costs"]["prediction_terms"])}
        if arm["outcomes"]!=expected:
            raise ValueError("independent audience outcomes differ")

def audit(root,summary):
    rows=[json.loads(path.read_bytes()) for path in sorted((root/"units").glob("*_points.json"))]
    for row in rows:
        audit_unit(root,row)
    count=verify(rows,summary)
    return {"execution_state":"completed","instrument_state":"valid","n_raw_units":len(rows),
            "reproduced_contrasts":count,"all_reported_aggregates_reproduced":True,"full_rollout_replay":False}
