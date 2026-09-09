"""Independent batch enumeration, physical retention and M02 scoring."""
import hashlib
from itertools import product
import json
import math
import random
from .audit_graphic_maker import seed,verify_maker,verify_production
from .audit_statistics import verify

def load(path,expected):
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected:
        raise ValueError("selection commitment mismatch")
    return json.loads(data)

def style_of(world,event):
    p=world["permutation"];board=event["artifact"]
    occupied={i for i in range(16) if (board//(2**i))%2}
    for style in [0,1]:
        base=set(p[:2])|set(p[4+2*style:6+2*style])
        if occupied in [base|{p[2]},base|{p[3]}]:
            return style
    raise ValueError("unknown independently interpreted graphic composition")

def order_probability(world,event,core):
    if event["first_action"] is None:
        return 1.0
    order=world["permutation"][:2].index(event["first_action"])
    return world["core_reuse"] if core==order else 1-world["core_reuse"]

def style_probability(world,style,acquired):
    return world["style_reuse"] if style==acquired else 1-world["style_reuse"]

def selected_probability(world,style,acquired,audience,n):
    total=0.0
    for candidate_styles in product([0,1],repeat=n):
        weight=math.prod(style_probability(world,item,acquired) for item in candidate_styles)
        position=next((i for i,item in enumerate(candidate_styles) if item==audience),0)
        if candidate_styles[position]==style:
            total+=weight
    return total

def independent_prediction(public,method):
    world=public["world"];rule=public["retention"];n=public["produced_count"]
    audiences=[0,1] if public["audience"] is None else [public["audience"]]
    entries=[];terms=0
    for core,acquired,audience in product([0,1],[0,1],audiences):
        weight=1/(4*len(audiences))
        if method!="population":
            for batch in public["history"]:
                full=batch["full_candidates"]
                if full is not None and method!="release-naive":
                    for item in full:
                        event=item["observation"]
                        weight*=style_probability(world,style_of(world,event),acquired)*order_probability(world,event,core)
                        terms+=1+int(event["first_action"] is not None)
                    kept=[i for i,item in enumerate(full) if item["retained"]]
                    expected=list(range(n)) if rule=="all" else (
                        [next((i for i,item in enumerate(full) if style_of(world,item["observation"])==audience),0)] if rule=="selected" else kept)
                    terms+=1
                    if kept!=expected:
                        weight=0.0
                        break
                    if rule=="random":
                        weight/=n
                else:
                    for event in batch["released"]:
                        observed=style_of(world,event)
                        if rule=="selected" and method!="release-naive":
                            weight*=selected_probability(world,observed,acquired,audience,n)
                        else:
                            weight*=style_probability(world,observed,acquired)
                        weight*=order_probability(world,event,core)
                        terms+=1+int(event["first_action"] is not None)
        entries.append((core,acquired,audience,weight))
    evidence=sum(weight for _,_,_,weight in entries)
    output={key:[0.0,0.0] for key in ["acquired_core","acquired_style","audience","future_raw_core","future_raw_style","future_release_style"]}
    for core,acquired,audience,weight in entries:
        weight/=evidence
        for key,bit in [("acquired_core",core),("acquired_style",acquired),("audience",audience)]:
            output[key][bit]+=weight
        for bit in [0,1]:
            output["future_raw_core"][bit]+=weight*(world["core_reuse"] if bit==core else 1-world["core_reuse"])
            raw=style_probability(world,bit,acquired)
            released=selected_probability(world,bit,acquired,audience,n) if rule=="selected" and method!="release-naive" else raw
            output["future_raw_style"][bit]+=weight*raw
            output["future_release_style"][bit]+=weight*released
    output["audience_choice"]=0 if output["audience"][0]>=output["audience"][1]-1e-12 else 1
    output["costs"]={"likelihood_terms":terms,"hypotheses":len(entries),"prediction_terms":6*len(entries)}
    return output

def audit_unit(root,row):
    uid=row["unit_id"]
    public=load(root/"public"/f"{uid}.json",row["public_hash"])
    private=load(root/"private"/f"{uid}.json",row["private_hash"])
    predictions=load(root/"predictions"/f"{uid}.json",row["prediction_hash"])
    if predictions["public_hash"]!=row["public_hash"]:
        raise ValueError("predictions refer to another public observation")
    world,maker=private["world"],private["maker"]
    condition=row["condition_spec"];namespace,index=row["lineage"],row["seed_components"]["index"]
    if public["world"]!={key:world[key] for key in ["permutation","style_reuse","core_reuse"]} or public["audience"]!=private["audience"]:
        raise ValueError("incorrect allowed context")
    if public["retention"]!=condition["retention"] or public["produced_count"]!=condition["count"]:
        raise ValueError("incorrect selection contract")
    audience=random.Random(seed(namespace,"audience",index)).randrange(2)
    if audience!=private["audience"]:
        raise ValueError("audience was not sampled independently")
    training,definitions=verify_maker(world,maker,namespace,index)
    def check_batch(batch,date,future=False):
        label="future-batch" if future else "batch"
        if len(batch["candidates"])!=condition["count"]:
            raise ValueError("produced work was discarded from the private record")
        cost=0
        for position,event in enumerate(batch["candidates"]):
            cost+=verify_production(world,event,maker,random.Random(seed(namespace,label,index,date,position)),date%2)
        rule=condition["retention"]
        if rule=="all":
            kept=list(range(condition["count"]))
        elif rule=="random":
            kept=[random.Random(seed(namespace,label+"-selection",index,date)).randrange(condition["count"])]
        else:
            kept=[next((i for i,event in enumerate(batch["candidates"]) if event["decoration"]==audience),0)]
        if kept!=batch["retained_indices"]:
            raise ValueError("retention decision differs from actual candidates")
        return cost
    if len(private["batches"])!=3 or len(public["history"])!=3:
        raise ValueError("incomplete production history")
    produced=sum(check_batch(batch,date) for date,batch in enumerate(private["batches"]))
    for batch,visible in zip(private["batches"],public["history"]):
        observations=[{"artifact":event["execution"]["artifact"],"first_action":event["program"][0] if condition["process"] else None}
                      for event in batch["candidates"]]
        kept=batch["retained_indices"]
        expected={"released":[observations[i] for i in kept],
            "full_candidates":[{"observation":event,"retained":i in kept} for i,event in enumerate(observations)] if condition["rejected"] else None}
        if visible!=expected:
            raise ValueError("paid candidate or process evidence differs")
    raw=private["raw_future"];release=private["release_future"]
    future_cost=verify_production(world,raw,maker,random.Random(seed(namespace,"raw-future",index)),1)+check_batch(release,0,True)
    retained_count=sum(len(batch["retained_indices"]) for batch in private["batches"])
    rejected_count=3*condition["count"]-retained_count
    visible_count=3*condition["count"] if condition["rejected"] else retained_count
    actual_style_one=world["style_reuse"] if maker["decoration"]["choice"]==1 else 1-world["style_reuse"]
    for method,arm in row["arms"].items():
        predicted=predictions["arms"][method]
        independent=independent_prediction(public,method)
        for key in ["acquired_core","acquired_style","audience","future_raw_core","future_raw_style","future_release_style"]:
            if any(abs(a-b)>1e-12 for a,b in zip(predicted[key],independent[key])):
                raise ValueError("independent finite selection prediction differs")
        if predicted["costs"]!=independent["costs"] or predicted["audience_choice"]!=independent["audience_choice"]:
            raise ValueError("selection inference decision or reference work differs")
        expected={
            "future_raw_style_log_score":math.log(predicted["future_raw_style"][raw["decoration"]]),
            "future_raw_core_log_score":math.log(predicted["future_raw_core"][raw["core_order"]]),
            "future_release_style_log_score":sum(math.log(predicted["future_release_style"][release["candidates"][i]["decoration"]])
                                               for i in release["retained_indices"])/len(release["retained_indices"]),
            "acquired_style_log_score":math.log(predicted["acquired_style"][maker["decoration"]["choice"]]),
            "acquired_core_log_score":math.log(predicted["acquired_core"][maker["core"]["choice"]]),
            "raw_style_absolute_probability_error":abs(predicted["future_raw_style"][1]-actual_style_one),
            "training_primitives":float(training),"library_definition_cost":float(definitions),
            "observed_production_primitives":float(produced),"future_production_primitives":float(future_cost),
            "retained_works":float(retained_count),"rejected_works":float(rejected_count),
            "paid_rejected_work_observations":float(rejected_count if condition["rejected"] else 0),
            "process_observations":float(visible_count if condition["process"] else 0),"audience_observations":1.0,
            "likelihood_terms":float(predicted["costs"]["likelihood_terms"]),"prediction_terms":float(predicted["costs"]["prediction_terms"])}
        if expected!=arm["outcomes"]:
            raise ValueError("independent selection outcome differs")

def audit(root,summary):
    rows=[json.loads(path.read_bytes()) for path in sorted((root/"units").glob("*_points.json"))]
    for row in rows:
        audit_unit(root,row)
    count=verify(rows,summary)
    return {"execution_state":"completed","instrument_state":"valid","n_raw_units":len(rows),
            "reproduced_contrasts":count,"all_reported_aggregates_reproduced":True,"full_rollout_replay":False}

