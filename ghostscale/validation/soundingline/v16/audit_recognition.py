"""Independent larger-graphic training, maker execution, prediction and scoring audit."""
from collections import Counter
import hashlib
from itertools import product
import json
import math
import random
from .graphic_reference import interpret
from .audit_statistics import verify

def seed(*parts):
    return int(hashlib.sha256(json.dumps(parts,sort_keys=True,separators=(",",":")).encode()).hexdigest()[:16],16)

def load(path,expected):
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected:
        raise ValueError("recognition commitment hash mismatch")
    return json.loads(data)

def independent_prediction(public,method):
    world=public["world"];p=world["permutation"]
    refs=public["references"] if method!="no-reference" else [[],[]]
    process=method!="surface-only"
    decoded=[]
    for group in refs+[public["anonymous"]]:
        observations=[]
        for event in group:
            occupied={i for i in range(16) if (event["artifact"]//(2**i))%2}
            style=0 if set(p[4:6])<=occupied else 1
            expected=set(p[:2])|set(p[4+2*style:6+2*style])
            if occupied not in [expected|{p[2]},expected|{p[3]}]:
                raise ValueError("observation outside independent graphic catalog")
            order=None if event["first_action"] is None else p[:2].index(event["first_action"])
            observations.append((style,order))
        decoded.append(observations)
    entries=[]
    for identity in [0,1]:
        for core0,style0,core1,style1 in product([0,1],repeat=4):
            states=[(core0,style0),(core1,style1)]
            probability=1/32
            for source,observations in zip([0,1,identity],decoded):
                core,style=states[source]
                for observed_style,observed_order in observations:
                    probability*=world["style_reuse"] if style==observed_style else 1-world["style_reuse"]
                    if process and observed_order is not None:
                        probability*=world["core_reuse"] if core==observed_order else 1-world["core_reuse"]
            entries.append((identity,states[identity],probability))
    normalizer=sum(value for _,_,value in entries)
    values={key:[0.0,0.0] for key in ["identity","acquired_core","acquired_decoration","future_core","future_decoration"]}
    for identity,(core,style),weight in entries:
        weight/=normalizer
        values["identity"][identity]+=weight
        values["acquired_core"][core]+=weight
        values["acquired_decoration"][style]+=weight
        for bit in [0,1]:
            values["future_core"][bit]+=weight*(world["core_reuse"] if bit==core else 1-world["core_reuse"])
            values["future_decoration"][bit]+=weight*(world["style_reuse"] if bit==style else 1-world["style_reuse"])
    order=decoded[-1][0][1] if process else None
    values["historical_core"]=list(values["future_core"]) if order is None else [float(bit==order) for bit in [0,1]]
    values["identity_choice"]=0 if values["identity"][0]>=values["identity"][1]-1e-12 else 1
    terms=32*sum(1+int(process and order is not None) for group in decoded for _,order in group)
    values["costs"]={"likelihood_terms":terms,"hypotheses":32,"prediction_terms":128}
    return values

def audit_unit(root,row):
    uid=row["unit_id"]
    public=load(root/"public"/f"{uid}.json",row["public_hash"])
    private=load(root/"private"/f"{uid}.json",row["private_hash"])
    predictions=load(root/"predictions"/f"{uid}.json",row["prediction_hash"])
    if predictions["public_hash"]!=row["public_hash"]:
        raise ValueError("reader commitment names another observation")
    world=private["world"];p=world["permutation"]
    if public["world"]!={key:world[key] for key in ["permutation","core_reuse","style_reuse"]}:
        raise ValueError("unpermitted or incorrect public world information")
    namespace,index=row["lineage"],row["seed_components"]["index"]
    source=random.Random(seed(namespace,"source",index)).randrange(2)
    if source!=private["source"]:
        raise ValueError("source identity was not independently sampled")
    training_cost=0;definition_cost=0
    for maker_id,maker in enumerate(private["makers"]):
        for kind,part in maker.items():
            options=[p[:2],list(reversed(p[:2]))] if kind=="core" else [p[4:6],p[6:8]]
            rng=random.Random(seed(namespace,"training",index,maker_id,kind))
            direction=rng.randrange(2)
            counts=Counter()
            for date,trial in enumerate(part["training"]):
                choice=direction if rng.random()<world["training_reliability"] else 1-direction
                result=interpret(options[choice])
                if trial!={"date":date,"program":options[choice],"target":result["artifact"],"feedback":result["legal"],"execution":result}:
                    raise ValueError("recorded training is not actual sampled primitive execution")
                counts[tuple(options[choice])]+=1
                training_cost+=result["primitive_cost"]
            eligible=[program for program,count in counts.items() if count>=4]
            routine=list(min(eligible,key=lambda program:(-counts[program],program)))
            expected={"library":[routine],"definition_cost":2,"processing_primitives":14}
            if part["routine"]!=routine or part["compiled"]!=expected or part["choice"]!=options.index(routine) or part["teaching_direction"]!=direction:
                raise ValueError("maker routine was not acquired from its own training")
            definition_cost+=2
    def check(event,maker,rng,topic,changed=False):
        core=maker["core"]["choice"]
        style=maker["decoration"]["choice"]
        core=core if rng.random()<world["core_reuse"] else 1-core
        style=1-style if changed else style
        style=style if rng.random()<world["style_reuse"] else 1-style
        core_program=p[:2] if core==0 else list(reversed(p[:2]))
        decor_program=p[4+2*style:6+2*style]
        program=core_program+[p[2+topic]]+decor_program
        result=interpret(program)
        expected={"program":program,"execution":result,"core_order":core,"decoration":style,"topic":topic,
                  "core_routine_used":core_program==maker["core"]["routine"],
                  "decoration_routine_used":decor_program==maker["decoration"]["routine"]}
        if event!=expected:
            raise ValueError("actual sampled graphic production did not regenerate")
        return result["primitive_cost"]
    observed_cost=0
    condition=row["condition_spec"]
    for maker,events in enumerate(private["references"]):
        for date,event in enumerate(events):
            observed_cost+=check(event,private["makers"][maker],random.Random(seed(namespace,"reference",index,maker,date)),0)
    for date,event in enumerate(private["anonymous"]):
        observed_cost+=check(event,private["makers"][source],random.Random(seed(namespace,"anonymous",index,date)),
                             int(condition["topic"]=="new-topic"),condition["surface"]=="changed-purpose")
    future=private["future"]
    future_cost=check(future,private["makers"][source],random.Random(seed(namespace,"future",index)),1)
    for actual_group,visible_group in zip(private["references"]+[private["anonymous"]],public["references"]+[public["anonymous"]]):
        for date,(actual,visible) in enumerate(zip(actual_group,visible_group)):
            if visible!={"artifact":actual["execution"]["artifact"],"first_action":actual["program"][0] if date<condition["dose"] else None}:
                raise ValueError("unpermitted observation or wrong evidence dose")
    for method,arm in row["arms"].items():
        predicted=predictions["arms"][method]
        independent=independent_prediction(public,method)
        for key in ["identity","acquired_core","acquired_decoration","future_core","future_decoration","historical_core"]:
            if any(abs(a-b)>1e-12 for a,b in zip(predicted[key],independent[key])):
                raise ValueError("independent joint predictive calculation differs")
        if predicted["costs"]!=independent["costs"]:
            raise ValueError("reader work accounting differs")
        # A predeclared numerical tie at 1e-12 uses source zero consistently.
        if predicted["identity_choice"]!=(0 if predicted["identity"][0]>=predicted["identity"][1]-1e-12 else 1):
            raise ValueError("identity decision differs from submitted posterior")
        expected={
            "identity_accuracy":float(predicted["identity_choice"]==source),
            "identity_log_score":math.log(predicted["identity"][source]),
            "future_core_log_score":math.log(predicted["future_core"][future["core_order"]]),
            "future_decoration_log_score":math.log(predicted["future_decoration"][future["decoration"]]),
            "historical_core_log_score":math.log(predicted["historical_core"][private["anonymous"][0]["core_order"]]),
            "acquired_core_log_score":math.log(predicted["acquired_core"][private["makers"][source]["core"]["choice"]]),
            "acquired_decoration_log_score":math.log(predicted["acquired_decoration"][private["makers"][source]["decoration"]["choice"]]),
            "training_primitives":float(training_cost),"library_definition_cost":float(definition_cost),
            "observed_production_primitives":float(observed_cost),"future_production_primitives":float(future_cost),
            "process_queries":float(3*condition["dose"]),"likelihood_terms":float(predicted["costs"]["likelihood_terms"]),
            "prediction_terms":float(predicted["costs"]["prediction_terms"])}
        if expected!=arm["outcomes"]:
            raise ValueError("independent recognition outcome differs")

def audit(root,summary):
    rows=[json.loads(path.read_bytes()) for path in sorted((root/"units").glob("*_points.json"))]
    for row in rows:
        audit_unit(root,row)
    contrasts=verify(rows,summary)
    return {"execution_state":"completed","instrument_state":"valid","n_raw_units":len(rows),
            "reproduced_contrasts":contrasts,"all_reported_aggregates_reproduced":True,"full_rollout_replay":False}
