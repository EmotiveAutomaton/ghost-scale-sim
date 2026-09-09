"""M02 actual retained/rejected batches and finite selection-aware inference."""
from itertools import product
import json
from .recognition import decode,probability

METHODS=["selection-aware","direct-table","release-naive","population"]
EXTENSION="ghostscale.validation.soundingline.v16.selection:public_reader"
DESIGN={
 "card_id":"M02","question":"Can selective publication create a personal-looking signature and bias prediction of the maker's unselected work?",
 "mechanism":"an editor chooses the first audience-matching artifact from an actually executed finite batch, or its first member if none matches",
 "strongest_rival":"direct joint observable prediction with the same release law, production count, audience and permitted rejected-work evidence",
 "access_arms":"identical serialized inputs within each paid evidence condition; naive release model ignores selection and rejected works; population ignores history",
 "target_realization":"makers acquire independent core/decorative routines from real trials; all candidates execute before any retention decision",
 "conditions":[{"id":f"{retention}-n-{count}-rejected-{rejected}-process-{process}","retention":retention,
                "count":count,"rejected":rejected,"process":process}
               for retention in ["all","random","selected"] for count in [1,4] for rejected in [False,True] for process in [False,True]],
 "arms":METHODS,
 "primary":[("selection-aware","release-naive","future_raw_style_log_score","nats_per_event",0.02),
            ("selection-aware","direct-table","future_raw_style_log_score","nats_per_event",0.02)],
 "secondary":["next selected release score","raw core-order score","inferred acquired style","raw style probability error",
              "retained and rejected counts","actual training/production costs","paid observation costs"],
 "generator_families":["W1 sixteen-cell acquired-routine production with finite first-match selection"],
 "paired_unit":"one independently acquired maker, three actual production batches and fresh raw/release predictions",
 "sample_rule":{"scout":64,"constructors":8,"expansion":256,"expansion_constructors":20,"final_expansion":1024},
 "dependencies":["large-graphic-physics","large-graphic-acquisition","selection-law","selection-information","independent-selection-scoring"],
 "adversaries":["X01","X03","X04","X06","X07","X08"],"repair_budget":1,
 "continuation":"expand a named selection-versus-production ambiguity; apparent signature explained by retention is a boundary, not erased",
 "coverage_limit":"known finite production and selection catalog; does not identify arbitrary real editors",
 "cost_contract":"paired arms receive identical public bytes; count all produced candidates, including rejected ones, and charge extra evidence"
}

def retained(events,retention,audience,rng):
    if retention=="all":
        return list(range(len(events)))
    if retention=="random":
        return [rng.randrange(len(events))]
    if retention=="selected":
        return [next((i for i,event in enumerate(events) if event["decoration"]==audience),0)]
    raise ValueError("unknown retention rule")

def selection_style_probability(style,acquired,audience,count,world):
    matching=world["style_reuse"] if audience==acquired else 1-world["style_reuse"]
    return 1-(1-matching)**count if style==audience else (1-matching)**count

def validate(public):
    if set(public)!={"schema_version","task_id","world","history","retention","produced_count","audience"} or public["schema_version"]!="v16.selection.1":
        raise ValueError("selection public schema violation")
    world=public["world"]
    if set(world)!={"permutation","style_reuse","core_reuse"} or sorted(world["permutation"])!=list(range(16)):
        raise ValueError("invalid selection world metadata")
    if any(type(i) is not int for i in world["permutation"]) or any(not 0<world[key]<1 for key in ["style_reuse","core_reuse"]):
        raise ValueError("invalid graphic interface or production probability")
    if public["retention"] not in {"all","random","selected"} or type(public["produced_count"]) is not int or not 1<=public["produced_count"]<=8:
        raise ValueError("unregistered selection law or production budget")
    if public["audience"] is not None and (type(public["audience"]) is not int or public["audience"] not in [0,1]):
        raise ValueError("invalid audience observation")
    for batch in public["history"]:
        if set(batch)!={"released","full_candidates"}:
            raise ValueError("undeclared batch field")
        expected_count=public["produced_count"] if public["retention"]=="all" else 1
        if len(batch["released"])!=expected_count:
            raise ValueError("wrong released-work count")
        for item in batch["released"]:
            decode(world,item)
        full=batch["full_candidates"]
        if full is not None:
            if len(full)!=public["produced_count"] or any(set(item)!={"observation","retained"} or type(item["retained"]) is not bool for item in full):
                raise ValueError("invalid paid candidate evidence")
            if [item["observation"] for item in full if item["retained"]]!=batch["released"]:
                raise ValueError("candidate and release evidence disagree")
            for item in full:
                decode(world,item["observation"])

def likelihood(public,core,style,audience,*,naive=False):
    world=public["world"];rule=public["retention"];n=public["produced_count"]
    weight=1.0;terms=0
    for batch in public["history"]:
        full=batch["full_candidates"]
        if full is not None and not naive:
            observations=[item["observation"] for item in full]
            value,cost=probability(observations,(core,style),world)
            weight*=value;terms+=cost
            kept=[i for i,item in enumerate(full) if item["retained"]]
            if rule=="all":
                expected=list(range(n))
            elif rule=="selected":
                expected=[next((i for i,item in enumerate(full) if decode(world,item["observation"])[0]==audience),0)]
            else:
                expected=kept
                weight/=n
            if kept!=expected:
                return 0.0,terms+1
            terms+=1
        elif rule!="selected" or naive:
            value,cost=probability(batch["released"],(core,style),world)
            weight*=value;terms+=cost
        else:
            decoration,order=decode(world,batch["released"][0])
            weight*=selection_style_probability(decoration,style,audience,n,world)
            terms+=1
            if order is not None:
                weight*=world["core_reuse"] if order==core else 1-world["core_reuse"]
                terms+=1
    return weight,terms

def infer(public,method):
    validate(public)
    if method not in METHODS:
        raise ValueError("unknown selection reader")
    audience_values=[0,1] if public["audience"] is None else [public["audience"]]
    entries=[];terms=0
    for core,style,audience in product([0,1],[0,1],audience_values):
        if method=="population":
            value,cost=1.0,0
        else:
            value,cost=likelihood(public,core,style,audience,naive=method=="release-naive")
        entries.append((core,style,audience,value/(4*len(audience_values))))
        terms+=cost
    evidence=sum(entry[3] for entry in entries)
    if evidence<=0:
        raise ValueError("selection history has zero model support")
    output={key:[0.0,0.0] for key in ["acquired_core","acquired_style","audience","future_raw_core","future_raw_style","future_release_style"]}
    world=public["world"]
    for core,style,audience,weight in entries:
        mass=weight if method=="direct-table" else weight/evidence
        for key,bit in [("acquired_core",core),("acquired_style",style),("audience",audience)]:
            output[key][bit]+=mass
        for bit in [0,1]:
            output["future_raw_core"][bit]+=mass*(world["core_reuse"] if bit==core else 1-world["core_reuse"])
            raw=world["style_reuse"] if bit==style else 1-world["style_reuse"]
            released=selection_style_probability(bit,style,audience,public["produced_count"],world) if (
                public["retention"]=="selected" and method!="release-naive") else raw
            output["future_raw_style"][bit]+=mass*raw
            output["future_release_style"][bit]+=mass*released
    if method=="direct-table":
        for key in output:
            output[key]=[value/evidence for value in output[key]]
    output["audience_choice"]=0 if output["audience"][0]>=output["audience"][1]-1e-12 else 1
    output["costs"]={"likelihood_terms":terms,"hypotheses":len(entries),"prediction_terms":6*len(entries)}
    return output

def public_reader(payload:bytes,method):
    return infer(json.loads(payload),method)

