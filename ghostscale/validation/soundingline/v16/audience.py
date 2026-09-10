"""M03 paid audience/rehearsal access and executable audience-facing reconstruction."""
from itertools import product
import json
from .selection import validate,likelihood
from .recognition import probability

POLICIES=["none","audience-only","rehearsal-only","both","direct-both"]
EXTENSION="ghostscale.validation.soundingline.v16.audience:public_reader"
DESIGN={
 "card_id":"M03","question":"Can audience knowledge change a useful reconstruction while the original production history stays fixed?",
 "mechanism":"paid audience context and actual unselected rehearsals update different parts of a finite acquired-craft account",
 "strongest_rival":"direct joint prediction and construction from exactly the same purchased observations",
 "access_arms":"paired audience/rehearsal information interventions with identical opportunities and explicit prices; direct-both and both have identical information",
 "target_realization":"original works execute and are committed before requests; paid rehearsals execute after requests; new reader composition executes after submission",
 "conditions":[{"id":f"{family}-{retention}-n-{count}","family":family,"retention":retention,"count":count}
               for family in ["learned-order","goal-indifferent"] for retention in ["random","selected"] for count in [1,4]],
 "arms":POLICIES,
 "primary":[("both","rehearsal-only","audience_success","success_fraction",0.05),
            ("both","audience-only","historical_core_log_score","nats_per_event",0.02),
            ("both","direct-both","audience_success","success_fraction",0.05)],
 "secondary":["joint retained core-order history score","acquired core score","fresh raw maker prediction",
              "audience-only versus none historical independence null","legal new composition","all source/reader/query costs"],
 "generator_families":["W1 bounded acquired-routine production with actual random/selected publication"],
 "paired_unit":"one acquired maker, fixed three-work release history, paired paid probes and reader constructions",
 "sample_rule":{"scout":64,"constructors":8,"expansion":256,"expansion_constructors":20,"final_expansion":1024},
 "dependencies":["audience-history-independence","audience-paid-queries","audience-construction","independent-audience-scoring"],
 "adversaries":["X01","X03","X04","X05","X07","X08"],"repair_budget":1,
 "continuation":"expand an audience-versus-history distinction or a paid rehearsal boundary; retained-history ambiguity remains explicit",
 "history_target":"core-order vector of three retained works, not all rejected candidates or the exact original training sequence",
 "value_boundary":"audience style requirement is a supplied task target, not a recovered human value",
 "cost_contract":"audience description costs one query; two rehearsals cost two observations plus actual maker execution; no equal-access claim across information ablations"
}

def requests(policy):
    return {"audience":policy in {"audience-only","both","direct-both"},
            "rehearsals":2 if policy in {"rehearsal-only","both","direct-both"} else 0}

def base_observation(public,audience):
    return {"schema_version":"v16.selection.1","task_id":public["task_id"],"world":public["world"],
            "history":public["history"],"retention":public["retention"],"produced_count":public["produced_count"],"audience":audience}

def public_reader(payload:bytes,policy):
    public=json.loads(payload)
    if set(public)!={"schema_version","task_id","phase","world","history","retention","produced_count","audience","rehearsals","target_topic"} or public["schema_version"]!="v16.audience.1":
        raise ValueError("audience public schema violation")
    if policy not in POLICIES or public["target_topic"]!=1:
        raise ValueError("unknown audience policy or composition brief")
    validate(base_observation(public,public["audience"]))
    if len(public["history"])!=3 or public["retention"] not in {"random","selected"} or any(
        batch["full_candidates"] is not None or any(event["first_action"] is not None for event in batch["released"])
        for batch in public["history"]):
        raise ValueError("audience study requires three artifact-only retained works")
    desired=requests(policy)
    if public["phase"]==1:
        if public["audience"] is not None or public["rehearsals"]:
            raise ValueError("answers appeared before paid requests")
        return {"queries":desired}
    if public["phase"]!=2 or len(public["rehearsals"])!=desired["rehearsals"]:
        raise ValueError("missing or unpurchased rehearsal evidence")
    if (public["audience"] is not None)!=desired["audience"]:
        raise ValueError("missing or unpurchased audience evidence")
    if any(event.get("first_action") is None for event in public["rehearsals"]):
        raise ValueError("rehearsal process observation is incomplete")
    audiences=[0,1] if public["audience"] is None else [public["audience"]]
    entries=[];terms=0
    for core,style,audience in product([0,1],[0,1],audiences):
        past,cost=likelihood(base_observation(public,audience),core,style,audience)
        rehearsal,rehearsal_cost=probability(public["rehearsals"],(core,style),public["world"])
        terms+=cost+rehearsal_cost
        entries.append((core,style,audience,past*rehearsal/(4*len(audiences))))
    evidence=sum(entry[3] for entry in entries)
    if evidence<=0:
        raise ValueError("audience evidence has zero model support")
    output={key:[0.0,0.0] for key in ["acquired_core","acquired_style","audience","future_core","future_style"]}
    output["historical_vector"]=[0.0]*8
    world=public["world"]
    for core,style,audience,weight in entries:
        mass=weight if policy=="direct-both" else weight/evidence
        for key,bit in [("acquired_core",core),("acquired_style",style),("audience",audience)]:
            output[key][bit]+=mass
        for bit in [0,1]:
            output["future_core"][bit]+=mass*(world["core_reuse"] if bit==core else 1-world["core_reuse"])
            output["future_style"][bit]+=mass*(world["style_reuse"] if bit==style else 1-world["style_reuse"])
        for index,trajectory in enumerate(product([0,1],repeat=3)):
            factor=1.0
            for order in trajectory:
                factor*=world["core_reuse"] if order==core else 1-world["core_reuse"]
            output["historical_vector"][index]+=mass*factor
    if policy=="direct-both":
        for key in output:
            output[key]=[value/evidence for value in output[key]]
    output["historical_core"]=list(output["future_core"])
    chosen=0 if output["audience"][0]>=output["audience"][1]-1e-12 else 1
    p=world["permutation"]
    output["new_composition"]=p[:2]+[p[3]]+p[4+2*chosen:6+2*chosen]
    output["audience_choice"]=chosen
    output["costs"]={"likelihood_terms":terms,"hypotheses":len(entries),"prediction_terms":28*len(entries)}
    return output
