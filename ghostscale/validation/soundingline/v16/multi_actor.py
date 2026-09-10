"""M04 role-relative predictions without public topology or routine-owner labels."""
from itertools import product
import json
import math
from .recognition import decode
from .selection import validate,selection_style_probability
from .graphic_world import execute
from .multi_actor_world import readout

POLICIES=["artifact-only","producer-view","revision-view","unselected-view","brief-flip","all","direct-all"]
EXTENSION="ghostscale.validation.soundingline.v16.multi_actor:public_reader"
ROLES=["producer_core","producer_style","revision","release","brief"]
DESIGN={
 "card_id":"M04","question":"Whose decisions can be predicted when production, revision, selection and a shared brief can all shape one artifact?",
 "mechanism":"actual acquired producer and self/other revision routines combine with optional selection and response to a shared brief",
 "strongest_rival":"self-revision instead of another editor; an equally informed direct joint predictor for every observable role target",
 "access_arms":"artifact-only and four paid probe policies; all/direct-all receive identical requested observations with the same prices",
 "target_realization":"all candidates execute, revisions remove/place from the actual prior artifact, retention selects actual candidates, brief intervention changes only the brief",
 "conditions":[{"id":f"{family}-{revision}-selector-{selector}-brief-{brief}","family":family,"revision":revision,
                "selector":selector,"shared_brief":brief}
               for family in ["learned-order","goal-indifferent"] for revision in ["none","self","other"]
               for selector in [False,True] for brief in [False,True]],
 "arms":POLICIES,"production_count":4,
 "primary":[("all",rival,f"future_{role}_log_score","nats_per_event",0.02)
            for role in ROLES for rival in ["artifact-only","direct-all"]],
 "secondary":["topology uncertainty","current core-order history","all candidate/revision/training costs","paid access and measured process resources"],
 "generator_families":["W1 larger graphic world; optional self/other revision, first-match selection and fixed own-purpose/shared-brief control"],
 "paired_unit":"one producer/editor acquisition pair, three releases and paired paid views/probes with fresh role-relative outcomes",
 "sample_rule":{"scout":64,"constructors":8,"expansion":256,"expansion_constructors":20,"final_expansion":1024},
 "dependencies":["multi-actor-physical","multi-actor-collisions","multi-actor-probes","independent-multi-actor-scoring"],
 "adversaries":["X01","X02","X03","X04","X05","X06","X07","X08"],"repair_budget":1,
 "continuation":"expand a named role/selection ambiguity; self and other revision can remain indistinguishable when acquired styles coincide",
 "coverage_limit":"known finite role catalog; a model-specific posterior is not proof of a real person's identity",
 "history_access":"producer/revision views describe the same first retained work; shared raw-style evidence is counted once",
 "revision_observation":"reader checks physical before/after and whether any primitive action occurred; macro names and routine-owner tags are private",
 "execution_bound":"each producer or revision stage is at most six primitives; a complete multistage episode can cost more than six"
}

def requests(policy):
    all_views=policy in {"all","direct-all"}
    return {"producer":all_views or policy=="producer-view","revision":all_views or policy=="revision-view",
            "unselected":all_views or policy=="unselected-view","brief_flip":all_views or policy=="brief-flip"}

def evidence(public,policy):
    if set(public)!={"schema_version","task_id","phase","world","brief","produced_count","history",
                     "producer_view","revision_view","unselected_view","brief_view"} or public["schema_version"]!="v16.multi-actor.1":
        raise ValueError("multi-actor public schema violation")
    if policy not in POLICIES or type(public["brief"]) is not int or public["brief"] not in [0,1]:
        raise ValueError("unknown policy or invalid public brief")
    validate({"schema_version":"v16.selection.1","task_id":public["task_id"],"world":public["world"],
              "history":[],"retention":"random","produced_count":public["produced_count"],"audience":None})
    if len(public["history"])!=3:
        raise ValueError("three independently produced releases are required")
    world=public["world"]
    history=[readout(world,value) for value in public["history"]]
    desired=requests(policy)
    fields={"producer":"producer_view","revision":"revision_view","unselected":"unselected_view","brief_flip":"brief_view"}
    if public["phase"]==1:
        if any(public[name] is not None for name in fields.values()):
            raise ValueError("query answers arrived before request")
        return None
    if public["phase"]!=2 or any((public[field] is not None)!=desired[key] for key,field in fields.items()):
        raise ValueError("missing or unpurchased role evidence")
    raw_style=None;first=None;acted=None
    if public["producer_view"] is not None:
        view=public["producer_view"]
        raw_style,first=decode(world,view)
        if first is None or readout(world,view["artifact"])[1]!=history[0][1]:
            raise ValueError("invalid producer process view")
    if public["revision_view"] is not None:
        view=public["revision_view"]
        if set(view)!={"before","after","program"} or view["after"]!=public["history"][0]:
            raise ValueError("undeclared revision fields or incorrect target work")
        before_style,before_topic=readout(world,view["before"])
        if before_topic!=history[0][1] or (raw_style is not None and raw_style!=before_style):
            raise ValueError("two views disagree about the same original work")
        result=execute(view["program"],initial=view["before"])
        if not result["legal"] or result["artifact"]!=view["after"]:
            raise ValueError("revision trace cannot produce its shown artifact")
        raw_style=before_style
        acted=bool(view["program"])
    unselected=None;changed=None
    if public["unselected_view"] is not None:
        view=public["unselected_view"]
        if set(view)!={"artifact"}:
            raise ValueError("undeclared unselected-view field")
        unselected=readout(world,view["artifact"])
    if public["brief_view"] is not None:
        view=public["brief_view"]
        if set(view)!={"new_brief","artifact"} or type(view["new_brief"]) is not int or view["new_brief"]!=1-public["brief"]:
            raise ValueError("incorrect brief intervention")
        changed=readout(world,view["artifact"])
    return {"history":history,"raw_style":raw_style,"first":first,"acted":acted,"unselected":unselected,"changed":changed}

def raw_probability(world,observed,acquired):
    return world["style_reuse"] if observed==acquired else 1-world["style_reuse"]

def weighted_states(public,observed):
    world=public["world"];b=public["brief"];n=public["produced_count"]
    entries=[];terms=0
    for core,style,other,target,revision,selector,shared,purpose in product([0,1],[0,1],[0,1],[0,1],range(3),[0,1],[0,1],[0,1]):
        base=other if revision==2 else style
        weight=1/384
        expected_topic=b if shared else purpose
        for visible_style,topic in observed["history"]:
            terms+=2
            weight*=int(topic==expected_topic)
            weight*=selection_style_probability(visible_style,base,target,n,world) if selector else raw_probability(world,visible_style,base)
        if observed["raw_style"] is not None:
            terms+=1
            weight*=int(observed["raw_style"]==observed["history"][0][0]) if revision==0 else raw_probability(world,observed["raw_style"],style)
        if observed["acted"] is not None:
            terms+=1
            weight*=int(observed["acted"]==(revision!=0))
        if observed["first"] is not None:
            terms+=1
            weight*=world["core_reuse"] if observed["first"]==core else 1-world["core_reuse"]
        if observed["unselected"] is not None:
            visible_style,topic=observed["unselected"]
            terms+=2
            weight*=raw_probability(world,visible_style,base)*int(topic==expected_topic)
        if observed["changed"] is not None:
            visible_style,topic=observed["changed"]
            terms+=2
            weight*=raw_probability(world,visible_style,base)*int(topic==((1-b) if shared else purpose))
        entries.append(((core,style,other,target,revision,selector,shared,purpose),weight))
    return entries,terms

def public_reader(payload:bytes,policy):
    public=json.loads(payload)
    observed=evidence(public,policy)
    if public["phase"]==1:
        return {"queries":requests(policy)}
    entries,terms=weighted_states(public,observed)
    evidence_mass=sum(weight for _,weight in entries)
    if evidence_mass<=0:
        raise ValueError("role observations have zero model support")
    output={key:[0.0,0.0] for key in ["producer_core","producer_style","release","brief","selector","shared_brief"]}
    output["revision"]=[0.0]*3
    output["revision_relation"]=[0.0]*3
    output["topology"]=[0.0]*12
    world=public["world"];b=public["brief"];n=public["produced_count"]
    prediction_terms=0
    for state,weight in entries:
        core,style,other,target,revision,selector,shared,purpose=state
        prediction_terms+=5+int(revision==0)+6+2*int(revision!=0)
        mass=weight if policy=="direct-all" else weight/evidence_mass
        base=other if revision==2 else style
        output["revision_relation"][revision]+=mass
        output["selector"][selector]+=mass
        output["shared_brief"][shared]+=mass
        output["topology"][revision*4+selector*2+shared]+=mass
        output["brief"][(1-b) if shared else purpose]+=mass
        if revision==0:
            output["revision"][0]+=mass
        for bit in [0,1]:
            output["producer_core"][bit]+=mass*(world["core_reuse"] if bit==core else 1-world["core_reuse"])
            output["producer_style"][bit]+=mass*raw_probability(world,bit,style)
            if revision!=0:
                output["revision"][1+bit]+=mass*raw_probability(world,bit,base)
            released=selection_style_probability(bit,base,target,n,world) if selector else raw_probability(world,bit,base)
            output["release"][bit]+=mass*released
    if policy=="direct-all":
        for key in output:
            output[key]=[value/evidence_mass for value in output[key]]
    output["historical_core"]=list(output["producer_core"]) if observed["first"] is None else [float(bit==observed["first"]) for bit in [0,1]]
    output["topology_entropy"]=-sum(value*math.log(value) for value in output["topology"] if value>0)
    output["compatible_topologies"]=sum(value>0 for value in output["topology"])
    output["costs"]={"likelihood_terms":terms,"hypotheses":384,"prediction_terms":prediction_terms,
                     "entropy_terms":output["compatible_topologies"]}
    return output
