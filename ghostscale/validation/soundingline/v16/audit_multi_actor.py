"""Independent M04 learning, all candidate/revision execution and role scores."""
import hashlib
import json
import math
import random
from .audit_graphic_maker import seed,verify_maker,verify_production
from .audit_selection import style_of
from .graphic_reference import interpret
from .multi_actor_reference import predict
from .audit_statistics import verify

def load(path,expected):
    payload=path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected:
        raise ValueError("multi-actor raw commitment differs")
    return json.loads(payload)

def verify_batch(construction,episode,namespace,index,label,date,*,count,brief,bypass):
    world=construction["world"];p=world["permutation"]
    if episode["count"]!=count or len(episode["candidates"])!=count or episode["brief"]!=brief or episode["selection_bypassed"]!=bypass:
        raise ValueError("candidate batch contract differs")
    total=0
    topic=brief if construction["shared_brief"] else construction["own_purpose"]
    for position,event in enumerate(episode["candidates"]):
        total+=verify_production(world,event["producer"],construction["producer"],
                                 random.Random(seed(namespace,label,index,date,position,"produce")),topic)
        before=event["producer"]["execution"]["artifact"]
        old_style=style_of(world,{"artifact":before})
        revision=construction["revision"]
        used=False
        if revision=="none":
            program=[]
        else:
            actor=construction["producer"] if revision=="self" else construction["editor"]
            part=actor["decoration"]
            rng=random.Random(seed(namespace,label,index,date,position,"revision"))
            choice=part["choice"] if rng.random()<world["style_reuse"] else 1-part["choice"]
            used=choice==part["choice"]
            fragment=part["routine"] if used else p[4+2*choice:6+2*choice]
            program=[16+cell for cell in p[4+2*old_style:6+2*old_style]]+fragment
        result=interpret(program,initial=before)
        if event["revision_program"]!=program or event["revision_execution"]!=result or event["revision_routine_used"]!=used:
            raise ValueError("revision did not execute the actually acquired actor routine")
        if event["artifact"]!=result["artifact"] or event["post_style"]!=style_of(world,{"artifact":result["artifact"]}) or event["topic"]!=topic:
            raise ValueError("multi-actor readout was not derived from actual execution")
        total+=result["primitive_cost"]
    if construction["selector"] and not bypass:
        kept=next((i for i,event in enumerate(episode["candidates"]) if event["post_style"]==construction["selection_target"]),0)
    else:
        kept=random.Random(seed(namespace,label,index,date,"retain")).randrange(count)
    if episode["retained_index"]!=kept:
        raise ValueError("selector did not retain the stated actual candidate")
    return total

def audit_unit(root,row):
    uid=row["unit_id"]
    public=load(root/"public"/f"{uid}.json",row["public_hash"])
    private=load(root/"private"/f"{uid}.json",row["private_hash"])
    original=load(root/"private"/f"{uid}-original.json",private["original_hash"])
    final=load(root/"predictions"/f"{uid}.json",row["prediction_hash"])
    first=load(root/"predictions"/f"{uid}-phase-1.json",final["phase_one_hash"])
    if first["public_hash"]!=row["public_hash"] or first["submitted_at"]>final["submitted_at"]:
        raise ValueError("query requests were not committed before predictions")
    construction=original["construction"];world=construction["world"]
    condition=row["condition_spec"];namespace,index=row["lineage"],row["seed_components"]["index"]
    constructor_id=f'constructor-{index%row["seed_components"]["constructors"]:03d}'
    rng=random.Random(seed(namespace,"constructor",constructor_id))
    p=list(range(16));rng.shuffle(p)
    expected_world={"permutation":p,"training_reliability":rng.uniform(0.70,0.95),"style_reuse":rng.uniform(0.75,0.95),
                    "core_reuse":rng.uniform(0.75,0.95) if condition["family"]=="learned-order" else 0.5}
    if world!=expected_world or constructor_id!=row["constructor_id"]:
        raise ValueError("independent constructor redraw differs")
    for key,label in [("brief","brief"),("own_purpose","own-purpose"),("selection_target","selection-target")]:
        if construction[key]!=random.Random(seed(namespace,label,index)).randrange(2):
            raise ValueError("purpose/brief/selection target sampling differs")
    if any(construction[key]!=condition[key] for key in ["revision","selector","shared_brief"]):
        raise ValueError("executed topology differs from registered condition")
    training=0;definitions=0
    for maker_id,key in enumerate(["producer","editor"]):
        cost,definition=verify_maker(world,construction[key],namespace,index,maker_id)
        training+=cost;definitions+=definition
    brief=construction["brief"]
    if len(original["history"])!=3 or public["produced_count"]!=4:
        raise ValueError("incorrect original production budget")
    original_cost=sum(verify_batch(construction,episode,namespace,index,"history",date,count=4,brief=brief,bypass=False)
                      for date,episode in enumerate(original["history"]))
    expected_public={**public,"phase":1,"world":{key:world[key] for key in ["permutation","style_reuse","core_reuse"]},
        "brief":brief,"produced_count":4,"history":[episode["candidates"][episode["retained_index"]]["artifact"] for episode in original["history"]],
        "producer_view":None,"revision_view":None,"unselected_view":None,"brief_view":None}
    if public!=expected_public:
        raise ValueError("initial reader input leaked an answer or differs from executed history")
    unselected_cost=verify_batch(construction,private["unselected_probe"],namespace,index,"probe-unselected",0,count=1,brief=brief,bypass=True)
    flipped_cost=verify_batch(construction,private["brief_probe"],namespace,index,"probe-brief",0,count=1,brief=1-brief,bypass=True)
    raw=private["future_raw"]
    future_cost=verify_production(world,raw,construction["producer"],random.Random(seed(namespace,"future-producer",index)),
                                  brief if construction["shared_brief"] else construction["own_purpose"])
    for key,label,count,new_brief,bypass in [("future_release","future-release",4,brief,False),
            ("future_revision","future-revision",1,brief,True),("future_brief","future-brief",1,1-brief,True)]:
        future_cost+=verify_batch(construction,private[key],namespace,index,label,0,count=count,brief=new_brief,bypass=bypass)
    current=original["history"][0]["candidates"][original["history"][0]["retained_index"]]
    def released(episode):
        return episode["candidates"][episode["retained_index"]]
    revision=private["future_revision"]["candidates"][0]
    targets={"producer_core":raw["core_order"],"producer_style":raw["decoration"],
        "revision":0 if not revision["revision_program"] else 1+revision["post_style"],
        "release":released(private["future_release"])["post_style"],"brief":released(private["future_brief"])["topic"]}
    topology=["none","self","other"].index(construction["revision"])*4+int(construction["selector"])*2+int(construction["shared_brief"])
    for policy,arm in row["arms"].items():
        all_views=policy in {"all","direct-all"}
        queries={"producer":all_views or policy=="producer-view","revision":all_views or policy=="revision-view",
                 "unselected":all_views or policy=="unselected-view","brief_flip":all_views or policy=="brief-flip"}
        if first["arms"][policy]!={"queries":queries}:
            raise ValueError("role query commitment differs from policy")
        expected={**public,"phase":2,
            "producer_view":{"artifact":current["producer"]["execution"]["artifact"],"first_action":current["producer"]["program"][0]} if queries["producer"] else None,
            "revision_view":{"before":current["producer"]["execution"]["artifact"],"after":current["artifact"],"program":current["revision_program"]} if queries["revision"] else None,
            "unselected_view":{"artifact":released(private["unselected_probe"])["artifact"]} if queries["unselected"] else None,
            "brief_view":{"new_brief":1-brief,"artifact":released(private["brief_probe"])["artifact"]} if queries["brief_flip"] else None}
        visible=final["public_inputs"][policy]
        if visible!=expected:
            raise ValueError("missing, unpurchased or incorrect role observation")
        prediction=final["arms"][policy]
        independent=predict(visible)
        for key in ["producer_core","producer_style","revision","release","brief","selector","shared_brief","revision_relation","topology","historical_core"]:
            if any(abs(a-b)>1e-12 for a,b in zip(prediction[key],independent[key])):
                raise ValueError("independent factorized role inference differs")
        if abs(prediction["topology_entropy"]-independent["topology_entropy"])>1e-12 or prediction["costs"]!=independent["costs"] or prediction["compatible_topologies"]!=independent["compatible_topologies"]:
            raise ValueError("role uncertainty or reference work differs")
        expected_outcomes={f"future_{role}_log_score":math.log(prediction[role][target]) for role,target in targets.items()}
        expected_outcomes.update({
            "historical_core_log_score":math.log(prediction["historical_core"][current["producer"]["core_order"]]),
            "topology_log_score":math.log(prediction["topology"][topology]),"topology_entropy":prediction["topology_entropy"],
            "compatible_topologies":float(prediction["compatible_topologies"]),"training_primitives":float(training),
            "library_definition_cost":float(definitions),"original_production_primitives":float(original_cost),
            "future_production_primitives":float(future_cost),"query_count":float(sum(queries.values())),
            "probe_production_primitives":float(unselected_cost*queries["unselected"]+flipped_cost*queries["brief_flip"]),
            "retained_works":3.0,"rejected_works":9.0,"likelihood_terms":float(prediction["costs"]["likelihood_terms"]),
            "prediction_terms":float(prediction["costs"]["prediction_terms"]),"entropy_terms":float(prediction["costs"]["entropy_terms"])})
        if arm["outcomes"]!=expected_outcomes:
            raise ValueError("independent role outcome differs")

def audit(root,summary):
    rows=[json.loads(path.read_bytes()) for path in sorted((root/"units").glob("*_points.json"))]
    for row in rows:
        audit_unit(root,row)
    count=verify(rows,summary)
    return {"execution_state":"completed","instrument_state":"valid","n_raw_units":len(rows),
            "reproduced_contrasts":count,"all_reported_aggregates_reproduced":True,"full_rollout_replay":False}
