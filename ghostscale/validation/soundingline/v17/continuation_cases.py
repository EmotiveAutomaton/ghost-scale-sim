"""Additional commissioned comparisons, retaining the original V17 consumers."""
import copy
import math
import os
import random
from . import recipient as b, adaptive as c, revision as d, craft_extension as a, observer as e
from .contracts import Costs, COST_KINDS, forecast_scores
from ..v16 import assembly
from ..v16.records import digest, seed_for

MODULES={"A2":a,"B":b,"C":c,"D":d,"E":e}
F_METHODS=("shared","separate","partial")

def identity(namespace,ci,hi,family,regime,public,private,structure):
    return dict(case_id=digest([namespace,ci,hi,family,regime]),constructor_id=digest([namespace,ci]),
        history_id=digest([namespace,ci,hi]),namespace=namespace,family=family,regime=regime,
        public=public,private=private,public_problem_sha256=digest(public),
        private_construction_sha256=digest([public,private]),structural_family=structure)

def make_pooled(namespace,ci,hi,regime):
    case=a.make_case(namespace,ci,hi,regime)
    public=case["public"]
    pool=[]
    for donor in range(4):
        other=a.make_case(namespace+"-pooled-training",ci*4+donor,hi,regime)
        # Both arms exclude the supplied held-out construction target.
        for row in other["public"]["training"]:
            if row["target"]!=public["target"]:
                # Assembly donor traces must execute under the actual receiver's laws.
                check=a.execute(public,row["program"],row["initial"])
                if check["legal"]:
                    pool.append(dict(row,target=check["state"]))
    pool=[t for t in pool if t["target"]!=public["target"]]
    if not pool: raise ValueError("pooled acquisition is empty")
    pooled=[copy.deepcopy(pool[i%len(pool)]) for i in range(len(public["training"]))]
    public["pooled_training"]=pooled
    case["private"]["pooled_acquisition_rule"]="four independently seeded donors, legal under recipient physics, first fixed count; exact target excluded"
    case.update(family="AP",public_problem_sha256=digest(public),
                private_construction_sha256=digest([public,case["private"]]))
    return case

def evaluate_craft(case,design):
    public=case["public"]
    unavailable=None
    try: learned=a.stitch.learn(public["training"],public["world_kind"])
    except Exception as exc:
        learned=None;unavailable=type(exc).__name__+": "+str(exc)
        for variable in ("GS_V17_STITCH_EXE","GS_V17_STITCH_CACHE"):
            path=os.environ.get(variable)
            if path:unavailable=unavailable.replace(path,"<isolated dependency path>")
    rows=[]
    for cap in design.get("memory_caps",[32]):
        for method in a.METHODS:
            dependent=method in ("stitch_abstractions","abstractions_and_exceptions")
            rep=a.acquire(public,method,cap,learned) if learned is not None or not dependent else None
            for budget in design.get("budgets_by_world",{}).get(public["world_kind"],design["budgets"]):
                if rep is None:
                    row=dict(program=None,execution=None,costs=Costs().receipt(16),task_success=False,
                        missing_output=True,invalid_program=False,attempted_primitives=0,failed_candidates=0,
                        representation=dict(definitions=[],fragments=[],episodes=[],memory_cap=cap),
                        apparatus_failure="Stitch unavailable: "+unavailable)
                else:
                    row=dict(a.solve(public,rep,budget),representation={k:v for k,v in rep.items() if k!="costs"})
                rows.append(dict(row,method=method+"@"+str(cap),target="construction_budget_"+str(budget),
                    budget=budget,memory_cap=cap,evidence_tier="maker acquisition and supplied construction goal",
                    stitch_learning=learned if dependent else None))
    return rows


def evaluate_pooled(case,design):
    rows=[]
    for condition in ("personal","pooled"):
        public=copy.deepcopy(case["public"])
        pool=public.pop("pooled_training")
        if condition=="pooled": public["training"]=pool
        other=dict(case,public=public)
        for row in evaluate_craft(other,design):
            row["method"]=condition+":"+row["method"]
            row["acquisition_distribution"]=condition
            rows.append(row)
    return rows

def assembly_options(world,initial):
    result=[]
    # Shared, public legal programs, including STOP; no private feasible set.
    for candidate in d.candidates(world,initial,max_steps=3):
        result.append(dict(id=candidate["id"],program=candidate["program"],
            artifact=d.encode(candidate["state"]),state=candidate["state"]))
    return result

def make_assembly_recipient(namespace,ci,hi,regime):
    if regime not in b.REGIMES: raise ValueError("unknown assembly recipient condition")
    world=assembly.constructor(namespace,ci)
    states=[]
    for program in assembly.legal_histories(world,max_steps=5):
        run=assembly.execute(world,program)
        if run["state"] not in states: states.append(run["state"])
    full=[s for s in states if all(v>=0 for v in s)]
    if len(full)<2: raise ValueError("no distinct legal recipient targets")
    rng=random.Random(seed_for(namespace,ci,hi,regime))
    templates=rng.sample(full,2)
    rw=dict(templates=[d.encode(s) for s in templates],visible=63,structure=str(world.parents))
    model=rng.choice(list(b.model_space(rw)))
    previous=copy.deepcopy(model)
    if regime=="changed_recipient": model=dict(model,order=list(reversed(model["order"])))
    initial=rng.choice(states)
    candidates=assembly_options(world,initial)
    desired=[float(k==hi%2) for k in range(2)] if hi%4!=3 else [.5,.5]
    calibration=[]
    for _ in range(12):
        board=d.encode(rng.choice(states))
        calibration.append(dict(artifact=board,response=b.draw(b.recipient(board,rw,model),rng)))
    role=("physical","communicator","producer_editor")[hi%3]
    knowledge="correct" if regime=="changed_recipient" else regime
    costs=Costs()
    edits=b.accepted_distribution(candidates,rw,desired,[(model,1)],role,knowledge,costs)
    chosen=b.draw(edits,rng)
    execution=assembly.execute(world,candidates[chosen]["program"],initial)
    response=b.recipient(d.encode(execution["state"]),rw,model)
    witness=assembly.plan(world,initial,budget=100000,max_steps=7)["program"]
    if witness is None: raise ValueError("unreachable initial assembly")
    public=dict(schema="v17.assembly-recipient.1",assembly_world=world.public(),initial=initial,world=rw,
        candidates=candidates,artifact=d.encode(initial),shared_brief=desired,role=role,
        editor_knowledge_condition=knowledge,calibration=calibration,
        query_cost=1,answer_support=[x["id"] for x in candidates])
    private=dict(maker_training=calibration,current_goal=desired,earlier_goals=[],
        actual_constraints=world.public(),believed_constraints=world.public(),considered_options=candidates,
        realized_history=assembly.execute(world,witness)["trace"],contributor_role=role,
        recipient_state=model,previous_recipient_state=previous,accepted_edit=candidates[chosen]["id"],
        edit_distribution=edits,execution=execution,recipient_distribution=response,
        recipient_response=str(b.draw(response,rng)),target_generation_costs=costs.receipt(),
        ratification_role="original maker" if role=="producer_editor" else None)
    return identity(namespace,ci,hi,"BA",regime,public,private,str(world.parents))

def predict_assembly(public,method,supplied=None):
    expected={"schema","assembly_world","initial","world","candidates","artifact","shared_brief","role",
        "editor_knowledge_condition","calibration","query_cost","answer_support"}
    if set(public)!=expected or public["schema"]!="v17.assembly-recipient.1": raise ValueError("assembly recipient public boundary")
    w=public["assembly_world"];world=assembly.World(tuple(w["parents"]),tuple(w["defaults"]))
    if public["candidates"]!=assembly_options(world,public["initial"]): raise ValueError("invalid shared assembly opportunity")
    if public["answer_support"]!=[x["id"] for x in public["candidates"]]: raise ValueError("invalid answer support")
    costs=Costs();costs.charge("training_acquisition",len(public["calibration"]))
    rw=public["world"];n=len(rw["templates"])
    if method=="inferred_recipient":
        models=b.infer_models(rw,public["calibration"],costs=costs)
        costs.charge("definition_storage",sum(2*len(m["order"])+2 for m,_ in models))
    elif method=="supplied_recipient_ceiling":
        if supplied is None: raise ValueError("ceiling requires explicit assistance")
        models=[(supplied,1)];costs.charge("definition_storage",2*n+1)
    elif method=="surface_task":
        models=[(dict(order=list(range(n)),precision=1.2,prior=[1/n]*n),1)]
        costs.charge("definition_storage",2*n+1)
    elif method=="retrieval":
        models=[];costs.charge("definition_storage",2*len(public["calibration"]))
    else: raise ValueError("unknown assembly reader")
    candidates=public["candidates"]
    if method=="retrieval":
        predicted=[];values=[]
        for candidate in candidates:
            distances=[(candidate["artifact"]^x["artifact"]).bit_count() for x in public["calibration"]]
            costs.charge("retrieval",2*len(distances))
            best=min(distances);counts=[1.]*n
            for example,distance in zip(public["calibration"],distances):
                costs.charge("selection")
                if distance==best: counts[example["response"]]+=1
            predicted.append(b.normalize(counts))
            values.append(b.outcome_value(predicted[-1],public["shared_brief"])-.025*len(candidate["program"]))
            costs.charge("proposal_generation")
        edits=b.softmax(values,5)
    else:
        edits=b.accepted_distribution(candidates,rw,public["shared_brief"],models,
            "physical" if method=="surface_task" else public["role"],public["editor_knowledge_condition"],costs)
        predicted=[b.mixture(x["artifact"],rw,models,costs) for x in candidates]
    effect=b.normalize([sum(w*p[k] for w,p in zip(edits,predicted)) for k in range(n)])
    return dict(edit=dict(zip(public["answer_support"],edits)),recipient={str(i):p for i,p in enumerate(effect)},costs=costs.receipt(16))

def evaluate_assembly(case,design):
    rows=[]
    for method in b.METHODS:
        result=predict_assembly(case["public"],method,case["private"]["recipient_state"] if method=="supplied_recipient_ceiling" else None)
        for target,key,truth in (("accepted_edit","edit",case["private"]["accepted_edit"]),
                               ("recipient_outcome","recipient",case["private"]["recipient_response"])):
            p=result[key]
            rows.append(dict(method=method,target=target,probabilities=p,**forecast_scores(p,list(p),truth),
                costs=result["costs"],task_success=max(p,key=p.get)==truth,missing_output=False,invalid_program=False,
                evidence_tier="supplied shared brief, assembly laws and recipient calibration",
                assistance="true recipient supplied" if method=="supplied_recipient_ceiling" else "recipient inferred or approximated"))
    return rows

def make_sharing(namespace,ci,hi,regime):
    if regime not in ("expert_partner","unfamiliar_partner"): raise ValueError("unknown sharing regime")
    original=b.make_case(namespace,ci,hi,"correct")
    world=original["public"]["world"];models=list(b.model_space(world));n=len(world["templates"])
    rng=random.Random(seed_for(namespace,ci,hi,"sharing"))
    own=rng.choice(models)
    partner=copy.deepcopy(own) if regime=="expert_partner" else dict(own,order=own["order"][1:]+own["order"][:1])
    def examples(model,count):
        records=[]
        for _ in range(count):
            board=rng.choice(world["templates"])
            records.append(dict(artifact=board,response=b.draw(b.recipient(board,world,model),rng)))
        return records
    own_training=examples(own,12);partner_training=examples(partner,2+(hi%2)*2)
    future=rng.choice(world["templates"])
    recipient_future=str(b.draw(b.recipient(future,world,partner),rng))
    possible=[dict(id=str(k),artifact=board,program=[c for c in range(16) if board&(1<<c)]) for k,board in enumerate(world["templates"])]
    own_truth={x["id"]:str(b.draw(b.recipient(x["artifact"],world,own),random.Random(seed_for(namespace,ci,hi,"own-outcome",x["id"])))) for x in possible}
    public=dict(schema="v17.sharing.1",world=world,own_training=own_training,partner_training=partner_training,
        future_artifact=future,personal_options=possible,personal_goal=0,answer_support=[str(i) for i in range(n)],
        sharing_prior=.75,query_cost=0)
    private=dict(own_model=own,partner_model=partner,partner_future=recipient_future,own_outcomes=own_truth,
        maker_training=own_training,earlier_goals=[],current_goal=0,actual_constraints=[],believed_constraints=[],
        considered_options=possible,realized_history=original["private"]["realized_history"],
        contributor_role="performer and partner observer",recipient_state=partner)
    return identity(namespace,ci,hi,"F",regime,public,private,world["structure"])

def sharing_posteriors(public,method,costs):
    if set(public)!={"schema","world","own_training","partner_training","future_artifact","personal_options","personal_goal","answer_support","sharing_prior","query_cost"}:
        raise ValueError("sharing public boundary")
    if public["schema"]!="v17.sharing.1" or method not in F_METHODS: raise ValueError("invalid sharing method")
    world=public["world"];models=list(b.model_space(world));m=len(models)
    logs=[]
    for examples in (public["own_training"],public["partner_training"]):
        costs.charge("training_acquisition",len(examples))
        ll=[]
        for model in models:
            value=0.
            for example in examples:
                p=b.recipient(example["artifact"],world,model,costs)[example["response"]]
                value+=math.log(p);costs.charge("learning")
            ll.append(value)
        logs.append(ll)
    if method=="shared":
        weights=b.softmax([x+y for x,y in zip(*logs)])
        own=other=list(zip(models,weights))
    elif method=="separate":
        own=list(zip(models,b.softmax(logs[0])));other=list(zip(models,b.softmax(logs[1])))
    else:
        rho=public["sharing_prior"]
        joint=[]
        for i in range(m):
            for j in range(m):
                prior=rho/m if i==j else (1-rho)/(m*(m-1))
                joint.append(logs[0][i]+logs[1][j]+math.log(prior));costs.charge("learning")
        weights=b.softmax(joint)
        own=list(zip(models,[sum(weights[i*m+j] for j in range(m)) for i in range(m)]))
        other=list(zip(models,[sum(weights[i*m+j] for i in range(m)) for j in range(m)]))
        costs.charge("selection",m*m)
    costs.charge("definition_storage",m*4+(1 if method=="partial" else 0))
    return own,other

def evaluate_sharing(case,design):
    p,private=case["public"],case["private"];rows=[]
    for method in F_METHODS:
        costs=Costs();own,partner=sharing_posteriors(p,method,costs)
        forecast=b.mixture(p["future_artifact"],p["world"],partner,costs)
        own_predictions=[]
        for option in p["personal_options"]:
            own_predictions.append(b.mixture(option["artifact"],p["world"],own,costs))
            costs.charge("proposal_generation");costs.charge("selection")
        chosen=max(range(len(own_predictions)),key=lambda i:own_predictions[i][p["personal_goal"]])
        option=p["personal_options"][chosen]
        execution=b.execute(option["program"])
        if not execution["legal"] or execution["artifact"]!=option["artifact"]: raise ValueError("personal execution failed physics")
        costs.charge("actual_execution",execution["primitive_cost"])
        for target,values,truth in (("partner_prediction",forecast,private["partner_future"]),
                                   ("personal_execution",own_predictions[chosen],private["own_outcomes"][option["id"]])):
            probs=dict(zip(p["answer_support"],values))
            rows.append(dict(method=method,target=target,probabilities=probs,**forecast_scores(probs,p["answer_support"],truth),
                costs=costs.receipt(16),task_success=max(probs,key=probs.get)==truth,missing_output=False,invalid_program=False,
                personal_goal_success=truth==str(p["personal_goal"]) if target=="personal_execution" else None,
                chosen_personal_option=option["id"],evidence_tier="own and partner demonstration collections; latent parameters withheld"))
    return rows

def make_case(design,ci,hi):
    f=design["family"];args=(design["namespace"],ci,hi,design["regime"])
    if f=="AP": return make_pooled(*args)
    if f=="BA": return make_assembly_recipient(*args)
    if f=="F": return make_sharing(*args)
    return MODULES[f].make_case(*args)

def evaluate_case(case,design):
    f=design["family"]
    if f=="A2": return evaluate_craft(case,design)
    if f=="AP": return evaluate_pooled(case,design)
    if f=="BA": return evaluate_assembly(case,design)
    if f=="F": return evaluate_sharing(case,design)
    return MODULES[f].evaluate_case(case,design)
