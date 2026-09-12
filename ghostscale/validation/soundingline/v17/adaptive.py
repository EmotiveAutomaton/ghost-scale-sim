"""C: counted precision, variable expansion and development-fitted strategy selection.

The finite decoder is shared with B. The epistemic labels describe constructed
decoder histories; they are not claims about human knowledge or architecture.
"""
import copy
import math
import random
from ..v16.records import digest, seed_for
from .contracts import Costs,COST_KINDS,forecast_scores
from . import recipient as b
REGIMES=("true_belief","false_belief","ignorance","accidentally_correct","noisy_evidence","changed_goal","misleading_old_cue","nonsocial")
METHODS=("anytime_inverse","variable_expansion","learned_controller","fixed_inverse","empirical","confidence_refinement","entropy_refinement")
OPTIONS=("cached","retrieval","inverse_3","inverse_9","inverse_full")


def make_case(namespace,constructor_index,history_index,regime):
    if regime not in REGIMES:
        raise ValueError("unknown adaptive regime")
    case=b.make_case(namespace,constructor_index,history_index,"correct")
    p=case["public"]
    n=len(p["world"]["templates"])
    rng=random.Random(seed_for(namespace,constructor_index,history_index,regime,"adaptive"))
    model=dict(order=list(range(n)),precision=3.0,prior=[1/n]*n)
    if regime in ("false_belief","misleading_old_cue","accidentally_correct"):
        model["order"]=model["order"][1:]+model["order"][:1]
    if regime=="ignorance":
        model["precision"]=0.0
    # An uninformed strong prior can accidentally identify this configuration,
    # while giving the wrong answer on a counterfactual configuration.
    if regime=="accidentally_correct":
        model["prior"]=[1-1e-9 if k==0 else 1e-9/(n-1) for k in range(n)]
    p["role"]="physical"
    p["editor_knowledge_condition"]="correct"
    p["shared_brief"]=[float(k==0) for k in range(n)]
    initial=p["world"]["templates"][0]
    p["artifact"]=initial
    p["candidates"]=b.opportunities(initial,p["world"])
    p["answer_support"]=[c["id"] for c in p["candidates"]]
    reliability=.5 if regime=="noisy_evidence" else 1.0
    demonstrations=[]
    for i in range(12):
        display=rng.choice(p["world"]["templates"])
        # The old cue worked in the first four observations, then changes.
        old=dict(model,order=list(range(n))) if regime=="misleading_old_cue" and i<4 else model
        q=b.recipient(display,p["world"],old)
        q=[reliability*x+(1-reliability)/n for x in q]
        demonstrations.append(dict(artifact=display,response=b.draw(q,rng)))
    p["calibration"]=demonstrations
    if regime=="changed_goal":
        p["shared_brief"]=[float(k==n-1) for k in range(n)]
    truth_cost=Costs()
    edits=b.accepted_distribution(p["candidates"],p["world"],p["shared_brief"],[(model,1)],p["role"],"correct",truth_cost)
    actual=b.draw(edits,rng)
    future=p["candidates"][actual]["artifact"]
    q=b.recipient(future,p["world"],model)
    truth=str(b.draw(q,rng))
    meta=dict(has_current_access=regime in ("true_belief","changed_goal","nonsocial"),
              reliability=reliability,change_announced=regime in ("changed_goal","misleading_old_cue"),
              computation_price=(0.0,.00002,.0002,.002)[history_index%4],
              hypothesis_limit=9 if history_index%4==2 else len(list(b.model_space(p["world"]))),
              cue_kind="mechanical" if regime=="nonsocial" else "recipient_observation")
    case.update(family="C",regime=regime,case_id=digest([namespace,constructor_index,history_index,regime]))
    case["public"]={"schema":"v17.adaptive.1","task":p,"resource_context":meta}
    case["private"].update(recipient_state=model,recipient_response=truth,
        recipient_distribution=q,accepted_edit=p["candidates"][actual]["id"],
        actual_belief_before=b.recipient(initial,p["world"],model),world_fact=0,
        belief_condition=regime,target_generation_costs=truth_cost.receipt())
    case["public_problem_sha256"]=digest(case["public"])
    case["private_construction_sha256"]=digest([case["public"],model])
    return case


def key(public):
    context=public["resource_context"]
    task=public["task"]
    return str((len(task["world"]["templates"]),context["has_current_access"],context["reliability"],
                context["change_announced"],len(task["calibration"])))


def option_prediction(public,option):
    task,meta=public["task"],public["resource_context"]
    if option=="cached":
        n=len(task["world"]["templates"])
        # Cheap access tags do not duplicate the recipient's belief content.
        if meta["has_current_access"]:
            result=b.predict(task,"surface_task")
            result["represented_variables"]=["visible_artifact","current_access"]
            return result
        counts=Costs()
        counts.charge("retrieval")
        return dict(recipient={str(k):1/n for k in range(n)},costs=counts.receipt(16),
                    represented_variables=["current_access"],hypotheses=0)
    if option=="retrieval":
        return b.predict(task,"retrieval")
    effort={"inverse_3":3,"inverse_9":9,"inverse_full":len(list(b.model_space(task["world"])))}[option]
    effort=min(effort,meta["hypothesis_limit"])
    return b.predict(task,"inferred_recipient",effort=effort)


def prepare(design):
    """Fit expected held-out decision loss on independent development environments.

    All five candidates are evaluated on the same fixed development allocation.
    These data select computations, never use the current screen's answer.
    """
    from collections import defaultdict
    groups=defaultdict(lambda:defaultdict(list))
    records=[]
    learning=0
    namespace=design["namespace"]+"-controller-development"
    for c in range(8):
        for h in range(4):
            for regime in REGIMES:
                case=make_case(namespace,c,h,regime)
                for prefix in (1,4,12):
                    public=copy.deepcopy(case["public"])
                    public["task"]["calibration"]=public["task"]["calibration"][:prefix]
                    for option in OPTIONS:
                        result=option_prediction(public,option)
                        probabilities=result["recipient"]
                        score=forecast_scores(probabilities,list(probabilities),case["private"]["recipient_response"])["brier_score"]
                        operations=result["costs"]["cold_total"]
                        learning+=operations+1
                        groups[key(public)][option].append((score,result["costs"]["repeat_online"]))
                        records.append(dict(case_id=case["case_id"],constructor=case["constructor_id"],
                            prefix=prefix,option=option,brier=score,cost=operations,feature_key=key(public)))
    table={k:{option:dict(expected_brier=sum(x[0] for x in xs)/len(xs),
             expected_operations=sum(x[1] for x in xs)/len(xs),n=len(xs)) for option,xs in group.items()} for k,group in groups.items()}
    return dict(schema="v17.controller.1",namespace=namespace,table=table,learning_operations=learning,
                records=records,method="out-of-screen empirical decision loss by observable feature bin")


def predict(public,method,training):
    if set(public)!={"schema","task","resource_context"} or public["schema"]!="v17.adaptive.1":
        raise ValueError("adaptive public schema violation")
    if method not in METHODS:
        raise ValueError("unknown adaptive strategy")
    meta=public["resource_context"]
    prefix=len(public["task"]["calibration"])
    prelim=None
    if method=="anytime_inverse":
        option="inverse_3" if prefix==1 else "inverse_9" if prefix==4 else "inverse_full"
    elif method=="fixed_inverse":
        option="inverse_full"
    elif method=="empirical":
        option="retrieval"
    elif method=="variable_expansion" and meta["has_current_access"]:
        option="cached"
    elif method in ("learned_controller","variable_expansion"):
        permitted=OPTIONS if method=="learned_controller" else ("cached","inverse_full")
        group=training["table"][key(public)]
        option=min(permitted,key=lambda o:group[o]["expected_brier"]+meta["computation_price"]*group[o]["expected_operations"])
    else:
        prelim=option_prediction(public,"inverse_3")
        values=list(prelim["recipient"].values())
        entropy=-sum(p*math.log(p) for p in values if p)
        expand=(max(values)<.8) if method=="confidence_refinement" else entropy>.5*math.log(len(values))
        option="inverse_full" if expand else "inverse_3"
    result=prelim if prelim and option=="inverse_3" else option_prediction(public,option)
    costs=Costs({k:result["costs"][k] for k in COST_KINDS})
    if prelim and option!="inverse_3":
        for kind in COST_KINDS:
            costs.charge(kind,prelim["costs"][kind])
    if method in ("learned_controller","variable_expansion"):
        costs.charge("learning",training["learning_operations"])
        costs.charge("definition_storage",sum(len(group)*3 for group in training["table"].values()))
        costs.charge("selection",len(OPTIONS) if method=="learned_controller" else 2)
    elif prelim:
        costs.charge("selection",len(prelim["recipient"]))
    result=dict(result,costs=costs.receipt(256),selected_strategy=option,
                variable_expanded=option.startswith("inverse"),hypothesis_limit=meta["hypothesis_limit"])
    if result["hypotheses"]>meta["hypothesis_limit"]:
        raise ValueError("actual hypothesis effort exceeded deadline allowance")
    return result


def evaluate_case(case,design):
    training=design["prepared"]
    rows=[]
    previous={}
    for prefix in (1,4,12):
        public=copy.deepcopy(case["public"])
        public["task"]["calibration"]=public["task"]["calibration"][:prefix]
        for method in METHODS:
            result=predict(public,method,training)
            p=result["recipient"]
            truth=case["private"]["recipient_response"]
            # Time from the announced observation-5 switch to this updated forecast
            # is recorded directly; a changed strategy is not equated with accuracy.
            rows.append(dict(method=method,target="recipient_after_"+str(prefix)+"_observations",
                probabilities=p,**forecast_scores(p,list(p),truth),costs=result["costs"],
                represented_variables=result["represented_variables"],hypotheses=result["hypotheses"],
                selected_strategy=result["selected_strategy"],variable_expanded=result["variable_expanded"],
                strategy_changed=method in previous and previous[method]!=result["selected_strategy"],
                observations_since_switch=max(0,prefix-4) if case["public"]["resource_context"]["change_announced"] else None,
                evidence_tier="supplied_brief_and_observed_calibration_prefix",missing_output=False,invalid_program=False,
                task_success=max(p,key=p.get)==truth,controller_namespace=training["namespace"]))
            previous[method]=result["selected_strategy"]
    return rows
