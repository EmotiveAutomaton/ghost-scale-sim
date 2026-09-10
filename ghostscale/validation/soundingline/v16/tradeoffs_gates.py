"""Known physical, null, selection and independent finite-profile controls."""
from itertools import product
import json
from .assembly import World,execute
from .tradeoffs_world import search,distribution,selected_distribution,features
from .tradeoffs_reference import menu,raw_law,retained_law,prediction
from .tradeoffs import public_reader

def fixture():
    world=World()
    training=[{"date":date,"program":[0,6,1,9],"target":[1,0,-1],"feedback":True} for date in range(8)]
    ctx={"date":0,"initial":[1,0,-1],"required_part":2,"orientation_standard":[0,0,0],"price":0.05,
        "max_steps":6,"search_budget":256,"consideration":"all","retention":"random","produced_count":2}
    law={"low":0.2,"high":0.8,"beta":8.0}
    history=[]
    for date in range(8):
        context={**ctx,"date":date}
        found=search(world.public(),context,[0,6,1],[1,0,-1])
        probabilities=distribution(found["options"],context,law,"changing-up")
        choice=max(range(len(probabilities)),key=lambda index:probabilities[index])
        history.append({"context":context,"state":found["options"][choice]["state"]})
    return {"schema_version":"v16.tradeoffs.1","task_id":"known-control","world":world.public(),"law":law,
        "training":training,"history":history,"future_context":{**ctx,"date":8,"produced_count":1}}

def run():
    public=fixture();config=public["world"];ctx=public["future_context"]
    original=search(config,ctx,[0,6,1],[1,0,-1])
    physical=[]
    independent=[]
    # 24 defined nuisance redraws; no unknown scout outcomes participate in gates.
    for defaults in product([0,1],repeat=3):
        for parents in [(-1,0,0),(-1,0,1),(-1,-1,1)]:
            world=World(parents,defaults)
            old=execute(world,[0,6,1])["state"]
            altered={**ctx,"initial":old}
            actual=search(world.public(),altered,[0,6,1],old)
            independent.append(actual==menu(world.public(),altered,[0,6,1],old))
            physical.extend(execute(world,option["program"],initial=old)["state"]==option["state"]
                and execute(world,option["program"],initial=old)["successfully_stopped"] for option in actual["options"])
    known_options=[{"state":[1,0,0],"program":[2,9]},{"state":[0,0,0],"program":[4,6,1,2,9]}]
    low=distribution(known_options,ctx,public["law"],"stable-low")
    high=distribution(known_options,ctx,public["law"],"stable-high")
    null=distribution(original["options"],ctx,public["law"],"absent")
    changed={**ctx,"orientation_standard":[1,1,1]}
    selected={**ctx,"retention":"selected","produced_count":2}
    probability=distribution(original["options"],selected,public["law"],"stable-low")
    analytical=selected_distribution(probability,original["options"],selected)
    enumerated=retained_law(probability,original["options"],selected)
    actual=public_reader(json.dumps(public).encode(),"chronological")
    direct=public_reader(json.dumps(public).encode(),"direct-chronological")
    reference=prediction(public,"chronological")
    checks={
      "tradeoff-physical":{"all_found_plans_actually_execute":all(physical),"24_nuisance_menus_match_independently":all(independent),
        "real_coverage_quality_conflict":low[1]>low[0] and high[0]>high[1],
        "consideration_is_not_feasibility":len(original["considered_targets"])>len(original["options"]),
        "bounded_search_restricts_found_plans":len(search(config,{**ctx,"search_budget":16},[0,6,1],[1,0,-1])["options"])<len(original["options"])},
      "tradeoff-null":{"absent_added_profile_ignores_orientation_standard":null==distribution(original["options"],changed,public["law"],"absent"),
        "singleton_history_is_uninformative":all(distribution(known_options[:1],ctx,public["law"],profile)==[1.0] for profile in ["stable-low","stable-high","changing-up","changing-down","absent"]),
        "named_dimensions_are_distinct":features([1,0,0],ctx)["coverage"]!=features([1,0,0],ctx)["quality"]},
      "tradeoff-selection":{"analytic_law_matches_all_batches":all(abs(a-b)<1e-12 for a,b in zip(analytical,enumerated)),
        "selection_normalizes":abs(sum(analytical)-1)<1e-12,
        "one_candidate_cannot_identify_selection":selected_distribution(probability,original["options"],{**selected,"produced_count":1})==probability},
      "tradeoff-reference":{"direct_joint_same_information":max(abs(a-b) for a,b in zip(actual["probabilities"],direct["probabilities"]))<1e-12,
        "independent_prediction_matches":max(abs(a-b) for a,b in zip(actual["probabilities"],reference["probabilities"]))<1e-12,
        "independent_raw_utility_matches":max(abs(a-b) for a,b in zip(probability,raw_law(original["options"],selected,public["law"],"stable-low")))<1e-12}}
    gates=[{"id":key,"checks":values,"instrument_state":"valid" if all(values.values()) else "failed","evidence_scope":"fixture"} for key,values in checks.items()]
    return {"gates":gates,"instrument_state":"valid" if all(row["instrument_state"]=="valid" for row in gates) else "failed"}
