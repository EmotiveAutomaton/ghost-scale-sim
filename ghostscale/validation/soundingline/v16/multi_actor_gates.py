"""M04 known collisions, actual actor interventions and probe separation."""
import copy
import json
from .multi_actor_world import prepare,batch,artifact
from .recognition import observe
from .multi_actor import public_reader
from .multi_actor_reference import predict

def fixture(policy="all",*,revision="self",selector=True,shared=True,count=4,family="learned-order",phase=2):
    condition={"family":family,"revision":revision,"selector":selector,"shared_brief":shared}
    namespace="multi-actor-known-controls"
    # Deliberately construct the known collision: independent training lands on
    # the same decorative routine. No scientific unknown-outcome selection.
    for index in range(32):
        construction=prepare(namespace,condition,index,8)
        if construction["producer"]["decoration"]["choice"]==construction["editor"]["decoration"]["choice"]:
            break
    else:
        raise ValueError("known matching-routine fixture was not constructed")
    history=[batch(construction,namespace,index,"history",date,count=count) for date in range(3)]
    unselected=batch(construction,namespace,index,"probe-unselected",0,count=1,bypass_selection=True)
    flipped=batch(construction,namespace,index,"probe-brief",0,count=1,brief=1-construction["brief"],bypass_selection=True)
    current=history[0]["candidates"][history[0]["retained_index"]]
    all_views=policy in {"all","direct-all"} and phase==2
    producer=(all_views or policy=="producer-view") and phase==2
    editor=(all_views or policy=="revision-view") and phase==2
    raw=(all_views or policy=="unselected-view") and phase==2
    changed=(all_views or policy=="brief-flip") and phase==2
    public={"schema_version":"v16.multi-actor.1","task_id":"known-role-collision","phase":phase,
        "world":{key:construction["world"][key] for key in ["permutation","style_reuse","core_reuse"]},
        "brief":construction["brief"],"produced_count":count,"history":[artifact(episode) for episode in history],
        "producer_view":observe(current["producer"],process=True) if producer else None,
        "revision_view":{"before":current["producer"]["execution"]["artifact"],"after":current["artifact"],"program":current["revision_program"]} if editor else None,
        "unselected_view":{"artifact":artifact(unselected)} if raw else None,
        "brief_view":{"new_brief":1-construction["brief"],"artifact":artifact(flipped)} if changed else None}
    return public

def run():
    own=fixture(revision="self")
    other=fixture(revision="other")
    result=public_reader(json.dumps(own).encode(),"all")
    direct=public_reader(json.dumps(own).encode(),"direct-all")
    independent=predict(own)
    singleton=public_reader(json.dumps(fixture(count=1)).encode(),"all")
    absent=public_reader(json.dumps(fixture(revision="none")).encode(),"all")
    indifferent=public_reader(json.dumps(fixture(family="goal-indifferent")).encode(),"all")
    unchanged=fixture(revision="self")
    changed=copy.deepcopy(unchanged)
    changed["revision_view"]["program"][:2]=reversed(changed["revision_view"]["program"][:2])
    recoded=public_reader(json.dumps(changed).encode(),"all")
    controlled=public_reader(json.dumps(fixture(shared=True)).encode(),"all")
    own_goal=public_reader(json.dumps(fixture(shared=False)).encode(),"all")
    arrays=["producer_core","producer_style","revision","release","brief","topology","revision_relation"]
    checks={
        "multi-actor-physical":{"real_revision_changes_or_rebuilds_existing_work":len(own["revision_view"]["program"])==4,
             "actual_no_revision_control":abs(absent["revision"][0]-1)<1e-12,
             "coordinate_indifferent_core_null":all(abs(value-0.5)<1e-12 for value in indifferent["producer_core"])},
        "multi-actor-collisions":{"self_and_other_have_identical_allowed_evidence":own==other,
             "reader_does_not_force_second_maker":result["revision_relation"][1]>0 and result["revision_relation"][2]>0,
             "single_candidate_cannot_reveal_selector":all(abs(value-0.5)<1e-12 for value in singleton["selector"]),
             "behavior_preserving_revision_reordering":all(abs(a-b)<1e-12 for key in arrays for a,b in zip(result[key],recoded[key]))},
        "multi-actor-probes":{"brief_following_identified_by_real_intervention":abs(controlled["shared_brief"][1]-1)<1e-12,
             "fixed_own_purpose_survives_changed_brief":abs(own_goal["shared_brief"][0]-1)<1e-12,
             "direct_joint_same_information":all(abs(a-b)<1e-12 for key in arrays for a,b in zip(result[key],direct[key])),
             "factorized_independent_reference":all(abs(a-b)<1e-12 for key in arrays for a,b in zip(result[key],independent[key]))}}
    gates=[{"id":key,"checks":values,"instrument_state":"valid" if all(values.values()) else "failed","evidence_scope":"fixture"}
           for key,values in checks.items()]
    return {"gates":gates,"instrument_state":"valid" if all(row["instrument_state"]=="valid" for row in gates) else "failed"}
