"""Selection nulls and positive information controls from finite batch enumeration."""
import copy
import json
from .graphic_reference import interpret
from .recognition_gates import run as inherited_gates
from .selection import public_reader,selection_style_probability
from .audit_selection import selected_probability

def fixture(*,count=4,full=False,retention="selected"):
    world={"permutation":list(range(16)),"core_reuse":0.9,"style_reuse":0.9}
    def obs(style):
        program=[0,1,2]+([4,5] if style==0 else [6,7])
        return {"artifact":interpret(program)["artifact"],"first_action":None}
    styles=[1]*(count-1)+[0]
    kept=list(range(count)) if retention=="all" else [count-1]
    events=[obs(style) for style in styles]
    batch={"released":[events[i] for i in kept],
           "full_candidates":[{"observation":event,"retained":i in kept} for i,event in enumerate(events)] if full else None}
    return {"schema_version":"v16.selection.1","task_id":"known-selection","world":world,
            "history":[copy.deepcopy(batch) for _ in range(3)],"retention":retention,"produced_count":count,"audience":0}

def run():
    world=fixture()["world"]
    errors=[]
    for n in [1,4,8]:
        for audience in [0,1]:
            for acquired in [0,1]:
                for style in [0,1]:
                    errors.append(abs(selection_style_probability(style,acquired,audience,n,world)-
                                      selected_probability(world,style,acquired,audience,n)))
    one=fixture(count=1)
    null=public_reader(json.dumps(one).encode(),"selection-aware")
    naive_null=public_reader(json.dumps(one).encode(),"release-naive")
    public=fixture()
    aware=public_reader(json.dumps(public).encode(),"selection-aware")
    naive=public_reader(json.dumps(public).encode(),"release-naive")
    direct=public_reader(json.dumps(public).encode(),"direct-table")
    paid=public_reader(json.dumps(fixture(full=True)).encode(),"selection-aware")
    checks={
      "selection-law":{"independent_exhaustive_batches":max(errors)<1e-12,
          "one_candidate_no_selection_effect":all(abs(a-b)<1e-12 for a,b in zip(null["future_raw_style"],naive_null["future_raw_style"])),
          "raw_and_released_targets_differ":abs(aware["future_raw_style"][0]-aware["future_release_style"][0])>1e-4},
      "selection-information":{"selection_changes_latent_evidence":aware["acquired_style"][1]>naive["acquired_style"][1]+1e-4,
          "rejected_work_can_reverse_account":paid["acquired_style"][1]>0.99,
          "same_information_direct_joint_matches":all(abs(a-b)<1e-12 for key in ["future_raw_style","future_release_style","acquired_style"]
                                                      for a,b in zip(aware[key],direct[key])),
          "unseen_independent_core_remains_ambiguous":all(abs(x-0.5)<1e-12 for x in aware["future_raw_core"])}}
    gates=[{"id":name,"checks":values,"instrument_state":"valid" if all(values.values()) else "failed","evidence_scope":"fixture"}
           for name,values in checks.items()]
    gates.extend(row for row in inherited_gates()["gates"] if row["id"] in {"large-graphic-physics","large-graphic-acquisition"})
    return {"gates":gates,"instrument_state":"valid" if all(row["instrument_state"]=="valid" for row in gates) else "failed"}

