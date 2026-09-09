"""M01 known-information, ambiguity, symmetry and data-use controls."""
from itertools import product
import json
from .graphic_world import execute,learn,construct
from .graphic_reference import interpret
from .recognition import public_reader

def fixture(*,process=False):
    world={"permutation":list(range(16)),"style_reuse":0.9,"core_reuse":0.9}
    def observation(core,style):
        program=([0,1] if core==0 else [1,0])+[2]+([4,5] if style==0 else [6,7])
        return {"artifact":interpret(program)["artifact"],"first_action":program[0] if process else None}
    return {"schema_version":"v16.recognition.1","task_id":"known-answer","world":world,
            "references":[[observation(0,0)]*3,[observation(1,1)]*3],
            "anonymous":[observation(0,0)]*3}

def run():
    unseen=fixture()
    output=public_reader(json.dumps(unseen).encode(),"craft")
    seen=fixture(process=True)
    conditioned=public_reader(json.dumps(seen).encode(),"craft")
    direct=public_reader(json.dumps(seen).encode(),"direct-table")
    prior=[0.0,0.0]
    for direction in [0,1]:
        for choices in product([0,1],repeat=7):
            weight=0.5
            training=[]
            for choice in choices:
                weight*=0.8 if choice==direction else 0.2
                program=[0,1] if choice==0 else [1,0]
                training.append({"program":program,"target":3,"feedback":True})
            routine=learn(training)["library"][0]
            prior[int(routine==[1,0])]+=weight
    library=[[0,1],[4,5]]
    built=construct(51,library,budget=256)
    primitive=construct(51,[],budget=256)
    cases={
      "large-graphic-physics":{"independent_execution":execute([0,1,2,4,5])==interpret([0,1,2,4,5]),
                               "larger_board_reaches_cell_fifteen":interpret([15])["artifact"]==32768,
                               "six_step_boundary":not execute([0]*7)["stopped"]},
      "large-graphic-acquisition":{"independent_acquisition_prior":all(abs(x-0.5)<1e-12 for x in prior),
          "executable_new_composition":not built["search_timeout"] and interpret(built["program"])["artifact"]==51,
          "counted_primitive_rival_misses_budget":primitive["search_timeout"]},
      "recognition-collision":{"familiar_style_identifies_source":output["identity"][0]>0.9,
          "style_does_not_reveal_independent_core":all(abs(x-0.5)<1e-12 for x in output["future_core"]),
          "current_route_remains_ambiguous":all(abs(x-0.5)<1e-12 for x in output["historical_core"])},
      "recognition-data-use":{"process_evidence_changes_prediction":conditioned["future_core"][0]>0.8,
          "direct_joint_rival_matches":all(abs(a-b)<1e-12 for key in ["identity","future_core","future_decoration"]
                                          for a,b in zip(conditioned[key],direct[key])),
          "data_ignoring_break_detected":max(abs(a-b) for a,b in zip(output["future_core"],conditioned["future_core"]))>0.2,
          "process_recovers_observed_current_order":conditioned["historical_core"]==[1.0,0.0]}}
    gates=[{"id":name,"checks":checks,"instrument_state":"valid" if all(checks.values()) else "failed","evidence_scope":"fixture"}
           for name,checks in cases.items()]
    return {"gates":gates,"instrument_state":"valid" if all(row["instrument_state"]=="valid" for row in gates) else "failed"}

