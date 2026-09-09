"""Production realization, misspecification and equal-information reference controls."""
from .mechanism_models import deterministic,board_prior
from .mechanism_reader import public_reader
from .records import canonical
from .assembly_gates import run as assembly_gates


def run():
    world={"family":"W1","option_training":[[0,1],[2,3]]*8}
    bounded=deterministic(world,((0,1),),12,"bounded",budget=4096,max_steps=3)
    habit=deterministic(world,((0,1),),12,"habit",budget=4096,max_steps=3)
    empty_bounded=deterministic(world,(),12,"bounded",budget=4096,max_steps=3)
    empty_habit=deterministic(world,(),12,"habit",budget=4096,max_steps=3)
    libraries,masses=board_prior(8)
    public={"schema_version":"v16.mechanism-reader.1","task_id":"opaque","world":world,"training":{"attempts":8},
            "current":{"goal":3,"artifact":3},"prior_works":[{"goal":3,"artifact":3}]*4,"future_goal":12,
            "production_budget":128,"max_steps":3,"beta":1.0,"length_cost":0.3,"reader_budget":128}
    mixture=public_reader(canonical(public),"mixture")
    direct=public_reader(canonical(public),"direct-mixture")
    public["prior_works"]=[{"goal":3,"artifact":0}]*4
    changed=public_reader(canonical(public),"mixture")
    public["prior_works"]=[]
    public["current"]["artifact"]=4
    reverse=public_reader(canonical(public),"bounded-model")
    broad=public_reader(canonical(public),"mixture")
    cases={"mechanism-production":{"bounded_executes_new_goal":bounded["task_success"],
            "committed_old_steps_can_block_goal":not habit["task_success"] and habit["habit_prefix"]==[0,1],
            "no_skill_habit_null":empty_bounded["program"]==empty_habit["program"],
            "ordered_libraries_retained":((0,1),(2,3)) in libraries and ((2,3),(0,1)) in libraries,
            "prior_normalization":abs(sum(masses)-1)<1e-10},
           "mechanism-inference":{"equal_information_joint_prediction":max(abs(a-b) for a,b in zip(mixture["future_probabilities"],direct["future_probabilities"]))<1e-10,
            "data_ignoring_break_detected":max(abs(mixture["family_posterior"][name]-changed["family_posterior"][name]) for name in mixture["family_posterior"])>1e-6,
            "reverse_off_model_is_not_uniform":reverse["model_mismatch"] and "future_probabilities" not in reverse,
            "broader_model_retains_support":not broad["model_mismatch"]}}
    gates=[{"id":name,"checks":checks,"instrument_state":"valid" if all(checks.values()) else "failed","evidence_scope":"fixture"}
           for name,checks in cases.items()]
    gates.extend(assembly_gates()["gates"])
    return {"gates":gates,"instrument_state":"valid" if all(row["instrument_state"]=="valid" for row in gates) else "failed"}
