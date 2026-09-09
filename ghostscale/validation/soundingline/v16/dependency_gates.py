"""Known aligned, hidden-orientation and dependency-removal controls."""
from .assembly import World,execute
from .assembly_reference import interpret
from .dependency_monitor import conflict,learned


def run():
    world=World()
    goal=[0,0,-1]
    subtle=execute(world,[0,6])["state"]
    useful=execute(world,[0,6,1])["state"]
    early=interpret(world.public(),[6,1,9],initial=subtle)
    late=interpret(world.public(),[4,6,1,9],initial=useful)
    cases={"dependency-control":{"orientation_conflict_is_real":conflict(subtle,goal),
        "shape_defaults_miss_orientation":not conflict([0,-1,-1],goal),
        "aligned_routine_has_no_conflict":not conflict([0,0,-1],goal),
        "unfinished_absence_is_not_false_alarm":not conflict([-1,-1,-1],goal),
        "empty_training_has_no_invented_routine":learned([])==[]},
        "dependency-inspection":{"early_repair_executes":early["successfully_stopped"] and early["state"]==goal,
            "late_repair_requires_real_removal":late["successfully_stopped"] and late["state"]==goal and late["trace"][0]["action"]==4,
            "blocked_rotation_is_detected":not interpret(world.public(),[6,9],initial=useful)["legal"],
            "independent_root_removes_dependency":interpret({"parents":[-1,-1,1],"defaults":[0,0,0]},[6,9],initial=useful)["legal"]}}
    gates=[{"id":name,"checks":checks,"instrument_state":"valid" if all(checks.values()) else "failed","evidence_scope":"fixture"}
           for name,checks in cases.items()]
    return {"gates":gates,"instrument_state":"valid" if all(row["instrument_state"]=="valid" for row in gates) else "failed"}
