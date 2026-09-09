"""W2 controls: independent physics, dependence, revision, stopping and acquisition."""
from itertools import product
from .assembly import World,execute,legal_histories,fragments,plan
from .assembly_reference import interpret,stopped_histories


def run():
    worlds=[World((-1,0,0),(0,0,0)),World((-1,0,1),(1,0,1)),World((-1,-1,1),(0,1,0))]
    physics=True
    count=0
    for world in worlds:
        # Include illegal and post-stop histories in addition to legal production.
        for length in range(4):
            for program in product(range(10),repeat=length):
                count+=1
                physics &= execute(world,program)==interpret(world.public(),program)
        physics &= set(legal_histories(world))=={program for program,result in stopped_histories(world.public())}
    world=worlds[0]
    work=[{"program":[0,1,9],"target":[0,0,-1]}]*4
    acquired=fragments(work,world)
    positive=plan(world,(0,0,0),library=acquired["library"],budget=4096)
    revision=execute(world,[4,6,1,9],initial=(0,0,-1))
    gate_sets={"assembly-physics":{"independent_interpreter_agrees":physics,
                  "missing_prerequisite_is_illegal":not execute(world,[1,9])["legal"],
                  "support_cannot_be_removed":not execute(world,[0,1,3,9])["legal"],
                  "support_cannot_be_rotated":not execute(world,[0,1,6,9])["legal"],
                  "post_stop_action_illegal":not execute(world,[9,0])["legal"],
                  "missing_stop_is_not_completion":not execute(world,[0,1])["successfully_stopped"]},
               "assembly-revision":{"disassembly_makes_revision_possible":revision["state"]==[1,0,-1] and revision["successfully_stopped"],
                  "downstream_dependency_has_cost":revision["primitive_cost"]==4,
                  "renderer_only_change_cannot_repair":not execute(world,[6,9],initial=(0,0,-1))["legal"]},
               "assembly-acquisition":{"repeated_real_fragment_acquired":acquired["library"]==[[0,1]],
                  "new_composition_executes":execute(world,positive["program"])["state"]==[0,0,0],
                  "failed_examples_do_not_acquire":fragments([{"program":[1,9],"target":[-1,0,-1]}]*4,world)["library"]==[]}}
    gates=[{"id":name,"checks":checks,"instrument_state":"valid" if all(checks.values()) else "failed",
            "evidence_scope":"fixture"} for name,checks in gate_sets.items()]
    return {"gates":gates,"independent_cases":count,
            "instrument_state":"valid" if all(row["instrument_state"]=="valid" for row in gates) else "failed"}
