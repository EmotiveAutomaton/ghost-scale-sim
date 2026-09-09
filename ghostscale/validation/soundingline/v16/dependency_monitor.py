"""S03 known acquired routine, coarse feedback and real assembly repair."""
from collections import Counter
import json
import random
from .assembly import World,constructor,execute,plan
from .assembly_reference import interpret
from .records import seed_for

POLICIES=["self-model","direct-simulation","shape-only","inspect-now","inspect-late"]
DESIGN={"card_id":"S03","question":"Which old procedures escape notice after a goal change, and what downstream repairs do they impose?",
    "conditions":[{"id":f"{kind}-budget-{budget}","kind":kind,"budget":budget}
                  for kind in ["aligned","obvious","subtle","initially-useful"] for budget in [128,512]],
    "arms":POLICIES,"primary":[("inspect-now","inspect-late","collateral_saving","primitive_evaluations",0.5),
        ("self-model","shape-only","success","success_fraction",0.05),
        ("self-model","direct-simulation","success","success_fraction",0.05)],
    "mechanism":"a learned old routine executes under a new purpose; coarse part presence hides orientation, and attached children can obstruct repair",
    "memory_scope":"complete own training and faithful routine-reuse memory; not general unknown-controller recovery",
    "access":"same coarse part presence, own executed training and public physics; inspection reveals orientations only after a paid committed request",
    "learner":"most frequent successful whole routine, at least3 repeats among8 actual trials; maximum3 actions before stop",
    "strong_rival":"ordinary direct forward simulation of the same own routine and known actions, with the same information",
    "sample_rule":{"scout":64,"constructors":8,"expansion":256,"expansion_constructors":20,"final_expansion":1024},
    "dependencies":["dependency-control","dependency-inspection","independent-dependency-scoring"],
    "adversaries":["X01","X04","X05","X08"],"repair_budget":1,
    "continuation":"expand a named noticeability/dependency-cost boundary; aligned routines remain a no-error control"}
EXTENSION="ghostscale.validation.soundingline.v16.dependency_monitor:public_reader"


def learned(training):
    counts=Counter(tuple(item["program"]) for item in training if item["feedback"])
    eligible=[program for program,count in counts.items() if count>=3]
    return list(min(eligible,key=lambda program:(-counts[program],program)))[:-1] if eligible else []


def prepare(namespace,condition,index,constructors):
    constructor_id=f"constructor-{index%constructors:03d}"
    world=constructor(namespace,constructor_id)
    reliability=random.Random(seed_for(namespace,"routine-reliability",constructor_id)).uniform(0.85,1.0)
    if condition["kind"]=="obvious":
        needed={0,2}
        for part in reversed(range(3)):
            if part in needed and world.parents[part]>=0:
                needed.add(world.parents[part])
        recipe=sorted(needed)
    else:
        recipe={"aligned":[0,1],"subtle":[0,6],"initially-useful":[0,6,1]}[condition["kind"]]
    old_goal=execute(world,recipe)["state"]
    training=[]
    for date in range(8):
        program=list(recipe)
        rng=random.Random(seed_for(namespace,condition["kind"],index,"old-training",date))
        if rng.random()>reliability:
            program[rng.randrange(len(program))]=rng.randrange(9)
        program.append(9)
        result=execute(world,program)
        training.append({"date":date,"program":program,"target":old_goal,"execution":result,
                         "feedback":result["legal"] and result["successfully_stopped"] and result["state"]==old_goal})
    routine=learned(training)
    current=execute(world,routine)
    goal=[world.defaults[0],world.defaults[1],-1]
    return {"world":world.public(),"reliability":reliability,"training":training,"routine":routine,
            "current_execution":current,"goal":goal,"constructor_id":constructor_id,
            "old_goal_error_change":sum(a!=b for a,b in zip([-1,-1,-1],goal))-sum(a!=b for a,b in zip(current["state"],goal))}


def conflict(state,goal):
    # Missing desired parts are ordinary unfinished work, not an erroneous commitment.
    return any(actual>=0 and actual!=desired for actual,desired in zip(state,goal))


def public_reader(payload:bytes,policy):
    public=json.loads(payload)
    if set(public)!={"schema_version","task_id","world","training","goal","visible_parts","phase","preparatory_program",
                     "inspected_state","search_budget","old_routine_reused"}:
        raise ValueError("dependency monitor schema violation")
    if public["schema_version"]!="v16.dependency-monitor.1" or policy not in POLICIES:
        raise ValueError("unknown dependency monitor contract")
    if set(public["world"])!={"parents","defaults"} or public["old_routine_reused"] is not True:
        raise ValueError("undeclared world fields or unsupported routine-memory contract")
    world=World(tuple(public["world"]["parents"]),tuple(public["world"]["defaults"]))
    if public["phase"]==1:
        preparation=[]
        if policy=="inspect-late":
            present=set(public["visible_parts"])
            for part in range(3):
                if public["goal"][part]>=0 and part not in present:
                    preparation.append(part)
                    present.add(part)
        return {"preparatory_program":preparation,"request_inspection":policy in {"inspect-now","inspect-late"},
                "costs":{"checking":3,"simulation_primitives":0}}
    if public["phase"]!=2:
        raise ValueError("unknown dependency monitor phase")
    routine=learned(public["training"])
    if public["inspected_state"] is not None:
        assumed=list(public["inspected_state"])
        simulations=0
    elif policy=="self-model":
        assumed=execute(world,routine+public["preparatory_program"])["state"]
        simulations=len(routine)+len(public["preparatory_program"])
    elif policy=="direct-simulation":
        # Same allowed history and physical information, ordinary forward error checking.
        assumed=interpret(world.public(),routine+public["preparatory_program"])["state"]
        simulations=len(routine)+len(public["preparatory_program"])
    else:
        assumed=[world.defaults[part] if part in public["visible_parts"] else -1 for part in range(3)]
        simulations=0
    repair=plan(world,public["goal"],initial=tuple(assumed),budget=public["search_budget"],max_steps=8)
    return {"assumed_state":assumed,"detected_conflict":conflict(assumed,public["goal"]),"repair":repair,
            "costs":{"checking":3,"simulation_primitives":simulations}}
