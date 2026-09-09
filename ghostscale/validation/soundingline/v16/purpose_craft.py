"""K03: acquired routines under changed artifact purposes."""
import json
import random
from .craft import acquire,draw_constructor,construct
from .learning import learn
from .world import execute
from .records import seed_for

DESIGN={"card_id":"K03","question":"When a purpose changes, does continuing, inhibiting or relearning acquired craft help?",
        "conditions":[{"id":f"{relation}-budget-{budget}","relation":relation,"budget":budget}
                      for relation in ["aligned","partially-aligned","opposed"] for budget in [8,32,128]],
        "arms":["continued","inhibited","relearned","primitive"],
        "primary":[("inhibited","continued","success","success_fraction",0.05),
                   ("relearned","continued","success","success_fraction",0.05)],
        "mechanism":"old acquired two-action fragments change search order; inhibition checks actual goal-worsening steps; recent real trials can replace the active library",
        "access":"all arms receive the same old and new training opportunities; library-update policies differ explicitly",
        "alignment_reference":"all 585 independent primitive histories; old/new negative Hamming distances and occupied-cell overlap",
        "training":"16 old plus 16 adaptation trials; all charged, including failures and arms retaining old library",
        "scope":"relearning may train the exact new purpose; this is adaptation, not a held-out composition claim",
        "sample_rule":{"scout":64,"constructors":8,"expansion":256,"expansion_constructors":20,"final_expansion":1024},
        "dependencies":["purpose-realization","purpose-inhibition","independent-purpose-costs"],
        "adversaries":["X04","X08"],"repair_budget":1,
        "continuation":"expand a named reuse/interference or checking-cost boundary; harmful habits are not a required result"}
EXTENSION="ghostscale.validation.soundingline.v16.purpose_craft:public_read"


def prepare(namespace,condition,index,constructors):
    constructor_id=f"constructor-{index%constructors:03d}"
    config=draw_constructor(namespace,constructor_id)
    direction=random.Random(seed_for(namespace,index,"old-topic")).randrange(2)
    _,old=acquire(random.Random(seed_for(namespace,index,"old-training")),config,direction,16)
    permutation=config["permutation"]
    previous=permutation[:2] if direction==0 else permutation[2:]
    other=permutation[2:] if direction==0 else permutation[:2]
    cells={"aligned":[*previous,other[0]],"partially-aligned":[previous[0],other[0]],"opposed":other}[condition["relation"]]
    target=sum(1<<cell for cell in cells)
    rng=random.Random(seed_for(namespace,condition["relation"],index,"adaptation-training"))
    attempts,targets,instructions=[],[],[]
    for trial in range(16):
        instruction=tuple(cells[-2:])
        performed=list(instruction)
        if rng.random()>config["instruction_reliability"]:
            performed[rng.randrange(2)]=rng.randrange(8)
        attempts.append(performed)
        targets.append(sum(1<<cell for cell in instruction))
        instructions.append(list(instruction))
    new={"attempts":attempts,"targets":targets,"instructions":instructions,"dates":list(range(16,32))}
    public={"schema_version":"v16.purpose-craft.1","task_id":"pending opaque transport","old_training":old,
            "new_training":new,"old_target":sum(1<<cell for cell in previous),"target":target,"budget":condition["budget"]}
    private={"constructor":config,"direction":direction,"old_goal_cells":previous,"new_goal_cells":cells}
    return public,private,constructor_id


def inhibition(library,target):
    retained=[]
    checked=[]
    for fragment in library:
        state=0
        worsens=False
        for action in fragment:
            result=execute((action,),start=state)
            before=(state^target).bit_count()
            after=(result.artifact^target).bit_count()
            checked.append({"before":state,"action":action,"after":result.artifact,"goal_error_before":before,
                            "goal_error_after":after,"primitive_cost":result.primitive_cost})
            worsens |= not result.legal or after>before
            state=result.artifact
        if not worsens:
            retained.append(fragment)
    return retained,checked


def public_read(payload:bytes):
    public=json.loads(payload)
    if set(public)!={"schema_version","task_id","old_training","new_training","old_target","target","budget"}:
        raise ValueError("purpose-craft public schema violation")
    if public["schema_version"]!="v16.purpose-craft.1":
        raise ValueError("wrong purpose-craft schema")
    old=learn(public["old_training"]["attempts"],public["old_training"]["targets"])
    new=learn(public["new_training"]["attempts"],public["new_training"]["targets"])
    retained,checks=inhibition(old.library,public["target"])
    arms={}
    for name,library in [("continued",old.library),("inhibited",retained),("relearned",new.library),("primitive",[])]:
        result=construct(public["target"],library,primitive_budget=public["budget"])
        arms[name]={"submission":result,"library":[list(fragment) for fragment in library],
                    "checking":checks if name=="inhibited" else [],
                    "costs":{"training_primitives":old.processing_cost+new.processing_cost,
                             "old_library_definition":old.definition_cost if name!="primitive" else 0,
                             "new_library_definition":new.definition_cost if name=="relearned" else 0,
                             "checking_primitives":sum(item["primitive_cost"] for item in checks) if name=="inhibited" else 0}}
    return arms
