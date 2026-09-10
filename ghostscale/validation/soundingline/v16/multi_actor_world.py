"""M04 actual producer, self/other revision, selection and shared-brief episodes."""
import random
from .recognition import constructor,acquire,produce,decode,recipes
from .graphic_world import execute
from .records import seed_for

def readout(world,artifact):
    style,_=decode(world,{"artifact":artifact,"first_action":None})
    topic=0 if artifact&(1<<world["permutation"][2]) else 1
    return style,topic

def prepare(namespace,condition,index,constructors):
    constructor_id=f"constructor-{index%constructors:03d}"
    world=constructor(namespace,constructor_id,condition)
    return {"world":world,"producer":acquire(namespace,index,0,world),"editor":acquire(namespace,index,1,world),
        "brief":random.Random(seed_for(namespace,"brief",index)).randrange(2),
        "own_purpose":random.Random(seed_for(namespace,"own-purpose",index)).randrange(2),
        "selection_target":random.Random(seed_for(namespace,"selection-target",index)).randrange(2),
        "revision":condition["revision"],"selector":condition["selector"],"shared_brief":condition["shared_brief"],
        "constructor_id":constructor_id}

def candidate(private,namespace,index,label,date,position,*,brief=None):
    world=private["world"]
    active_brief=private["brief"] if brief is None else brief
    topic=active_brief if private["shared_brief"] else private["own_purpose"]
    production=produce(world,private["producer"],random.Random(seed_for(namespace,label,index,date,position,"produce")),topic)
    before=production["execution"]["artifact"]
    old_style,_=readout(world,before)
    revision=private["revision"]
    routine_used=False
    if revision=="none":
        program=[]
    else:
        actor=private["producer"] if revision=="self" else private["editor"]
        acquired=actor["decoration"]
        rng=random.Random(seed_for(namespace,label,index,date,position,"revision"))
        chosen=acquired["choice"] if rng.random()<world["style_reuse"] else 1-acquired["choice"]
        routine_used=chosen==acquired["choice"]
        fragment=list(acquired["routine"]) if routine_used else recipes(world,"decoration")[chosen]
        program=[16+cell for cell in recipes(world,"decoration")[old_style]]+fragment
    revised=execute(program,initial=before)
    style,actual_topic=readout(world,revised["artifact"])
    return {"producer":production,"revision_program":program,"revision_execution":revised,
            "revision_routine_used":routine_used,"artifact":revised["artifact"],"post_style":style,"topic":actual_topic}

def batch(private,namespace,index,label,date,*,count=4,brief=None,bypass_selection=False):
    candidates=[candidate(private,namespace,index,label,date,position,brief=brief) for position in range(count)]
    if private["selector"] and not bypass_selection:
        kept=next((i for i,event in enumerate(candidates) if event["post_style"]==private["selection_target"]),0)
    else:
        kept=random.Random(seed_for(namespace,label,index,date,"retain")).randrange(count)
    return {"candidates":candidates,"retained_index":kept,"count":count,
            "brief":private["brief"] if brief is None else brief,"selection_bypassed":bypass_selection}

def artifact(batch):
    return batch["candidates"][batch["retained_index"]]["artifact"]
