"""Independent checks of actual large-graphic maker learning and routine expansion."""
from collections import Counter
import hashlib
import json
import random
from .graphic_reference import interpret

def seed(*parts):
    return int(hashlib.sha256(json.dumps(parts,sort_keys=True,separators=(",",":")).encode()).hexdigest()[:16],16)

def verify_maker(world,maker,namespace,index,maker_id=0):
    p=world["permutation"]
    if set(maker)!={"core","decoration"}:
        raise ValueError("unknown acquired routine family")
    cost=0;definitions=0
    for kind,part in maker.items():
        options=[p[:2],list(reversed(p[:2]))] if kind=="core" else [p[4:6],p[6:8]]
        rng=random.Random(seed(namespace,"training",index,maker_id,kind))
        direction=rng.randrange(2)
        counts=Counter()
        if len(part["training"])!=7:
            raise ValueError("incomplete training history")
        for date,trial in enumerate(part["training"]):
            choice=direction if rng.random()<world["training_reliability"] else 1-direction
            result=interpret(options[choice])
            if trial!={"date":date,"program":options[choice],"target":result["artifact"],"feedback":result["legal"],"execution":result}:
                raise ValueError("training does not match actual sampled primitive executions")
            counts[tuple(options[choice])]+=1
            cost+=result["primitive_cost"]
        available=[program for program,count in counts.items() if count>=4]
        routine=list(min(available,key=lambda program:(-counts[program],program)))
        expected={"library":[routine],"definition_cost":2,"processing_primitives":14}
        if part["routine"]!=routine or part["compiled"]!=expected or part["choice"]!=options.index(routine) or part["teaching_direction"]!=direction:
            raise ValueError("routine acquisition did not regenerate")
        definitions+=2
    return cost,definitions

def verify_production(world,event,maker,rng,topic,changed=False):
    p=world["permutation"]
    core=maker["core"]["choice"];style=maker["decoration"]["choice"]
    core=core if rng.random()<world["core_reuse"] else 1-core
    style=1-style if changed else style
    style=style if rng.random()<world["style_reuse"] else 1-style
    core_program=p[:2] if core==0 else list(reversed(p[:2]))
    decoration=p[4+2*style:6+2*style]
    program=core_program+[p[2+topic]]+decoration
    result=interpret(program)
    if event!={"program":program,"execution":result,"core_order":core,"decoration":style,"topic":topic,
               "core_routine_used":core_program==maker["core"]["routine"],
               "decoration_routine_used":decoration==maker["decoration"]["routine"]}:
        raise ValueError("actual maker production did not independently regenerate")
    return result["primitive_cost"]

