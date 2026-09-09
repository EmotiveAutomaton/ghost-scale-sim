"""W2: dependency-constrained assembly, revision and explicit stopping.

Parts have binary orientation. Supporting parts cannot be removed or rotated
while a dependent remains attached. A stopped object rejects further actions.
The observation is a finished part/relation graph, not a state-averaged emission.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import json
import random
from .records import seed_for

ATTACH=tuple(range(3))
REMOVE=tuple(range(3,6))
ROTATE=tuple(range(6,9))
STOP=9
ACTIONS=tuple(range(10))
EMPTY=(-1,-1,-1)


@dataclass(frozen=True)
class World:
    parents:tuple=(-1,0,0)
    defaults:tuple=(0,0,0)

    def __post_init__(self):
        if len(self.parents)!=3 or len(self.defaults)!=3 or any(value not in (0,1) for value in self.defaults):
            raise ValueError("W2 requires three binary-orientation parts")
        if any(parent not in range(-1,part) for part,parent in enumerate(self.parents)):
            raise ValueError("W2 dependency parents must precede their children")

    def public(self):
        return {"parents":list(self.parents),"defaults":list(self.defaults)}


def step(world,state,stopped,action):
    if stopped or type(action) is not int or action not in ACTIONS:
        return state,stopped,False
    if action==STOP:
        return state,True,True
    part=action%3
    child_present=any(parent==part and state[child]>=0 for child,parent in enumerate(world.parents))
    updated=list(state)
    if action in ATTACH:
        parent=world.parents[part]
        if state[part]>=0 or (parent>=0 and state[parent]<0):
            return state,stopped,False
        updated[part]=world.defaults[part]
    elif action in REMOVE:
        if state[part]<0 or child_present:
            return state,stopped,False
        updated[part]=-1
    else:
        if state[part]<0 or child_present:
            return state,stopped,False
        updated[part]=1-state[part]
    return tuple(updated),False,True


def artifact(world,state):
    return {"parts":[[part,orientation] for part,orientation in enumerate(state) if orientation>=0],
            "relations":[[parent,child] for child,parent in enumerate(world.parents)
                         if parent>=0 and state[child]>=0]}


def execute(world,program,initial=EMPTY):
    if len(initial)!=3 or any(type(value) is not int or value not in (-1,0,1) for value in initial) or any(
            initial[part]>=0 and parent>=0 and initial[parent]<0 for part,parent in enumerate(world.parents)):
        raise ValueError("invalid initial assembly state")
    state,stopped=tuple(initial),False
    trace=[]
    for action in program:
        next_state,next_stopped,legal=step(world,state,stopped,action)
        trace.append({"action":action,"before":list(state),"after":list(next_state),"legal":legal,
                      "stopped":next_stopped})
        if not legal:
            return {"state":list(state),"artifact":artifact(world,state),"legal":False,"stopped":stopped,
                    "successfully_stopped":False,"primitive_cost":len(trace),"trace":trace}
        state,stopped=next_state,next_stopped
    return {"state":list(state),"artifact":artifact(world,state),"legal":True,"stopped":stopped,
            "successfully_stopped":stopped,"primitive_cost":len(trace),"trace":trace}


def legal_histories(world,max_steps=4):
    def visit(state,program):
        if len(program)<max_steps:
            yield program+(STOP,)
        if len(program)>=max_steps-1:
            return
        for action in ACTIONS[:-1]:
            after,_,legal=step(world,state,False,action)
            if legal:
                yield from visit(after,program+(action,))
    yield from visit(EMPTY,())


def mismatch(state,target):
    return sum(actual!=desired for actual,desired in zip(state,target))


def fragments(training,world):
    counts={}
    processing=0
    for example in training:
        program=tuple(example["program"])
        execution=execute(world,program)
        processing+=execution["primitive_cost"]
        if not execution["legal"] or not execution["successfully_stopped"] or execution["state"]!=example["target"]:
            continue
        for offset in range(len(program)-1):
            fragment=program[offset:offset+2]
            if STOP not in fragment:
                counts[fragment]=counts.get(fragment,0)+1
    # Zero-arity consecutive two-action procedures, two observed uses, fixed cap4.
    library=sorted((part for part,count in counts.items() if count>=2),key=lambda part:(-counts[part],part))[:4]
    return {"library":[list(part) for part in library],"definition_cost":sum(map(len,library)),
            "training_primitives":processing,"training_examples":len(training)}


def plan(world,target,*,initial=EMPTY,library=(),budget=128,max_steps=8):
    """Bounded breadth-first search over executable macro and primitive extensions."""
    if budget<0 or max_steps<1:
        raise ValueError("invalid bounded assembly search")
    vocabulary=[tuple(fragment) for fragment in library]+[(action,) for action in ACTIONS]
    seen={tuple(initial)}
    queue=[(tuple(initial),())]
    cursor,spent=0,0
    attempts=[]
    while cursor<len(queue):
        state,program=queue[cursor]
        cursor+=1
        for fragment in vocabulary:
            if len(program)+len(fragment)>max_steps:
                continue
            next_state,stopped=state,False
            legal=True
            for action in fragment:
                if spent>=budget:
                    return {"program":[],"search_timeout":True,"successor_evaluations":spent,"frontier_states":len(queue),"attempts":attempts}
                spent+=1
                before=next_state
                next_state,stopped,legal=step(world,next_state,stopped,action)
                attempts.append({"before":list(before),"action":action,"after":list(next_state),"legal":legal,"stopped":stopped})
                if not legal:
                    break
            if not legal:
                continue
            proposal=program+fragment
            if stopped:
                if next_state==tuple(target):
                    return {"program":list(proposal),"search_timeout":False,"successor_evaluations":spent,
                            "frontier_states":len(queue),"attempts":attempts}
                continue
            if next_state not in seen:
                seen.add(next_state)
                queue.append((next_state,proposal))
    return {"program":[],"search_timeout":False,"unreachable":True,"successor_evaluations":spent,
            "frontier_states":len(queue),"attempts":attempts}


def public_construct(payload:bytes):
    public=json.loads(payload)
    if set(public)!={"schema_version","task_id","world","training","targets","budget","initial","max_steps"}:
        raise ValueError("assembly public schema violation")
    if public["schema_version"]!="v16.assembly.1":
        raise ValueError("wrong assembly schema")
    world=World(tuple(public["world"]["parents"]),tuple(public["world"]["defaults"]))
    acquired=fragments(public["training"],world)
    return {name:{"submissions":[plan(world,target,initial=tuple(public["initial"]),library=library,
                           budget=public["budget"],max_steps=public["max_steps"]) for target in public["targets"]],
                  "acquisition":acquired}
            for name,library in [("learned",acquired["library"]),("primitive",[])]}


def constructor(namespace,constructor_id):
    rng=random.Random(seed_for(namespace,"assembly-constructor",constructor_id))
    return World(rng.choice([(-1,0,0),(-1,0,1),(-1,-1,1)]),tuple(rng.randrange(2) for _ in range(3)))
