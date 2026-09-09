"""Different executed production mechanisms under one declared finite public world."""
from functools import lru_cache
from itertools import product
from collections import defaultdict
import math
import numpy as np
from . import assembly as a
from . import assembly_inference as ai
from .world import execute as board_execute,histories
from .learning import learn,encoding_cost
from .craft import construct
from .reconstruction import RELIABILITIES,TOPIC_PROBABILITIES
from .records import canonical
from .options import observed_transitions,discover,plan as option_plan

FAMILIES=("softmax","bounded","habit","options")


def families(public_world):
    return FAMILIES if public_world["family"]=="W1" else FAMILIES[:3]


@lru_cache(maxsize=32)
def board_prior(attempts):
    """Retain learned library ORDER; habitual first-fragment production needs it."""
    masses=defaultdict(float)
    for count0 in range(attempts+1):
        for count1 in range(attempts-count0+1):
            failure=attempts-count0-count1
            traces=[(0,1)]*count0+[(2,3)]*count1+[(0,)]*failure
            targets=[3]*count0+[12]*count1+[3]*failure
            library=learn(traces,targets).library
            coefficient=math.factorial(attempts)/(math.factorial(count0)*math.factorial(count1)*math.factorial(failure))
            for reliability,topic,direction in product(RELIABILITIES,TOPIC_PROBABILITIES,range(2)):
                p0=reliability*(topic if direction==0 else 1-topic)
                p1=reliability-p0
                masses[library]+=coefficient*p0**count0*p1**count1*(1-reliability)**failure/18
    libraries=tuple(sorted(masses))
    return libraries,tuple(masses[library] for library in libraries)


def prior(public_world,training):
    if public_world["family"]=="W1":
        return board_prior(training["attempts"])
    world=a.World(tuple(public_world["parents"]),tuple(public_world["defaults"]))
    return ai.prior(world,**training)


def execute(public_world,program):
    if public_world["family"]=="W1":
        result=board_execute(program)
        return {"state":result.artifact,"artifact":result.artifact,"legal":result.legal,
                "finished":result.legal,"primitive_cost":result.primitive_cost}
    world=a.World(tuple(public_world["parents"]),tuple(public_world["defaults"]))
    result=a.execute(world,program)
    return {"state":result["state"],"artifact":result["artifact"],"legal":result["legal"],
            "finished":result["successfully_stopped"],"primitive_cost":result["primitive_cost"]}


def deterministic(public_world,library,goal,family,*,budget=128,max_steps=4):
    if family not in {"bounded","habit","options"}:
        raise ValueError("expected an executed deterministic maker")
    if public_world["family"]=="W1":
        prefix=list(library[0]) if family=="habit" and library else []
        before=board_execute(prefix)
        if not before.legal:
            raise ValueError("learned W1 initial habitual fragment is not executable")
        if family=="options":
            if "option_training" not in public_world:
                raise ValueError("option production requires permitted executed exploration")
            graph=discover(observed_transitions(public_world["option_training"]))
            search=option_plan(goal,options=graph,primitive_budget=budget,max_steps=3)
        else:
            search=construct(goal,library,primitive_budget=budget,start=before.artifact)
        candidate=prefix+search["program"]
        fallback=search["search_timeout"] or len(candidate)>3
        program=prefix if fallback else candidate
    else:
        if family=="options":
            raise ValueError("W2 option production is not admitted by the W1 option instrument")
        world=a.World(tuple(public_world["parents"]),tuple(public_world["defaults"]))
        prefix=list(library[0]) if family=="habit" and library else []
        before=a.execute(world,prefix)
        if not before["legal"]:
            raise ValueError("learned W2 initial habitual fragment is not executable")
        remaining=max_steps-len(prefix)
        search=a.plan(world,goal,initial=tuple(before["state"]),library=library,budget=budget,max_steps=remaining)
        fallback=search["search_timeout"] or search.get("unreachable",False)
        program=prefix+[9] if fallback else prefix+search["program"]
    execution=execute(public_world,program)
    if not execution["legal"] or not execution["finished"]:
        raise ValueError("production mechanism failed to generate a legal finished work")
    return {"program":program,"execution":execution,"habit_prefix":prefix,
            "completion_search":search,"stopped_without_goal_solution":fallback,
            "task_success":execution["state"]==(list(goal) if isinstance(goal,tuple) else goal)}


@lru_cache(maxsize=2048)
def _distribution(encoded_world,library,goal,family,budget,max_steps,beta,length_cost):
    import json
    public_world=json.loads(encoded_world)
    if public_world["family"]=="W1":
        support=tuple(range(16))
        if family=="softmax":
            programs=tuple(histories())
            weights=np.array([math.exp(-beta*(board_execute(program).artifact^goal).bit_count()-
                                       length_cost*encoding_cost(program,library)) for program in programs])
            weights/=weights.sum()
            values=np.zeros(16)
            for program,weight in zip(programs,weights):
                values[board_execute(program).artifact]+=weight
            return support,tuple(float(x) for x in values),programs,tuple(float(x) for x in weights)
    else:
        world=a.World(tuple(public_world["parents"]),tuple(public_world["defaults"]))
        reference=ai.kernel(world,library,goal,max_steps,beta,length_cost)
        support=reference["support"]
        if family=="softmax":
            return support,tuple(map(float,reference["artifact_probabilities"])),reference["programs"],tuple(map(float,reference["route_probabilities"]))
    generated=deterministic(public_world,library,goal,family,budget=budget,max_steps=max_steps)
    state=generated["execution"]["state"]
    state=tuple(state) if isinstance(state,list) else state
    if state not in support:
        raise ValueError("executed deterministic artifact lies outside common route support")
    probabilities=tuple(float(candidate==state) for candidate in support)
    return support,probabilities,(tuple(generated["program"]),),(1.0,)


def distribution(public_world,library,goal,family,*,budget=128,max_steps=4,beta=1.0,length_cost=0.2):
    encoded=canonical(public_world).decode()
    return _distribution(encoded,tuple(map(tuple,library)),tuple(goal) if isinstance(goal,list) else goal,
                         family,budget,max_steps,beta,length_cost)


def draw(public_world,library,goal,family,*,rng,budget=128,max_steps=4,beta=1.0,length_cost=0.2):
    if family!="softmax":
        return deterministic(public_world,library,goal,family,budget=budget,max_steps=max_steps)
    support,probabilities,programs,weights=distribution(public_world,library,goal,family,budget=budget,
                                                       max_steps=max_steps,beta=beta,length_cost=length_cost)
    program=list(rng.choices(programs,weights=weights,k=1)[0])
    execution=execute(public_world,program)
    return {"program":program,"execution":execution,"habit_prefix":[],"completion_search":None,
            "stopped_without_goal_solution":False,
            "task_success":execution["state"]==(list(goal) if isinstance(goal,tuple) else goal)}
