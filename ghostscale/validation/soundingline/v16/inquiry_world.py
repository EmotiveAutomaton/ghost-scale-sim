"""Executed practice and unseen composition/reader tasks for changing competence."""
from itertools import product
import math
import random
from .inquiry import PRIOR,PERMUTATIONS,GOALS,construction,command_for_goal,update,prediction
from .records import seed_for
from .world import execute


def constructor(namespace,constructor_id):
    rng=random.Random(seed_for(namespace,"constructor",constructor_id))
    mappings=[list(rng.choice(PERMUTATIONS)) for _ in range(2)]
    probabilities=[]
    for _ in range(2):
        favorite=rng.randrange(4)
        strength=rng.uniform(0.35,0.85)
        probabilities.append([strength if cell==favorite else (1-strength)/3 for cell in range(4)])
    return {"mappings":mappings,"offer_probabilities":probabilities}


def outcome(world,domain,command,*,noisy,rng):
    cell=rng.randrange(4) if noisy else world["mappings"][domain][command]
    result=execute((cell,))
    return {"command":command,"cell":cell,"program":[cell],"artifact":result.artifact,
            "legal":result.legal,"primitive_cost":result.primitive_cost}


def initial_learning(world,examples,*,namespace,index,noisy_domains):
    beliefs=[tuple(PRIOR),tuple(PRIOR)]
    records=[]
    for domain in range(2):
        order=list(range(4))
        random.Random(seed_for(namespace,index,"initial-order",domain)).shuffle(order)
        for example in range(examples):
            command=order[example%4]
            rng=random.Random(seed_for(namespace,index,"initial-example",domain,example))
            work=outcome(world,domain,command,noisy=domain in noisy_domains,rng=rng)
            beliefs[domain]=update(beliefs[domain],command,work["cell"])
            records.append({"domain":domain,"kind":"permitted demonstration",**work})
    return beliefs,records


def execute_tests(world,programs,*,namespace,index,noisy_domains):
    """All six untrained two-cell compositions in each domain remain one maker cluster."""
    outcomes=[]
    if {(item["domain"],item["goal"]) for item in programs}!=set(product(range(2),GOALS)) or len(programs)!=12:
        raise ValueError("held-out program submission is incomplete")
    for item in programs:
        domain,goal,commands=item["domain"],item["goal"],item["commands"]
        if len(commands)!=2 or any(type(command) is not int or command not in range(4) for command in commands):
            raise ValueError("invalid command program")
        primitives=[]
        for step,command in enumerate(commands):
            # Common counterfactual random draws across reader arms.
            rng=random.Random(seed_for(namespace,index,"heldout",domain,goal,step))
            cell=rng.randrange(4) if domain in noisy_domains else world["mappings"][domain][command]
            primitives.append(cell)
        executed=execute(primitives)
        outcomes.append({"domain":domain,"goal":goal,"commands":commands,"program":primitives,
                         "artifact":executed.artifact,"legal":executed.legal,
                         "success":executed.legal and executed.artifact==goal,
                         "primitive_cost":executed.primitive_cost})
    return outcomes


def maker_task(world,belief,*,namespace,index):
    from .learning import learn
    rng=random.Random(seed_for(namespace,index,"unseen-maker-acquisition"))
    kind=rng.randrange(2)
    commands=(0,1) if kind==0 else (2,3)
    actual=tuple(world["mappings"][0][command] for command in commands)
    target=execute(actual).artifact
    acquisition=learn([actual]*4,[target]*4)
    if acquisition.library!=(actual,):
        raise ValueError("new maker did not acquire the repeated procedure")
    return {"artifact":target,"true_program":list(actual),
            "maker_acquisition":{"attempts":[list(actual)]*4,
                "targets":[target]*4,"learned_library":[list(item) for item in acquisition.library]},
            "new_maker_id":seed_for(namespace,index,"new-maker-id")}
