"""Independent W2 interpreter and scalar likelihood enumeration.

No import from the primary assembly implementation; the duplicated physics is
intentional so an implementation mistake can be detected by disagreement.
"""
from itertools import product
from math import exp


def interpret(config,program,initial=(-1,-1,-1)):
    if len(initial)!=3 or any(type(value) is not int or value not in (-1,0,1) for value in initial):
        raise ValueError("invalid initial part vector")
    for child,parent in enumerate(config["parents"]):
        if initial[child]>=0 and parent>=0 and initial[parent]<0:
            raise ValueError("initial graph contains an unsupported part")
    parts={index:orientation for index,orientation in enumerate(initial) if orientation>=0}
    parents=config["parents"]
    halted=False
    trace=[]
    valid=True
    for command in program:
        before=[parts.get(index,-1) for index in range(3)]
        allowed=not halted and type(command) is int and 0<=command<=9
        if allowed and command!=9:
            index=command%3
            supported=any(child in parts and parent==index for child,parent in enumerate(parents))
            if command<3:
                allowed=index not in parts and (parents[index]<0 or parents[index] in parts)
                if allowed:
                    parts[index]=config["defaults"][index]
            elif command<6:
                allowed=index in parts and not supported
                if allowed:
                    del parts[index]
            else:
                allowed=index in parts and not supported
                if allowed:
                    parts[index]=1-parts[index]
        elif allowed:
            halted=True
        trace.append({"action":command,"before":before,"after":[parts.get(index,-1) for index in range(3)],
                      "legal":allowed,"stopped":halted})
        if not allowed:
            valid=False
            break
    artifact={"parts":[[index,parts[index]] for index in sorted(parts)],
              "relations":[[parent,child] for child,parent in enumerate(parents) if parent>=0 and child in parts]}
    return {"state":[parts.get(index,-1) for index in range(3)],"artifact":artifact,"legal":valid,
            "stopped":halted,"successfully_stopped":valid and halted,"primitive_cost":len(trace),"trace":trace}


def stopped_histories(config,max_steps=4):
    for length in range(1,max_steps+1):
        for prefix in product(range(9),repeat=length-1):
            program=prefix+(9,)
            result=interpret(config,program)
            if result["legal"]:
                yield program,result


def scalar_likelihood(config,target,artifact,*,max_steps=4,beta=1.0,length_cost=0.2):
    total,matching=0.0,0.0
    for program,result in stopped_histories(config,max_steps):
        distance=sum(a!=b for a,b in zip(result["state"],target))
        weight=exp(-beta*distance-length_cost*len(program))
        total+=weight
        if result["artifact"]==artifact:
            matching+=weight
    return matching/total
