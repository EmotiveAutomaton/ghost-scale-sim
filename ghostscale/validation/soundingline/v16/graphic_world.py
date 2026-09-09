"""Larger W1 profile: sixteen cells, thirty-two primitives and six steps."""
from collections import Counter
from itertools import product

def execute(program, *, initial=0, max_steps=6):
    if type(max_steps) is not int or max_steps<0:
        raise ValueError("invalid execution limit")
    if type(initial) is not int or not 0 <= initial < 65536:
        raise ValueError("invalid sixteen-cell starting state")
    board=initial
    trace=[]
    for index,action in enumerate(program):
        if index>=max_steps:
            return {"artifact":board,"legal":False,"stopped":False,"error":"timeout",
                    "primitive_cost":index,"trace":trace}
        before=board
        legal=type(action) is int and 0<=action<32
        if legal:
            board=board | (1<<action) if action<16 else board & ~(1<<(action-16))
        trace.append({"before":before,"action":action,"after":board,"legal":legal})
        if not legal:
            return {"artifact":board,"legal":False,"stopped":True,"error":"illegal primitive",
                    "primitive_cost":index+1,"trace":trace}
    return {"artifact":board,"legal":True,"stopped":True,"error":None,
            "primitive_cost":len(trace),"trace":trace}

def learn(training):
    counts=Counter()
    processing=0
    for trial in training:
        result=execute(trial["program"])
        processing+=result["primitive_cost"]
        if trial["feedback"] and result["legal"] and result["artifact"]==trial["target"]:
            counts[tuple(trial["program"])]+=1
    eligible=[p for p,count in counts.items() if len(p)==2 and count>=4]
    library=[list(min(eligible,key=lambda p:(-counts[p],p)))] if eligible else []
    return {"library":library,"definition_cost":sum(map(len,library)),"processing_primitives":processing}

def expand(tokens,library):
    result=[]
    for token in tokens:
        if type(token) is int:
            result.append(token)
        elif isinstance(token,str) and token.startswith("m") and token[1:].isdigit():
            index=int(token[1:])
            if index>=len(library):
                raise ValueError("unknown acquired routine")
            result.extend(library[index])
        else:
            raise ValueError("invalid program token")
    return result

def construct(target,library,*,budget,initial=0):
    if type(budget) is not int or budget<0 or type(target) is not int or not 0<=target<65536:
        raise ValueError("invalid construction target or budget")
    tokens=[f"m{i}" for i in range(len(library))]+list(range(32))
    attempts=[]
    spent=0
    for length in range(7):
        for code in product(tokens,repeat=length):
            program=expand(code,library)
            charge=min(len(program),6)
            if spent+charge>budget:
                return {"program":[],"tokens":[],"attempts":attempts,"search_primitives":spent,"search_timeout":True}
            result=execute(program,initial=initial)
            spent+=result["primitive_cost"]
            attempts.append({"tokens":list(code),"execution":result})
            if result["legal"] and result["artifact"]==target:
                return {"program":program,"tokens":list(code),"attempts":attempts,
                        "search_primitives":spent,"search_timeout":False}
    return {"program":[],"tokens":[],"attempts":attempts,"search_primitives":spent,"search_timeout":True}
