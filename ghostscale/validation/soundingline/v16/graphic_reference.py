"""Independent list-state interpreter for the sixteen-cell graphic profile."""
def interpret(program,*,initial=0,max_steps=6):
    if type(max_steps) is not int or max_steps<0:
        raise ValueError("invalid execution limit")
    if type(initial) is not int or initial<0 or initial>=2**16:
        raise ValueError("invalid sixteen-cell starting state")
    cells=[(initial//(2**i))%2 for i in range(16)]
    trace=[]
    def artifact():
        return sum(value*(2**i) for i,value in enumerate(cells))
    for index,action in enumerate(program):
        if index==max_steps:
            return {"artifact":artifact(),"legal":False,"stopped":False,"error":"timeout",
                    "primitive_cost":index,"trace":trace}
        before=artifact()
        legal=type(action) is int and action in range(32)
        if legal:
            cells[action%16]=int(action<16)
        trace.append({"before":before,"action":action,"after":artifact(),"legal":legal})
        if not legal:
            return {"artifact":artifact(),"legal":False,"stopped":True,"error":"illegal primitive",
                    "primitive_cost":index+1,"trace":trace}
    return {"artifact":artifact(),"legal":True,"stopped":True,"error":None,"primitive_cost":len(trace),"trace":trace}
