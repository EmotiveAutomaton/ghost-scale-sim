"""Native projector-based options from observed transitions only.

The normalized-Laplacian motivation follows Machado, Bellemare and Bowling
(2017), https://arxiv.org/abs/1703.00956. The basis-invariant extrema and directed
shortest-path policy are this finite adaptation, not their full eigenbehavior RL.
"""
from collections import deque
import numpy as np
from .world import execute


def observed_transitions(attempts):
    records=[]
    for episode,program in enumerate(attempts):
        state=0
        for step,action in enumerate(program):
            result=execute((action,),start=state)
            records.append({"episode":episode,"step":step,"before":state,"action":action,
                            "after":result.artifact,"legal":result.legal,"primitive_cost":result.primitive_cost})
            if not result.legal:
                break
            state=result.artifact
    return records


def extrema(projector,tolerance=1e-10):
    pairs=[]
    for left in range(len(projector)):
        for right in range(left+1,len(projector)):
            distance=float(np.sum((projector[left]-projector[right])**2))
            pairs.append((distance,left,right))
    if not pairs:
        return []
    maximum=max(item[0] for item in pairs)
    chosen=min((left,right) for distance,left,right in pairs if distance>=maximum-tolerance)
    return list(chosen)


def discover(transitions,*,option_cap=4):
    nodes=sorted({record[side] for record in transitions for side in ["before","after"]})
    edges={}
    adjacency={node:set() for node in nodes}
    for record in transitions:
        if not record["legal"]:
            continue
        source,target,action=record["before"],record["after"],record["action"]
        key=(source,action)
        if key in edges and edges[key]!=target:
            raise ValueError("deterministic options received contradictory transitions")
        edges[key]=target
        if source!=target:
            adjacency[source].add(target)
            adjacency[target].add(source)
    components=[]
    remaining=set(nodes)
    while remaining:
        frontier=[min(remaining)]
        component=set()
        while frontier:
            node=frontier.pop()
            if node in component:
                continue
            component.add(node)
            frontier.extend(adjacency[node]-component)
        remaining-=component
        components.append(sorted(component))
    options=[]
    diagnostics=[]
    for component in sorted(components,key=lambda values:(-len(values),values)):
        if len(component)<2:
            diagnostics.append({"nodes":component,"eigenvalues":[0.0],"first_nonzero_multiplicity":0,
                                "targets":[],"status":"isolated observed state"})
            continue
        matrix=np.array([[float(right in adjacency[left]) for right in component] for left in component])
        degree=matrix.sum(axis=1)
        inverse=1/np.sqrt(degree)
        laplacian=np.eye(len(component))-inverse[:,None]*matrix*inverse[None,:]
        eigenvalues,vectors=np.linalg.eigh(laplacian)
        positive=[i for i,value in enumerate(eigenvalues) if value>1e-10]
        lowest=eigenvalues[positive[0]]
        indices=[i for i,value in enumerate(eigenvalues) if abs(value-lowest)<1e-10]
        basis=vectors[:,indices]
        projector=basis@basis.T
        targets=[component[i] for i in extrema(projector)]
        diagnostics.append({"nodes":component,"eigenvalues":eigenvalues.tolist(),
                            "first_nonzero_multiplicity":len(indices),"targets":targets,
                            "projector":projector.tolist(),"status":"observed subspace"})
        for target in targets:
            if len(options)>=option_cap:
                break
            distances={target:0}
            queue=deque([target])
            while queue:
                destination=queue.popleft()
                for (source,action),after in sorted(edges.items()):
                    if after==destination and source not in distances:
                        distances[source]=distances[destination]+1
                        queue.append(source)
            policy={}
            for source in sorted(distances):
                if source==target:
                    continue
                candidates=[action for (before,action),after in edges.items()
                            if before==source and distances.get(after)==distances[source]-1]
                policy[str(source)]=min(candidates)
            options.append({"target":target,"policy":policy,"distances":{str(k):v for k,v in distances.items()},
                            "initiation":sorted(policy,key=int),"unreachable_observed_states":sorted(set(nodes)-set(distances))})
    return {"options":options,"observed_nodes":nodes,"observed_directed_state_actions":len(edges),
            "observed_state_fraction":len(nodes)/16,"component_count":len(components),"spectral_components":diagnostics,
            "exploration_primitives":sum(record["primitive_cost"] for record in transitions),
            "tabular_policy_entries":sum(len(option["policy"]) for option in options),
            "option_definition_cost":sum(len(option["policy"])+1 for option in options),
            "edges":[{"before":source,"action":action,"after":target} for (source,action),target in sorted(edges.items())]}


def expand(option,state,edges):
    known={(record["before"],record["action"]):record["after"] for record in edges}
    program=[]
    start=state
    for _ in range(len(option["policy"])+1):
        if state==option["target"]:
            return {"program":program,"terminated":True,"unsupported":False,"start":start,"end":state}
        if str(state) not in option["policy"]:
            return {"program":program,"terminated":False,"unsupported":True,"start":start,"end":state}
        action=option["policy"][str(state)]
        if (state,action) not in known:
            raise ValueError("option policy requests an unobserved transition")
        program.append(action)
        state=known[(state,action)]
    raise ValueError("learned option failed its finite termination bound")


def plan(target,*,library=(),options=None,primitive_budget=128,max_steps=3,start=0):
    attempts,option_expansions=[],[]
    def result(program,timeout,spent,calls,**extra):
        return {"program":program,"search_timeout":timeout,"successor_evaluations":spent,"option_calls":calls,
                "option_policy_lookups":sum(len(item["expansion"]["program"]) for item in option_expansions),
                "attempts":attempts,"option_expansions":option_expansions,**extra}
    if target==start:
        return result([],False,0,0)
    queue=deque([(start,[])])
    seen={start}
    spent,calls=0,0
    while queue:
        state,program=queue.popleft()
        candidates=[]
        if options is not None:
            for option in options["options"]:
                proposal=expand(option,state,options["edges"])
                calls+=1
                option_expansions.append({"state":state,"target":option["target"],"expansion":proposal})
                if proposal["terminated"] and proposal["program"]:
                    candidates.append(proposal["program"])
        candidates.extend([list(fragment) for fragment in library])
        candidates.extend([[action] for action in range(8)])
        for candidate in candidates:
            if len(program)+len(candidate)>max_steps:
                continue
            if spent+len(candidate)>primitive_budget:
                return result([],True,spent,calls)
            execution=execute(candidate,start=state)
            spent+=execution.primitive_cost
            attempts.append({"before":state,"program":candidate,"after":execution.artifact,
                             "legal":execution.legal,"primitive_cost":execution.primitive_cost})
            if not execution.legal:
                continue
            combined=program+candidate
            if execution.artifact==target:
                return result(combined,False,spent,calls)
            if execution.artifact not in seen:
                seen.add(execution.artifact)
                queue.append((execution.artifact,combined))
    return result([],False,spent,calls,unreachable=True)
