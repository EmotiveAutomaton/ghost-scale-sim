"""Independent observed-edge, option termination, primitive-cost and outcome audit."""
import hashlib
import json
import numpy as np
from .reference import interpret
from .audit_statistics import verify


def load(path,expected):
    payload=path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected:
        raise ValueError("option raw hash mismatch")
    return json.loads(payload)


def graph_check(graph,training):
    edges={}
    nodes=set()
    for program in training["attempts"]:
        state=0
        for action in program:
            result=interpret([action],start=state)
            nodes.update([state,result["artifact"]])
            if result["legal"]:
                edges[(state,action)]=result["artifact"]
            state=result["artifact"]
    actual={(row["before"],row["action"]):row["after"] for row in graph["edges"]}
    if actual!=edges or set(graph["observed_nodes"])!=nodes:
        raise ValueError("option graph differs from allowed exploration")
    if graph["observed_state_fraction"]!=len(nodes)/16:
        raise ValueError("observed graph coverage denominator mismatch")
    for component in graph["spectral_components"]:
        members=component["nodes"]
        if len(members)<2:
            continue
        matrix=np.zeros((len(members),len(members)))
        for (source,action),target in edges.items():
            if source!=target and source in members and target in members:
                a,b=members.index(source),members.index(target)
                matrix[a,b]=matrix[b,a]=1
        degree=matrix.sum(axis=1)
        operator=np.diag(degree)-matrix
        for a in range(len(members)):
            for b in range(len(members)):
                operator[a,b]/=np.sqrt(degree[a]*degree[b])
        projector=np.array(component["projector"])
        eigenvalue=min(value for value in component["eigenvalues"] if value>1e-10)
        if max(np.max(np.abs(projector@projector-projector)),np.max(np.abs(operator@projector-eigenvalue*projector)))>1e-10:
            raise ValueError("spectral projector algebra fails")
        if abs(np.trace(projector)-component["first_nonzero_multiplicity"])>1e-10:
            raise ValueError("degenerate eigenspace rank mismatch")
    for option in graph["options"]:
        for initial in map(int,option["policy"]):
            state=initial
            visited=set()
            while state!=option["target"]:
                if state in visited or str(state) not in option["policy"]:
                    raise ValueError("option does not terminate from its declared initiation")
                visited.add(state)
                action=option["policy"][str(state)]
                if (state,action) not in edges:
                    raise ValueError("option uses an unobserved edge")
                after=edges[(state,action)]
                if option["distances"][str(after)]!=option["distances"][str(state)]-1:
                    raise ValueError("option does not descend its finite distance")
                state=after


def audit_unit(root,row):
    uid=row["unit_id"]
    public=load(root/"public"/f"{uid}.json",row["public_hash"])
    saved=load(root/"predictions"/f"{uid}.json",row["prediction_hash"])
    load(root/"private"/f"{uid}.json",row["private_hash"])
    for name,prediction in saved["arms"].items():
        graph=prediction["graph"]
        training=public["pooled_training"] if name=="generic-options" else public["own_training"]
        if graph is not None:
            graph_check(graph,training)
        executions=[]
        successful=[]
        for target,submission in zip(public["targets"],prediction["submissions"]):
            for attempt in submission["attempts"]:
                result=interpret(attempt["program"],start=attempt["before"])
                if (result["artifact"],result["legal"],result["primitive_cost"])!=(attempt["after"],attempt["legal"],attempt["primitive_cost"]):
                    raise ValueError("option planner primitive trace mismatch")
            spent=sum(attempt["primitive_cost"] for attempt in submission["attempts"])
            if spent!=submission["successor_evaluations"] or spent>public["budget"]:
                raise ValueError("option primitive search cost mismatch")
            for call in submission["option_expansions"]:
                state=call["state"]
                if graph is None:
                    raise ValueError("option expansion without acquired graph")
                edges={(item["before"],item["action"]):item["after"] for item in graph["edges"]}
                for action in call["expansion"]["program"]:
                    if (state,action) not in edges:
                        raise ValueError("option call used an unobserved transition")
                    state=edges[(state,action)]
                if state!=call["expansion"]["end"] or (call["expansion"]["terminated"] and state!=call["target"]):
                    raise ValueError("option call termination mismatch")
            if submission["option_policy_lookups"]!=sum(len(call["expansion"]["program"]) for call in submission["option_expansions"]):
                raise ValueError("option lookup cost mismatch")
            execution=interpret(submission["program"])
            executions.append(execution)
            successful.append(execution["legal"] and not submission["search_timeout"] and execution["artifact"]==target)
        if row["arms"][name]["executions"]!=executions:
            raise ValueError("option submitted execution mismatch")
        count=len(executions)
        definition=sum(len(option["policy"])+1 for option in graph["options"]) if graph else sum(map(len,prediction["library"]))
        expected={"success":sum(successful)/count,"legal":sum(item["legal"] for item in executions)/count,
            "search_cost":sum(item["successor_evaluations"] for item in prediction["submissions"])/count,
            "option_policy_lookups":sum(item["option_policy_lookups"] for item in prediction["submissions"])/count,
            "execution_primitives":sum(item["primitive_cost"] for item in executions)/count,
            "training_primitives":float(sum(map(len,training["attempts"]))),"definition_cost":float(definition),
            "observed_graph_fraction":graph["observed_state_fraction"] if graph else 0.0,
            "observed_components":float(graph["component_count"]) if graph else 0.0}
        if row["arms"][name]["outcomes"]!=expected:
            raise ValueError("independent option outcome or cost mismatch")


def audit(root,summary):
    rows=[json.loads(path.read_bytes()) for path in sorted((root/"units").glob("*_points.json"))]
    for row in rows:
        audit_unit(root,row)
    count=verify(rows,summary)
    return {"execution_state":"completed","instrument_state":"valid","n_raw_units":len(rows),
            "reproduced_contrasts":count,"all_reported_aggregates_reproduced":True,"full_rollout_replay":False}
