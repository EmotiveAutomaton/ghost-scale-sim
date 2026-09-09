"""Independent attention allocation, feedback, acquisition and transfer audit."""
from collections import Counter
import hashlib
import json
import random
from .reference import interpret
from .audit_statistics import verify


def rng_for(*parts):
    payload=json.dumps(parts,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return random.Random(int(hashlib.sha256(payload).hexdigest()[:16],16))


def load(path,expected):
    payload=path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected:
        raise ValueError("attention raw commitment mismatch")
    return json.loads(payload)


def audit_unit(root,row):
    uid=row["unit_id"]
    public=load(root/"public"/f"{uid}.json",row["public_hash"])
    private=load(root/"private"/f"{uid}.json",row["private_hash"])
    prediction=load(root/"predictions"/f"{uid}.json",row["prediction_hash"])
    design=load(root/"private"/f"{uid}-acquisition-design.json",private["acquisition_design_hash"])
    allocation=load(root/"public"/f"{uid}-allocation.json",private["allocation_hash"])
    learning=load(root/"predictions"/f"{uid}-acquisition.json",private["learning_hash"])
    visible=public["learning"]
    construction=public["construction"]
    if learning["public_input"]!=visible or construction["acquisition"]!=learning["arms"]:
        raise ValueError("construction differs from prior acquired-library commitment")
    if set(visible)!={"schema_version","task_id","offered_topics","allocations","instruction_access","feedback_access","processed"}:
        raise ValueError("future task information entered acquisition")
    if allocation!={"task_id":visible["task_id"],"offered_topics":visible["offered_topics"],"allocations":visible["allocations"]}:
        raise ValueError("attention allocation differs from pre-trial commitment")
    if visible["allocations"]!=design["allocations"] or len(visible["offered_topics"])!=row["condition_spec"]["offers"]:
        raise ValueError("opportunity exposure or allocated attempts mismatch")
    for name,records in visible["processed"].items():
        allocation=visible["allocations"][name]
        expected_counts=[12,4] if name=="focal-effort" else [4,12]
        if [sum(visible["offered_topics"][offer]==topic for offer in allocation) for topic in range(2)]!=expected_counts:
            raise ValueError("actual attention allocation differs from intervention")
        if [item["offer"] for item in records]!=allocation:
            raise ValueError("processed record does not match allocated trials")
        counts=Counter()
        for item,truth in zip(records,private["trial_truth"][name]):
            offer=design["offers"][item["offer"]]
            rng=rng_for(row["lineage"],row["seed_components"]["index"],"attempt",item["offer"])
            proposal=list(offer["instruction"]) if visible["instruction_access"] else [rng.randrange(8),rng.randrange(8)]
            program=list(proposal)
            if rng.random()>design["constructor"]["student_reliability"]:
                program[rng.randrange(2)]=rng.randrange(8)
            execution=interpret(program)
            success=execution["legal"] and execution["artifact"]==offer["intended_target"]
            expected={"offer":item["offer"],"topic":offer["topic"],"instruction":offer["instruction"] if visible["instruction_access"] else None,
                      "proposed":proposal,"program":program,"artifact":execution["artifact"],
                      "feedback":success if visible["feedback_access"] else None}
            if item!=expected or truth!={"offer":item["offer"],"intended_target":offer["intended_target"],"actual_success":success}:
                raise ValueError("actual training action or returned feedback mismatch")
            if item["feedback"] is not False:
                counts[tuple(program)]+=1
        eligible=sorted((program for program,count in counts.items() if count>=3),key=lambda program:(-counts[program],program))
        library=[list(eligible[0])] if eligible else []
        arm=prediction["arms"][name]
        if library!=arm["library"] or library!=learning["arms"][name]["library"]:
            raise ValueError("active library differs from permitted processed evidence")
        successes={};executions={}
        for phase in ["pretest","transfer"]:
            successes[phase]=[];executions[phase]=[]
            for target,submission in zip(construction[phase+"_targets"],arm[phase]):
                for attempt in submission["attempted_programs"]:
                    expanded=[]
                    for token in attempt["tokens"]:
                        expanded.extend([token] if isinstance(token,int) else library[int(token[1:])])
                    result=interpret(expanded)
                    if (result["artifact"],result["legal"],result["primitive_cost"])!=(attempt["artifact"],attempt["legal"],attempt["cost"]):
                        raise ValueError("attention search primitive cost mismatch")
                spent=sum(attempt["cost"] for attempt in submission["attempted_programs"])
                if spent!=submission["search_primitives"] or spent>construction[phase+"_budget"]:
                    raise ValueError("attention search budget mismatch")
                execution=interpret(submission["program"])
                executions[phase].append(execution)
                successes[phase].append(execution["legal"] and not submission["search_timeout"] and execution["artifact"]==target)
        if executions!=row["arms"][name]["executions"]:
            raise ValueError("attention held-out execution mismatch")
        costs={"training_primitives":sum(len(item["program"]) for item in records),"instruction_queries":16*int(visible["instruction_access"]),
               "feedback_queries":16*int(visible["feedback_access"]),"definition_cost":sum(map(len,library)),
               "offered_opportunities":len(visible["offered_topics"]),"processed_trials":16}
        if costs!=arm["costs"]:
            raise ValueError("attention training or information cost mismatch")
        expected={"pretest_success":sum(successes["pretest"])/6,"focal_transfer":sum(successes["transfer"][:2])/2,
            "foil_transfer":sum(successes["transfer"][2:])/2,"balanced_transfer":sum(successes["transfer"])/4,
            "pretest_search_cost":sum(item["search_primitives"] for item in arm["pretest"])/6,
            "transfer_search_cost":sum(item["search_primitives"] for item in arm["transfer"])/4,
            "pretest_execution_cost":sum(item["primitive_cost"] for item in executions["pretest"])/6,
            "transfer_execution_cost":sum(item["primitive_cost"] for item in executions["transfer"])/4,
            **{key:float(value) for key,value in costs.items()}}
        if row["arms"][name]["outcomes"]!=expected:
            raise ValueError("independent attention outcome mismatch")
    equality=row["arms"]["focal-effort"]["outcomes"]["pretest_success"]==row["arms"]["other-effort"]["outcomes"]["pretest_success"]
    if private["equal_pretest_score"]!=equality:
        raise ValueError("measured skill matching indicator differs")


def audit(root,summary):
    rows=[json.loads(path.read_bytes()) for path in sorted((root/"units").glob("*_points.json"))]
    for row in rows:
        audit_unit(root,row)
    count=verify(rows,summary)
    return {"execution_state":"completed","instrument_state":"valid","n_raw_units":len(rows),
            "reproduced_contrasts":count,"all_reported_aggregates_reproduced":True,"full_rollout_replay":False}
