"""Independent acquisition, maker production, scored prediction and cost audit."""
from collections import Counter
from itertools import product
import hashlib
import json
import math
import random
from .reference import interpret,token_cost
from .assembly_reference import interpret as assembly_interpret,stopped_histories
from .audit_purpose import learned as board_learned
from .audit_statistics import verify


def load(path,expected):
    payload=path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected:
        raise ValueError("mechanism raw commitment mismatch")
    return json.loads(payload)


def rng_for(*parts):
    payload=json.dumps(parts,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return random.Random(int(hashlib.sha256(payload).hexdigest()[:16],16))


def acquisition_library(world,record):
    if world["family"]=="W1":
        return board_learned({"attempts":record["attempts"],"targets":record["goals"]})
    counts=Counter()
    spent=0
    for example in record["records"]:
        result=assembly_interpret(world,example["program"])
        if result!=example["execution"]:
            raise ValueError("W2 actual training execution mismatch")
        spent+=result["primitive_cost"]
        if result["legal"] and result["successfully_stopped"] and result["state"]==example["target"]:
            for offset in range(len(example["program"])-1):
                fragment=tuple(example["program"][offset:offset+2])
                if 9 not in fragment:
                    counts[fragment]+=1
    library=sorted((fragment for fragment,count in counts.items() if count>=2),key=lambda part:(-counts[part],part))[:4]
    if record["learned"]!={"library":[list(part) for part in library],"definition_cost":sum(map(len,library)),
                            "training_primitives":spent,"training_examples":len(record["records"])}:
        raise ValueError("W2 acquisition accounting mismatch")
    return library


def execution(world,program):
    if world["family"]=="W1":
        value=interpret(program)
        return {"state":value["artifact"],"artifact":value["artifact"],"legal":value["legal"],
                "finished":value["legal"],"primitive_cost":value["primitive_cost"]}
    value=assembly_interpret(world,program)
    return {"state":value["state"],"artifact":value["artifact"],"legal":value["legal"],
            "finished":value["successfully_stopped"],"primitive_cost":value["primitive_cost"]}


def check_search(world,search,library,*,start=None,budget):
    if world["family"]=="W1" and "attempted_programs" in search:
        spent=0
        for attempt in search["attempted_programs"]:
            program=[]
            for token in attempt["tokens"]:
                program.extend([token] if isinstance(token,int) else library[int(token[1:])])
            value=interpret(program,start=0 if start is None else start)
            if (value["artifact"],value["legal"],value["primitive_cost"])!=(attempt["artifact"],attempt["legal"],attempt["cost"]):
                raise ValueError("W1 mechanism search primitive mismatch")
            spent+=value["primitive_cost"]
        if spent!=search["search_primitives"]:
            raise ValueError("W1 mechanism search cost mismatch")
    elif world["family"]=="W1":
        spent=0
        for attempt in search["attempts"]:
            value=interpret(attempt["program"],start=attempt["before"])
            if (value["artifact"],value["legal"],value["primitive_cost"])!=(attempt["after"],attempt["legal"],attempt["primitive_cost"]):
                raise ValueError("option mechanism search primitive mismatch")
            spent+=value["primitive_cost"]
        if spent!=search["successor_evaluations"]:
            raise ValueError("option mechanism search cost mismatch")
        edges={}
        for program in world["option_training"]:
            state=0
            for action in program:
                after=interpret([action],start=state)["artifact"]
                edges[(state,action)]=after
                state=after
        for call in search["option_expansions"]:
            state=call["state"]
            for action in call["expansion"]["program"]:
                if (state,action) not in edges:
                    raise ValueError("option maker used an unobserved transition")
                state=edges[(state,action)]
            if state!=call["expansion"]["end"]:
                raise ValueError("option maker expansion differs from observed graph")
        if search["option_policy_lookups"]!=sum(len(item["expansion"]["program"]) for item in search["option_expansions"]):
            raise ValueError("option maker lookup accounting mismatch")
    else:
        spent=len(search["attempts"])
        for attempt in search["attempts"]:
            value=assembly_interpret(world,[attempt["action"]],initial=attempt["before"])
            if (value["state"],value["legal"],value["stopped"])!=(attempt["after"],attempt["legal"],attempt["stopped"]):
                raise ValueError("W2 mechanism search primitive mismatch")
        if spent!=search["successor_evaluations"]:
            raise ValueError("W2 mechanism search cost mismatch")
    if spent>budget:
        raise ValueError("mechanism exceeded primitive search cap")
    return spent


def check_work(world,work,library,goal,family,parameters,rng):
    actual=execution(world,work["program"])
    if actual!=work["execution"] or not actual["legal"] or not actual["finished"]:
        raise ValueError("actual maker program is not its recorded finished artifact")
    target=list(goal) if isinstance(goal,tuple) else goal
    if work["task_success"]!=(actual["state"]==target):
        raise ValueError("maker task-success mismatch")
    if family=="softmax":
        if world["family"]=="W1":
            programs=[program for length in range(4) for program in product(range(8),repeat=length)]
        else:
            programs=[program for program,value in stopped_histories(world,parameters["max_steps"])]
            # Match the declared depth-first proposal order using independently
            # enumerated programs: terminate precedes each extension at a node.
            programs.sort(key=lambda program:tuple(-1 if action==9 else action for action in program))
        weights=[]
        for program in programs:
            state=execution(world,program)["state"]
            distance=(state^goal).bit_count() if world["family"]=="W1" else sum(a!=b for a,b in zip(state,goal))
            weights.append(math.exp(-parameters["beta"]*distance-parameters["length_cost"]*token_cost(program,library)))
        selected=rng.choices(programs,weights=weights,k=1)[0]
        if list(selected)!=work["program"]:
            raise ValueError("independent seeded softmax maker draw mismatch")
    else:
        prefix=list(library[0]) if family=="habit" and library else []
        if work["habit_prefix"]!=prefix:
            raise ValueError("habit was not actually committed before planning")
        before=execution(world,prefix)["state"]
        check_search(world,work["completion_search"],library,start=before,budget=parameters["budget"])
        search=work["completion_search"]
        if world["family"]=="W1":
            fallback=search["search_timeout"] or len(prefix+search["program"])>3
            proposed=prefix if fallback else prefix+search["program"]
        else:
            fallback=search["search_timeout"] or search.get("unreachable",False)
            proposed=prefix+[9] if fallback else prefix+search["program"]
        if proposed!=work["program"] or fallback!=work["stopped_without_goal_solution"]:
            raise ValueError("maker completion differs from retained search and commitment")


def audit_unit(root,row):
    uid=row["unit_id"]
    public=load(root/"public"/f"{uid}.json",row["public_hash"])
    private=load(root/"private"/f"{uid}.json",row["private_hash"])
    predictions=load(root/"predictions"/f"{uid}.json",row["prediction_hash"])["arms"]
    world=public["world"]
    library=acquisition_library(world,private["acquisition"])
    if [list(fragment) for fragment in library]!=private["ordered_learned_library"]:
        raise ValueError("maker's ordered acquired library differs from actual training")
    if world["family"]=="W1" and world["option_training"]!=private["generic_exploration"]["attempts"]:
        raise ValueError("option exploration was not the declared generic training")
    for index,(observation,work) in enumerate(zip([*public["prior_works"],public["current"]],private["observed_works"])):
        rng=rng_for(row["lineage"],row["condition"],row["seed_components"]["index"],"observed-work",index)
        check_work(world,work,library,observation["goal"],private["actual_family"],private["production_parameters"],rng)
        if work["execution"]["artifact"]!=observation["artifact"]:
            raise ValueError("reader observation differs from actual finished maker work")
    rng=rng_for(row["lineage"],row["condition"],row["seed_components"]["index"],"hidden-future")
    future=private["hidden_future"]
    check_work(world,future,library,public["future_goal"],private["actual_family"],private["production_parameters"],rng)
    truth=future["execution"]["state"]
    for name,prediction in predictions.items():
        if prediction.get("capability_state")=="not_admitted":
            expected={"model_mismatch":False,"predictive_zero_support":False,"finite_log_score":None,"state":"not_admitted"}
            if row["diagnostics"][name]!=expected:
                raise ValueError("unadmitted capability was scored as a scientific mismatch")
            continue
        mismatch=prediction["model_mismatch"]
        probability=0.0 if mismatch or truth not in prediction["future_support"] else prediction["future_probabilities"][prediction["future_support"].index(truth)]
        if name in row["diagnostics"]:
            expected={"model_mismatch":mismatch,"predictive_zero_support":not mismatch and probability<=0,
                      "finite_log_score":math.log(probability) if probability>0 else None,
                      "state":"finite" if probability>0 else "unsupported"}
            if row["diagnostics"][name]!=expected:
                raise ValueError("reverse-direction support/score mismatch")
            continue
        reconstruction=prediction["reconstruction"]
        check_search(world,reconstruction,prediction["inferred_library"],budget=public["reader_budget"])
        checked=interpret(reconstruction["program"]) if world["family"]=="W1" else assembly_interpret(world,reconstruction["program"])
        stopped=checked["legal"] if world["family"]=="W1" else checked["successfully_stopped"]
        success=checked["legal"] and stopped and not reconstruction["search_timeout"] and checked["artifact"]==public["current"]["artifact"]
        expected={"future_log_score":math.log(probability),"reconstruction_success":float(success),
            "legal_finished_reconstruction":float(checked["legal"] and stopped),
            "search_cost":float(reconstruction.get("search_primitives",reconstruction.get("successor_evaluations"))),
            "likelihood_route_evaluations":float(prediction["costs"]["likelihood_route_evaluations"]),
            "future_route_evaluations":float(prediction["costs"]["future_route_evaluations"]),
            "prior_work_queries":float(prediction["costs"]["prior_work_queries"]),
            "grammatical_route_count":float(prediction["grammatically_compatible_routes"])}
        if row["arms"][name]["execution"]!=checked or row["arms"][name]["outcomes"]!=expected:
            raise ValueError("independent mechanism scoring mismatch")


def audit(root,summary):
    rows=[json.loads(path.read_bytes()) for path in sorted((root/"units").glob("*_points.json"))]
    for row in rows:
        audit_unit(root,row)
    count=verify(rows,summary)
    for condition,reported in summary["conditions"].items():
        selected=[row for row in rows if row["condition"]==condition]
        for name,value in reported["diagnostics"].items():
            admitted=[row["diagnostics"][name] for row in selected if row["diagnostics"][name]["state"]!="not_admitted"]
            finite=[item["finite_log_score"] for item in admitted if item["finite_log_score"] is not None]
            expected={"n_total":len(selected),"n_finite":len(finite),"n_admitted":len(admitted),
                "n_not_admitted":len(selected)-len(admitted),
                "model_mismatch_rate":sum(item["model_mismatch"] for item in admitted)/len(admitted) if admitted else None,
                "predictive_zero_support_rate":sum(item["predictive_zero_support"] for item in admitted)/len(admitted) if admitted else None,
                "conditional_finite_log_score":sum(finite)/len(finite) if finite else None,
                "interpretation":"conditional on finite support, not an overall proper-score mean"}
            for key in expected:
                left,right=value[key],expected[key]
                if isinstance(right,float):
                    if abs(left-right)>1e-10:
                        raise ValueError("reverse-direction conditional aggregate mismatch")
                elif left!=right:
                    raise ValueError("reverse-direction denominator/state mismatch")
    return {"execution_state":"completed","instrument_state":"valid","n_raw_units":len(rows),
            "reproduced_contrasts":count,"all_reported_aggregates_reproduced":True,
            "maker_production_draws_independently_regenerated":True,"full_rollout_replay":False}
