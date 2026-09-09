"""Independent command physics, evidence flow, episode costs and aggregate audit."""
import hashlib
import itertools
import json
import math
import random
from .audit_statistics import verify

MAPS=list(itertools.permutations(range(4)))
PRIOR=[0.9/24]*24+[0.1]


def rng_for(*parts):
    encoded=json.dumps(parts,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return random.Random(int(hashlib.sha256(encoded).hexdigest()[:16],16))


def checked(path,expected=None):
    payload=path.read_bytes()
    if expected is not None and hashlib.sha256(payload).hexdigest()!=expected:
        raise ValueError("inquiry raw commitment hash mismatch")
    return json.loads(payload)


def close(actual,expected):
    if isinstance(expected,dict):
        if set(actual)!=set(expected):
            raise ValueError("independent inquiry record keys differ")
        for key in expected:
            close(actual[key],expected[key])
    elif isinstance(expected,(list,tuple)):
        if len(actual)!=len(expected):
            raise ValueError("independent inquiry record lengths differ")
        for left,right in zip(actual,expected):
            close(left,right)
    elif isinstance(expected,(int,float)):
        if not math.isfinite(actual) or abs(actual-expected)>1e-10:
            raise ValueError("independent inquiry numeric mismatch")
    elif actual!=expected:
        raise ValueError("independent inquiry value mismatch")


def posterior(belief,command,cell):
    weights=[weight*int(mapping[command]==cell) for weight,mapping in zip(belief[:24],MAPS)]
    weights.append(belief[24]/4)
    total=sum(weights)
    return [weight/total for weight in weights]


def check_learning(frame,beliefs,examples,forget=()):
    close(frame["public"],{"beliefs":beliefs,"examples":[{key:item[key] for key in ["domain","command","cell"]}
                                                        for item in examples],"forget_domains":list(forget)})
    after=[list(belief) for belief in beliefs]
    for domain in forget:
        after[domain]=list(PRIOR)
    for item in examples:
        after[item["domain"]]=posterior(after[item["domain"]],item["command"],item["cell"])
    close(frame["result"],{"beliefs":after,"logical_model_evaluations":25*len(examples)})
    return after


def execute(world,programs,namespace,index,noisy):
    records=[]
    for program in programs:
        domain,goal,commands=program["domain"],program["goal"],program["commands"]
        physical=[]
        artifact=0
        for step,command in enumerate(commands):
            rng=rng_for(namespace,index,"heldout",domain,goal,step)
            cell=rng.randrange(4) if domain in noisy else world["mappings"][domain][command]
            physical.append(cell)
            artifact|=1<<cell
        records.append({"domain":domain,"goal":goal,"commands":commands,"program":physical,
                        "artifact":artifact,"legal":True,"success":artifact==goal,"primitive_cost":len(physical)})
    return records


def success(records,weights):
    total=0.0
    for domain in range(2):
        selected=[item for item in records if item["domain"]==domain]
        if len(selected)!=6:
            raise ValueError("held-out denominator is not six compositions per domain")
        total+=weights[domain]*sum(item["success"] for item in selected)/6
    return total


def audit_unit(root,row):
    uid,namespace,index=row["unit_id"],row["lineage"],row["seed_components"]["index"]
    private=checked(root/"private"/f"{uid}.json",row["truth_hash"])
    public=checked(root/"public"/f"{uid}.json",row["observation_hash"])
    final=checked(root/"predictions"/f"{uid}.json",row["prediction_hash"])
    initial=checked(root/"predictions"/f"{uid}-initial.json",private["initial_prediction_hash"])
    transport=checked(root/"private"/f"{uid}-transport.json")
    spec,world=row["condition_spec"],private["world"]
    rng=rng_for(namespace,"constructor",row["constructor_id"])
    mappings=[list(rng.choice(MAPS)) for _ in range(2)]
    probabilities=[]
    for domain in range(2):
        favorite=rng.randrange(4)
        strength=rng.uniform(0.35,0.85)
        probabilities.append([strength if cell==favorite else (1-strength)/3 for cell in range(4)])
    close(world,{"mappings":mappings,"offer_probabilities":probabilities})
    initial_beliefs=check_learning(initial["requests"][0],[list(PRIOR),list(PRIOR)],public["initial_examples"])
    close(initial["requests"][1]["public"],{"beliefs":initial_beliefs})
    baseline=execute(world,initial["requests"][1]["result"]["programs"],namespace,index,spec["noisy_domains"])
    close(private["baseline_executions"],baseline)
    if row["card_id"]=="R05":
        for name,arm in row["arms"].items():
            frames=final["requests"][name]
            examples=private["practice_examples"][str(1 if name=="unrelated" else 0)]
            beliefs=check_learning(frames["learning"],initial_beliefs,examples)
            for example in examples:
                if example["cell"]!=world["mappings"][example["domain"]][example["command"]]:
                    raise ValueError("practice output differs from actual command physics")
            close(frames["construction"]["public"],{"beliefs":beliefs})
            close(frames["reading"]["public"],{"belief":beliefs[0],"artifact":private["unseen_maker"]["artifact"]})
            executions=execute(world,frames["construction"]["result"]["programs"],namespace,index,[])
            close(arm["executions"],executions)
            truth=private["unseen_maker"]["true_program"][0]
            close(private["unseen_maker"]["hidden_next_cell"],truth)
            skill,base=success(executions,spec["future_weights"]),success(baseline,spec["future_weights"])
            work=sum(item["result"]["logical_model_evaluations"] for item in frames.values())
            expected={"success":skill,"initial_success":base,"gain":skill-base,
                "future_log_score":math.log(frames["reading"]["result"]["future_probabilities"][truth]),
                "model_evaluations":float(work),"practice_examples":float(spec["practice_examples"]),
                "enacted_primitives":float(spec["practice_examples"] if name!="observe" else 0),
                "observed_examples":float(spec["practice_examples"] if name=="observe" else 0),"heldout_primitives":24.0}
            close(arm["outcomes"],expected)
        # This is a known information-equivalence control, including zero examples.
        for frame in ["learning","construction","reading"]:
            close(final["requests"]["enact"][frame],final["requests"]["observe"][frame])
        return
    states={name:{"beliefs":[list(belief) for belief in initial_beliefs],"histories":[[],[]],
                  "pending":[[],[]],"committed":None,"queries":0,"domain_queries":[0,0],
                  "evaluations":0,"learning_evaluations":0,"feedback_count":0,"stopped":False}
            for name in row["arms"]}
    if len(private["frames"])!=8:
        raise ValueError("inquiry horizon incomplete")
    for tick,frame in enumerate(private["frames"]):
        submitted=checked(root/"predictions"/f"{uid}-tick-{tick}.json",frame["prediction_hash"])
        offers=[rng_for(namespace,index,"offer",tick,domain).choices(range(4),weights=probabilities[domain],k=1)[0]
                for domain in range(2)]
        tie=rng_for(transport["reader_seed"],"tie",tick).random()
        for name,state in states.items():
            memory=submitted["requests"][name]["memory"]
            forget=[0] if tick in spec["forget_steps"] else []
            state["beliefs"]=check_learning(memory,state["beliefs"],[],forget)
            if forget:
                state["pending"][0]=[]
                state["committed"]=None
            decision_frame=submitted["requests"][name]["decision"]
            expected_public={"schema_version":"v16.inquiry.1","task_id":transport["public_id"],
                "beliefs":state["beliefs"],"offers":offers,"histories":state["histories"],
                "familiarity":spec["familiarity"],"pending_commands":[[item["command"] for item in batch] for batch in state["pending"]],
                "feedback_batches":spec["feedback_batches"],"future_weights":spec["future_weights"],
                "opportunity_cost":spec["opportunity_cost"],"remaining_interactions":8-tick,"tie_draw":tie,
                "committed_domain":state["committed"],"offer_probabilities":probabilities}
            close(decision_frame["public"],expected_public)
            close(decision_frame["options"],{"policy":"decline" if state["stopped"] else name})
            choice=decision_frame["result"]
            domain=choice["domain"]
            response=frame["responses"][name]
            state["evaluations"]+=choice["logical_model_evaluations"]
            if domain is None:
                state["stopped"]=True
                state["committed"]=None
                close(response,{"execution":None,"released":[],"learning":None})
                continue
            command=choice["commands"]
            cell=rng_for(namespace,index,"feedback",tick,domain).randrange(4) if domain in spec["noisy_domains"] else mappings[domain][command]
            example={"domain":domain,"target_cell":offers[domain],"command":command,"cell":cell,
                     "program":[cell],"artifact":1<<cell,"legal":True,"primitive_cost":1}
            close(response["execution"],example)
            state["pending"][domain].append(example)
            state["queries"]+=1
            state["domain_queries"][domain]+=1
            state["committed"]=domain if choice["commit_next"] else None
            released=[]
            if len(state["pending"][domain])>=spec["feedback_batches"][domain]:
                released=list(state["pending"][domain])
                for item in released:
                    belief=state["beliefs"][domain]
                    p=sum(weight for weight,mapping in zip(belief[:24],MAPS) if mapping[item["command"]]==item["cell"])+belief[24]/4
                    state["histories"][domain].append({"success":item["cell"]==item["target_cell"],"surprise":-math.log(p)})
                    state["beliefs"][domain]=posterior(belief,item["command"],item["cell"])
                    state["learning_evaluations"]+=25
                state["feedback_count"]+=len(released)
                state["pending"][domain]=[]
            close(response["released"],released)
            close(response["beliefs_after"],state["beliefs"])
    close(private["final_states"],states)
    for name,state in states.items():
        prediction=final["requests"][name]
        close(prediction["public"],{"beliefs":state["beliefs"]})
        executions=execute(world,prediction["result"]["programs"],namespace,index,spec["noisy_domains"])
        close(row["arms"][name]["executions"],executions)
        skill,base=success(executions,spec["future_weights"]),success(baseline,spec["future_weights"])
        work=state["evaluations"]+state["learning_evaluations"]+prediction["result"]["logical_model_evaluations"]
        expected={"success":skill,"initial_success":base,"gain":skill-base,
            "utility":skill-spec["opportunity_cost"]*state["queries"]-1e-7*work,
            "queries":float(state["queries"]),"domain0_queries":float(state["domain_queries"][0]),
            "domain1_queries":float(state["domain_queries"][1]),"feedback_examples":float(state["feedback_count"]),
            "model_evaluations":float(work),"practice_primitives":float(state["queries"]),
            "initial_example_primitives":float(2*spec["initial_examples"]),"heldout_primitives":24.0,
            "stopped":float(state["stopped"])}
        close(row["arms"][name]["outcomes"],expected)


def audit(root,summary):
    rows=[json.loads(path.read_bytes()) for path in sorted((root/"units").glob("*_points.json"))]
    for row in rows:
        audit_unit(root,row)
    count=verify(rows,summary)
    return {"execution_state":"completed","instrument_state":"valid","n_raw_units":len(rows),
            "reproduced_contrasts":count,"all_reported_aggregates_reproduced":True,
            "reader_predictions_reexecuted":False,"full_rollout_replay":False}
