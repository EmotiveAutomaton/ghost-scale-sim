"""Bounded R-family tie repair; original inquiry_study.py remains frozen.
"""
"""Prospective, separately executed inquiry episodes; evaluator truth stays private."""
import copy
import math
import random
import secrets
import uuid
import time
from .records import canonical,digest,read,write,now,seed_for
from .inquiry_stable import PRIOR,prediction
from .inquiry_world import constructor,initial_learning,outcome,execute_tests,maker_task
from .inquiry_designs import DESIGNS,condition_values
from .estimands import Estimand,paired_summary


STABLE_OPERATIONS = {
    "inquiry": "ghostscale.validation.soundingline.v16.inquiry_stable:choose",
    "inquiry-learning": "ghostscale.validation.soundingline.v16.inquiry_stable:learn_public",
    "inquiry-construction": "ghostscale.validation.soundingline.v16.inquiry_stable:construct_public",
    "inquiry-reading": "ghostscale.validation.soundingline.v16.inquiry_stable:read_maker_public"}


def request(reader,kind,public,**options):
    kind = STABLE_OPERATIONS[kind]
    result=reader.request(kind,public,**options)
    return {"kind":kind,"public":copy.deepcopy(public),"options":options,"result":result}


def learning_request(reader,beliefs,examples=(),forget=()):
    return request(reader,"inquiry-learning",{"beliefs":beliefs,
                   "examples":[{key:item[key] for key in ["domain","command","cell"]} for item in examples],
                   "forget_domains":list(forget)})


def weighted_success(executions,weights):
    return sum(weights[domain]*sum(item["success"] for item in executions if item["domain"]==domain)/6
               for domain in range(2))


def execute_unit(root,card,condition,index,*,namespace,packet,reader,constructors=8,scope="discovery"):
    uid=digest([namespace,card,condition["id"],index])[:24]
    destination=root/"units"/f"{uid}_points.json"
    if destination.exists():
        row=read(destination)
        if row["packet_hash"]!=packet["packet_hash"]:
            raise ValueError("existing inquiry unit differs from lock")
        return row
    started_wall=time.perf_counter()
    started_cpu=time.process_time()
    spec=condition_values(condition)
    constructor_id=index%constructors
    world=constructor(namespace,constructor_id)
    transport_path=root/"private"/f"{uid}-transport.json"
    history_transport=root/"private"/f"history-{digest([namespace,index])[:24]}-transport.json"
    shared=read(history_transport) if history_transport.exists() else {"reader_seed":secrets.randbits(128)}
    write(history_transport,shared)
    transport=read(transport_path) if transport_path.exists() else {
        "public_id":uuid.uuid4().hex,"reader_seed":shared["reader_seed"]}
    # The independent random tie seed cannot be inverted to a small private unit index.
    write(transport_path,transport)
    _,examples=initial_learning(world,spec["initial_examples"],namespace=namespace,index=index,
                                noisy_domains=spec["noisy_domains"])
    learned=learning_request(reader,[list(PRIOR),list(PRIOR)],examples)
    beliefs=learned["result"]["beliefs"]
    initial=request(reader,"inquiry-construction",{"beliefs":beliefs})
    public={"task_id":transport["public_id"],"initial_examples":examples,
            "curriculum_probabilities":world["offer_probabilities"],
            "cue_exposures":[{"domain":domain,"cue_strength":1.0} for domain in range(2)
                             for _ in range(spec["familiarity"][domain])],
            "objective_weights":spec["future_weights"],"opportunity_cost":spec["opportunity_cost"]}
    public_hash=write(root/"public"/f"{uid}.json",public)
    initial_path=root/"predictions"/f"{uid}-initial.json"
    submitted=read(initial_path)["submitted_at"] if initial_path.exists() else now()
    initial_hash=write(initial_path,{"submitted_at":submitted,"requests":[learned,initial]})
    baseline=execute_tests(world,initial["result"]["programs"],namespace=namespace,index=index,
                           noisy_domains=spec["noisy_domains"])
    if card=="R05":
        private,arms,prediction_hash=practice(root,uid,reader,world,spec,index,namespace,beliefs,baseline)
    else:
        private,arms,prediction_hash=episodes(root,uid,reader,world,spec,index,namespace,beliefs,baseline,
                                            transport,DESIGNS[card]["arms"])
    private.update(world=world,baseline_executions=baseline,initial_prediction_hash=initial_hash,
                   transport_hash=digest(transport))
    truth_hash=write(root/"private"/f"{uid}.json",private)
    row={"unit_id":uid,"card_id":card,"condition":condition["id"],"unit_kind":"inquiry",
         "constructor_id":constructor_id,"maker_history_id":digest([namespace,index])[:24],
         "evidence_scope":scope,"lineage":namespace,"seed_components":{"index":index,"constructors":constructors},
         "packet_hash":packet["packet_hash"],"commission_hash":packet["identity"]["commission_hash"],
         "condition_spec":spec,"observation_hash":public_hash,"prediction_hash":prediction_hash,
         "truth_hash":truth_hash,"arms":arms,"failures":[],"prediction_submitted_at":submitted,"scored_at":now()}
    row["runtime"]={"wall_seconds":time.perf_counter()-started_wall,
                    "evaluator_cpu_seconds":time.process_time()-started_cpu,
                    "reader_process_cpu":"per-request CPU, wall and RSS retained in the separate resource receipt",
                    "logical_work":"hypothesis terms evaluated in a cache-free reference computation; not measured CPU instructions"}
    if hasattr(reader, "samples"):
        resource_hash = write(root/"private"/f"{uid}-resources.json",
            {"unit_id": uid, "requests": reader.samples, "startup_cost_included": False,
             "scope": "actual reader CPU/wall/RSS in this amended discovery; no retrospective old-run estimate"})
        row["reader_resources_sha256"] = resource_hash
    write(destination,row)
    return row


def episodes(root,uid,reader,world,spec,index,namespace,initial_beliefs,baseline,transport,names):
    states={name:{"beliefs":copy.deepcopy(initial_beliefs),"histories":[[],[]],"pending":[[],[]],
                  "committed":None,"queries":0,"domain_queries":[0,0],"evaluations":0,
                  "learning_evaluations":0,"feedback_count":0,"stopped":False} for name in names}
    frames=[]
    for tick in range(8):
        offers=[]
        for domain in range(2):
            rng=random.Random(seed_for(namespace,index,"offer",tick,domain))
            offers.append(rng.choices(range(4),weights=world["offer_probabilities"][domain],k=1)[0])
        tie=random.Random(seed_for(transport["reader_seed"],"tie",tick)).random()
        requests={}
        for name,state in states.items():
            forget=[0] if tick in spec["forget_steps"] else []
            memory=learning_request(reader,state["beliefs"],forget=forget)
            state["beliefs"]=memory["result"]["beliefs"]
            if forget:
                # Erase unreleased examples too; raw evaluator history remains retained.
                state["pending"][0]=[]
                state["committed"]=None
            public={"schema_version":"v16.inquiry.1","task_id":transport["public_id"],
                    "beliefs":state["beliefs"],"offers":offers,"histories":state["histories"],
                    "familiarity":spec["familiarity"],"pending_commands":[[item["command"] for item in batch]
                        for batch in state["pending"]],"feedback_batches":spec["feedback_batches"],
                    "future_weights":spec["future_weights"],"opportunity_cost":spec["opportunity_cost"],
                    "remaining_interactions":8-tick,"tie_draw":tie,"committed_domain":state["committed"],
                    "offer_probabilities":world["offer_probabilities"]}
            decision=request(reader,"inquiry",public,policy="decline" if state["stopped"] else name)
            requests[name]={"memory":memory,"decision":decision}
        prediction_path=root/"predictions"/f"{uid}-tick-{tick}.json"
        submitted=read(prediction_path)["submitted_at"] if prediction_path.exists() else now()
        prediction_hash=write(prediction_path,{"submitted_at":submitted,"requests":requests})
        # All arms have committed their inquiry action before any response at this tick.
        responses={}
        for name,state in states.items():
            decision=requests[name]["decision"]["result"]
            domain=decision["domain"]
            state["evaluations"]+=decision["logical_model_evaluations"]
            if domain is None:
                state["stopped"]=True
                state["committed"]=None
                responses[name]={"execution":None,"released":[],"learning":None}
                continue
            command=decision["commands"]
            rng=random.Random(seed_for(namespace,index,"feedback",tick,domain))
            work=outcome(world,domain,command,noisy=domain in spec["noisy_domains"],rng=rng)
            example={"domain":domain,"target_cell":offers[domain],**work}
            state["pending"][domain].append(example)
            state["queries"]+=1
            state["domain_queries"][domain]+=1
            state["committed"]=domain if decision["commit_next"] else None
            released=[]
            feedback=None
            if len(state["pending"][domain])>=spec["feedback_batches"][domain]:
                released=copy.deepcopy(state["pending"][domain])
                for item in released:
                    p=prediction(state["beliefs"][domain],item["command"])[item["cell"]]
                    state["histories"][domain].append({"success":item["cell"]==item["target_cell"],
                                                        "surprise":-math.log(p)})
                    feedback=learning_request(reader,state["beliefs"],[item])
                    state["beliefs"]=feedback["result"]["beliefs"]
                    state["learning_evaluations"]+=feedback["result"]["logical_model_evaluations"]
                state["feedback_count"]+=len(released)
                state["pending"][domain]=[]
            responses[name]={"execution":example,"released":released,"beliefs_after":copy.deepcopy(state["beliefs"])}
        frames.append({"tick":tick,"prediction_hash":prediction_hash,"responses":responses})
    final={name:request(reader,"inquiry-construction",{"beliefs":state["beliefs"]})
           for name,state in states.items()}
    final_path=root/"predictions"/f"{uid}.json"
    submitted=read(final_path)["submitted_at"] if final_path.exists() else now()
    prediction_hash=write(final_path,{"submitted_at":submitted,"requests":final})
    arms={}
    for name,state in states.items():
        executions=execute_tests(world,final[name]["result"]["programs"],namespace=namespace,index=index,
                                 noisy_domains=spec["noisy_domains"])
        skill=weighted_success(executions,spec["future_weights"])
        base=weighted_success(baseline,spec["future_weights"])
        work=state["evaluations"]+state["learning_evaluations"]+final[name]["result"]["logical_model_evaluations"]
        arms[name]={"executions":executions,"outcomes":{
            "success":skill,"initial_success":base,"gain":skill-base,
            "utility":skill-spec["opportunity_cost"]*state["queries"]-1e-7*work,
            "queries":float(state["queries"]),"domain0_queries":float(state["domain_queries"][0]),
            "domain1_queries":float(state["domain_queries"][1]),"feedback_examples":float(state["feedback_count"]),
            "model_evaluations":float(work),"practice_primitives":float(state["queries"]),
            "initial_example_primitives":float(2*spec["initial_examples"]),"heldout_primitives":24.0,
            "stopped":float(state["stopped"])}}
    return {"frames":frames,"final_states":states},arms,prediction_hash


def practice(root,uid,reader,world,spec,index,namespace,beliefs,baseline):
    _,all_examples=initial_learning(world,spec["practice_examples"],namespace=namespace,index=index,
                                    noisy_domains=[])
    domain_examples={domain:[item for item in all_examples if item["domain"]==domain] for domain in range(2)}
    unseen=maker_task(world,None,namespace=namespace,index=index)
    requests={}
    for name in ["enact","observe","unrelated"]:
        examples=domain_examples[1 if name=="unrelated" else 0]
        learned=learning_request(reader,beliefs,examples)
        after=learned["result"]["beliefs"]
        requests[name]={"learning":learned,"construction":request(reader,"inquiry-construction",{"beliefs":after}),
                        "reading":request(reader,"inquiry-reading",{"belief":after[0],"artifact":unseen["artifact"]})}
    final_path=root/"predictions"/f"{uid}.json"
    submitted=read(final_path)["submitted_at"] if final_path.exists() else now()
    prediction_hash=write(final_path,{"submitted_at":submitted,"requests":requests})
    unseen["hidden_next_cell"]=unseen["maker_acquisition"]["learned_library"][0][0]
    arms={}
    for name,frames in requests.items():
        executions=execute_tests(world,frames["construction"]["result"]["programs"],namespace=namespace,
                                 index=index,noisy_domains=[])
        skill=weighted_success(executions,spec["future_weights"])
        base=weighted_success(baseline,spec["future_weights"])
        model_cost=sum(item["result"]["logical_model_evaluations"] for item in frames.values())
        arms[name]={"executions":executions,"outcomes":{"success":skill,"initial_success":base,
            "gain":skill-base,"future_log_score":math.log(frames["reading"]["result"]["future_probabilities"][unseen["hidden_next_cell"]]),
            "model_evaluations":float(model_cost),"practice_examples":float(spec["practice_examples"]),
            "enacted_primitives":float(spec["practice_examples"] if name!="observe" else 0),
            "observed_examples":float(spec["practice_examples"] if name=="observe" else 0),"heldout_primitives":24.0}}
    return {"practice_examples":domain_examples,"unseen_maker":unseen},arms,prediction_hash


def summarize(card,rows):
    conditions={}
    for condition in DESIGNS[card]["conditions"]:
        selected=[row for row in rows if row["condition"]==condition["id"]]
        specs=[Estimand(f"{card}-{a}-minus-{b}-{metric}",metric,a,b,units,bar,f"{a} minus {b}: {metric}")
               for a,b,metric,units,bar in DESIGNS[card]["primary"]]
        conditions[condition["id"]]={"contrasts":[paired_summary(selected,spec) for spec in specs],
            "arms":{name:{metric:sum(row["arms"][name]["outcomes"][metric] for row in selected)/len(selected)
                          for metric in selected[0]["arms"][name]["outcomes"]} for name in DESIGNS[card]["arms"]}}
    return {"card_id":card,"conditions":conditions,"n_maker_packets":len(rows),"evidence_scope":rows[0]["evidence_scope"],
            "instrument_state":"valid","independent_reaggregation":"pending","heldout_tasks_are_nested":True,
            "scope_limit":DESIGNS[card].get("scope_limit","finite command-map ecology; no human inference")}
