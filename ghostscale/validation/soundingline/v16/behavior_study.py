"""Retained paired opportunity and self-monitoring units."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import copy
import math
import random
import uuid
from .records import canonical,digest,read,write,now,seed_for
from .behavior_designs import DESIGNS
from .estimands import Estimand,paired_summary
from . import opportunity as opportunities
from . import self_monitor as selves
from . import self_trajectory as trajectories
from .reference import interpret


def estimands(card):
    return [Estimand(f"{card}-{a}-minus-{b}-{target}",target,a,b,units,bar,f"{a} minus {b}: {target}")
            for a,b,target,units,bar in DESIGNS[card]["primary"]]


def critic_reader(payload):
    import json
    public = json.loads(payload)
    if set(public) != {"schema_version","goal","action_cost","actual","beliefs","search_budget"}:
        raise ValueError("critic resource schema violation")
    candidates = [0,1][:public["search_budget"]]
    known = [candidate for candidate in candidates if candidate in public["beliefs"]]
    choice = max(known,key=lambda a:(int(a==public["goal"])-public["action_cost"]*a,-a))
    return {"program":[0] if choice else [],"chosen_option":choice,
            "costs":{"search_evaluations":len(candidates)},
            "access_tier":"supplied-state resource benchmark; not learned-reader admission"}


def execute_unit(root:Path,card,condition,index,*,namespace,packet,constructors=8,evidence_scope="discovery"):
    uid = digest([namespace,card,condition["id"],index])[:24]
    unit_path = root/"units"/f"{uid}_points.json"
    if unit_path.exists():
        row = read(unit_path)
        if row["packet_hash"] != packet["packet_hash"]:
            raise ValueError("existing behavioral unit differs from packet lock")
        return row
    family = DESIGNS[card]["family"]
    if family in {"opportunity","critic"}:
        public,private,maker,constructor = opportunities.generate(namespace,condition,index,constructors)
    else:
        public,private,constructor = selves.generate(namespace,condition,index,constructors)
        maker = None
    public_path = root/"public"/f"{uid}.json"
    opaque = read(public_path)["task_id"] if public_path.exists() else uuid.uuid4().hex
    public["task_id"] = opaque
    inputs = {}
    if family=="critic":
        goal = maker.purpose
        views = {"own-resources":((0,1),(0,1)),
                 "actual-constraints":(maker.actual,maker.actual),
                 "known-repertoire":(maker.actual,maker.beliefs)}
        for name,(actual,beliefs) in views.items():
            inputs[name] = {"schema_version":"v16.critic.1","goal":goal,"action_cost":maker.action_cost,
                            "actual":list(actual),"beliefs":list(beliefs),"search_budget":2}
    else:
        for name in DESIGNS[card]["arms"]:
            inputs[name] = copy.deepcopy(public)
            if name=="without-probe":
                inputs[name]["observations"] = []
                inputs[name]["query_cost"] = 0
            if name=="memory-only":
                inputs[name]["artifacts"] = []
    observation_record = {"task_id":opaque,"base":public,"arms":inputs}
    public_hash = write(public_path,observation_record)
    if family=="self-trajectory":
        return execute_trajectory(root,uid,card,condition,index,namespace,packet,constructors,
                                  evidence_scope,constructor,public,private,public_hash)
    if family=="opportunity":
        predictions = {name:opportunities.public_reader(canonical(observation),
                       model="latent-menu" if name=="without-probe" else name)
                       for name,observation in inputs.items()}
    elif family=="critic":
        predictions = {name:critic_reader(canonical(observation)) for name,observation in inputs.items()}
    else:
        predictions = {name:selves.public_reader(canonical(observation),
                       strategy="self-model" if name=="memory-only" else name)
                       for name,observation in inputs.items()}
    prediction_path = root/"predictions"/f"{uid}.json"
    submitted = read(prediction_path)["submitted_at"] if prediction_path.exists() else now()
    prediction_hash = write(prediction_path,{"submitted_at":submitted,"observation_hash":public_hash,
                                             "arms":predictions})
    rng = random.Random(seed_for(namespace,condition["id"],index,"scoring"))
    if family=="opportunity":
        private["hidden_continuation"] = opportunities.enact(maker,public["target_probe"],rng)
    elif family=="self":
        private["reset_coin"],private["future_coin"] = rng.random(),rng.random()
        private["future_before_repair"] = int(private["controller"] != (private["future_coin"] < private["execution_error"]))
    scored = {}
    for name,prediction in predictions.items():
        if prediction.get("model_mismatch"):
            raise ValueError("in-model behavioral unit has impossible evidence")
        if family=="opportunity":
            truth = private["hidden_continuation"]["artifact"]
            outcomes = {"future_log_score":math.log(prediction["future_probabilities"][truth]),
                        "future_success":float(truth==1),
                        "state_evaluations":float(prediction["costs"]["state_evaluations"]),
                        "query_cost":float(prediction["costs"]["query_cost"])}
            scored[name] = {**prediction,"outcomes":outcomes}
        elif family=="critic":
            view = inputs[name]
            actual_primitives = [0] if 1 in view["actual"] else []
            execution = interpret(prediction["program"],feasible=actual_primitives)
            success = execution["legal"] and execution["artifact"]==view["goal"]
            scored[name] = {**prediction,"execution":execution,
                            "outcomes":{"success":float(success),"legal":float(execution["legal"]),
                                        "utility":float(success)-view["action_cost"]*prediction["chosen_option"],
                                        "search_evaluations":float(prediction["costs"]["search_evaluations"]),
                                        "primitive_cost":float(execution["primitive_cost"])}}
        else:
            result = selves.evaluate(private,prediction,reset_coin=private["reset_coin"],future_coin=private["future_coin"])
            outcomes = {key:float(result[key]) for key in
                        ["net_repair","original_goal_success","adopted_goal_success","original_goal_recovery",
                         "continued_original_goal","legal"]}
            outcomes["controller_log_score"] = math.log(prediction["controller_probabilities"][private["controller"]])
            outcomes["future_log_score"] = math.log(prediction["future_probabilities"][private["future_before_repair"]])
            outcomes.update(state_evaluations=float(prediction["costs"]["state_evaluations"]),
                            monitoring_cost=float(prediction["costs"]["monitoring"]),
                            repair_cost=float(prediction["costs"]["repair"]))
            scored[name] = {**prediction,"execution":result,"outcomes":outcomes}
    private_hash = write(root/"private"/f"{uid}.json",private)
    row = base_row(uid,card,condition,index,namespace,packet,constructors,evidence_scope,constructor)
    row.update(public=observation_record,private=private,arms=scored,observation_hash=public_hash,
               prediction_hash=prediction_hash,truth_hash=private_hash,
               prediction_submitted_at=submitted,scored_at=now())
    write(unit_path,row)
    return row


def base_row(uid,card,condition,index,namespace,packet,constructors,scope,constructor):
    return {"unit_id":uid,"card_id":card,"condition":condition["id"],"unit_kind":DESIGNS[card]["family"],
            "constructor_id":constructor,"maker_history_id":uid,"evidence_scope":scope,
            "lineage":namespace,"seed_components":{"index":index,"constructors":constructors},
            "packet_hash":packet["packet_hash"],"commission_hash":packet["identity"]["commission_hash"],
            "environment_hash":digest(packet["identity"]["environment"]),"failures":[]}


def execute_trajectory(root,uid,card,condition,index,namespace,packet,constructors,scope,constructor,
                       public,private,public_hash):
    views = {name:{"schema_version":"v16.self-trajectory.1","task_id":public["task_id"],
                  "base":copy.deepcopy(public),"history":[]} for name in DESIGNS[card]["arms"]}
    controllers = {name:private["controller"] for name in views}
    rounds = []
    for round_index in range(3):
        round_path = root/"predictions"/f"{uid}-round-{round_index}.json"
        predictions = {name:trajectories.reader(canonical(view),name) for name,view in views.items()}
        submitted = read(round_path)["submitted_at"] if round_path.exists() else now()
        receipt = {"round":round_index,"submitted_at":submitted,"public_inputs":copy.deepcopy(views),
                   "arms":predictions}
        round_hash = write(round_path,receipt)
        rng = random.Random(seed_for(namespace,condition["id"],index,"scoring",round_index))
        reset_coin,future_coin = rng.random(),rng.random()
        executions = {}
        for name,prediction in predictions.items():
            state = {**private,"controller":controllers[name]}
            result = selves.evaluate(state,prediction,reset_coin=reset_coin,future_coin=future_coin)
            after = result["controller_after"]
            artifact = 1-after if future_coin<private["execution_error"] else after
            controllers[name] = after
            executions[name] = result
            views[name]["history"].append({"reset_requested":prediction["repair"],
                                           "reset_goal":prediction["adopted_goal"],
                                           "observed_artifact":artifact})
        rounds.append({"prediction_hash":round_hash,"predictions":predictions,
                       "reset_coin":reset_coin,"future_coin":future_coin,"executions":executions,
                       "submitted_at":submitted})
    final = {name:trajectories.reader(canonical(view),name) for name,view in views.items()}
    scored = {}
    original = private["actual_original_goal"]
    for name,prediction in final.items():
        last = rounds[-1]["executions"][name]
        scored[name] = {**prediction,"outcomes":{
            "original_goal_success":float(last["original_goal_success"]),
            "adopted_goal_success":float(last["adopted_goal_success"]),
            "original_goal_recovery":float(int(prediction["original_goal_probabilities"][1]>
                                              prediction["original_goal_probabilities"][0])==original),
            "net_repair":sum(item["executions"][name]["net_repair"] for item in rounds)/3,
            "original_goal_log_score":math.log(prediction["original_goal_probabilities"][original]),
            "monitoring_cost":sum(item["predictions"][name]["costs"]["monitoring"] for item in rounds),
            "repair_cost":sum(item["executions"][name]["repair_cost"] for item in rounds),
            "state_evaluations":sum(item["predictions"][name]["costs"]["state_evaluations"] for item in rounds)
                                +prediction["costs"]["state_evaluations"]}}
    private["rounds"] = rounds
    private_hash = write(root/"private"/f"{uid}.json",private)
    prediction_hash = write(root/"predictions"/f"{uid}.json",
                            {"round_prediction_hashes":[item["prediction_hash"] for item in rounds],
                             "final_public_inputs":views,"final_predictions":final})
    row = base_row(uid,card,condition,index,namespace,packet,constructors,scope,constructor)
    row.update(public=read(root/"public"/f"{uid}.json"),private=private,arms=scored,
               observation_hash=public_hash,prediction_hash=prediction_hash,truth_hash=private_hash,
               prediction_submitted_at=rounds[0]["submitted_at"],scored_at=now())
    write(root/"units"/f"{uid}_points.json",row)
    return row


def summarize(card,rows):
    conditions = {}
    for design in DESIGNS[card]["conditions"]:
        selected = [row for row in rows if row["condition"]==design["id"]]
        if not selected:
            raise ValueError("missing registered behavioral condition")
        conditions[design["id"]] = {"contrasts":[paired_summary(selected,e) for e in estimands(card)],
             "arms":{name:{metric:sum(row["arms"][name]["outcomes"][metric] for row in selected)/len(selected)
                           for metric in selected[0]["arms"][name]["outcomes"]}
                     for name in DESIGNS[card]["arms"]}}
    return {"card_id":card,"conditions":conditions,"n_maker_packets":len(rows),
            "evidence_scope":rows[0]["evidence_scope"],"instrument_state":"valid",
            "independent_reaggregation":"pending","scope_limit":DESIGNS[card].get("scope_limit"),
            "naive_self_scope":"intentionally misspecified feedback attack, excluded from promotion"}
