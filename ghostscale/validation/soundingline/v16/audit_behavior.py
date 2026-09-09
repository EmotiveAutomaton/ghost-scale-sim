"""Independent executable behavioral scoring, raw integrity and reaggregation."""
import hashlib
import json
import math
import random
from pathlib import Path
from .audit_statistics import verify


def rng_for(*parts):
    payload = json.dumps(parts,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return random.Random(int(hashlib.sha256(payload).hexdigest()[:16],16))


def self_outcome(private,prediction,before,reset_coin,future_coin):
    adopted = prediction["adopted_goal"]
    after = before
    if prediction["repair"]:
        after = int(adopted) if reset_coin>=0.05 else 1-int(adopted)
    goal = private["actual_original_goal"]
    artifact = after if future_coin>=private["execution_error"] else 1-after
    net = int(before!=adopted and after==adopted)-int(before==adopted and after!=adopted)
    original_guess = int(prediction["original_goal_probabilities"][1]>prediction["original_goal_probabilities"][0])
    return {"net_repair":net,"correct_repairs":int(before!=adopted and after==adopted),
            "harmful_repairs":int(before==adopted and after!=adopted),"controller_after":after,
            "adopted_goal_success":int(artifact==adopted),"original_goal_success":int(artifact==goal),
            "original_goal_recovery":int(original_guess==goal),"continued_original_goal":int(adopted==goal),
            "legal":True,"primitive_cost":1,"repair_cost":prediction["costs"]["repair"]},artifact


def opportunity_future(private,probe,rng):
    state = dict(private["maker"])
    if probe=="reminder":
        state["considered"]=[0,1]
    elif probe=="demonstration":
        state["beliefs"]=[0,1] if 1 in state["actual"] else [0]
        state["considered"]=[0,1]
    elif probe=="tool":
        state.update(actual=[0,1],beliefs=[0,1],considered=[0,1])
    elif probe=="retarget":
        state["purpose"]=1
    elif probe=="search":
        state["search_budget"]=2
    elif probe=="higher-reward":
        state["reward"]=2.0
    examined = state["considered"][:state["search_budget"]]
    candidates = [a for a in examined if a in state["beliefs"]]
    intended = max(candidates,key=lambda a:(state["reward"]*int(a==state["purpose"])-state["action_cost"]*a,-a))
    flip = probe!="baseline" and rng.random()<state["lapse"]
    attempt = 1-intended if flip else intended
    artifact = int(attempt==1 and 1 in state["actual"])
    continuation = private["hidden_continuation"]
    if continuation["state_after"]!=state or continuation["attempted_option"]!=attempt or continuation["artifact"]!=artifact:
        raise ValueError("independent intervention execution mismatch")
    return artifact


def audit(root:Path,summary):
    rows = [json.loads(path.read_bytes()) for path in (root/"units").glob("*_points.json")]
    for row in rows:
        sources = {}
        for directory,key in [("public","observation_hash"),("private","truth_hash"),("predictions","prediction_hash")]:
            payload = (root/directory/(row["unit_id"]+".json")).read_bytes()
            if hashlib.sha256(payload).hexdigest()!=row[key]:
                raise ValueError("raw reference hash mismatch")
            sources[directory]=json.loads(payload)
        if row["public"]!=sources["public"] or row["private"]!=sources["private"]:
            raise ValueError("embedded raw data differs from separate file")
        private = sources["private"]
        family = row["unit_kind"]
        if family=="self-trajectory":
            verify_trajectory(root,row,sources)
            continue
        rng = rng_for(row["lineage"],row["condition"],row["seed_components"]["index"],"scoring")
        if family=="opportunity":
            truth = opportunity_future(private,row["public"]["base"]["target_probe"],rng)
        elif family=="self":
            reset_coin,future_coin = rng.random(),rng.random()
            if (reset_coin,future_coin)!=(private["reset_coin"],private["future_coin"]):
                raise ValueError("scoring seed lineage mismatch")
            expected_controller = int(private["acquisition_record"]["successful_action_counts"][1]>
                                      private["acquisition_record"]["successful_action_counts"][0])
            if expected_controller!=private["controller"]:
                raise ValueError("compiled controller disagrees with acquisition counts")
        for name,arm in row["arms"].items():
            prediction = sources["predictions"]["arms"][name]
            for key,value in prediction.items():
                if arm[key]!=value:
                    raise ValueError("scored arm differs from submitted prediction")
            if family=="opportunity":
                outcomes={"future_log_score":math.log(prediction["future_probabilities"][truth]),
                          "future_success":float(truth==1),
                          "state_evaluations":float(prediction["costs"]["state_evaluations"]),
                          "query_cost":float(prediction["costs"]["query_cost"])}
            elif family=="critic":
                view=row["public"]["arms"][name]
                if view["goal"]!=private["maker"]["purpose"]:
                    raise ValueError("critic goal changed across resources")
                if name!="own-resources" and view["actual"]!=private["maker"]["actual"]:
                    raise ValueError("critic resource matching failed")
                artifact,legal=0,True
                for action in prediction["program"]:
                    if action!=0 or 1 not in view["actual"]:
                        legal=False
                        break
                    artifact=1
                success=legal and artifact==view["goal"]
                outcomes={"success":float(success),"legal":float(legal),
                          "utility":float(success)-view["action_cost"]*prediction["chosen_option"],
                          "search_evaluations":float(prediction["costs"]["search_evaluations"]),
                          "primitive_cost":float(len(prediction["program"]))}
            else:
                execution,artifact=self_outcome(private,prediction,private["controller"],reset_coin,future_coin)
                if execution!=arm["execution"]:
                    raise ValueError("independent reset/goal scoring mismatch")
                outcomes={key:float(execution[key]) for key in
                          ["net_repair","original_goal_success","adopted_goal_success","original_goal_recovery",
                           "continued_original_goal","legal"]}
                before_future=private["controller"] if future_coin>=private["execution_error"] else 1-private["controller"]
                outcomes["controller_log_score"]=math.log(prediction["controller_probabilities"][private["controller"]])
                outcomes["future_log_score"]=math.log(prediction["future_probabilities"][before_future])
                outcomes.update(state_evaluations=float(prediction["costs"]["state_evaluations"]),
                                monitoring_cost=float(prediction["costs"]["monitoring"]),
                                repair_cost=float(prediction["costs"]["repair"]))
            if outcomes!=arm["outcomes"]:
                raise ValueError("independent behavioral outcome mismatch")
    count=verify(rows,summary)
    return {"execution_state":"completed","instrument_state":"valid","n_raw_units":len(rows),
            "reproduced_contrasts":count,"all_reported_aggregates_reproduced":True,"full_rollout_replay":False}


def verify_trajectory(root,row,sources):
    private=row["private"]
    controllers={name:private["controller"] for name in row["arms"]}
    histories={name:[] for name in row["arms"]}
    executions={name:[] for name in row["arms"]}
    predictions={name:[] for name in row["arms"]}
    for round_index,round_record in enumerate(private["rounds"]):
        payload=(root/"predictions"/f"{row['unit_id']}-round-{round_index}.json").read_bytes()
        if hashlib.sha256(payload).hexdigest()!=round_record["prediction_hash"]:
            raise ValueError("round prediction hash mismatch")
        submitted=json.loads(payload)
        if submitted["arms"]!=round_record["predictions"]:
            raise ValueError("round actions differ from pre-outcome commitment")
        rng=rng_for(row["lineage"],row["condition"],row["seed_components"]["index"],"scoring",round_index)
        reset_coin,future_coin=rng.random(),rng.random()
        if (reset_coin,future_coin)!=(round_record["reset_coin"],round_record["future_coin"]):
            raise ValueError("trajectory seed mismatch")
        for name,prediction in submitted["arms"].items():
            if submitted["public_inputs"][name]["history"]!=histories[name]:
                raise ValueError("reader's intervention history differs from actual past actions")
            outcome,artifact=self_outcome(private,prediction,controllers[name],reset_coin,future_coin)
            if outcome!=round_record["executions"][name]:
                raise ValueError("independent trajectory execution mismatch")
            controllers[name]=outcome["controller_after"]
            histories[name].append({"reset_requested":prediction["repair"],"reset_goal":prediction["adopted_goal"],
                                    "observed_artifact":artifact})
            executions[name].append(outcome)
            predictions[name].append(prediction)
    for name,arm in row["arms"].items():
        final=sources["predictions"]["final_predictions"][name]
        if sources["predictions"]["final_public_inputs"][name]["history"]!=histories[name]:
            raise ValueError("final public history differs")
        for key,value in final.items():
            if arm[key]!=value:
                raise ValueError("final belief record differs")
        last=executions[name][-1]
        goal=private["actual_original_goal"]
        expected={"original_goal_success":float(last["original_goal_success"]),
                  "adopted_goal_success":float(last["adopted_goal_success"]),
                  "original_goal_recovery":float(int(final["original_goal_probabilities"][1]>
                                                     final["original_goal_probabilities"][0])==goal),
                  "net_repair":sum(item["net_repair"] for item in executions[name])/3,
                  "original_goal_log_score":math.log(final["original_goal_probabilities"][goal]),
                  "monitoring_cost":sum(item["costs"]["monitoring"] for item in predictions[name]),
                  "repair_cost":sum(item["repair_cost"] for item in executions[name]),
                  "state_evaluations":sum(item["costs"]["state_evaluations"] for item in predictions[name])
                                      +final["costs"]["state_evaluations"]}
        if expected!=arm["outcomes"]:
            raise ValueError("independent trajectory aggregate mismatch")
