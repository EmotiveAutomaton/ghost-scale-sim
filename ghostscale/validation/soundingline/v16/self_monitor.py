"""Acquired binary controller with accessible goal memory and executable resets."""
from __future__ import annotations
from itertools import product
from math import comb
from dataclasses import dataclass, asdict
import json
import random
from .records import seed_for
from .world import execute

TRAINING_STEPS = 8
TRAINING_ALIGNMENTS = (0.65,0.8,0.95)
EXECUTION_ERRORS = (0.02,0.1,0.2)
RESET_ERROR = 0.05


def controller_prior(original_goal, alignment):
    probability_one = alignment if original_goal else 1-alignment
    return sum(comb(TRAINING_STEPS,count)*probability_one**count *
               (1-probability_one)**(TRAINING_STEPS-count) for count in range(5,TRAINING_STEPS+1))


def acquire(namespace, constructor_id, maker_id):
    constructor_rng = random.Random(seed_for(namespace,"constructor",constructor_id))
    alignment = constructor_rng.choice(TRAINING_ALIGNMENTS)
    error = constructor_rng.choice(EXECUTION_ERRORS)
    rng = random.Random(seed_for(namespace,maker_id,"acquisition"))
    original_goal = rng.randrange(2)
    instructions = [original_goal if rng.random() < alignment else 1-original_goal
                    for _ in range(TRAINING_STEPS)]
    attempts = [(0,) if instruction else (4,) for instruction in instructions]
    feedback = [execute(attempt).artifact == goal for attempt,goal in zip(attempts,instructions)]
    counts = [sum(int(success and goal == action) for success,goal in zip(feedback,instructions))
              for action in range(2)]
    controller = int(counts[1] > counts[0])  # frozen tie convention
    return {"original_goal":original_goal,"controller":controller,"execution_error":error,
            "constructor":{"alignment":alignment,"execution_error":error},
            "acquisition_record":{"dates":list(range(TRAINING_STEPS)),
                                  "instructions":instructions,"attempts":[list(a) for a in attempts],
                                  "feedback":feedback,"successful_action_counts":counts,
                                  "compiled_controller":controller,"primitive_cost":TRAINING_STEPS}}


def produce(controller,error,rng):
    action = 1-controller if rng.random() < error else controller
    program = (0,) if action else (4,)
    execution = execute(program)
    return {"artifact":execution.artifact,"program":list(program),"legal":execution.legal,
            "primitive_cost":execution.primitive_cost,"controller":controller}


def public_reader(payload: bytes, *, strategy="self-model"):
    public = json.loads(payload)
    if set(public) != {"schema_version","task_id","access_tier","memory","artifacts",
                       "current_goal","retargeted","memory_reliability","artifact_reliability",
                       "monitoring_cost","repair_cost","target_request"} or public["schema_version"] != "v16.self.1":
        raise ValueError("self-monitor public schema violation")
    if strategy not in {"self-model","memory-only","direct-error","bayes-error","direct-completion"}:
        raise ValueError("unknown self-monitor")
    if strategy == "bayes-error":
        return direct_bayesian_error(public)
    memory = public["memory"]
    weights,states = [],[]
    for original,alignment,error,controller in product(range(2),TRAINING_ALIGNMENTS,EXECUTION_ERRORS,range(2)):
        prior_one = controller_prior(original,alignment)
        weight = 0.5/9*(prior_one if controller else 1-prior_one)
        for name,value in [("original_goal",original),("controller",controller)]:
            remembered = memory.get(name)
            if remembered is not None:
                reliability = public["memory_reliability"]
                weight *= reliability if remembered == value else 1-reliability
        if strategy != "memory-only":
            for artifact in public["artifacts"]:
                reliability = public["artifact_reliability"]
                correct_signal = (1-error) if artifact == controller else error
                weight *= reliability*correct_signal+(1-reliability)*(1-correct_signal)
        states.append((original,controller,error))
        weights.append(weight)
    total = sum(weights)
    if total == 0:
        return {"model_mismatch":True,"reason":"contradictory accessible memory"}
    weights = [weight/total for weight in weights]
    original_posterior = [sum(weight for weight,state in zip(weights,states) if state[0] == goal)
                          for goal in range(2)]
    controller_posterior = [sum(weight for weight,state in zip(weights,states) if state[1] == control)
                            for control in range(2)]
    declared_goal = public["current_goal"]
    if public["retargeted"] and declared_goal is not None:
        adopted_goal = declared_goal
    elif strategy in {"self-model","memory-only"}:
        adopted_goal = int(original_posterior[1] > original_posterior[0])
    else:
        adopted_goal = declared_goal if declared_goal is not None else (
            public["artifacts"][-1] if public["artifacts"] else 0)
    wrong_probability = 1-controller_posterior[adopted_goal]
    # Repair is an action with reset risk and a declared cost. Direct monitoring
    # evaluates current-goal error; it does not infer or replace an original goal.
    if strategy == "direct-error":
        if memory.get("controller") is not None and public["memory_reliability"] == 1:
            expected_error = float(memory["controller"] != adopted_goal)
        elif public["artifacts"]:
            # Serious ordinary error comparator uses the accessible signal and a
            # cost-sensitive noise correction, with the public error-rate prior.
            mismatches = sum(value != adopted_goal for value in public["artifacts"])
            signal_error = mismatches/len(public["artifacts"])
            average_error = sum(EXECUTION_ERRORS)/len(EXECUTION_ERRORS)
            expected_error = max(0.0,min(1.0,(signal_error-average_error)/(1-2*average_error)))
        else:
            expected_error = 0.5
        repair = expected_error*(1-RESET_ERROR)-(1-expected_error)*RESET_ERROR > public["repair_cost"]
    elif strategy == "direct-completion":
        repair = True
    else:
        repair = wrong_probability*(1-RESET_ERROR)-(1-wrong_probability)*RESET_ERROR > public["repair_cost"]
    future_probability_one = sum(weight*((1-error) if controller else error)
                                 for weight,(_,controller,error) in zip(weights,states))
    return {"model_mismatch":False,"reader":strategy,
            "controller_probabilities":controller_posterior,"original_goal_probabilities":original_posterior,
            "future_probabilities":[1-future_probability_one,future_probability_one],
            "adopted_goal":adopted_goal,"repair":bool(repair),
            "costs":{"monitoring":public["monitoring_cost"],
                     "repair":public["repair_cost"] if repair else 0,
                     "state_evaluations":len(states)*(1+len(public["artifacts"]))}}


def generate(namespace,condition,index,constructors=8):
    constructor_id = f"constructor-{index%constructors:03d}"
    maker = acquire(namespace,constructor_id,(condition["id"],index))
    rng = random.Random(seed_for(namespace,condition["id"],index,"observation"))
    memory_mode = condition.get("memory","partial")
    memory = {"original_goal":maker["original_goal"],"controller":maker["controller"] if memory_mode == "intact" else None}
    if memory_mode == "absent":
        memory["original_goal"] = None
    if memory_mode == "misleading":
        memory["original_goal"] = 1-maker["original_goal"]
    observations = [produce(maker["controller"],maker["execution_error"],rng)
                    for _ in range(condition.get("action_evidence",1))]
    edited = condition.get("edited_artifact",False)
    artifacts = [1-work["artifact"] if edited else work["artifact"] for work in observations]
    retarget = condition.get("retarget",False)
    if condition.get("notes",False):
        memory["original_goal"] = maker["original_goal"]
    # A newly assigned goal is independent of the forgotten goal: publishing a
    # guaranteed binary complement would itself reveal the original goal.
    current_goal = rng.randrange(2) if retarget else memory["original_goal"]
    public = {"schema_version":"v16.self.1","task_id":None,"access_tier":f"self-{memory_mode}",
              "memory":memory,"artifacts":artifacts,"current_goal":current_goal,"retargeted":retarget,
              "memory_reliability":0.8 if memory_mode == "misleading" else 1.0,
              "artifact_reliability":condition.get("assumed_artifact_reliability",1.0),
              "monitoring_cost":1.0,"repair_cost":0.2,
              "target_request":"predict controller, original goal, next action; submit an optional reset"}
    private = {**maker,"observations":observations,"artifact_edited":edited,
               "actual_original_goal":maker["original_goal"],
               "externally_adopted_goal":current_goal,
               "memory_mode":memory_mode,"constructor_id":constructor_id}
    return public,private,constructor_id


def evaluate(private,prediction,*,reset_coin,future_coin):
    original = private["actual_original_goal"]
    before = private["controller"]
    adopted = prediction["adopted_goal"]
    after = before
    if prediction["repair"]:
        after = 1-adopted if reset_coin < RESET_ERROR else adopted
    changed_helpfully = before != adopted and after == adopted
    changed_harmfully = before == adopted and after != adopted
    actual_action = 1-after if future_coin < private["execution_error"] else after
    executed = execute((0,) if actual_action else (4,))
    return {"net_repair":int(changed_helpfully)-int(changed_harmfully),
            "correct_repairs":int(changed_helpfully),"harmful_repairs":int(changed_harmfully),
            "controller_after":after,"adopted_goal_success":int(executed.artifact == adopted),
            "original_goal_success":int(executed.artifact == original),
            "original_goal_recovery":int(int(prediction["original_goal_probabilities"][1] >
                                              prediction["original_goal_probabilities"][0]) == original),
            "continued_original_goal":int(adopted == original),
            "legal":executed.legal,"primitive_cost":executed.primitive_cost,
            "repair_cost":prediction["costs"]["repair"]}


def direct_bayesian_error(public):
    """Direct expected-goal-error calculation with the complete allowed finite prior.

    This is the strong ordinary-monitoring rival. Its binary error state may be
    mathematically sufficient for the same repair decision as explicit self inference;
    that equivalence is an intended boundary, not an untested superiority claim.
    """
    target = public["current_goal"]
    # A missing externally supplied goal needs a separate inferred-goal decision.
    # Enumerate each candidate original goal before deciding; no private target.
    masses = []
    for old_goal in (0,1):
        for alignment in TRAINING_ALIGNMENTS:
            p_one = sum(comb(8,k)*(alignment if old_goal else 1-alignment)**k *
                        (1-alignment if old_goal else alignment)**(8-k) for k in range(5,9))
            for noise in EXECUTION_ERRORS:
                for action_error in (0,1):
                    reference_goal = target if target is not None else old_goal
                    control = reference_goal ^ action_error
                    mass = (p_one if control else 1-p_one)/18
                    for name,value in (("original_goal",old_goal),("controller",control)):
                        claim = public["memory"].get(name)
                        if claim is not None:
                            mass *= public["memory_reliability"] if claim == value else 1-public["memory_reliability"]
                    for seen in public["artifacts"]:
                        emitted = 1-noise if seen == control else noise
                        mass *= public["artifact_reliability"]*emitted+(1-public["artifact_reliability"])*(1-emitted)
                    masses.append((old_goal,control,noise,mass))
    denominator = sum(item[3] for item in masses)
    if denominator == 0:
        return {"model_mismatch":True,"reason":"contradictory accessible memory"}
    original = [sum(mass for goal,control,noise,mass in masses if goal == candidate)/denominator
                for candidate in (0,1)]
    adopted = target if public["retargeted"] and target is not None else int(original[1] > original[0])
    control_probabilities = [sum(mass for goal,control,noise,mass in masses if control == candidate)/denominator
                             for candidate in (0,1)]
    error_probability = sum(mass for goal,control,noise,mass in masses if control != adopted)/denominator
    repair = error_probability*(1-RESET_ERROR)-(1-error_probability)*RESET_ERROR > public["repair_cost"]
    future = sum(mass*((1-noise) if control else noise) for goal,control,noise,mass in masses)/denominator
    return {"model_mismatch":False,"reader":"bayes-error","controller_probabilities":control_probabilities,
            "original_goal_probabilities":original,"future_probabilities":[1-future,future],
            "adopted_goal":adopted,"repair":repair,
            "costs":{"monitoring":public["monitoring_cost"],"repair":public["repair_cost"] if repair else 0,
                     "state_evaluations":len(masses)*(1+len(public["artifacts"]))}}
