"""Exact controlled-state filtering for self-evidence after one's own resets."""
from itertools import product
import json
from .self_monitor import (TRAINING_ALIGNMENTS,EXECUTION_ERRORS,RESET_ERROR,
                           controller_prior)


def infer(public,*,ignore_interventions=False):
    base = public["base"]
    states = {}
    for goal,q,error,control in product(range(2),TRAINING_ALIGNMENTS,EXECUTION_ERRORS,range(2)):
        p = controller_prior(goal,q)
        weight = (p if control else 1-p)/18
        for name,value in (("original_goal",goal),("controller",control)):
            memory = base["memory"].get(name)
            if memory is not None:
                weight *= base["memory_reliability"] if memory == value else 1-base["memory_reliability"]
        states[(goal,q,error,control)] = weight
    evaluations = len(states)
    def observe(artifact):
        nonlocal states,evaluations
        changed = {}
        for state,weight in states.items():
            goal,q,error,control = state
            p = 1-error if artifact == control else error
            reliability = base["artifact_reliability"]
            changed[state] = weight*(reliability*p+(1-reliability)*(1-p))
        total = sum(changed.values())
        if total == 0:
            raise ValueError("impossible self-trajectory evidence")
        states = {state:weight/total for state,weight in changed.items()}
        evaluations += len(states)
    total = sum(states.values())
    if total == 0:
        raise ValueError("contradictory self-trajectory memory")
    states = {state:weight/total for state,weight in states.items()}
    for artifact in base["artifacts"]:
        observe(artifact)
    for step in public["history"]:
        if step["reset_requested"] and not ignore_interventions:
            changed = {}
            for (goal,q,error,control),weight in states.items():
                for after in range(2):
                    probability = 1-RESET_ERROR if after == step["reset_goal"] else RESET_ERROR
                    key = (goal,q,error,after)
                    changed[key] = changed.get(key,0.0)+weight*probability
                    evaluations += 1
            states = changed
        observe(step["observed_artifact"])
    original = [sum(w for (g,q,e,c),w in states.items() if g == value) for value in range(2)]
    controller = [sum(w for (g,q,e,c),w in states.items() if c == value) for value in range(2)]
    return states,original,controller,evaluations


def reader(payload:bytes,strategy="self-model"):
    public = json.loads(payload)
    if set(public) != {"schema_version","task_id","base","history"} or public["schema_version"] != "v16.self-trajectory.1":
        raise ValueError("trajectory public schema violation")
    if strategy not in {"self-model","bayes-error","direct-completion","direct-error","naive-self"}:
        raise ValueError("unknown trajectory reader")
    states,original,controller,cost = infer(public,ignore_interventions=strategy=="naive-self")
    base = public["base"]
    if base["retargeted"] and base["current_goal"] is not None:
        adopted = base["current_goal"]
    elif strategy in {"self-model","bayes-error","naive-self"}:
        adopted = int(original[1]>original[0])
    else:
        adopted = base["current_goal"] if base["current_goal"] is not None else (
            public["history"][-1]["observed_artifact"] if public["history"] else
            base["artifacts"][-1] if base["artifacts"] else 0)
    if strategy == "direct-completion":
        repair = True
    elif strategy == "direct-error":
        latest = public["history"][-1]["observed_artifact"] if public["history"] else (
            base["artifacts"][-1] if base["artifacts"] else adopted)
        repair = latest != adopted
    elif strategy == "bayes-error":
        # Direct expected goal-error reduction, without requiring an old-controller
        # identity decision. It can be sufficient for the same repair policy.
        wrong = sum(weight for (g,q,error,control),weight in states.items() if control != adopted)
        repair = wrong*(1-RESET_ERROR)-(1-wrong)*RESET_ERROR > base["repair_cost"]
    else:
        wrong = 1-controller[adopted]
        repair = wrong*(1-RESET_ERROR)-(1-wrong)*RESET_ERROR > base["repair_cost"]
    future_one = sum(weight*((1-error) if control else error)
                     for (g,q,error,control),weight in states.items())
    return {"reader":strategy,"model_mismatch":False,"original_goal_probabilities":original,
            "controller_probabilities":controller,"future_probabilities":[1-future_one,future_one],
            "adopted_goal":adopted,"repair":bool(repair),
            "costs":{"state_evaluations":cost,"monitoring":base["monitoring_cost"],
                     "repair":base["repair_cost"] if repair else 0},
            "evidence_scope":"attack-diagnostic" if strategy=="naive-self" else "eligible"}


def gates():
    from .records import canonical
    base = {"schema_version":"v16.self.1","task_id":"opaque","access_tier":"self-absent",
            "memory":{"original_goal":None,"controller":None},"artifacts":[],
            "current_goal":None,"retargeted":False,"memory_reliability":1,
            "artifact_reliability":1,"monitoring_cost":1,"repair_cost":0.2,"target_request":"fixture"}
    public = {"schema_version":"v16.self-trajectory.1","task_id":"opaque","base":base,
              "history":[{"reset_requested":True,"reset_goal":1,"observed_artifact":1}]}
    correct = reader(canonical(public))
    naive = reader(canonical(public),"naive-self")
    error = sum(EXECUTION_ERRORS)/len(EXECUTION_ERRORS)
    exact_control = (1-RESET_ERROR)*(1-error)/((1-RESET_ERROR)*(1-error)+RESET_ERROR*error)
    checks = {"known_reset_posterior":abs(correct["controller_probabilities"][1]-exact_control)<1e-10,
              "own_reset_does_not_reveal_forgotten_goal":abs(correct["original_goal_probabilities"][1]-0.5)<1e-10,
              "naive_feedback_break_detected":abs(naive["original_goal_probabilities"][1]-0.5)>1e-3,
              "old_goal_boundary_retained":all(p>0 for p in correct["original_goal_probabilities"])}
    return {"id":"self-feedback-causality","evidence_scope":"fixture","checks":checks,
            "instrument_state":"valid" if all(checks.values()) else "failed"}
