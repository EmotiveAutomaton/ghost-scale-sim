"""A finite causal opportunity world with executable interventions.

Physical feasibility, beliefs, considered candidates, goals and search effort are
different state components. False beliefs need not be subsets of feasibility.
"""
from __future__ import annotations
from dataclasses import dataclass, replace, asdict
from functools import lru_cache
from itertools import product
import json
import random
from .world import execute
from .records import seed_for

CAUSES = ("physical","knowledge","consideration","purpose","search","false-affordance")
COSTS = (0.1,0.5,1.25)
LAPSES = (0.02,0.08,0.2)
PROBES = ("baseline","reminder","demonstration","tool","retarget","search","higher-reward")


@dataclass(frozen=True)
class Maker:
    actual: tuple[int,...]
    beliefs: tuple[int,...]
    considered: tuple[int,...]
    purpose: int
    search_budget: int
    action_cost: float
    lapse: float
    reward: float = 1.0


def make(cause, cost=0.1, lapse=0.02):
    maker = Maker((0,1),(0,1),(0,1),1,2,cost,lapse)
    if cause == "physical":
        return replace(maker,actual=(0,),beliefs=(0,))
    if cause == "knowledge":
        return replace(maker,beliefs=(0,))
    if cause == "consideration":
        return replace(maker,considered=(0,))
    if cause == "purpose":
        return replace(maker,purpose=0)
    if cause == "search":
        return replace(maker,search_budget=1)
    if cause == "false-affordance":
        return replace(maker,actual=(0,))
    raise ValueError("unknown cause")


def intervene(maker, probe):
    if probe == "baseline":
        return maker
    if probe == "reminder":
        return replace(maker,considered=(0,1))
    if probe == "demonstration":
        # An actual attempted tool demonstration teaches its observed availability.
        demo = execute((0,),feasible=(0,) if 1 in maker.actual else ())
        learned = (0,1) if demo.legal else (0,)
        return replace(maker,beliefs=learned,considered=(0,1))
    if probe == "tool":
        return replace(maker,actual=(0,1),beliefs=(0,1),considered=(0,1))
    if probe == "retarget":
        return replace(maker,purpose=1)
    if probe == "search":
        return replace(maker,search_budget=2)
    if probe == "higher-reward":
        return replace(maker,reward=2.0)
    raise ValueError("unknown intervention")


def decide(maker):
    evaluated = list(maker.considered[:maker.search_budget])
    known = [option for option in evaluated if option in maker.beliefs]
    def utility(option):
        return maker.reward*int(option == maker.purpose)-maker.action_cost*option
    selected = max(known,key=lambda option:(utility(option),-option)) if known else 0
    return selected, evaluated


def enact(maker, probe, rng=None, *, lapse_event=None):
    changed = intervene(maker,probe)
    selected,evaluated = decide(changed)
    # Baseline is the controlled same-choice anchor; later motor lapses are actual
    # executable errors, not noise added to an evaluator label.
    if lapse_event is None:
        lapse_event = probe != "baseline" and rng is not None and rng.random() < changed.lapse
    attempted = 1-selected if lapse_event else selected
    execution = execute((0,) if attempted else (),
                        feasible=(0,) if 1 in changed.actual else ())
    return {"artifact":execution.artifact,"legal":execution.legal,
            "attempted_option":attempted,"intended_option":selected,
            "search_evaluations":len(evaluated),"considered_candidates":list(changed.considered),
            "evaluated_candidates":evaluated,"primitive_cost":execution.primitive_cost,
            "state_before":asdict(maker),"state_after":asdict(changed),
            "intervention_realized":changed != maker,
            "actual_goal_success":execution.artifact == changed.purpose}


@lru_cache(maxsize=512)
def response_probability(cause,cost,lapse,probe):
    maker = make(cause,cost,lapse)
    normal = enact(maker,probe,lapse_event=False)["artifact"]
    slipped = enact(maker,probe,lapse_event=True)["artifact"]
    probability = 0.0 if probe == "baseline" else lapse
    return (1-probability)*normal + probability*slipped


def public_reader(payload: bytes, model="latent-menu"):
    public = json.loads(payload)
    allowed = {"schema_version","task_id","access_tier","final_artifact","observations",
               "allowed_causes","target_probe","query_cost","reader_search_budget"}
    if set(public) != allowed or public["schema_version"] != "v16.opportunity.1":
        raise ValueError("opportunity public schema violation")
    causes = public["allowed_causes"]
    if model == "fixed-menu":
        causes = [cause for cause in causes if cause != "consideration"]
    elif model == "budget-only":
        causes = [cause for cause in causes if cause in {"search","purpose"}]
    elif model == "accurate-belief":
        causes = [cause for cause in causes if cause != "false-affordance"]
    elif model not in {"latent-menu","generic","direct-table"}:
        raise ValueError("unknown opportunity reader")
    if not causes:
        return {"model_mismatch":True,"reason":"reader has no represented cause"}
    states = list(product(causes,COSTS,LAPSES))
    if len(states)*(len(public["observations"])+1) > public["reader_search_budget"]:
        raise ValueError("declared finite-reader budget is insufficient")
    weights = [1/len(states)]*len(states)
    for observed in public["observations"]:
        for index,(cause,cost,lapse) in enumerate(states):
            probability = response_probability(cause,cost,lapse,observed["probe"])
            weights[index] *= probability if observed["artifact"] else 1-probability
        total = sum(weights)
        if total == 0:
            return {"model_mismatch":True,"reason":"impossible opportunity evidence"}
        weights = [weight/total for weight in weights]
    if model == "generic":
        weights = [1/len(states)]*len(states)
    probability = sum(weight*response_probability(cause,cost,lapse,public["target_probe"])
                      for weight,(cause,cost,lapse) in zip(weights,states))
    cause_posterior = {cause:sum(weight for weight,state in zip(weights,states) if state[0] == cause)
                       for cause in causes}
    return {"model_mismatch":False,"reader":model,"future_probabilities":[1-probability,probability],
            "cause_posterior":None if model == "direct-table" else cause_posterior,
            "costs":{"state_evaluations":len(states)*(len(public["observations"])+1),
                     "query_cost":public["query_cost"]},
            "reconstruction":None}


def generate(namespace,condition,index,constructors=8):
    constructor_id = f"constructor-{index%constructors:03d}"
    constants = random.Random(seed_for(namespace,"constructor",constructor_id))
    cost,lapse = constants.choice(COSTS),constants.choice(LAPSES)
    rng = random.Random(seed_for(namespace,condition["id"],index,"maker"))
    causes = condition.get("causes",list(CAUSES[:5]))
    cause = condition.get("force_cause") or rng.choice(causes)
    maker = make(cause,cost,lapse)
    baseline = enact(maker,"baseline",lapse_event=False)
    probes = condition.get("observed_probes",["reminder"])
    private_observations = [{"probe":probe,"execution":enact(maker,probe,rng)} for probe in probes]
    public = {"schema_version":"v16.opportunity.1","task_id":None,"access_tier":"artifact-plus-paid-probes",
              "final_artifact":baseline["artifact"],
              "observations":[{"probe":item["probe"],"artifact":item["execution"]["artifact"]}
                              for item in private_observations],
              "allowed_causes":causes,"target_probe":condition.get("target_probe","demonstration"),
              "query_cost":len(probes),"reader_search_budget":1024}
    # The false-affordance card has its own baseline cases. For O01's four causes,
    # low-cost states demonstrably share the same initial intended choice.
    private = {"cause":cause,"constructor_id":constructor_id,"maker":asdict(maker),
               "original_goal":maker.purpose,"current_goal":maker.purpose,
               "baseline_execution":baseline,"intervention_outcomes":private_observations,
               "actual_feasibility":list(maker.actual),"maker_beliefs":list(maker.beliefs),
               "considered_alternatives":list(maker.considered),
               "search_effort":baseline["search_evaluations"],"acquisition_record":"belief acquisition is separately manipulated here"}
    return public,private,maker,constructor_id
