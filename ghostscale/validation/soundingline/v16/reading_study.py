"""Prospective reconstruction packets K02/P01/P02/P03; fixed before their scouts."""
from __future__ import annotations
from pathlib import Path
import math
import random
import uuid
from .reconstruction import prepare, reader, sample_work
from .records import canonical, digest, read, write, now, seed_for
from .reference import interpret
from .estimands import Estimand, paired_summary

SAMPLE_RULE = {"scout": 64, "scout_constructors": 8, "expansion": 256,
               "expansion_constructors": 20, "final_expansion": 1024}
DESIGNS = {
    "K02": {"question": "Does inferred personal acquisition predict hidden production beyond generic knowledge?",
            "conditions": [{"id":"prior-4", "prior_works":4}],
            "mechanism": "Acquisition changes executable route costs; earlier finished works update a hidden repertoire mixture.",
            "primary": [("maker", "generic", "future_log_score", "nats_per_event", 0.02),
                        ("maker", "direct-table", "future_log_score", "nats_per_event", 0.02)],
            "target_realization": "Repertoires are learned from executed, dated attempts; true libraries never enter observer bytes.",
            "continuation": "Keep generic and direct-table explanations when personal reconstruction adds no prediction."},
    "P01": {"question": "Can executable reconstruction succeed while the historical route remains ambiguous?",
            "conditions": [{"id":f"budget-{budget}", "search_budget":budget} for budget in [8,32,128]],
            "mechanism": "Multiple legal place/remove histories terminate in the same visible artifact.",
            "primary": [("maker", "primitive", "success", "success_fraction", 0.05)],
            "target_realization": "Retain all colliding true-route candidates; independent interpreter executes submissions.",
            "continuation": "Retain useful reconstruction with unresolved history; do not force route identification."},
    "P02": {"question": "Which paid observation separates histories sharing a finished artifact?",
            "conditions": [{"id":query,"query":query} for query in ["none","process","continuation","tool","prior-work"]],
            "mechanism": "A true process prefix, independent continuation, changed tool set or another work constrains latent craft.",
            "primary": [("maker", "without-query", "future_log_score", "nats_per_event", 0.02),
                        ("maker", "direct-table", "future_log_score", "nats_per_event", 0.02)],
            "target_realization": "Each revealed item is generated under its declared intervention; tool removal changes legal route support.",
            "continuation": "Probe only future-relevant distinctions; retain indistinguishable histories and charge query costs."},
    "P03": {"question": "Does permitted evidence across works improve a prospective individual model?",
            "conditions": [{"id":f"dose-{dose}","prior_works":dose} for dose in [0,1,2,4,8]],
            "mechanism": "All prior artifacts share one independently acquired maker repertoire; likelihoods preserve this dependence.",
            "primary": [("maker", "generic", "future_log_score", "nats_per_event", 0.02),
                        ("maker", "direct-table", "future_log_score", "nats_per_event", 0.02)],
            "target_realization": "Evidence counts are numeric; independent training/history and constructor seeds are retained.",
            "continuation": "Expand a named evidence boundary or rival separation; recognition is not process prediction."},
}
for card, design in DESIGNS.items():
    design.update({"card_id":card, "strongest_rival":"direct prediction table with the same public finite model and evidence",
                   "access_arms":"same finished artifact, declared task context and permitted prior/query observations; without-query is explicitly paid-access ablation",
                   "secondary":["repertoire log score","posterior entropy","route collision class size","legality",
                                "reconstruction success","search/training/query costs"],
                   "generator_families":["W1 learned-library softmax route production"],
                   "paired_unit":"one independent acquisition history with paired reader predictions on one held-out maker event",
                   "sample_rule":SAMPLE_RULE, "repair_budget":1,
                   "dependencies":["W1-execution","exact-training-prior","artifact-marginalization",
                                   "data-use-joint-calibration","physical-reader-boundary","independent-reaggregation"],
                   "adversaries":["X01","X02","X03","X04","X07","X08"]})
DESIGNS["P02"]["adversaries"].append("X06")


def estimands(card):
    return [Estimand(f"{card}-{arm}-minus-{rival}-{target}", target, arm, rival, units, bar,
                     f"{arm} minus {rival}: {target}")
            for arm,rival,target,units,bar in DESIGNS[card]["primary"]]


def execute_unit(root: Path, card, condition, index, *, namespace, packet_hash,
                 constructors=8, evidence_scope="discovery"):
    uid = digest([namespace,card,condition["id"],index])[:24]
    path = root / "units" / f"{uid}_points.json"
    if path.exists():
        row = read(path)
        if row["packet_hash"] != packet_hash:
            raise ValueError("completed reading unit has different source")
        return row
    public, private, constructor_id = prepare(namespace,card,condition,index,constructors)
    public_path = root / "public" / f"{uid}.json"
    public["task_id"] = read(public_path)["task_id"] if public_path.exists() else uuid.uuid4().hex
    public_hash = write(public_path,public)
    arms = {name:reader(canonical(public),name) for name in ["maker","generic","direct-table","primitive"]}
    ablated = read(public_path)
    ablated["declared_context"]["current_observation"]["prefix"] = []
    if condition.get("query","none") in {"continuation","tool","prior-work"}:
        ablated["permitted_prior_artifacts"] = ablated["permitted_prior_artifacts"][:-1]
    ablated["query_costs"]["diagnostic"] = 0
    arms["without-query"] = reader(canonical(ablated),"maker")
    prediction_path = root / "predictions" / f"{uid}.json"
    submitted_at = read(prediction_path)["submitted_at"] if prediction_path.exists() else now()
    prediction_hash = write(prediction_path,{"submitted_at":submitted_at,"observation_hash":public_hash,
                                            "arms":arms,"ablation_observation":ablated})
    future = sample_work(random.Random(seed_for(namespace,condition["id"],index,"future")),
                         private["acquisition_record"]["library_index"],12)
    private["hidden_continuations"] = future
    private_hash = write(root / "private" / f"{uid}.json",private)
    scored = {}
    for name,prediction in arms.items():
        if prediction["model_mismatch"]:
            raise ValueError("in-model reading packet generated impossible evidence; preserve partial raw records")
        execution = interpret(prediction["reconstruction"]["program"])
        legal = execution["legal"] and not prediction["reconstruction"]["search_timeout"]
        probabilities = prediction["future_probabilities"]
        if probabilities[future["artifact"]] <= 0:
            raise ValueError("zero predictive support: explicit mismatch requires its own interpretation")
        historical = prediction["history_probabilities"]
        scored[name] = {**prediction,"execution":execution,
                       "outcomes":{"future_log_score":math.log(probabilities[future["artifact"]]),
                                   "success":float(legal and execution["artifact"] == public["final_artifact"]),
                                   "legal":float(legal)},
                       "historical_log_score":None if historical is None or
                           historical[private["acquisition_record"]["library_index"]] == 0 else math.log(
                               historical[private["acquisition_record"]["library_index"]]),
                       "historical_model_mismatch":historical is not None and
                           historical[private["acquisition_record"]["library_index"]] == 0,
                       "historical_entropy":None if historical is None else -sum(
                           weight*math.log(weight) for weight in historical if weight > 0)}
    row = {"unit_id":uid,"card_id":card,"condition":condition["id"],
           "constructor_id":constructor_id,"maker_history_id":uid,"evidence_scope":evidence_scope,
           "lineage":namespace,"seed_components":{"index":index,"constructors":constructors},
           "packet_hash":packet_hash,"observation_hash":public_hash,"prediction_hash":prediction_hash,
           "truth_hash":private_hash,"public":public,"private":private,"arms":scored,
           "failures":[],"prediction_submitted_at":submitted_at,"scored_at":now()}
    write(path,row)
    return row


def summarize(card,rows):
    output = {}
    for condition in DESIGNS[card]["conditions"]:
        selected = [row for row in rows if row["condition"] == condition["id"]]
        if not selected:
            raise ValueError("missing registered condition")
        output[condition["id"]] = {
            "contrasts":[paired_summary(selected,estimand) for estimand in estimands(card)],
            "arms":{name:{key:sum(row["arms"][name]["outcomes"][key] for row in selected)/len(selected)
                          for key in ["future_log_score","success","legal"]}
                    for name in selected[0]["arms"]},
            "mean_route_collision_class_size":sum(len(row["private"]["equivalence_classes"]) for row in selected)/len(selected)}
    return {"card_id":card,"evidence_scope":rows[0]["evidence_scope"],"conditions":output,
            "n_maker_packets":len(rows),"instrument_state":"valid",
            "independent_reaggregation":"pending","historical_target":"acquired repertoire class, never a unique program string",
            "direct_table_rival":"same finite generative model and evidence; prediction without explicit personal reconstruction"}
