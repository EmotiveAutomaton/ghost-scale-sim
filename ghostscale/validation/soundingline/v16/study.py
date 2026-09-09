"""Immutable, resumable acquisition discovery; predictions precede evaluator reveal."""
from __future__ import annotations
from pathlib import Path
from .records import canonical, digest, read, write, now, file_digest
from .craft import prepare_unit, construct_public, DESIGN
from .reference import interpret
from .estimands import Estimand, paired_summary


ESTIMANDS = [Estimand(f"personal-minus-{rival}-success", "success", "personal", rival,
                      "success_fraction", 0.05, "personal success minus rival success")
             for rival in ["pooled", "primitive"]]


def unit(root, condition, index, *, packet_hash, namespace, constructors, evidence_scope):
    uid = digest([namespace, condition["id"], index])[:24]
    path = root / "units" / f"{uid}_points.json"
    if path.exists():
        saved = read(path)
        if saved["packet_hash"] != packet_hash:
            raise ValueError("unit implementation lock differs")
        return saved
    generated = prepare_unit(condition, index, namespace=namespace, constructors=constructors,
                             evidence_scope=evidence_scope)
    public = generated.pop("public")
    private = generated.pop("private")
    observation_hash = write(root / "public" / f"{uid}.json", public)
    predictions = construct_public(canonical(public))
    prediction_path = root / "predictions" / f"{uid}.json"
    submitted_at = read(prediction_path)["submitted_at"] if prediction_path.exists() else now()
    prediction_hash = write(prediction_path, {"submitted_at": submitted_at,
                                             "observation_hash": observation_hash, "arms": predictions})
    truth_hash = write(root / "private" / f"{uid}.json", private)
    arms = {}
    for name, result in predictions.items():
        executions = [interpret(submission["program"]) for submission in result["submissions"]]
        legal = [execution["legal"] and not submission["search_timeout"]
                 for execution, submission in zip(executions, result["submissions"])]
        success = [valid and execution["artifact"] == target
                   for valid, execution, target in zip(legal, executions, public["targets"])]
        arms[name] = {**result, "executions": executions,
                      "outcomes": {"success": sum(success)/len(success), "legal": sum(legal)/len(legal),
                                   "search_cost": sum(item["search_primitives"] for item in result["submissions"])/len(success)},
                      "task_success": success, "task_legal": legal}
    row = {**generated, "packet_hash": packet_hash, "public": public, "private": private,
           "observation_hash": observation_hash, "prediction_hash": prediction_hash,
           "truth_hash": truth_hash, "arms": arms, "failures": [],
           "prediction_submitted_at": submitted_at, "scored_at": now()}
    write(path, row)
    return row


def summarize(rows):
    conditions = sorted({row["condition"] for row in rows}, key=lambda name: int(name.split("-")[-1]))
    summaries = {}
    for condition in conditions:
        subset = [row for row in rows if row["condition"] == condition]
        summaries[condition] = {
            "contrasts": [paired_summary(subset, estimand) for estimand in ESTIMANDS],
            "arms": {arm: {"success_mean": sum(row["arms"][arm]["outcomes"]["success"] for row in subset)/len(subset),
                           "legal_mean": sum(row["arms"][arm]["outcomes"]["legal"] for row in subset)/len(subset),
                           "search_primitives_mean": sum(row["arms"][arm]["outcomes"]["search_cost"] for row in subset)/len(subset),
                           "training_primitives_mean": sum(row["arms"][arm]["costs"]["training_primitives"] for row in subset)/len(subset),
                           "definition_cost_mean": sum(row["arms"][arm]["costs"]["library_definition"] for row in subset)/len(subset)}
                     for arm in ["personal", "pooled", "primitive"]}}
    return {"card_id": "K01", "conditions": summaries, "n_maker_packets": len(rows),
            "instrument_state": "valid", "evidence_scope": rows[0]["evidence_scope"],
            "historical_inference": "not measured by acquisition construction",
            "independent_reaggregation": "pending"}


def raw_manifest(root):
    return {"schema_version": "v16.raw.1",
            "files": {str(path.relative_to(root)).replace("\\", "/"): file_digest(path)
                      for folder in ["units", "public", "private", "predictions"]
                      for path in sorted((root / folder).glob("*.json"))},
            "archive_location": "this packet directory", "retention": "retain through final verified archival handoff",
            "units_policy": "immutable; failures retained; no excluded unsuccessful attempts"}
