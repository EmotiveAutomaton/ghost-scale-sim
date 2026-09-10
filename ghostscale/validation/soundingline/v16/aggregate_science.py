"""Independent closeout calculations over retained scientific and export records.

The primary study/reducer modules are not imported here. Frozen-design JSON is
the estimand authority; separately implemented physical/statistical auditors
supply calculations, not the reporting functions that made the summaries.
"""
import importlib
import json
import math
from collections import Counter
from pathlib import Path
from .records import read, digest, file_digest

PREFIX = "ghostscale.validation.soundingline.v16."
AUDITORS = {
    "K01": "craft", "K02": "reading", "K03": "purpose", "K04": "options",
    "K05": "attention", "P01": "reading", "P02": "reading", "P03": "reading",
    "P04": "mechanism", "O01": "behavior", "O02": "behavior", "O03": "behavior",
    "O04": "behavior", "S01": "behavior", "S02": "behavior", "S03": "dependency",
    "S04": "behavior", "S05": "behavior", "M01": "recognition",
    "M02": "selection", "M03": "audience", "M04": "multi_actor",
    "V01": "tradeoffs", "V02": "tradeoffs", "V03": "preference_probe",
    **{f"R{i:02d}": "inquiry_stable" for i in range(1, 6)}}


def registered_designs(value):
    """Find actual nested design objects, never infer registration from output."""
    found = {}
    if isinstance(value, dict):
        if "card_id" in value and "conditions" in value and any(key in value for key in ["estimands", "primary", "primary_estimands"]):
            found[value["card_id"]] = value
        else:
            for nested in value.values():
                found.update(registered_designs(nested))
    elif isinstance(value, list):
        for nested in value:
            found.update(registered_designs(nested))
    return found


def validate_layout(rows, summary, design, n_per_condition=None):
    conditions = {item["id"] for item in design["conditions"]}
    if set(summary["conditions"]) != conditions or {row["condition"] for row in rows} != conditions:
        raise ValueError("registered condition set differs from raw or summary")
    if len({row["unit_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate raw units")
    if n_per_condition is not None and Counter(row["condition"] for row in rows) != dict.fromkeys(conditions, n_per_condition):
        raise ValueError("registered per-condition denominator differs")
    fields = ["arm", "rival", "target", "units", "practical_bar"]
    if "estimands" in design:
        expected = [tuple(item[key] for key in fields) for item in design["estimands"]]
    elif "primary" in design:
        expected = [tuple(item) for item in design["primary"]]
    elif design["card_id"] == "K01":
        # K01's compact design names estimands defined in its frozen study source.
        # This independent transcription also checks against the expanded scout
        # registration during the same pass; it does not call that reducer.
        known = {"personal-minus-"+rival+"-success": ("personal", rival, "success", "success_fraction", .05)
                 for rival in ["pooled", "primitive"]}
        expected = [known[name] for name in design["primary_estimands"]]
    else:
        raise ValueError("no independently specified estimand definition")
    if design["card_id"] == "M03":
        # Frozen audience_study.py:97 includes this explicit secondary contrast
        # in addition to DESIGN.primary. Retain and reproduce it as secondary.
        expected.append(("audience-only", "none", "historical_core_log_score", "nats_per_event", .02))
    for condition, report in summary["conditions"].items():
        observed = report["contrasts"]
        if len({item["estimand"]["id"] for item in observed}) != len(observed):
            raise ValueError("duplicate contrast identity")
        actual = [tuple(item["estimand"][key] for key in fields) for item in observed]
        if Counter(actual) != Counter(expected):
            raise ValueError("registered contrast omitted or added")


def scientific(base, summary_path, design, n_per_condition=None, *, invalid_original=False):
    summary = read(summary_path)
    rows = [read(path) for path in sorted((base/"units").glob("*_points.json"))]
    validate_layout(rows, summary, design, n_per_condition)
    name = "inquiry" if invalid_original else AUDITORS[summary["card_id"]]
    module = importlib.import_module(PREFIX+"audit_"+name)
    checked = module.audit(base, summary)
    if checked["instrument_state"] != "valid":
        raise ValueError("independent numerical reproduction failed")
    return {"summary_sha256": file_digest(summary_path), "card_id": summary["card_id"],
            "calculation": checked, "scientific_instrument_state": "invalid preserved original" if invalid_original else "valid",
            "qualification": "Reproducing an invalid instrument's saved arithmetic does not validate that instrument." if invalid_original else None,
            "auditor_source_sha256": file_digest(Path(module.__file__)),
            "frozen_secondary_contrasts": ["audience-only minus none: historical core log score"] if summary["card_id"] == "M03" else [],
            "registered_conditions": len(design["conditions"])}


def native_fixture(base):
    reported = read(base/"AGGREGATE.json")
    expected = {"n": 0, "legal_count": 0, "success_count": 0,
                "ambiguous_history_count": 0, "future_log_score_sum": 0.0}
    rows = [read(path) for path in sorted((base/"units").glob("*_points.json"))]
    if not rows or len({row["unit_id"] for row in rows}) != len(rows):
        raise ValueError("empty or duplicate native fixture")
    for row in rows:
        prediction, truth, public = row["prediction"], row["truth"], row["public"]
        uid = row["unit_id"]
        for folder, key in [("predictions", "prediction_hash"), ("private", "truth_hash")]:
            if file_digest(base/folder/(uid+".json")) != row[key]:
                raise ValueError("native raw reference differs")
        saved = read(base/"predictions"/(uid+".json"))
        if saved["prediction"] != prediction or read(base/"private"/(uid+".json")) != truth or read(base/"public"/(uid+".json")) != public or digest(public) != row["observation_hash"]:
            raise ValueError("native embedded source differs")
        probabilities = prediction["future_probabilities"]
        if any(not math.isfinite(p) or p < 0 for p in probabilities) or abs(sum(probabilities)-1) > 1e-10:
            raise ValueError("native probability distribution differs")
        program = prediction["reconstruction"]["program"]
        valid = not prediction["reconstruction"].get("search_timeout", False)
        state = [0]*4
        for position, action in enumerate(program):
            if position >= 3 or type(action) is not int or not 0 <= action < 8:
                valid = False
                break
            state[action % 4] = int(action < 4)
        artifact = sum(value << position for position, value in enumerate(state))
        probability = prediction["future_probabilities"][truth["future_artifact"]]
        if row["outcomes"]["legal"] != valid or row["outcomes"]["success"] != (valid and artifact == public["final_artifact"]) or abs(row["outcomes"]["future_log_score"]-math.log(probability)) > 1e-10:
            raise ValueError("native primary score differs from physical reconstruction")
        expected["n"] += 1
        expected["legal_count"] += int(valid)
        expected["success_count"] += int(valid and artifact == public["final_artifact"])
        expected["ambiguous_history_count"] += int(len(truth["equivalence_classes"]) > 1)
        expected["future_log_score_sum"] += math.log(probability)
    for key, value in expected.items():
        if not math.isclose(value, reported[key], rel_tol=0, abs_tol=1e-12):
            raise ValueError("native fixture aggregate differs")
    return {"reproduced": expected, "scientific_sample": False}


def transfer(base):
    """Recount unique public/private/prediction joins and their saved CPU sum."""
    public = read(base/"PUBLIC_MANIFEST.json")
    commit = read(base/"PREDICTIONS_COMMITTED.json")
    report = read(base/"PUBLIC_REPORT.json")
    completion = read(base/"COMPLETION.json")
    if commit["manifest_sha256"] != file_digest(base/"PUBLIC_MANIFEST.json"):
        raise ValueError("transfer public manifest differs")
    observations, predictions, tasks, cases = {}, {}, set(), set()
    for relative, expected in public["observations"].items():
        if file_digest(base/relative) != expected:
            raise ValueError("transfer observation differs")
        row = read(base/relative)
        if row["task_id"] in observations:
            raise ValueError("duplicate transfer observation")
        observations[row["task_id"]] = row
    cpu = 0.0
    for relative, expected in commit["prediction_files"].items():
        if file_digest(base/relative) != expected:
            raise ValueError("transfer prediction differs")
        row = read(base/relative)
        task = Path(relative).stem
        predictions[task] = row
        cpu += row["runtime"]["reader_cpu_seconds"]
    for relative, expected in read(base/"private/EVALUATION_MANIFEST.json").items():
        if file_digest(base/relative) != expected:
            raise ValueError("transfer truth differs")
        row = read(base/relative)
        if row["case_id"] in cases or set(row["task_ids"]) != set(row["expected_predictions"]):
            raise ValueError("transfer case or task layout differs")
        cases.add(row["case_id"])
        for task, expected_prediction in row["expected_predictions"].items():
            if task in tasks or predictions[task]["result"] != expected_prediction:
                raise ValueError("transfer prediction join differs")
            if predictions[task]["observation_sha256"] != digest(observations[task]):
                raise ValueError("transfer prediction used another observation")
            tasks.add(task)
    if set(observations) != tasks or set(predictions) != tasks:
        raise ValueError("transfer orphan or missing task")
    for item in [public, report, completion]:
        if (item["n_cases"], item["n_tasks"]) != (len(cases), len(tasks)):
            raise ValueError("transfer reported denominator differs")
    if not math.isclose(cpu, commit["reader_cpu_seconds"], abs_tol=1e-10):
        raise ValueError("transfer CPU arithmetic differs")
    return {"n_cases": len(cases), "n_tasks": len(tasks), "saved_reader_cpu_seconds": cpu,
            "scope": "Independent recount and exact committed prediction join; platform measurements are retained, not regenerated clocks."}


def archive(base, *, required_recurrences=18):
    from .archive_reference import verify_case, verify_summary
    search = sorted((read(path) for path in (base/"units").glob("*_points.json")),
                    key=lambda row: (["fixed", "adaptive"].index(row["method"]), row["replicate"], row["step"]))
    if digest(search) != read(base/"SEARCH_FREEZE.json")["candidate_journal_hash"]:
        raise ValueError("archive search freeze differs")
    seen, candidates = {}, set()
    for journal in search:
        receipt_path = base/journal["candidate_receipt"]
        if file_digest(receipt_path) != journal["candidate_sha256"]:
            raise ValueError("archive journal source differs")
        receipt = read(receipt_path)
        source = receipt_path.parent.parent
        uid = receipt["case"]["unit_id"]
        rowpath = source/"units"/(uid+"_points.json")
        if file_digest(rowpath) != receipt["unit_sha256"] or uid in candidates:
            raise ValueError("archive missing or duplicate source unit")
        checked = verify_case(source, read(rowpath), receipt["case"])
        candidates.add(uid)
        group = seen.setdefault((journal["method"], journal["replicate"]), set())
        novel = sorted(set(checked["semantic_ids"])-group)
        if sorted(journal["new_cases"]) != novel:
            raise ValueError("archive novelty was not independently reproduced")
        group.update(novel)
        for key in ["physical_primitives", "separate_verification_primitives"]:
            if checked[key] != journal[key]:
                raise ValueError("archive journal cost differs")
    followup = []
    source = base/"private/followup"
    for path in sorted((source/"private").glob("*-archive-case.json")):
        receipt = read(path)
        uid = receipt["case"]["unit_id"]
        rowpath = source/"units"/(uid+"_points.json")
        if file_digest(rowpath) != receipt["unit_sha256"] or uid in candidates:
            raise ValueError("archive follow-up source differs")
        row = read(rowpath)
        verify_case(source, row, receipt["case"])
        candidates.add(uid)
        followup.append({"condition": row["condition"], "cases": receipt["case"]["cases"],
                         "index": row["seed_components"]["index"]})
    actual_raw = {path.stem.removesuffix("_points") for folder in ["search", "followup"]
                  for path in (base/"private"/folder).glob("**/units/*_points.json")}
    if actual_raw != candidates:
        raise ValueError("archive unaccounted candidate source")
    checked = verify_summary(search, followup, read(base/"AGGREGATE.json"), required_recurrences)
    completion = read(base/"COMPLETION.json")
    if (completion["candidate_maker_records"], completion["followup_maker_condition_records"]) != (len(search), len(followup)):
        raise ValueError("archive completion denominator differs")
    return checked
