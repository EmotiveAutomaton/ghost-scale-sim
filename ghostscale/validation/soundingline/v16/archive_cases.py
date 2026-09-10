"""Semantic case identities from actual acquired production and public predictions."""
from .records import read
from .selection import EXTENSION
from .recoding import close
from .graphic_reference import interpret
from .archive_design import CASE_IDS


def classify(public, private, predictions, release_only):
    aware = predictions["selection-aware"]
    naive = predictions["release-naive"]
    direct = predictions["direct-table"]
    style = private["maker"]["decoration"]["choice"]
    keys = ["acquired_core", "acquired_style", "audience", "future_raw_core", "future_raw_style", "future_release_style"]
    result = {"direct-equivalence": all(close(aware[key], direct[key]) for key in keys),
        "artifact-core-ambiguity": all(abs(value-0.5) <= 1e-10 for value in aware["acquired_core"]),
        "process-narrows-core": max(aware["acquired_core"]) >= 0.75,
        "selection-changes-raw-prediction": abs(aware["future_raw_style"][0]-naive["future_raw_style"][0]) >= 0.05,
        "naive-confident-error": naive["acquired_style"][style] < 0.2 and aware["acquired_style"][style] > 0.5,
        "rejected-work-corrects-account": any(batch["full_candidates"] is not None for batch in public["history"])
            and aware["acquired_style"][style]-release_only["acquired_style"][style] >= 0.05}
    if set(result) != set(CASE_IDS):
        raise ValueError("case target changed")
    return result


def case_record(base, row, reader):
    import copy
    uid = row["unit_id"]
    public = read(base/"public"/(uid+".json"))
    private = read(base/"private"/(uid+".json"))
    predictions = read(base/"predictions"/(uid+".json"))["arms"]
    release_only = copy.deepcopy(public)
    for batch in release_only["history"]:
        batch["full_candidates"] = None
    counterfactual = reader.request(EXTENSION, release_only, method="selection-aware")
    cases = classify(public, private, predictions, counterfactual)
    # Visible final cells admit both core orders; this independently executed
    # witness concerns historical order and does not infer a full maker program.
    example = private["batches"][0]["candidates"][private["batches"][0]["retained_indices"][0]]
    original = example["program"]
    changed = [original[1], original[0], *original[2:]]
    collision = {"original": original, "alternative": changed,
                 "original_execution": interpret(original), "alternative_execution": interpret(changed)}
    if collision["original_execution"]["artifact"] != collision["alternative_execution"]["artifact"]:
        raise ValueError("archive core collision did not physically realize")
    return {"cases": cases, "unit_id": uid, "condition": row["condition"],
        "source_packet": row["packet_hash"], "source_public_sha256": row["public_hash"],
        "source_predictions_sha256": row["prediction_hash"], "source_private_sha256": row["private_hash"],
        "release_only_request": {"kind": EXTENSION, "public": release_only,
            "options": {"method": "selection-aware"}, "result": counterfactual},
        "derived_ablation_scope": "same retained public history with rejected observations removed; no future answer supplied and no new prospective sample claimed",
        "physical_collision": collision, "verification_primitive_steps": 10,
        "sampling_scope": "multiple semantic cases from one maker remain one dependent case packet"}
