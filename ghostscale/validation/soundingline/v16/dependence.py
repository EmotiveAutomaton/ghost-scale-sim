"""X07 immutable source identity, nested sampling and retained/rejected accounting."""
from collections import Counter
from copy import deepcopy
import math
from .records import digest, read, file_digest
from .record_integrity import independent_units
from .estimands import Estimand, paired_summary


def source_bound_sample(rows, admitted):
    """Validate exact retained source rows before interpreting their sampling IDs.

This catches a copied history even if all its displayed identity fields are
renamed: it has no matching admitted source digest. It is not a claim to detect
adversarially falsified provenance outside the frozen archive trust boundary.
"""
    if not rows:
        raise ValueError("empty source sample")
    for row in rows:
        if admitted.get(row["unit_id"]) != digest(row):
            raise ValueError("unit differs from its admitted immutable source")
    return independent_units(rows)


def duplicate_attacks(row):
    fields = ("unit_id", "lineage", "card_id", "condition", "constructor_id", "maker_history_id", "seed_components")
    original = {key: deepcopy(row[key]) for key in fields}
    results = []
    for attack in ["renamed-rendering", "renamed-history", "renamed-seed", "entire-identity-forged"]:
        copied = deepcopy(original)
        copied["unit_id"] = "duplicate-"+row["unit_id"]
        if attack in {"renamed-history", "entire-identity-forged"}:
            copied["maker_history_id"] = "renamed-history"
        if attack in {"renamed-seed", "entire-identity-forged"}:
            copied["seed_components"]["index"] += 1000000
        if attack == "entire-identity-forged":
            copied["lineage"] += "-forged"
            checker = lambda: source_bound_sample([original, copied], {original["unit_id"]: digest(original)})
        else:
            checker = lambda: independent_units([original, copied])
        error = None
        try:
            checker()
        except ValueError as caught:
            error = str(caught)
        results.append({"attack": attack, "original_sampling_identity": original,
            "copied_sampling_identity": copied, "actual_rejection": error, "rejected": error is not None})
    return {"instrument_state": "valid" if all(item["rejected"] for item in results) else "failed", "attacks": results,
            "scope": "source hashes plus sampling identities; identical artifacts from truly fresh histories are permitted"}


def partition(batch, expected_count):
    candidates = batch["candidates"]
    if len(candidates) != expected_count:
        raise ValueError("produced candidates discarded or duplicated")
    kept = batch.get("retained_indices", [batch["retained_index"]] if "retained_index" in batch else None)
    if kept is None or not kept or len(set(kept)) != len(kept) or any(type(index) is not int or not 0 <= index < len(candidates) for index in kept):
        raise ValueError("retained indices are not a valid production partition")
    rejected = [index for index in range(len(candidates)) if index not in kept]
    return {"produced": len(candidates), "retained": len(kept), "rejected": len(rejected),
            "retained_indices": kept, "rejected_indices": rejected,
            "candidate_digests": [digest(item) for item in candidates],
            "sampling_scope": "one production batch inside one maker packet; candidates are not independent maker units"}


def batch_inventory(value, path=""):
    records = []
    if isinstance(value, dict):
        if "candidates" in value and ("retained_index" in value or "retained_indices" in value):
            records.append({"path": path, **partition(value, len(value["candidates"]))})
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                records.extend(batch_inventory(item, path+"/"+key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            records.extend(batch_inventory(item, path+"/"+str(index)))
    return records


def unit_batch_inventory(root, row):
    """Follow all separately retained private unit files, including original work."""
    mapping = read(root/"RAW_MANIFEST.json")["files"]
    inventory = []
    for path in sorted((root/"private").glob(row["unit_id"]+"*.json")):
        relative = "private/"+path.name
        if mapping.get(relative) != file_digest(path):
            raise ValueError("private production archive differs from its raw manifest")
        inventory.extend(batch_inventory(read(path), relative))
    return inventory


def cluster_fixture(*, constructors=8, makers_per_constructor=8):
    rows = []
    for constructor in range(constructors):
        for maker in range(makers_per_constructor):
            index = constructor*makers_per_constructor+maker
            positive = constructor < constructors//2
            rows.append({"unit_id": f"known-{index}", "lineage": "known-shared-variation", "card_id": "fixture",
                "condition": "shared-constructors", "constructor_id": f"constructor-{constructor}",
                "maker_history_id": f"independently-acquired-{index}", "seed_components": {"index": index},
                "evidence_scope": "fixture", "nested_episodes": 32,
                "arms": {"a": {"outcomes": {"success": float(positive)}}, "b": {"outcomes": {"success": float(not positive)}}}})
    return rows


def hierarchical_control():
    rows = cluster_fixture()
    spec = Estimand("constructor-variation", "success", "a", "b", "success_fraction", 0.05, "paired maker success difference")
    valid = paired_summary(rows, spec, seed=991, replicates=999)
    identity = source_bound_sample(rows, {row["unit_id"]: digest(row) for row in rows})
    # Intentionally wrong benchmark: count the 32 nested episodes as independent
    # after discarding the constructor and maker hierarchy.
    flat_n = sum(row["nested_episodes"] for row in rows)
    flat_half_width = 1.96/math.sqrt(flat_n)
    checks = {"paired_null_mean_is_zero": valid["mean"] == 0,
        "makers_not_episodes_are_the_denominator": valid["n_makers"] == identity["n_retained_unit_records"] == 64,
        "independent_constructor_count_is_eight": valid["n_constructors"] == 8,
        "shared_variation_keeps_interval_wide": valid["interval_95"][1]-valid["interval_95"][0] > 1.0,
        "intentionally_flat_reducer_is_overprecise": 2*flat_half_width < 0.1,
        "nested_false_precision_does_not_inherit_valid_criterion": valid["criterion_state"] == "inconclusive"}
    return {"checks": checks, "valid_summary": valid,
        "intentionally_broken_summary": {"n": flat_n, "interval_95": [-flat_half_width, flat_half_width],
            "fault": "constructor and maker dependence discarded; not an admissible inferential result"},
        "instrument_state": "valid" if all(checks.values()) else "failed"}


def sampling_groups(rows):
    report = independent_units(rows)
    contexts = {}
    for row in rows:
        key = (row["lineage"], row["card_id"], row["condition"])
        item = contexts.setdefault(key, {"indices": [], "constructors": Counter(), "source_unit_ids": []})
        item["indices"].append(row["seed_components"]["index"])
        item["constructors"][row["constructor_id"]] += 1
        item["source_unit_ids"].append(row["unit_id"])
    return {"retained_units": report["n_retained_unit_records"], "conditions": [
        {"lineage": lineage, "card_id": card, "condition": condition,
         "makers": len(item["indices"]), "constructors": len(item["constructors"]),
         "makers_per_constructor": dict(item["constructors"]), "seed_indices": sorted(item["indices"]),
         "unit_ids_sha256": digest(sorted(item["source_unit_ids"]))}
        for (lineage, card, condition), item in sorted(contexts.items())],
        "scope": "separate per-condition constructor/maker groups; sharing across conditions cannot be ignored in a cross-condition interaction"}
