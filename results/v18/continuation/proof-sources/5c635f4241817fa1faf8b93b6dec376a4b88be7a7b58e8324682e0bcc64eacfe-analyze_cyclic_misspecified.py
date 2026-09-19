"""Full raw reconstruction of the candidate-family misspecification screen."""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys


root, output = map(Path, sys.argv[1:])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


plan = json.loads((root / "PLAN.json").read_bytes())
complete = json.loads((root / "COMPLETE.json").read_bytes())
assert plan["branch"] == "g2-cyclic-misspecified"
assert complete["plan_sha256"] == sha(root / "PLAN.json")
groups = defaultdict(lambda: defaultdict(list))
families = defaultdict(set)
bindings = {}
rows = 0
for name in complete["blocks"]:
    receipt = json.loads((root / "blocks" / f"{name}.json").read_bytes())
    path = root / "raw" / f"{name}_points.json.gz"
    bindings[name] = sha(path)
    block = json.loads(gzip.decompress(path.read_bytes()))
    assert bindings[name] == receipt["raw_sha256"]
    assert digest(block) == receipt["content_sha256"]
    assert block["plan_sha256"] == sha(root / "PLAN.json")
    for unit in block["units"]:
        case = unit["case"]
        public = case["public"]
        truth = case["private"]["true_world"]
        assert case["truth_excluded"] and truth not in public["models"]
        families[case["family"]].add(case["structural_unit"])
        evidence = {}
        for row in unit["rows"]:
            rows += 1
            count = row["requested_queries"]
            value = digest(
                [row["observation_record"], row["query_indices"], row["acquisition_operations"]]
            )
            assert count not in evidence or evidence[count] == value
            evidence[count] = value
            assert row["query_policy"] == "misspecification-action"
            assert row["truth_in_candidate_family"] is False
            assert row["candidate_family_supplied"]
            assert row["acquisition_operations"] <= row["costs"]["total_online"] <= row["budget"]
            if row["candidate_aware"] and row["evidence_inconsistent_with_candidate_family"]:
                assert row["misspecification_detected"]
                assert row["abstained_on_inconsistency"]
                assert row["program"] is None
                assert not row["unsafe_action_attempted_after_inconsistency"]
            if row["method"] == "forced-candidate-direct" and row["evidence_inconsistent_with_candidate_family"]:
                assert row["unsafe_action_attempted_after_inconsistency"] == (row["program"] is not None)
            groups[row["method"], count][case["structural_unit"]].append(row)

metrics = (
    "success",
    "invalid",
    "missing_output",
    "candidate_compatible_laws",
    "evidence_inconsistent_with_candidate_family",
    "misspecification_detected",
    "abstained_on_inconsistency",
    "unsafe_action_attempted_after_inconsistency",
    "acquisition_operations",
    "forecast_brier",
)
cells = []
for (method, count), units in sorted(groups.items()):
    assert len(units) == 192 and all(len(values) == 4 for values in units.values())
    cell = {
        "method": method,
        "requested_queries": count,
        "contexts": 192,
        "rows": 768,
    }
    metric_presence = {
        metric: {metric in row for nested in units.values() for row in nested}
        for metric in metrics
    }
    assert all(len(presence) == 1 for presence in metric_presence.values())
    selected_metrics = [metric for metric, presence in metric_presence.items() if True in presence]
    cell.update(
        {
            f"mean_{metric}": mean(
                mean(row[metric] for row in nested) for nested in units.values()
            )
            for metric in selected_metrics
        }
    )
    cell["metrics_not_applicable"] = [
        metric for metric, presence in metric_presence.items() if False in presence
    ]
    cell["mean_online_operations"] = mean(
        mean(row["costs"]["total_online"] for row in nested) for nested in units.values()
    )
    cells.append(cell)

assert rows == 9216
assert {key: len(value) for key, value in families.items()} == {
    "chain": 64,
    "fork": 64,
    "groups": 64,
}
result = {
    "schema": "v18.1.candidate-family-misspecification-diagnostic.1",
    "scope": (
        "descriptive constructed mechanism; deliberately truth-excluded supplied family; "
        "miniature architecture untested; no third primary"
    ),
    "plan_sha256": sha(root / "PLAN.json"),
    "rows_checked": rows,
    "distinct_context_units": 192,
    "new_structural_context_units": 192,
    "histories_per_context": 4,
    "family_context_units": {key: len(value) for key, value in families.items()},
    "cells": cells,
    "raw_bindings": bindings,
    "aggregation": (
        "equal structural-context means; four histories nested; methods share evidence "
        "within each query count"
    ),
    "limits": [
        "Misspecification is constructed by excluding evaluator truth from a supplied three-law family in one miniature assembly architecture.",
        "Query states are apparatus-provided and their physical preparation is not charged in this screen.",
        "Detection means the retained supplied compatible set becomes empty; it does not establish open-world or human anomaly detection.",
        "The forced-candidate method is a labeled unsafe negative control, not a serious equally informed rival.",
    ],
}
output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
print(json.dumps(cells, sort_keys=True))
