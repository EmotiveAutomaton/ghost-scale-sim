"""Independent raw aggregation for charged physical candidate-family misspecification."""
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
assert plan["branch"] == "g2-cyclic-misspecified-physical"
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
            value = digest([
                row["observation_record"], row["query_indices"],
                row["physical_setup_paths"], row["physical_setup_records"],
                row["acquisition_operations"],
            ])
            assert count not in evidence or evidence[count] == value
            evidence[count] = value
            assert row["query_policy"] == "misspecification-action-physical"
            assert row["truth_in_candidate_family"] is False
            assert row["candidate_family_supplied"] and not row["setup_uses_evaluator_truth"]
            assert row["acquisition_operations"] <= row["costs"]["total_online"] <= row["budget"]
            assert row["acquired_queries"] == sum(
                record["query_executed"] for record in row["physical_setup_records"]
            )
            assert row["physical_setup_failures"] == sum(
                not record["reached_query_state"] for record in row["physical_setup_records"]
            )
            if row["candidate_aware"] and row["evidence_inconsistent_with_candidate_family"]:
                assert row["misspecification_detected"]
                assert row["abstained_on_inconsistency"]
                assert row["program"] is None
                assert not row["unsafe_action_attempted_after_inconsistency"]
            if row["candidate_aware"] and not row["evidence_inconsistent_with_candidate_family"]:
                assert not row["misspecification_detected"]
                assert not row["abstained_on_inconsistency"]
            if (row["method"] == "forced-candidate-direct"
                    and row["evidence_inconsistent_with_candidate_family"]):
                assert row["unsafe_action_attempted_after_inconsistency"] == (row["program"] is not None)
            groups[case["family"], row["method"], count][case["structural_unit"]].append(row)

metrics = (
    "success",
    "invalid",
    "missing_output",
    "candidate_compatible_laws",
    "evidence_inconsistent_with_candidate_family",
    "misspecification_detected",
    "abstained_on_inconsistency",
    "unsafe_action_attempted_after_inconsistency",
    "acquired_queries",
    "physical_query_attempts",
    "query_exhausted",
    "acquisition_operations",
    "physical_setup_operations",
    "physical_setup_failures",
    "forecast_brier",
)
cells = []
for (family, method, count), units in sorted(groups.items()):
    assert len(units) == 64 and all(len(values) == 4 for values in units.values())
    cell = {
        "family": family,
        "method": method,
        "requested_queries": count,
        "contexts": 64,
        "rows": 256,
    }
    metric_presence = {
        metric: {metric in row for nested in units.values() for row in nested}
        for metric in metrics
    }
    assert all(len(presence) == 1 for presence in metric_presence.values())
    selected_metrics = [metric for metric, presence in metric_presence.items() if True in presence]
    cell.update({
        f"mean_{metric}": mean(
            mean(row[metric] for row in nested) for nested in units.values()
        )
        for metric in selected_metrics
    })
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
    "schema": "v18.1.physical-candidate-family-misspecification-diagnostic.1",
    "scope": (
        "exposed descriptive constructed mechanism; deliberately truth-excluded supplied family; "
        "charged candidate-relative physical preparation; miniature architecture untested; no third primary"
    ),
    "plan_sha256": sha(root / "PLAN.json"),
    "rows_checked": rows,
    "distinct_context_units": 192,
    "new_structural_context_units": 0,
    "histories_per_context": 4,
    "family_context_units": {key: len(value) for key, value in families.items()},
    "cells": cells,
    "raw_bindings": bindings,
    "aggregation": (
        "equal structural-context means within topology; four histories nested; methods share evidence "
        "within each requested-query count"
    ),
    "limits": [
        "The exact queue-14 truth-excluded contexts and relevant outcomes were exposed before this diagnostic was frozen; no new support was added.",
        "Misspecification remains constructed inside a supplied three-law family in one miniature assembly architecture.",
        "Every attempted query receives a fresh task-initial object; provisioning and reset remain outside the count.",
        "The shared 32,768-operation envelope includes candidate filtering, setup search, candidate simulation, physical setup outcomes, intended probes and downstream task execution.",
        "Detection means retained supplied support becomes empty; surviving wrong support or budget exhaustion is not relabeled as detection.",
    ],
}
output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
print(json.dumps(cells, sort_keys=True))
