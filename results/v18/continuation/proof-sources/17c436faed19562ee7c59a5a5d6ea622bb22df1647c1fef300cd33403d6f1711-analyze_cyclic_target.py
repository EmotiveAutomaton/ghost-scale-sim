"""Outcome-independent aggregation for the V18.1 cyclic target-query screen."""
from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys


ROOT = Path(sys.argv[1])
OUTPUT = Path(sys.argv[2])


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


plan_path = ROOT / "PLAN.json"
plan = json.loads(plan_path.read_bytes())
complete = json.loads((ROOT / "COMPLETE.json").read_bytes())
plan_sha256 = file_digest(plan_path)
assert plan["branch"] == "g2-cyclic-target"
assert complete["plan_sha256"] == plan_sha256
assert len(complete["blocks"]) == 96

cell_units: dict[tuple, dict[str, list[dict]]] = defaultdict(
    lambda: defaultdict(list)
)
evidence_by_case: dict[tuple, tuple] = {}
families: dict[str, set[str]] = defaultdict(set)
rows_checked = 0
raw_bindings: dict[str, str] = {}

for block_name in complete["blocks"]:
    receipt_path = ROOT / "blocks" / f"{block_name}.json"
    raw_path = ROOT / "raw" / f"{block_name}_points.json.gz"
    receipt = json.loads(receipt_path.read_bytes())
    raw_sha256 = file_digest(raw_path)
    assert raw_sha256 == receipt["raw_sha256"]
    block = json.loads(gzip.decompress(raw_path.read_bytes()))
    assert digest(block) == receipt["content_sha256"]
    assert block["plan_sha256"] == plan_sha256
    assert receipt["rows"] == sum(len(unit["rows"]) for unit in block["units"])
    raw_bindings[block_name] = raw_sha256
    for unit in block["units"]:
        case = unit["case"]
        structural_unit = case["structural_unit"]
        family = case["family"]
        families[family].add(structural_unit)
        assert case["n"] == 7
        assert case["target_parts"] == [1, 2]
        assert case["coverage"]["candidate_laws"] == 4
        assert case["coverage"]["independent_reciprocal_cycles"] == 2
        for row in unit["rows"]:
            rows_checked += 1
            policy = row["query_policy"]
            requested = row["requested_queries"]
            method = row["method"]
            assert policy in ("fixed", "decision", "target-aware")
            assert requested in (1, 2)
            assert row["budget"] == 32768
            evidence_key = (case["case_id"], policy, requested)
            evidence_value = (
                digest(row["observation_record"]),
                tuple(row["query_indices"]),
                row["query_exhausted"],
                row["acquisition_operations"],
                row["query_compatible_laws"],
                row["query_isolates_truth"],
            )
            if evidence_key in evidence_by_case:
                assert evidence_by_case[evidence_key] == evidence_value
            else:
                evidence_by_case[evidence_key] = evidence_value
            metrics = {
                "success": row["success"],
                "missing_output": row["missing_output"],
                "query_exhausted": row["query_exhausted"],
                "acquired_queries": row["acquired_queries"],
                "acquisition_operations": row["acquisition_operations"],
                "query_compatible_laws": row["query_compatible_laws"],
                "query_isolates_truth": row["query_isolates_truth"],
                "online_operations": row["costs"]["total_online"],
            }
            if row.get("forecast_brier") is not None:
                metrics["forecast_brier"] = row["forecast_brier"]
            if row.get("dependency_recovered") is not None:
                metrics["dependency_recovered"] = row["dependency_recovered"]
            cell_units[(method, policy, requested)][structural_unit].append(metrics)

assert rows_checked == 18432
assert {family: len(units) for family, units in families.items()} == {
    "chain": 64,
    "fork": 64,
    "groups": 64,
}

cells = []
for (method, policy, requested), units in sorted(cell_units.items()):
    assert len(units) == 192
    assert all(len(records) == 4 for records in units.values())
    metric_names = tuple(next(iter(units.values()))[0])
    cell = {
        "method": method,
        "query_policy": policy,
        "requested_queries": requested,
        "distinct_context_units": len(units),
        "rows": sum(len(records) for records in units.values()),
    }
    for metric in metric_names:
        assert all(metric in row for records in units.values() for row in records)
        cell[f"mean_{metric}"] = mean(
            mean(row[metric] for row in records) for records in units.values()
        )
    cells.append(cell)


def get(method: str, policy: str, requested: int) -> dict:
    return next(
        cell
        for cell in cells
        if cell["method"] == method
        and cell["query_policy"] == policy
        and cell["requested_queries"] == requested
    )


target_two = get("dependencies", "target-aware", 2)
target_one = get("dependencies", "target-aware", 1)
fixed_two = get("dependencies", "fixed", 2)
decision_two = get("dependencies", "decision", 2)
target_direct = get("conditioned-direct", "target-aware", 2)
target_primitive = get("candidate-set-primitive", "target-aware", 2)

assert target_two["mean_query_isolates_truth"] == 1
assert target_two["mean_success"] == target_direct["mean_success"] == 1
assert target_two["mean_acquisition_operations"] == 43
assert target_one["mean_query_compatible_laws"] == 2
assert target_one["mean_query_isolates_truth"] == 0
assert target_one["mean_success"] == 0
assert fixed_two["mean_query_compatible_laws"] == 4
assert fixed_two["mean_success"] == 0
assert decision_two["mean_query_exhausted"] == 2 / 3
assert decision_two["mean_success"] == 1 / 3

result = {
    "schema": "v18.1.cyclic-target-diagnostic.1",
    "scope": (
        "descriptive fresh seven-part, two-target cyclic query screen; "
        "both final contrasts already frozen; no primary"
    ),
    "plan_sha256": plan_sha256,
    "rows_checked": rows_checked,
    "distinct_context_units": 192,
    "histories_per_context": 4,
    "family_context_units": {family: len(units) for family, units in families.items()},
    "cells": cells,
    "finding": {
        "target_aware_two_query_truth_isolation_fraction": target_two[
            "mean_query_isolates_truth"
        ],
        "target_aware_two_query_dependency_success_fraction": target_two[
            "mean_success"
        ],
        "target_aware_two_query_direct_success_fraction": target_direct[
            "mean_success"
        ],
        "target_aware_two_query_primitive_success_fraction": target_primitive[
            "mean_success"
        ],
        "target_aware_two_query_acquisition_operations": target_two[
            "mean_acquisition_operations"
        ],
        "target_aware_one_query_compatible_laws": target_one[
            "mean_query_compatible_laws"
        ],
        "target_aware_one_query_dependency_success_fraction": target_one[
            "mean_success"
        ],
        "fixed_two_query_compatible_laws": fixed_two[
            "mean_query_compatible_laws"
        ],
        "fixed_two_query_dependency_success_fraction": fixed_two[
            "mean_success"
        ],
        "decision_two_query_exhausted_fraction": decision_two[
            "mean_query_exhausted"
        ],
        "decision_two_query_dependency_success_fraction": decision_two[
            "mean_success"
        ],
    },
    "aggregation": (
        "equal structural-context means; four histories nested; methods share each "
        "case's evidence receipt"
    ),
    "raw_bindings": raw_bindings,
}

payload = canonical(result) + b"\n"
if OUTPUT.exists():
    assert OUTPUT.read_bytes() == payload
else:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(payload)
print(json.dumps(result["finding"], sort_keys=True))
