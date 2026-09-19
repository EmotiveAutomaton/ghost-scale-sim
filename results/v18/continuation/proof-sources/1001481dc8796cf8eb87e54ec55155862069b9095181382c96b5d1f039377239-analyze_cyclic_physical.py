"""Full raw reconstruction of the charged physical query-preparation diagnostic."""
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
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


plan = json.loads((root / "PLAN.json").read_bytes())
complete = json.loads((root / "COMPLETE.json").read_bytes())
assert plan["branch"] == "g2-cyclic-physical"
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
    assert bindings[name] == receipt["raw_sha256"] and digest(block) == receipt["content_sha256"]
    assert block["plan_sha256"] == sha(root / "PLAN.json")
    for unit in block["units"]:
        case = unit["case"]
        public = case["public"]
        evidence = {}
        families[case["family"]].add(case["structural_unit"])
        for row in unit["rows"]:
            rows += 1
            key = row["requested_queries"]
            value = digest([
                row["observation_record"], row["physical_setup_paths"],
                row["acquisition_operations"],
            ])
            assert key not in evidence or evidence[key] == value
            evidence[key] = value
            assert row["query_policy"] == "target-action-physical"
            assert row["physical_setup_operations"] == sum(map(len, row["physical_setup_paths"]))
            assert row["acquisition_operations"] <= row["costs"]["total_online"] <= row["budget"]
            assert not row["setup_uses_evaluator_truth"] and row["candidate_family_supplied"]
            assert len(row["observation_record"][len(public["observations"]):]) == row["acquired_queries"]
            groups[row["method"], row["requested_queries"]][case["structural_unit"]].append(row)

cells = []
metrics = (
    "success", "query_exhausted", "query_compatible_laws", "query_isolates_truth",
    "acquisition_operations", "physical_setup_operations",
)
for (method, count), units in sorted(groups.items()):
    assert len(units) == 192 and all(len(values) == 4 for values in units.values())
    cell = {
        "method": method,
        "requested_queries": count,
        "contexts": 192,
        "rows": 768,
    }
    cell.update({
        f"mean_{metric}": mean(mean(row[metric] for row in nested) for nested in units.values())
        for metric in metrics
    })
    cell["mean_online_operations"] = mean(
        mean(row["costs"]["total_online"] for row in nested) for nested in units.values()
    )
    cells.append(cell)

assert rows == 6144
assert {key: len(value) for key, value in families.items()} == {
    "chain": 64, "fork": 64, "groups": 64,
}
result = {
    "schema": "v18.1.physical-query-preparation-diagnostic.1",
    "scope": "exposed descriptive constructed mechanism; queue-12 contexts reused; miniature architecture untested; no third primary",
    "plan_sha256": sha(root / "PLAN.json"),
    "rows_checked": rows,
    "distinct_context_units": 192,
    "new_structural_context_units": 0,
    "histories_per_context": 4,
    "family_context_units": {key: len(value) for key, value in families.items()},
    "cells": cells,
    "raw_bindings": bindings,
    "aggregation": "equal structural-context means; four histories nested; all methods receive identical evidence",
    "limits": [
        "Each query receives a fresh object in the public task initial state; provisioning/reset is not counted.",
        "The supplied four-law family and already exposed complete action menu remain apparatus assumptions.",
        "Robust setup planning, candidate simulation, physical setup actions and the observed action share the 32,768-operation envelope.",
        "The reused outcomes were exposed before freeze, so this diagnostic cannot revise either held final contrast.",
    ],
}
output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
print(json.dumps(cells, sort_keys=True))
