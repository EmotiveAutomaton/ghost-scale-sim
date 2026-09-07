"""Audit saved V15 records without running science or changing the live runner.

Checks every executed coverage block, rather than just the first 64 definitions.
Separates record integrity from experimental and runtime failures. This is not a
fresh-seed replication or regeneration of per-card aggregates from raw rollouts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ghostscale.validation.soundingline.v15 import coverage as CV

REPO = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Audit timestamps must include their timezone")
    return parsed.astimezone(timezone.utc)


def check_block(record, index):
    """Check the actual cells and their arithmetic against the locked definition."""
    expected = CV.block(index)
    problems = []
    if record.get("block") != index or record.get("digest") != CV.block_digest(index):
        problems.append("block order or definition digest mismatch")
    if record.get("secondary") != expected["secondary"]:
        problems.append("secondary settings mismatch")
    cells = record.get("cells", [])
    if (record.get("n_ok") != CV.BLOCK_CELLS
            or record.get("n_cells") != CV.BLOCK_CELLS or len(cells) != CV.BLOCK_CELLS):
        problems.append("incomplete balance block")
    if [c.get("cell_id") for c in cells] != [c["cell_id"] for c in expected["cells"]]:
        problems.append("cell identity, duplication, or order mismatch")
    for actual, wanted in zip(cells, expected["cells"]):
        for key in ("block", "family", "kappa", "dose"):
            if actual.get(key) != wanted[key]:
                problems.append(f"{actual.get('cell_id')}: {key} mismatch")
        if actual.get("secondary") != expected["secondary"]:
            problems.append(f"{actual.get('cell_id')}: secondary settings mismatch")
        architectures = actual.get("by_architecture", {})
        if set(architectures) != set(CV.COVERAGE_ARCHITECTURES):
            problems.append(f"{actual.get('cell_id')}: incomplete architecture comparison")
            continue
        if actual.get("n_rows") != sum(a.get("n", 0) for a in architectures.values()):
            problems.append(f"{actual.get('cell_id')}: row count mismatch")
        for name, values in architectures.items():
            if (values.get("n", 0) <= 0 or not 0 <= values.get("accuracy", -1) <= 1
                    or not all(math.isfinite(values.get(k, float("nan"))) for k in
                               ("log_score", "accuracy", "likelihood_evaluations"))):
                problems.append(f"{actual.get('cell_id')}:{name}: invalid score or count")
        difference = architectures["joint_exact"]["log_score"] - architectures["independent"]["log_score"]
        reported = actual.get("joint_minus_independent")
        if reported is None or not math.isclose(difference, reported, rel_tol=1e-12, abs_tol=1e-12):
            problems.append(f"{actual.get('cell_id')}: comparison arithmetic mismatch")
    return problems


def waiting_receipt(checkpoints, registry, deadline, now):
    """Reconcile the completed fixed packet with the subsequent integrity boundary.

No estimate of utilization is invented. This applies only to an unamended packet
whose last scientific checkpoint is confirmation and whose integrity stage ran.
"""
    science = [r for r in checkpoints if r.get("kind") in {"card", "coverage_block", "confirmation"}]
    integrity = [r for r in checkpoints if r.get("kind") == "integrity"]
    packet = registry.get("packet", [])
    results = registry.get("results", [])
    eligible = (science and integrity and science[-1].get("kind") == "confirmation"
                and packet and len(results) == len(packet) and not registry.get("amendments"))
    if not eligible:
        return {"reconciled": False, "reason": "completed unamended confirmation and integrity checkpoints required"}
    start = utc(science[-1]["t"])
    end = min(utc(deadline["opened"]) + timedelta(hours=deadline["confirmation_end_hour"]),
              utc(integrity[0]["t"]), now)
    seconds = max(0.0, (end - start).total_seconds())
    return {"reconciled": True, "from_utc": start.isoformat(), "to_utc": end.isoformat(),
            "waiting_hours": seconds / 3600, "RUNTIME_FAILED": seconds > 0,
            "basis": "last confirmation checkpoint to hour 166; fixed packet exhausted, then scheduler wait loop",
            "limitation": "does not reconstruct earlier downtime or measured worker utilization"}


def audit(repo=REPO):
    root = repo / "results/v15"
    problems, failures = [], []
    entries = load(root / "COMPLETION.json")["entries"]
    lanes = {}
    for key, entry in entries.items():
        path = (repo / entry["verdict_path"]).resolve()
        if not path.is_relative_to(repo.resolve()) or not path.is_file():
            problems.append(f"{key}: missing or external verdict")
            continue
        if sha(path) != entry["verdict_sha256"]:
            problems.append(f"{key}: verdict hash mismatch")
        verdict = load(path)
        for field in ("state", "criterion_status"):
            if verdict.get(field) != entry.get(field):
                problems.append(f"{key}: {field} differs from ledger")
        lanes[entry["lane"]] = lanes.get(entry["lane"], 0) + 1
        if verdict.get("state") == "INSTRUMENT_FAILED":
            failures.append({"entry": key, "failed_gates": verdict.get("gates", {}).get("failed_names", [])})
    blocks_path = root / "coverage/blocks.jsonl"
    count = cells = 0
    chain = hashlib.sha256(b"v15-coverage").hexdigest()
    if blocks_path.exists():
        with blocks_path.open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                record = json.loads(line)
                problems.extend(f"coverage block {index}: {p}" for p in check_block(record, index))
                chain = hashlib.sha256((chain + CV.block_digest(index)).encode()).hexdigest()
                count += 1
                cells += len(record.get("cells", []))
    else:
        problems.append("local executed coverage stream unavailable")
    occupancy = load(root / "WORKER_OCCUPANCY.json")
    if count != occupancy.get("coverage_blocks_executed") or cells != occupancy.get("coverage_cells_executed"):
        problems.append("executed coverage counts differ from occupancy receipt")
    checkpoints_path = root / "CHECKPOINTS.jsonl"
    checkpoints = [json.loads(line) for line in checkpoints_path.read_text(encoding="utf-8").splitlines()]
    waiting = waiting_receipt(checkpoints, load(root / "CONFIRMATION_REGISTRY.json"),
                              load(root / "DEADLINE.json"), datetime.now(timezone.utc))
    return {"program": "v15", "checked_utc": datetime.now(timezone.utc).isoformat(),
            "record_integrity_ok": not problems, "record_problems": problems,
            "completion_entries_checked": len(entries), "lane_counts": lanes,
            "completion_ledger_sha256": sha(root / "COMPLETION.json"),
            "instrument_failures": failures,
            "coverage": {"blocks_verified": count, "cells_verified": cells,
                         "full_executed_definition_chain": chain,
                         "raw_stream_sha256": sha(blocks_path) if blocks_path.exists() else None,
                         "score_arithmetic_checked": True, "rollouts_regenerated": False},
            "runtime_reconciliation": waiting,
            "original_occupancy_claim": {k: occupancy.get(k) for k in
                                          ("RUNTIME_FAILED", "waited_for_deadline_hours", "occupancy_ratio")},
            "scientific_aggregate_regeneration": {"complete": False,
                "reason": "full per-card unit outputs were not retained by the scientific runner; verdict hashes and cell arithmetic are not independent regeneration"},
            "scope": "saved-record integrity and runtime reconciliation; no new scientific runs"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["record_integrity_ok"] else 1)
