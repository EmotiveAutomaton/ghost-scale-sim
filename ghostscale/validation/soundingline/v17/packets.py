"""Small immutable JSONL chunks, source locks, resume and descriptive reaggregation."""
from collections import defaultdict
import math
import os
from pathlib import Path
import random
import sys
import time

from ..v16.records import canonical, digest, file_digest, now, read, write
from ..v16.runtime import local_owner
from .craft import METHODS, REGIMES, PILOT_BUDGETS, evaluate_case, make_case
REPO = Path(__file__).resolve().parents[4]
SOURCE_FILES = [
    "runners/run_v17.py",
    "tests/test_v17_foundation.py",
    *["ghostscale/validation/soundingline/v17/" + n for n in
      ("__init__.py", "contracts.py", "programs.py", "craft.py", "packets.py", "validity.py")],
    *["ghostscale/validation/soundingline/v16/" + n for n in
      ("__init__.py", "records.py", "runtime.py", "campaign_ownership.py", "graphic_world.py")],
    "docs/versions/v17-adaptive-appreciation/CODING_PACKAGE.md",
]


def source_identity():
    return {name: file_digest(REPO / name) for name in SOURCE_FILES}


def specification(stage, budgets=None):
    fixtures = stage == "fixture"
    pilot = stage == "pilot"
    return {"schema": "v17.packet.1", "stage": stage,
            "namespace": "v17-a-" + stage + "-1",
            "constructors": 2 if fixtures else 4 if pilot else 16,
            "histories_per_constructor": 2 if fixtures else 4,
            "regimes": list(REGIMES), "methods": list(METHODS),
            "budgets": list(budgets or (PILOT_BUDGETS if pilot else (128, 512, 2048))),
            "memory_cap": 32, "cases_per_chunk": 4 if fixtures else 16,
            "claim_status": "discarded_development" if pilot or fixtures else "descriptive",
            "fresh_namespaces_reserved": ["v17-a-fresh-1", "v17-b-fresh-1", "v17-c-fresh-1", "v17-d-fresh-1"],
            "sampling_limit": "constructor permutations share one structural motif family; histories can duplicate",
            "scope": "A graphic maker-assisted baseline slice; not whole A or whole V17"}


def case_sequence(design):
    for c in range(design["constructors"]):
        for h in range(design["histories_per_constructor"]):
            for regime in design["regimes"]:
                yield make_case(design["namespace"], c, h, regime)


def chunk_payload(cases, design):
    return b"".join(canonical({"case": case, "rows": [{**row, "claim_status": design["claim_status"]}
                    for row in evaluate_case(case, design["budgets"], design["memory_cap"])]}) + b"\n"
                    for case in cases)


def write_bytes_once(path, payload):
    """Reuse atomic create-if-absent semantics; never truncate a retained chunk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def load_cases(root, index):
    import json
    for chunk in index["chunks"]:
        path = root / chunk["path"]
        if file_digest(path) != chunk["sha256"]:
            raise ValueError("retained chunk hash mismatch")
        values = [json.loads(line) for line in path.read_bytes().splitlines()]
        if len(values) != chunk["cases"]:
            raise ValueError("retained chunk count mismatch")
        yield from values


def aggregate(items, design):
    groups = defaultdict(list)
    identity_counts = {"case_ids": set(), "public_problems": set(), "private_constructions": set(),
                       "constructors": set(), "histories": set()}
    attempted_cases = 0
    private_units = {}
    for item in items:
        case = item["case"]
        attempted_cases += 1
        private_units[case["case_id"]] = case["private_construction_sha256"]
        for name, key in (("case_ids", "case_id"), ("public_problems", "public_problem_sha256"),
                          ("private_constructions", "private_construction_sha256"),
                          ("constructors", "constructor_id"), ("histories", "history_id")):
            identity_counts[name].add(case[key])
        for row in item["rows"]:
            groups[(row["regime"], row["budget"], row["method"])].append(row)
    table = []
    for (regime, budget, method), rows in sorted(groups.items()):
        cluster = defaultdict(list)
        for row in rows:
            cluster[row["constructor_id"]].append(float(row["task_success"]))
        means = [sum(values)/len(values) for _, values in sorted(cluster.items())]
        mean = sum(means)/len(means)
        rng = random.Random(int(digest([design["namespace"], regime, budget, method])[:16], 16))
        boot = sorted(sum(rng.choices(means, k=len(means)))/len(means) for _ in range(2000))
        table.append({"family": "A", "regime": regime, "budget": budget, "method": method,
                      "attempted": len(rows), "valid_interface": sum(not r["invalid_program"] for r in rows),
                      "unique_history_ids": len({r["history_id"] for r in rows}),
                      "unique_private_units": len({private_units[r["case_id"]] for r in rows}),
                      "constructor_clusters": len(cluster), "solve_rate": mean,
                      "descriptive_95_interval": [boot[49], boot[1949]],
                      "interval_method": "percentile bootstrap of constructor means, 2000 draws; few-cluster limitation",
                      "mean_cold_cost": sum(r["costs"]["cold_total"] for r in rows)/len(rows),
                      "mean_repeat_online_cost": sum(r["costs"]["repeat_online"] for r in rows)/len(rows),
                      "mean_storage_tokens": sum(r["costs"]["definition_storage"] for r in rows)/len(rows),
                      "missing_outputs": sum(r["missing_output"] for r in rows),
                      "evidence_tier": "maker_acquisition_and_supplied_target", "claim_status": design["claim_status"]})
    contrasts = []
    for regime in design["regimes"]:
        for budget in design["budgets"]:
            for left, right in ((METHODS[1], METHODS[0]), (METHODS[2], METHODS[0]), (METHODS[2], METHODS[1])):
                lrows = {r["history_id"]: r for r in groups[(regime, budget, left)]}
                rrows = {r["history_id"]: r for r in groups[(regime, budget, right)]}
                if set(lrows) != set(rrows):
                    raise ValueError("unpaired comparison")
                clusters = defaultdict(list)
                for key, row in lrows.items():
                    clusters[row["constructor_id"]].append(float(row["task_success"])-float(rrows[key]["task_success"]))
                differences = [sum(xs)/len(xs) for _, xs in sorted(clusters.items())]
                rng = random.Random(int(digest([design["namespace"], regime, budget, left, right])[:16], 16))
                boot = sorted(sum(rng.choices(differences, k=len(differences)))/len(differences) for _ in range(2000))
                contrasts.append({"regime": regime, "budget": budget, "left": left, "right": right,
                                  "paired_solve_rate_difference": sum(differences)/len(differences),
                                  "descriptive_95_interval": [boot[49], boot[1949]], "paired_histories": len(lrows),
                                  "constructor_clusters": len(clusters), "claim_status": design["claim_status"]})
    return {"schema": "v17.comparisons.1", "attempted_cases": attempted_cases, "contrasts": contrasts,
            "uniqueness": {key: len(value) for key, value in identity_counts.items()},
            "structural_families": 1, "comparison_table": table,
            "qualifications": ["Absolute cell permutations are not independent structural architectures.",
                               "Repeated history/order aliases do not add independent worlds.",
                               "These intervals are descriptive, not confirmation or equivalence.",
                               "Stitch, hybrid, assembly and observer comparisons have not run in this packet."]}


def run_packet(root, design, stop_after_chunks=None):
    root = Path(root)
    if root.exists() and not (root/"LOCK.json").exists() and any(root.iterdir()):
        raise ValueError("refusing nonempty output without a V17 lock")
    with local_owner(root):
        lock_path = root / "LOCK.json"
        source = source_identity()
        if lock_path.exists():
            lock = read(lock_path)
            if lock["design"] != design or lock["source_files"] != source:
                raise ValueError("frozen design or source changed; use a new packet, retain the old")
        else:
            lock = {"schema": "v17.lock.1", "design": design, "source_files": source, "frozen_at": now()}
            write(lock_path, lock)
        index_path = root/"INDEX.json"
        index = read(index_path) if index_path.exists() else {"schema": "v17.index.1", "lock_sha256": file_digest(lock_path), "chunks": []}
        if index["lock_sha256"] != file_digest(lock_path):
            raise ValueError("index lock mismatch")
        # Validate all retained material before calculating or extending it.
        list(load_cases(root, index))
        cases = list(case_sequence(design))
        size = design["cases_per_chunk"]
        begin, cpu_begin = time.perf_counter(), time.process_time()
        added = 0
        try:
            for chunk_index, start in enumerate(range(0, len(cases), size)):
                part = cases[start:start+size]
                relative = f"raw/chunk-{chunk_index:05d}_points.jsonl"
                path = root/relative
                if chunk_index < len(index["chunks"]):
                    record = index["chunks"][chunk_index]
                    if record["path"] != relative or record["case_ids"] != [c["case_id"] for c in part]:
                        raise ValueError("chunk identity/order mismatch")
                    continue
                payload = chunk_payload(part, design)
                if path.exists():
                    # A crash after atomic chunk creation, before index creation.
                    if path.read_bytes() != payload:
                        raise ValueError("unindexed chunk differs from deterministic replay")
                else:
                    write_bytes_once(path, payload)
                index["chunks"].append({"path": relative, "sha256": file_digest(path),
                                        "bytes": path.stat().st_size, "cases": len(part),
                                        "rows": len(part)*len(design["budgets"])*len(METHODS),
                                        "case_ids": [c["case_id"] for c in part]})
                write(index_path, index, immutable=False)
                added += 1
                write(root/"STATUS.json", {"schema": "v17.status.1", "pid": os.getpid(), "heartbeat": now(),
                                           "state": "running", "chunks": len(index["chunks"]), "cases": start+len(part)},
                      immutable=False)
                if stop_after_chunks is not None and added >= stop_after_chunks:
                    write(root/"STATUS.json", {"schema": "v17.status.1", "pid": os.getpid(), "heartbeat": now(),
                                               "state": "paused_at_chunk_boundary", "chunks": len(index["chunks"])},
                          immutable=False)
                    return None
            summary = aggregate(load_cases(root, index), design)
            write(root/"COMPARISONS.json", summary)
            completion = {"schema": "v17.completion.1", "state": "completed", "campaign_complete": False,
                          "lock_sha256": file_digest(lock_path), "index_sha256": file_digest(index_path),
                          "summary_sha256": file_digest(root/"COMPARISONS.json"),
                          "cases": len(cases), "rows": sum(c["rows"] for c in index["chunks"])}
            write(root/"COMPLETION.json", completion)
            write(root/"STATUS.json", {"schema": "v17.status.1", "pid": os.getpid(), "heartbeat": now(),
                                       "state": "completed", "campaign_complete": False}, immutable=False)
            return completion
        except Exception as exc:
            write(root/("FAILURE-" + str(time.time_ns()) + ".json"), {"recorded_at": now(), "error": type(exc).__name__,
                                                                     "message": str(exc), "retained_chunks": len(index["chunks"])})
            raise
        finally:
            write(root/("TIME-" + str(time.time_ns()) + ".json"),
                  {"wall_seconds": time.perf_counter()-begin, "process_cpu_seconds": time.process_time()-cpu_begin,
                   "new_chunks": added, "gpu_used": False, "workers": 1,
                   "scope": "this packet entry, including generation and analysis; not whole campaign"})
