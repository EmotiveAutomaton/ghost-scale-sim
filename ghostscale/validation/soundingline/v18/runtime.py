"""Finite immutable blocks; no closed campaign is prepared or resumed."""
from datetime import datetime, timedelta, timezone
import gzip
import os
from pathlib import Path
import platform
import sys
import time
import uuid

from ..v16.records import canonical, digest, file_digest, now, read, write
from ..v16.runtime import local_owner
from .study import make_case, evaluate

REPO = Path(__file__).resolve().parents[4]
SOURCE_FILES = [
    "ghostscale/__init__.py", "ghostscale/validation/__init__.py",
    "ghostscale/validation/soundingline/__init__.py",
    *["ghostscale/validation/soundingline/v16/" + s for s in
      ("__init__.py", "records.py", "runtime.py", "campaign_ownership.py", "world.py", "learning.py",
       "craft.py", "purpose_craft.py", "attention_craft.py")],
    *["ghostscale/validation/soundingline/v18/" + s for s in
      ("__init__.py", "study.py", "runtime.py", "verify.py")],
    "runners/run_v18.py", "runners/report_v18.py", "runners/launch_background.py",
    "tests/test_v18.py", "tests/test_background_launch.py",
    "docs/versions/v18-selective-acquisition/CODING_PACKAGE.md",
    "docs/versions/v18-selective-acquisition/README.md",
    "docs/versions/v18-selective-acquisition/PRERUN_CLARIFICATION.md",
]


def sources():
    return {p: file_digest(REPO / p) for p in SOURCE_FILES}


def freeze(root, *, namespace, constructors, histories, started_at, admission=None, development=False):
    root = Path(root)
    if constructors < 1 or histories < 1 or (not development and (constructors, histories) != (64, 4)):
        raise ValueError("scientific sample must be the registered 64 x 4")
    if development and (constructors > 8 or histories > 2):
        raise ValueError("development cap exceeded")
    if not development and (not admission or admission.get("passed") is not True or admission.get("sources") != sources()):
        raise ValueError("science requires passing source-bound admission")
    start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    if start.tzinfo is None:
        raise ValueError("start needs a timezone")
    sunday = datetime(2026, 9, 20, 15, tzinfo=timezone.utc)
    deadline = start + timedelta(hours=48) if development else min(start + timedelta(hours=48), sunday)
    cutoff = deadline - timedelta(hours=2)
    plan = {"schema": "v18.plan.1", "namespace": namespace, "constructors": list(range(constructors)),
            "histories": list(range(histories)), "block_size": 8, "core_budget": 32, "extension_budget": 128,
            "seed_components": [[namespace, c, h, constructors] for c in range(constructors) for h in range(histories)],
            "started_at": started_at, "admission_cutoff": cutoff.isoformat(), "delivery_deadline": deadline.isoformat(),
            "cpu_ceiling_seconds": 18 * 3600, "sources": sources(),
            "claim_status": "discarded_development" if development else "descriptive_constructed_mechanism",
            "admission": admission, "environment": {"python": sys.version, "platform": platform.platform(),
                                                       "scientific_dependencies": "Python standard library only"},
            "operator": "Codex in the commissioning conversation", "child_cpu_seconds": 0,
            "process_contract": "one CPU worker, no scientific child processes, single-thread environment"}
    write(root / "PLAN.json", plan)
    return plan


def validate(root):
    plan = read(root / "PLAN.json")
    if plan["sources"] != sources():
        raise ValueError("frozen source changed")
    return plan


def load_block(root, name):
    path = root / "raw" / (name + "_points.json.gz")
    receipt = read(root / "blocks" / (name + ".json"))
    if file_digest(path) != receipt["raw_sha256"]:
        raise ValueError("raw block corruption")
    block = __import__("json").loads(gzip.decompress(path.read_bytes()))
    if digest(block) != receipt["content_sha256"]:
        raise ValueError("block content mismatch")
    return block, receipt


def save_block(root, name, block, wall, cpu):
    raw = gzip.compress(canonical(block), mtime=0)
    path = root / "raw" / (name + "_points.json.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("orphan block differs; preserve and investigate")
    else:
        temporary = path.with_suffix(".tmp")
        with temporary.open("xb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path); temporary.unlink()
    receipt = {"name": name, "raw_sha256": file_digest(path), "content_sha256": digest(block),
               "cases": len(block["units"]), "rows": sum(len(u["rows"]) for u in block["units"]),
               "completed_at": now(), "wall_seconds": wall, "worker_cpu_seconds": cpu, "child_cpu_seconds": 0}
    write(root / "blocks" / (name + ".json"), receipt)


def spent_cpu(root):
    return sum(read(p)["worker_cpu_seconds"] for p in (root / "attempts").glob("*.json"))


def extension_decision(root, plan):
    path = root / "EXTENSION.json"
    if path.exists():
        return read(path)
    core = [read(p) for p in sorted((root / "blocks").glob("core-*.json"))]
    remaining = (datetime.fromisoformat(plan["admission_cutoff"]) - datetime.now(timezone.utc)).total_seconds()
    # Four times measured core time, plus one minute, reserves substantial margin.
    # It is a cost-only rule and never consults successes, signs or intervals.
    estimate_wall = 4 * sum(r["wall_seconds"] for r in core) + 60
    estimate_cpu = 4 * sum(r["worker_cpu_seconds"] for r in core) + 60
    decision = {"admitted": remaining > estimate_wall and spent_cpu(root) + estimate_cpu < plan["cpu_ceiling_seconds"],
                "basis": "four times core wall/CPU plus 60 seconds; no outcome read", "written_at": now(),
                "estimated_seconds": estimate_wall, "estimated_cpu_seconds": estimate_cpu,
                "remaining_seconds": remaining, "core_receipts": {r["name"]: r["content_sha256"] for r in core}}
    write(path, decision)
    return decision


def run(root, *, stop_after_blocks=None):
    root = Path(root).resolve()
    with local_owner(root):
        plan = validate(root)
        if (root / "RUN_COMPLETE.json").exists():
            for name in read(root / "RUN_COMPLETE.json")["blocks"]:
                load_block(root, name)
            return read(root / "RUN_COMPLETE.json")
        started, cpu_start = time.monotonic(), time.process_time()
        attempt = uuid.uuid4().hex
        cpu_before = spent_cpu(root)
        completed, new = [], 0
        status = {"schema": "v18.status.1", "pid": os.getpid(), "parent_pid": os.getppid(), "started_at": now(),
                  "python": sys.executable, "source_root": str(REPO), "plan_sha256": file_digest(root / "PLAN.json"),
                  "state": "running", "blocks": completed}
        def emit(**changes):
            status.update(changes); status["heartbeat"] = now()
            write(root / "STATUS.json", status, immutable=False)
            write(root / "attempts" / (attempt + ".json"),
                  {"pid": os.getpid(), "started_at": status["started_at"], "updated_at": now(),
                   "wall_seconds": time.monotonic() - started, "worker_cpu_seconds": time.process_time() - cpu_start,
                   "child_cpu_seconds": 0, "state": status["state"]}, immutable=False)
        emit()
        try:
            for phase, budget in (("core", 32), ("extension", 128)):
                if phase == "extension" and not extension_decision(root, plan)["admitted"]:
                    break
                for block_index, start in enumerate(range(0, len(plan["constructors"]), plan["block_size"])):
                    name = f"{phase}-{block_index:03d}"
                    if (root / "blocks" / (name + ".json")).exists():
                        load_block(root, name); completed.append(name); continue
                    if stop_after_blocks is not None and new >= stop_after_blocks:
                        emit(state="checkpointed"); return status
                    if datetime.now(timezone.utc) >= datetime.fromisoformat(plan["admission_cutoff"]) or cpu_before + time.process_time() - cpu_start >= plan["cpu_ceiling_seconds"]:
                        emit(state="resource_cutoff"); return status
                    emit(active_block=name)
                    block_wall, block_cpu = time.monotonic(), time.process_time()
                    if phase == "extension":
                        core, _ = load_block(root, f"core-{block_index:03d}")
                        cases = [u["case"] for u in core["units"]]
                    else:
                        cases = [make_case(plan["namespace"], c, h, len(plan["constructors"]))
                                 for c in plan["constructors"][start:start + plan["block_size"]] for h in plan["histories"]]
                    units = []
                    for case in cases:
                        units.append({"case": case, "rows": evaluate(case, budget)})
                        emit(active_case=case["case_id"])
                    block = {"plan_sha256": file_digest(root / "PLAN.json"), "name": name,
                             "budget": budget, "units": units}
                    save_block(root, name, block, time.monotonic() - block_wall, time.process_time() - block_cpu)
                    completed.append(name); new += 1; emit()
            result = {"execution_state": "completed", "instrument_state": "pending_independent_verification",
                      "criterion_state": "not_applicable", "claim_status": plan["claim_status"],
                      "plan_sha256": file_digest(root / "PLAN.json"), "blocks": completed, "completed_at": now()}
            write(root / "RUN_COMPLETE.json", result); emit(state="completed")
            return result
        except BaseException as exc:
            emit(state="failed", error=f"{type(exc).__name__}: {exc}")
            write(root / "failures" / (attempt + ".json"), status)
            raise
        finally:
            emit()
