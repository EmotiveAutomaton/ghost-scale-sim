"""Finite job eligibility, preserved failure bounds and file-bound closeout.

No scientific estimator, criterion or seed is changed here. A completed worker
does not establish confirmation or any of the four required closeout proofs.
"""
from pathlib import Path
from .records import read, write, digest, file_digest, now
from .runtime import campaign, remaining_seconds
from .completion_guard import assess, bound_receipt, PROOFS

STAGES = {"preflight", "pilot", "discovery", "transfer", "confirmation", "close", "resume"}
PROOF_FIELDS = {"raw_archive": "verified_complete_accessible",
    "independent_aggregates": "full_aggregate_regeneration",
    "scientific_replay": "whole_unit_replay", "documentary_write_through": "write_through_verified"}


def validate_plan(plan):
    jobs = plan["jobs"]
    ids = [row["job_id"] for row in jobs]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("empty or duplicate finite job inventory")
    by_id = {row["job_id"]: row for row in jobs}
    for row in jobs:
        if row["stage"] not in STAGES-{"preflight", "resume"}:
            raise ValueError("unknown scientific/closeout job stage")
        if not row.get("contribution") or row.get("unit_cap", 0) < 1:
            raise ValueError("job requires a finite cap and expected contribution")
        if any(name not in by_id or name == row["job_id"] for name in row["dependencies"]):
            raise ValueError("missing or self-referential job dependency")
        for field in ["completion", "admission"]:
            path = Path(row[field])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("job evidence path escapes campaign")
    seen = set()
    def visit(name, pending):
        if name in pending:
            raise ValueError("cyclic job eligibility")
        if name not in seen:
            for dependency in by_id[name]["dependencies"]:
                visit(dependency, pending|{name})
            seen.add(name)
    for name in ids:
        visit(name, set())
    return by_id


def freeze_plan(root, plan):
    validate_plan(plan)
    accepted = campaign(root)
    identity = {"commission_sha256": accepted["commission_sha256"],
        "accepted_at": accepted["accepted_at"], "deadline": accepted["deadline"], "plan": plan}
    record = {"identity": identity, "sha256": digest(identity)}
    write(root/"operations/QUEUE_PLAN.json", record)
    return record


def failures(root):
    return [read(path) for path in sorted((root/"operations/failures").glob("*.json"))]


def record_failure(root, job_id, error, *, root_cause, family, kind="worker"):
    if kind not in {"worker", "instrument"} or not root_cause or not family:
        raise ValueError("failure needs an explicit kind, family and root-cause identity")
    previous = failures(root)
    repeated = sum(row["job_id"] == job_id and row["root_cause"] == root_cause and row["kind"] == kind for row in previous)+1
    row = {"job_id": job_id, "kind": kind, "family": family, "root_cause": root_cause,
        "error": str(error), "recorded_at": now(), "same_root_failures": repeated,
        "execution_state": "quarantined" if kind == "worker" and repeated >= 3 else "failed",
        "scientific_sources_changed": False, "accepted_clock": campaign(root)["accepted_at"]}
    path = root/"operations/failures"/f"{len(previous)+1:04d}.json"
    write(path, row)
    return row


def repair_state(root, family):
    plans = [read(path) for path in (root/"repairs").glob("*/PLAN.json")]
    plans = [row for row in plans if row.get("family") == family]
    used = len(plans)
    causes = {row["root_cause"] for row in failures(root) if row["kind"] == "instrument" and row["family"] == family}
    causes.update(row.get("root_cause", row.get("cause", row["repair_id"])) for row in plans)
    substantive = len(causes)
    return {"family": family, "repairs_used": used, "remaining_repairs": max(0, 1-used),
        "state": "closed" if substantive >= 2 or used > 1 else "one bounded repair available" if used == 0 else "repair exhausted",
        "rule": "a second substantive defect closes this instrument family; no source rewrite or reserve replacement is authorized by this status"}


def queue_state(root, plan):
    jobs = validate_plan(plan)
    history = failures(root)
    output = {}
    for name, job in jobs.items():
        completion = root/job["completion"]
        admission = root/job["admission"]
        if any(row["job_id"] == name and row["execution_state"] == "quarantined" for row in history):
            state = "quarantined"
        elif completion.exists():
            receipt = read(completion)
            if receipt.get("execution_state") == "completed" and receipt.get("instrument_state") in {"valid", "not_applicable"}:
                state = "completed"
            elif receipt.get("execution_state") in {"failed", "blocked", "quarantined"}:
                state = receipt["execution_state"]
            else:
                state = "unresolved completion"
        elif not admission.exists() or read(admission).get("instrument_state") != "valid":
            state = "awaiting admission"
        else:
            state = "admitted"
        output[name] = {"job_id": name, "stage": job["stage"], "state": state,
            "unit_cap": job["unit_cap"], "contribution": job["contribution"], "dependencies": job["dependencies"]}
    resolved = set()
    def resolve(name):
        if name in resolved:
            return
        row = output[name]
        for dependency in row["dependencies"]:
            resolve(dependency)
        missing = [dep for dep in row["dependencies"] if output[dep]["state"] != "completed"]
        row["unresolved_dependencies"] = missing
        if row["state"] == "admitted":
            row["state"] = "eligible" if not missing else "blocked by dependencies"
        elif row["state"] == "completed" and missing:
            row["state"] = "invalid dependency completion"
        resolved.add(name)
    for name in output:
        resolve(name)
    return {"jobs": list(output.values()), "eligible": [name for name,row in output.items() if row["state"] == "eligible"],
        "campaign_complete": False, "rule": "an empty eligible list never establishes scientific closeout"}


def dispatchable(root, job, forecast):
    if forecast.get("kind") != "measured forecast" or forecast.get("measured_seconds", 0) <= 0 or not forecast.get("source_sha256"):
        raise ValueError("job forecast lacks measured work and source identity")
    source = (root/forecast["source_path"]).resolve()
    if not source.is_relative_to(root.resolve()) or file_digest(source) != forecast["source_sha256"]:
        raise ValueError("forecast measured-work source changed")
    if forecast.get("conservative_seconds", 0) <= 0 or forecast.get("closeout_reserve_seconds", -1) < 0:
        raise ValueError("job forecast lacks finite work and closeout allowances")
    remaining = remaining_seconds(root)
    fits = forecast["conservative_seconds"]+forecast["closeout_reserve_seconds"] < remaining
    return {"job_id": job["job_id"], "eligible_within_deadline": fits,
        "remaining_seconds": remaining, "forecast": forecast, "accepted_clock_changed": False}


def verify_closeout(root, inputs):
    required = [row["card_id"] for row in read(root/"COMMISSION_MANIFEST.json")["cards"]]
    cards = []
    for original in inputs["cards"]:
        row = dict(original)
        evidence = row.get("evidence", [])
        checked = []
        for item in evidence:
            checked.append(bound_receipt(root, item["path"], item["sha256"]))
        row["evidence_verified"] = bool(checked)
        if row.get("execution_state") == "completed" and any(item.get("instrument_state") == "failed" for item in checked):
            row["instrument_state"] = "failed"
        cards.append(row)
    proofs = {}
    for name in PROOFS:
        item = inputs.get("proofs", {}).get(name)
        if item is None:
            proofs[name] = {"verified": False}
            continue
        receipt = bound_receipt(root, item["path"], item["sha256"])
        proofs[name] = {"verified": receipt.get("execution_state") == "completed" and receipt.get("instrument_state") == "valid"
            and receipt.get(PROOF_FIELDS[name]) is True, "path": item["path"], "sha256": item["sha256"]}
    result = assess(required, cards, inputs["expansions"], proofs)
    return {**result, "execution_state": "completed" if result["campaign_closed"] else "blocked",
        "instrument_state": "valid", "proofs": proofs, "card_count": len(cards),
        "commission_sha256": campaign(root)["commission_sha256"],
        "deadline": campaign(root)["deadline"], "recorded_at": now()}
