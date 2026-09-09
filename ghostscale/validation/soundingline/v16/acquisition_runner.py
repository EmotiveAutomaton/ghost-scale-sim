"""Finite K01 packet execution and explicit independent-analysis admission."""
from pathlib import Path
from .runtime import PACKAGE, freeze, remaining_seconds
from .records import read, write, now, file_digest
from .craft import DESIGN
from .craft_gates import run as gates
from .study import unit, summarize, raw_manifest, ESTIMANDS
from .audit_craft import audit
from .admission import craft_admission


def execute(root: Path, heartbeat, *, resume=False):
    dependencies = craft_admission(root)
    packet_id = "k01-scout-1"
    lock_path = root / "packets" / f"{packet_id}.json"
    if resume and not lock_path.exists():
        raise ValueError("resume never prepares a discovery packet")
    design = {**DESIGN, "scope": "discovery", "n_per_condition": 64, "constructors": 8,
              "estimands": [estimator.record() for estimator in ESTIMANDS],
              "dependency_joins": dependencies}
    modules = ["world.py", "reference.py", "learning.py", "records.py", "craft.py",
               "estimands.py", "study.py", "craft_gates.py", "audit_craft.py",
               "acquisition_runner.py", "admission.py"]
    packet = freeze(root, packet_id, [PACKAGE / name for name in modules], design)
    output = root / packet_id
    gate_path = output / "GATES.json"
    receipt = read(gate_path) if gate_path.exists() else gates()
    write(gate_path, receipt)
    if receipt["instrument_state"] != "valid":
        raise ValueError("K01 acquisition instrument failed")
    rows = []
    for condition in DESIGN["conditions"]:
        for index in range(64):
            if remaining_seconds(root) <= 0:
                heartbeat(execution_state="checkpointed", reason="immutable elapsed ceiling",
                          completed_units=len(rows))
                return {"execution_state": "checkpointed", "completed_units": len(rows)}
            row = unit(output, condition, index, packet_hash=packet["packet_hash"],
                       namespace=DESIGN["seed_namespace"], constructors=8, evidence_scope="discovery")
            rows.append(row)
            heartbeat(packet_id=packet_id, card_id="K01", condition=condition["id"],
                      completed_units=len(rows), planned_units=192, last_unit=row["unit_id"])
    summary_path = output / "SUMMARY.json"
    summary = summarize(rows)
    receipt = audit(output, summary)
    summary["independent_reaggregation"] = "valid"
    write(output / "REAGGREGATION.json", receipt)
    write(output / "RAW_MANIFEST.json", raw_manifest(output))
    write(summary_path, summary)
    write(output / "COMPLETION.json", {
        "execution_state": "completed", "instrument_state": "valid",
        "criterion_state": "see each paired contrast", "evidence_scope": "discovery",
        "warrant": "DESCRIPTIVE ONLY", "pursuit": "OPENED",
        "packet_hash": packet["packet_hash"], "summary_sha256": file_digest(summary_path),
        "n_maker_packets": len(rows), "n_conditions": 3, "constructors_per_condition": 8,
        "expansion_state": "planned; assess the preregistered capability or boundary rule",
        "confirmation_state": "untested", "closeout_state": "incomplete"})
    return {"packet_id": packet_id, "execution_state": "completed",
            "instrument_state": "valid", "n_maker_packets": len(rows),
            "independent_reaggregation": "valid", "campaign_complete": False}
