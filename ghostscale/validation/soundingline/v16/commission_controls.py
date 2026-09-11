"""Bounded supplemental execution of omitted commissioned source controls.

Original packets and outcomes remain frozen. These are known-answer controls,
not new scientific makers or a replacement confirmation family.
"""
from pathlib import Path
from .records import read, write, file_digest, digest, now
from .completion_guard import bound_receipt
from .runtime import REPO, freeze, remaining_seconds
from .record_integrity import source_locks
from .reader_process import ReaderProcess
from .expansion_runner import EXTENSIONS
from .consumer_frames import requests
from .packet_controls import ActualReader
from .attack_collisions import check_case as collision, FAMILIES as COLLISIONS
from .attack_misspecification import check_case as misspecification, FAMILIES as MISSPECIFICATIONS

PACKET = "commission-controls-1"
CONTROLS = {"X03": (collision, COLLISIONS), "X06": (misspecification, MISSPECIFICATIONS)}


def specification(root):
    manifest = {row["card_id"]: row for row in read(root/"COMMISSION_MANIFEST.json")["cards"]}
    cards = read(root/"study-products-1/NATIVE_CARDS.json")["cards"]
    gaps = []
    for card in cards:
        phase = card["final_source"].split("/")[0]
        if phase not in {"constructor-expansion-1", "boundary-expansion-1"}:
            continue
        cid = card["card_id"]
        control = phase.replace("expansion", "controls")+"/"+cid+"/COMPLETION.json"
        saved = read(root/control)
        missing = sorted(set(manifest[cid]["dependency_ids"])-set(saved["required_adversaries"]))
        if not missing:
            continue
        if saved["instrument_state"] != "valid" or set(missing)-set(CONTROLS):
            raise ValueError("supplement does not implement the missing commissioned control")
        conditions = []
        for item in saved["conditions"]:
            original = bound_receipt(root, item["receipt"], item["sha256"])
            path = card["final_source"]+"/units/"+original["source_unit_id"]+"_points.json"
            unit = bound_receipt(root, path, original["source_unit_sha256"])
            if unit["seed_components"]["index"] != 0 or unit["condition"] != item["condition"] or unit["card_id"] != cid:
                raise ValueError("supplement must retain original index-zero condition selection")
            conditions.append({"condition": item["condition"], "unit": {"path": path, "sha256": original["source_unit_sha256"]},
                "original_condition": {"path": item["receipt"], "sha256": item["sha256"]}})
        if len({row["condition"] for row in conditions}) != len(conditions):
            raise ValueError("duplicate original source conditions")
        gaps.append({"card_id": cid, "source": card["final_source"], "missing_adversaries": missing,
            "original_control": {"path": control, "sha256": file_digest(root/control)}, "conditions": conditions})
    if not gaps or len(gaps) > 2 or sum(len(row["conditions"]) for row in gaps) > 40:
        raise ValueError("supplement exceeds the observed two-card, forty-condition bound")
    return {"gaps": gaps, "commission_sha256": file_digest(root/"COMMISSION_MANIFEST.json"),
        "selection": "All affected original conditions, immutable index zero; no outcome-based selection",
        "scope": "Missing commissioned known-answer checks only; original science, controls and failed closeout remain retained"}


def execute(root):
    source_locks(root, REPO)
    spec = specification(root)
    files = set(read(root/"packets/boundary-controls-1.json")["identity"]["files"])
    files.update(["ghostscale/validation/soundingline/v16/commission_controls.py",
        "ghostscale/validation/soundingline/v16/commission_control_audit.py",
        "runners/control_v16_commission_gaps.py", "tests/test_v16_commission_controls.py"])
    packet = freeze(root, PACKET, [REPO/name for name in sorted(files)], spec)
    output = root/PACKET
    if (output/"COMPLETION.json").exists():
        raise ValueError("completed supplemental controls must not be rerun")
    records = []
    for gap in spec["gaps"]:
        with ReaderProcess(output/gap["card_id"]/"reader", extensions=EXTENSIONS) as raw:
            reader = ActualReader(raw)
            for item in gap["conditions"]:
                if remaining_seconds(root) < 600:
                    raise ValueError("supplement reached its immutable closeout reserve")
                path = output/gap["card_id"]/"conditions"/(digest(item["condition"])+"_points.json")
                unit = bound_receipt(root, item["unit"]["path"], item["unit"]["sha256"])
                identity = {"packet_hash": packet["packet_hash"], "card_id": gap["card_id"], **item}
                if path.exists():
                    saved = read(path)
                    if saved["identity"] != identity or saved["instrument_state"] != "valid":
                        raise ValueError("supplement resume source differs or prior condition failed")
                else:
                    baselines = []
                    for frame in requests(root/gap["source"], unit):
                        actual = reader.request(frame["kind"], frame["public"], **frame["options"])
                        baselines.append({**frame, "actual": actual})
                        if actual != frame["result"]:
                            write(path, {"identity": identity, "instrument_state": "failed", "baselines": baselines})
                            raise ValueError("supplement actual source prediction differs")
                    checks = {}
                    for attack in gap["missing_adversaries"]:
                        producer, families = CONTROLS[attack]
                        checks[attack] = producer(reader, families[gap["card_id"]])
                    valid = bool(baselines) and all(value["instrument_state"] == "valid" for value in checks.values())
                    saved = {"identity": identity, "execution_state": "completed", "instrument_state": "valid" if valid else "failed",
                        "baselines": baselines, "controls": checks, "completed_at": now(), "reader_pid": raw.child.pid}
                    write(path, saved)
                    if not valid:
                        raise ValueError("supplement known-answer control failed; evidence preserved")
                records.append({"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)})
        print({"supplemented_card": gap["card_id"], "conditions": len(gap["conditions"])}, flush=True)
    from .commission_control_audit import verify_records
    audit = verify_records(root, spec, records)
    write(output/"INDEPENDENT_AUDIT.json", audit)
    result = {"execution_state": "completed", "instrument_state": "valid", "completed_at": now(),
        "packet_hash": packet["packet_hash"], "specification": spec, "records": records,
        "independent_audit": {"path": PACKET+"/INDEPENDENT_AUDIT.json", "sha256": file_digest(output/"INDEPENDENT_AUDIT.json")},
        "commission_dependencies_verified": True, "condition_controls": len(records),
        "scope": "Retrospective supplemental controls, not an amendment of original outcomes or additional independent maker samples"}
    write(output/"COMPLETION.json", result)
    return result
