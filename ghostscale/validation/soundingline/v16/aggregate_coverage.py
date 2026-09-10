"""Coverage joins for independently calculated native and source-control ledgers."""
from pathlib import Path
from .records import read, file_digest, digest
from .completion_guard import bound_receipt
from .aggregate_science import AUDITORS, registered_designs

PHASES = {"development", "original_controls", "constructor", "constructor_controls", "boundary", "boundary_controls",
          "catalogue", "confirmation_native", "confirmation_primary", "confirmation_controls", "confirmation_power"}


def producer(root, repo, reference):
    receipt = bound_receipt(root, reference["path"], reference["sha256"])
    if receipt.get("execution_state") != "completed" or receipt.get("instrument_state") != "valid" or not receipt.get("sources"):
        raise ValueError("aggregate coverage requires a valid completed independent producer")
    for name, expected in receipt["sources"].items():
        path = (repo/name).resolve()
        if not path.is_relative_to(repo.resolve()) or file_digest(path) != expected:
            raise ValueError("independent aggregate producer source changed")
    return receipt


def require_phases(references):
    if set(references) != PHASES:
        raise ValueError("full aggregate coverage requires exactly all eleven declared calculation phases")


def scouts(root):
    work = []
    for packet in sorted((root/"packets").glob("*-scout-*.json")):
        spec = read(packet)["identity"]["design"]
        designs = registered_designs(spec)
        base = root/packet.stem
        for card, design in sorted(designs.items()):
            candidates = [path for path in [base/"SUMMARY.json", base/card/"SUMMARY.json"] if path.exists() and read(path).get("card_id") == card]
            if len(candidates) != 1:
                raise ValueError("registered scout has no unique scientific summary")
            work.append((candidates[0], design, spec.get("n_per_condition", 64), packet.stem == "inquiry-scout-1"))
    if not work:
        raise ValueError("native scout coverage is empty")
    return work


def native_reports(root, repo, reports, work):
    expected = {path.relative_to(root).as_posix() for path,_,_,_ in work}
    if set(reports) != expected or len(expected) != len(work):
        raise ValueError("independent aggregate coverage omitted or duplicated a registered summary")
    inventory, manifests, counts = {}, {}, []
    for path, design, n, invalid in work:
        name = path.relative_to(root).as_posix()
        report = reports[name]
        calculation = report["calculation"]
        expected_n = len(design["conditions"])*n
        if report["summary_sha256"] != file_digest(path) or report["card_id"] != design["card_id"] or report["registered_conditions"] != len(design["conditions"]):
            raise ValueError("independent aggregate summary identity or condition coverage changed")
        if calculation.get("execution_state") != "completed" or calculation.get("instrument_state") != "valid" or calculation.get("all_reported_aggregates_reproduced") is not True or calculation["n_raw_units"] != expected_n:
            raise ValueError("independent aggregate arithmetic or denominator is incomplete")
        state = "invalid preserved original" if invalid else "valid"
        if report["scientific_instrument_state"] != state or (invalid and not report.get("qualification")):
            raise ValueError("independent aggregate erased a preserved invalid instrument")
        auditor = "inquiry" if invalid else AUDITORS[design["card_id"]]
        if file_digest(repo/"ghostscale/validation/soundingline/v16"/("audit_"+auditor+".py")) != report["auditor_source_sha256"]:
            raise ValueError("independent native calculation source changed")
        manifest_path = path.parent/"RAW_MANIFEST.json"
        raw = read(manifest_path)["files"]
        completion = read(path.parent/"COMPLETION.json")
        if "raw_manifest_sha256" in completion and completion["raw_manifest_sha256"] != file_digest(manifest_path):
            raise ValueError("native raw manifest changed after its completed source packet")
        units = sum(Path(member).parts[0] == "units" and member.endswith("_points.json") for member in raw)
        if units != expected_n:
            raise ValueError("native raw manifest omits the independent calculation's denominator")
        for member, sha in raw.items():
            target = (path.parent/member).resolve()
            if not target.is_relative_to(path.parent.resolve()):
                raise ValueError("native raw manifest escapes its scientific packet")
            relative = target.relative_to(repo.resolve()).as_posix()
            if relative in inventory and inventory[relative] != sha:
                raise ValueError("independent calculations disagree on retained raw input bytes")
            inventory[relative] = sha
        manifests[manifest_path.relative_to(root).as_posix()] = file_digest(manifest_path)
        counts.append({"summary": name, "card_id": design["card_id"], "condition_records": expected_n,
            "conditions": len(design["conditions"]), "scientific_instrument_state": state})
    return {"summaries": counts, "condition_records": sum(row["condition_records"] for row in counts),
        "raw_manifests": manifests, "raw_inputs": inventory}


def source_controls(root, receipt, name, work):
    if receipt["control_packet"] != name:
        raise ValueError("independent source-control calculation belongs to another packet")
    expected = {design["card_id"]: (len(design["conditions"]), n*len(design["conditions"])) for _,design,n in work}
    cards = receipt["cards"]
    if len(cards) != len(expected) or {row["card_id"] for row in cards} != set(expected):
        raise ValueError("independent source-control coverage omitted or duplicated a native card")
    for row in cards:
        conditions, records = expected[row["card_id"]]
        if row["instrument_state"] != "valid" or row["condition_controls"] != conditions or row["source_condition_records"] != records or len(row["conditions"]) != conditions:
            raise ValueError("independent source-control condition denominator differs")
        destination = root/name/row["card_id"]
        if file_digest(destination/"COMPLETION.json") != row["control_completion_sha256"] or file_digest(destination/"source_inventory_points.json") != row["source_inventory_sha256"]:
            raise ValueError("independent source-control input identity changed")
        for checked in row["conditions"]:
            path = destination/"conditions"/digest(checked["condition"])[:16]/"RECEIPT.json"
            if file_digest(path) != checked["receipt_sha256"]:
                raise ValueError("independently checked source-control condition changed")
    if receipt["condition_controls"] != sum(row[0] for row in expected.values()) or receipt["source_condition_records"] != sum(row[1] for row in expected.values()):
        raise ValueError("independent full source-control denominator differs")
    return {"cards": len(cards), "conditions": receipt["condition_controls"], "condition_records": receipt["source_condition_records"]}
