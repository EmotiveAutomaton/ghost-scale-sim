"""Join the completed native calculation proof with the separately recounted controls."""
from pathlib import Path
from ghostscale.validation.soundingline.v16.runtime import REPO
from ghostscale.validation.soundingline.v16.records import read, write, file_digest, now
from ghostscale.validation.soundingline.v16.completion_guard import bound_receipt
from ghostscale.validation.soundingline.v16.commission_control_audit_v2 import verify
from ghostscale.validation.soundingline.v16.record_integrity import source_locks


def join(root, output, parent_ref, supplement_ref):
    if output.exists():
        raise ValueError("preserve existing aggregate supplement")
    parent = bound_receipt(root, parent_ref["path"], parent_ref["sha256"])
    if parent.get("execution_state") != "completed" or parent.get("instrument_state") != "valid" or parent.get("full_aggregate_regeneration") is not True:
        raise ValueError("aggregate supplement requires completed parent arithmetic")
    supplement = verify(root, supplement_ref)
    ref = parent["raw_input_baseline"]
    baseline = bound_receipt(root, ref["path"], ref["sha256"])
    files = dict(baseline["files"])
    for item in supplement["records"]:
        name = "results/v16/"+item["path"]
        if name in files and files[name] != item["sha256"]:
            raise ValueError("supplement conflicts with original raw baseline")
        files[name] = item["sha256"]
    write(output/"raw_inputs_points.json", {"files": files,
        "parent_baseline": ref, "supplemental_control_records": len(supplement["records"]),
        "scope": "Original independently regenerated inputs plus separately recounted supplemental control records; no original input omitted or replaced"})
    result = {**parent, "recorded_at": now(), "parent_aggregate_proof": parent_ref,
        "supplemental_commission_controls": supplement_ref,
        "supplemental_control_audit": supplement["independent_audit"],
        "supplemental_condition_controls": supplement["condition_controls"],
        "raw_input_baseline": {"path": (output/"raw_inputs_points.json").relative_to(root).as_posix(), "sha256": file_digest(output/"raw_inputs_points.json")},
        "supplement_producer_sha256": file_digest(Path(__file__)),
        "scope": "Eleven completed independent native/control calculation phases retained unchanged, joined with the actual forty-condition supplemental recount; shared known-answer definitions are not an independent physical model"}
    write(output/"RECEIPT.json", result)
    return result


def main():
    root = REPO/"results/v16"
    source_locks(root, REPO)
    reference = lambda name: {"path": name, "sha256": file_digest(root/name)}
    result = join(root, root/"aggregate-final-2", reference("aggregate-final-1/RECEIPT.json"), reference("commission-controls-1/COMPLETION.json"))
    print({"full_aggregate_regeneration": result["full_aggregate_regeneration"],
        "supplemental_condition_controls": result["supplemental_condition_controls"]}, flush=True)


if __name__ == "__main__":
    main()
