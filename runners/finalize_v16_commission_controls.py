"""Finalize already executed controls with a separately frozen serialization repair."""
from ghostscale.validation.soundingline.v16.runtime import REPO, freeze
from ghostscale.validation.soundingline.v16.records import read, write, file_digest, digest, now
from ghostscale.validation.soundingline.v16.record_integrity import source_locks
from ghostscale.validation.soundingline.v16.commission_control_audit_v2 import verify_records, verify


def main():
    root = REPO/"results/v16"
    output = root/"commission-controls-1"
    if (output/"COMPLETION.json").exists():
        raise ValueError("preserve completed supplemental finalization")
    source_locks(root, REPO)
    original = read(root/"packets/commission-controls-1.json")
    spec = original["identity"]["design"]
    files = set(original["identity"]["files"])
    files.update(["ghostscale/validation/soundingline/v16/commission_control_audit_v2.py",
        "runners/finalize_v16_commission_controls.py", "tests/test_v16_commission_control_serialization.py"])
    finalizer = freeze(root, "commission-controls-finalization-1", [REPO/name for name in sorted(files)],
        {"original_packet_hash": original["packet_hash"], "correction": "Compare canonical persisted JSON witnesses; retain original tuple/list comparison failure",
         "scientific_units_rerun": False, "known_answer_controls_rerun": False})
    refs = []
    for gap in spec["gaps"]:
        for item in gap["conditions"]:
            name = "commission-controls-1/"+gap["card_id"]+"/conditions/"+digest(item["condition"])+"_points.json"
            row = read(root/name)
            if row["identity"]["packet_hash"] != original["packet_hash"]:
                raise ValueError("retained control packet changed")
            refs.append({"path": name, "sha256": file_digest(root/name)})
    audit = verify_records(root, spec, refs)
    write(output/"INDEPENDENT_AUDIT.json", audit)
    result = {"execution_state": "completed", "instrument_state": "valid", "completed_at": now(),
        "packet_hash": original["packet_hash"], "finalizer_packet_hash": finalizer["packet_hash"],
        "specification": spec, "records": refs,
        "independent_audit": {"path": "commission-controls-1/INDEPENDENT_AUDIT.json", "sha256": file_digest(output/"INDEPENDENT_AUDIT.json")},
        "commission_dependencies_verified": True, "condition_controls": len(refs),
        "scope": "Retrospective supplemental controls; separately corrected serialization recount; no new scientific samples or repeated confirmation"}
    write(output/"COMPLETION.json", result)
    verify(root, {"path": "commission-controls-1/COMPLETION.json", "sha256": file_digest(output/"COMPLETION.json")})
    print({"commission_dependencies_verified": True, "condition_controls": len(refs), "recount": audit}, flush=True)


if __name__ == "__main__":
    main()
