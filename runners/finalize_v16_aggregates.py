"""Join every independent calculation phase to the final frozen scientific inventory."""
import argparse
from pathlib import Path
import os
import time
from ghostscale.validation.soundingline.v16.runtime import REPO
from ghostscale.validation.soundingline.v16.records import read, write, now, file_digest
from ghostscale.validation.soundingline.v16.record_integrity import source_locks
from ghostscale.validation.soundingline.v16.aggregate_coverage import require_phases, producer, scouts, native_reports, source_controls
from ghostscale.validation.soundingline.v16.aggregate_science import native_fixture, transfer, archive
from ghostscale.validation.soundingline.v16.aggregate_controls import run as original_controls
from runners.reaggregate_v16_packet import inventory


def run(root, output, input_path):
    if output.exists() or not output.resolve().is_relative_to(root.resolve()) or output.resolve().is_relative_to((root/"closeout").resolve()):
        raise ValueError("aggregate proof requires a new retained campaign directory included in the final raw snapshot")
    started = time.perf_counter()
    references = read(input_path)["phases"]
    require_phases(references)
    receipts = {name: producer(root, REPO, reference) for name,reference in references.items()}
    checks = source_locks(root, REPO)
    completed = read(root/"expansion-closure/COMPLETION.json")
    if completed.get("execution_state") != "completed" or completed.get("instrument_state") != "valid":
        raise ValueError("final calculations require the exhausted finite expansion ladder")
    native, raw_inputs = {}, {}
    def add_native(name, reports, work):
        result = native_reports(root, REPO, reports, work)
        for path, sha in result.pop("raw_inputs").items():
            if path in raw_inputs and raw_inputs[path] != sha:
                raise ValueError("independent calculation phases disagree on raw inputs")
            raw_inputs[path] = sha
        native[name] = result
        print({"joined_native_calculations": name, "summaries": len(result["summaries"]), "condition_records": result["condition_records"]}, flush=True)
    add_native("development", receipts["development"]["scientific_summaries"], scouts(root))
    profile = root/"expansion-setup/profile/campaign"
    _, _, profile_work = inventory(profile, "expansion-fixture-profile")
    add_native("profile_fixtures", receipts["development"]["profile_fixtures"], [(p,d,n,False) for p,d,n in profile_work])
    jobs = {"constructor": "constructor-expansion-1", "boundary": "boundary-expansion-1", "confirmation_native": "confirmation-1"}
    work_by_phase = {}
    for phase, packet in jobs.items():
        packet_path, completion_path, work = inventory(root, packet)
        report = receipts[phase]
        if report["packet_id"] != packet or report["packet_sha256"] != file_digest(packet_path) or report["completion_sha256"] != file_digest(completion_path):
            raise ValueError("independent native phase completion/packet identity changed")
        add_native(phase, report["scientific_summaries"], [(p,d,n,False) for p,d,n in work])
        if native[phase]["condition_records"] != report["raw_records"]:
            raise ValueError("independent native phase total denominator differs")
        work_by_phase[phase] = work
    control_counts = {}
    for phase, name, source in [("constructor_controls", "constructor-controls-1", "constructor"),
                                ("boundary_controls", "boundary-controls-1", "boundary"),
                                ("confirmation_controls", "confirmation-controls", "confirmation_native")]:
        control_counts[phase] = source_controls(root, receipts[phase], name, work_by_phase[source])
    # These original compact producers did not carry a complete input checksum
    # ledger. Recount them in this final process instead of trusting old status.
    controls = original_controls(root, lambda name,value: print({"final_original_control_recount": name}, flush=True))
    if controls != receipts["original_controls"]["results"]:
        raise ValueError("original independent control calculation changed")
    native_check = native_fixture(root/"native-fixture-1")
    transfers = {name: transfer(root/name) for name in ["transfer-fixture-1", "transfer-fixture-2"]}
    search = archive(root/"archive-search-1")
    development = receipts["development"]
    if native_check != development["native_fixture"] or transfers != development["transfer_recounts"] or search != development["archive_comparison"]:
        raise ValueError("independently recounted native fixture, B01 or B02 changed")
    catalogue = receipts["catalogue"]
    if catalogue["selection_sha256"] != file_digest(root/"case-catalogue-2/SELECTION.json") or catalogue["source_completion_sha256"] != file_digest(root/"case-catalogue-2/COMPLETION.json"):
        raise ValueError("independently checked descriptive catalogue changed")
    claims = read(root/"confirmation-selection/CLAIMS.json")["selected"]
    primary, power = receipts["confirmation_primary"], receipts["confirmation_power"]
    if primary["primary_count"] != len(claims) or power["claim_count"] != len(claims) or len(power["claims"]) != len(claims) or len(claims) > 3 or set(primary["claims"]) != {claim["claim_id"] for claim in claims} or {claim["claim_id"] for claim in power["claims"]} != set(primary["claims"]):
        raise ValueError("independent confirmation primary/power coverage differs")
    if primary["multiplicity_sha256"] != file_digest(root/"confirmation/MULTIPLICITY.json") or power["claims_sha256"] != file_digest(root/"confirmation-selection/CLAIMS.json"):
        raise ValueError("independent frozen selection or multiplicity input changed")
    for claim in claims:
        if primary["claims"][claim["claim_id"]]["primary_sha256"] != file_digest(root/"confirmation"/claim["card_id"]/"PRIMARY.json"):
            raise ValueError("independently calculated confirmation primary changed")
    raw_path = output/"raw_inputs_points.json"
    write(raw_path, {"files": raw_inputs, "scope": "Original raw-manifest SHA-256 values of every independently regenerated native summary; final raw archive must match each current file against this exact baseline"})
    result = {"execution_state": "completed", "instrument_state": "valid", "recorded_at": now(), "process_id": os.getpid(),
        "full_aggregate_regeneration": True, "source_checks": checks, "phases": references, "native_calculations": native,
        "source_control_calculations": control_counts, "original_control_packets": len(controls),
        "native_fixture": native_check, "transfer_recounts": transfers, "archive_comparison": search,
        "catalogue": {key:catalogue[key] for key in ["source_cards", "bounded_candidates", "selected_cases", "found_lenses", "missing_lenses_in_bounded_search"]},
        "confirmation_primaries": len(claims), "raw_input_baseline": {"path": raw_path.relative_to(root).as_posix(), "sha256": file_digest(raw_path)},
        "input_snapshot_scope": "Scientific raw manifest identities are frozen here; final archive verification also requires every current raw file to match this baseline",
        "qualification": "The original invalid inquiry instrument remains invalid despite reproduced arithmetic. Fixture records, discovery phases, selected examples and confirmation histories are never pooled into one scientific sample.",
        "platform_scope": "Measured CPU/wall/memory values and saved arithmetic are retained; no reconstruction of historical clocks or continuous occupancy is claimed",
        "campaign_complete": False, "wall_seconds": time.perf_counter()-started,
        "sources": {path.relative_to(REPO).as_posix(): file_digest(path) for path in [Path(__file__).resolve(), REPO/"ghostscale/validation/soundingline/v16/aggregate_coverage.py"]}}
    write(output/"RECEIPT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=REPO/"results/v16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("existing aggregate proof attempt is retained")
    try:
        result = run(args.root.resolve(), args.output.resolve(), args.inputs.resolve())
    except Exception as error:
        if args.output.resolve().is_relative_to(args.root.resolve()):
            write(args.output/"FAILURE.json", {"execution_state": "failed", "instrument_state": "unresolved", "recorded_at": now(), "error": repr(error)})
        raise
    print({"execution_state": "completed", "full_aggregate_regeneration": result["full_aggregate_regeneration"]})


if __name__ == "__main__":
    main()
