"""Recalculate immutable completed cards while later frozen cards finish.

No unfinished card is opened. This read-only analysis never owns campaign status
and emits a completed phase receipt only after the full native packet completes.
"""
import argparse
import os
from pathlib import Path
import time
from ghostscale.validation.soundingline.v16.runtime import REPO, campaign
from ghostscale.validation.soundingline.v16.records import read, write, file_digest, now
from ghostscale.validation.soundingline.v16.record_integrity import source_locks
from ghostscale.validation.soundingline.v16.aggregate_science import scientific, registered_designs
from runners.reaggregate_v16_packet import inventory


def completed_card(root, name, packet, entry):
    base = root/name/entry["card_id"]
    path = base/"COMPLETION.json"
    if not path.exists():
        return None
    completed = read(path)
    if completed.get("execution_state") != "completed":
        return None
    if completed.get("instrument_state") != "valid" or completed["packet_hash"] != packet["packet_hash"]:
        raise ValueError("incremental calculation source is invalid or belongs to another packet")
    summary = base/"AGGREGATE.json"
    manifest = base/"RAW_MANIFEST.json"
    if file_digest(summary) != completed["aggregate_sha256"] or file_digest(manifest) != completed["raw_manifest_sha256"]:
        raise ValueError("completed native summary or raw manifest changed")
    return {item.relative_to(root).as_posix(): file_digest(item) for item in [path, summary, manifest]}


def cached(root, output, card, expected):
    path = output/"items"/card/"RECEIPT.json"
    if not path.exists():
        return None
    receipt = read(path)
    report_path = path.parent/"CALCULATION.json"
    if receipt.get("execution_state") != "completed" or receipt.get("instrument_state") != "valid" or receipt["source_hashes"] != expected or file_digest(report_path) != receipt["calculation_sha256"]:
        raise ValueError("retained independent calculation or its native source changed")
    for name, sha in expected.items():
        if file_digest(root/name) != sha:
            raise ValueError("cached calculation source bytes changed")
    return read(report_path)


def run(root, output, name, *, resume=False):
    accepted = campaign(root)
    known = name == "expansion-fixture-profile" and accepted["campaign_id"].startswith("fixture-")
    if name != "boundary-expansion-1" and not known:
        raise ValueError("incremental calculation is bounded to final discovery or the known profile fixture")
    if output.exists() != resume:
        raise ValueError("use a new calculation directory or explicitly resume its preserved attempt")
    began, cpu = time.perf_counter(), time.process_time()
    checks = source_locks(root, REPO)
    packet_path = root/"packets"/(name+".json")
    packet = read(packet_path)
    entries = packet["identity"]["design"]["cards"]
    designs = registered_designs(packet["identity"]["design"])
    if not entries or len(entries) != len(designs) or {entry["card_id"] for entry in entries} != set(designs):
        raise ValueError("incremental calculation omitted or duplicated a frozen scientific card")
    source_paths = [Path(__file__).resolve(), REPO/"ghostscale/validation/soundingline/v16/aggregate_science.py", REPO/"runners/reaggregate_v16_packet.py"]
    sources = {path.relative_to(REPO).as_posix(): file_digest(path) for path in source_paths}
    plan = {"packet_id": name, "packet_sha256": file_digest(packet_path), "accepted_at": accepted["accepted_at"],
        "deadline": accepted["deadline"], "sources": sources, "cards": entries,
        "scope": "All original frozen cards in original order, each only after its valid native completion; no unfinished outcomes or new scientific samples"}
    write(output/"PLAN.json", plan)
    reports = {}
    waiting = None
    for entry in entries:
        card = entry["card_id"]
        bound = completed_card(root, name, packet, entry)
        if bound is None:
            waiting = card
            break
        base = root/name/card
        label = (base/"AGGREGATE.json").relative_to(root).as_posix()
        result = cached(root, output, card, bound)
        if result is None:
            card_began, card_cpu = time.perf_counter(), time.process_time()
            result = scientific(base, base/"AGGREGATE.json", entry["design"], entry["n_per_condition"])
            if completed_card(root, name, packet, entry) != bound:
                raise ValueError("completed card source changed during independent calculation")
            destination = output/"items"/card
            write(destination/"CALCULATION.json", result)
            write(destination/"RECEIPT.json", {"execution_state": "completed", "instrument_state": "valid",
                "recorded_at": now(), "process_id": os.getpid(), "source_hashes": bound,
                "calculation_sha256": file_digest(destination/"CALCULATION.json"),
                "wall_seconds": time.perf_counter()-card_began, "cpu_seconds": time.process_time()-card_cpu})
        reports[label] = result
        print({"independently_recalculated": label, "raw_records": result["calculation"]["n_raw_units"]}, flush=True)
    if not waiting and (output/"RECEIPT.json").exists():
        _, completion_path, work = inventory(root, name)
        prior = read(output/"RECEIPT.json")
        if prior["scientific_summaries"] != reports or prior["completion_sha256"] != file_digest(completion_path):
            raise ValueError("completed independent phase changed before resume")
        return prior
    attempt = {"recorded_at": now(), "process_id": os.getpid(), "checked_cards": len(reports), "waiting_card": waiting,
        "wall_seconds": time.perf_counter()-began, "cpu_seconds": time.process_time()-cpu,
        "execution_state": "checkpointed" if waiting or not (root/name/"COMPLETION.json").exists() else "completed",
        "instrument_state": "valid", "full_aggregate_regeneration": False, "campaign_complete": False}
    write(output/"attempts"/f"{len(list((output/'attempts').glob('*.json')))+1:03d}.json", attempt)
    if attempt["execution_state"] == "checkpointed":
        return attempt
    # The original final driver checks full packet/card/denominator identities.
    _, completion_path, work = inventory(root, name)
    if set(reports) != {path.relative_to(root).as_posix() for path,_,_ in work}:
        raise ValueError("completed incremental calculation does not cover the full native packet")
    if any(file_digest(REPO/path) != sha for path,sha in sources.items()):
        raise ValueError("independent calculation producer changed during execution")
    result = {"execution_state": "completed", "instrument_state": "valid", "recorded_at": now(),
        "process_id": os.getpid(), "clean_analysis_entry": "runners.reaggregate_v16_incremental", "packet_id": name,
        "source_checks": checks, "packet_sha256": file_digest(packet_path), "completion_sha256": file_digest(completion_path),
        "scientific_summaries": reports, "raw_records": sum(row["calculation"]["n_raw_units"] for row in reports.values()),
        "sources": sources, "plan_sha256": file_digest(output/"PLAN.json"),
        "card_receipts": {entry["card_id"]: file_digest(output/"items"/entry["card_id"]/"RECEIPT.json") for entry in entries},
        "process_attempts": {path.name: file_digest(path) for path in (output/"attempts").glob("*.json")},
        "full_aggregate_regeneration": False, "campaign_complete": False,
        "scope": "Complete final native packet independently recalculated from already completed immutable card outputs; source controls, global phase coverage and archive input-baseline checks remain separate"}
    if (output/"RECEIPT.json").exists():
        return read(output/"RECEIPT.json")
    write(output/"RECEIPT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=REPO/"results/v16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packet", choices=["boundary-expansion-1", "expansion-fixture-profile"], required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.root, args.output, args.packet, resume=args.resume)
    except Exception as error:
        write(args.output/"failures"/(now().replace(":", "-")+".json"),
            {"execution_state": "failed", "instrument_state": "unresolved", "error": repr(error), "process_id": os.getpid()})
        raise
    print({key: result.get(key) for key in ["execution_state", "raw_records", "waiting_card"]}, flush=True)


if __name__ == "__main__":
    main()
