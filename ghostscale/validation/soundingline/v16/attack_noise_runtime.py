"""X08 source-specific misleading-progress and honest completion controls."""
from collections import Counter
from .records import read, write, digest, file_digest, now
from .runtime import remaining_seconds
from .active_consumers import selected_units, attack_consumers, requests
from .reader_process import ReaderProcess
from .recoded_interface import OPERATIONS
from .collision_cases import Recorder
from .noise_cases import noise_case
from .runtime_interruption import interruption_control
from .completion_guard import from_progress
from .fairness import audit_unit

SNAPSHOT = "results/v16/noise-runtime-setup/PROGRESS_SNAPSHOT.json"
DESIGN = {"card_id": "X08", "scope": "fixture", "selection": "source index zero in every required current native condition",
    "noise": "executed iid physical noise with known selected rise and decline streaks; source-bound corrected inquiry policies and independent scalar reference",
    "runtime": "actual owned native child interruption, immutable saved units and clocks, exact resume reaggregation",
    "closure": "actual incomplete campaign snapshot with no ready worker still cannot pass finite completion guard",
    "source_snapshot": SNAPSHOT, "pending_consumers": ["B02", "B03", "B04"],
    "scope_limit": "known adversarial noise witnesses and bounded runtime controls, not a prevalence estimate or whole-campaign closure",
    "future_consumers": "archive and confirmation require their own source-specific runtime and dependence joins"}


def execute(root, output, heartbeat, packet):
    required = set(attack_consumers(root)["X08"])-set(DESIGN["pending_consumers"])
    selected = list(selected_units(root, required))
    if {row["card_id"] for _, _, row in selected} != required:
        raise ValueError("missing required noise/runtime source")
    sources = {name: read(root/"packets"/(name+".json"))["packet_hash"] for name, _, _ in selected}
    runtime = interruption_control(output/"runtime-fixture", sources)
    if runtime["instrument_state"] != "valid":
        raise ValueError("actual interruption control failed; evidence retained")
    progress = read(root/"noise-runtime-setup/PROGRESS_SNAPSHOT.json")
    closure = from_progress(progress)
    if closure["campaign_closed"] or not closure["blocking_reasons"]:
        raise ValueError("incomplete actual campaign falsely passed completion")
    closure_path = output/"private/EMPTY_QUEUE_points.json"
    write(closure_path, {"snapshot_sha256": file_digest(root/"noise-runtime-setup/PROGRESS_SNAPSHOT.json"),
        "commissioned_cards": [row["card_id"] for row in progress["cards"]],
        "ready_jobs": [], "actual_completion_guard": closure,
        "broken_queue_only_completer_would_pass": not [],
        "scope": "known empty dispatch list applied to actual unfinished campaign accounting"})
    manifests = {"private/EMPTY_QUEUE_points.json": file_digest(closure_path)}
    for relative, expected in runtime["files"].items():
        manifests["runtime-fixture/"+relative] = expected
    manifests["runtime-fixture/RECEIPT.json"] = file_digest(output/"runtime-fixture/RECEIPT.json")
    counts, baseline_count, controls = Counter(), 0, 0
    with ReaderProcess(output/"public/reader", extensions=[name for name in OPERATIONS if ":" in name]) as reader:
        for ordinal, (source_packet, source, row) in enumerate(selected):
            if remaining_seconds(root) <= 0:
                return {"execution_state": "checkpointed", "reason": "immutable ceiling", "campaign_complete": False}
            frames = list(requests(source.parent.parent, row))
            identity = {"source_packet": source_packet, "source_sha256": file_digest(source),
                        "request_sha256": digest(frames), "packet_hash": packet["packet_hash"]}
            relative = f'units/{row["unit_id"]}_points.json'
            destination = output/relative
            if destination.exists():
                record = read(destination)
                if record["identity"] != identity or record["instrument_state"] != "valid":
                    raise ValueError("saved noise/runtime control changed or failed")
            else:
                for frame in frames:
                    actual = reader.request(frame["kind"], frame["public"], **frame["options"])
                    if actual != frame["result"]:
                        write(output/"failures"/f'{row["unit_id"]}_points.json', {"identity": identity, "frame": frame, "actual": actual})
                        raise ValueError("noise/runtime source baseline changed")
                audit_unit(source.parent.parent, row, frames)
                cases = []
                if row["card_id"].startswith("R"):
                    for direction in ["rise", "decline"]:
                        recorder = Recorder(reader)
                        checks, witness = noise_case(recorder, direction)
                        cases.append({"checks": checks, "witness": witness, "requests": recorder.frames})
                record = {"identity": identity, "card_id": row["card_id"], "condition": row["condition"],
                    "cases": cases, "source_reader_requests": len(frames), "runtime_receipt": "runtime-fixture/RECEIPT.json",
                    "runtime_source_packet_join": sources[source_packet], "closure_guard_passed_incomplete_campaign": False,
                    "instrument_state": "valid" if all(all(case["checks"].values()) for case in cases) else "failed",
                    "calibrations_are_not_fresh_maker_samples": True, "completed_at": now()}
                write(destination, record)
                if record["instrument_state"] != "valid":
                    raise ValueError("noise control failed; complete evidence retained")
            counts[row["card_id"]] += 1
            baseline_count += len(frames)
            controls += sum(len(case["requests"]) for case in record["cases"])
            manifests[relative] = file_digest(destination)
            heartbeat(completed_units=ordinal+1, planned_units=len(selected), card_id=row["card_id"])
    result = {"card_id": "X08", "execution_state": "completed", "instrument_state": "valid",
        "native_consumer_condition_calibrations": dict(sorted(counts.items())),
        "actual_source_reader_requests": baseline_count, "known_noise_reader_requests": controls,
        "runtime_controls": runtime["checks"], "empty_queue_campaign_completion_rejected": True,
        "pending_consumers": DESIGN["pending_consumers"], "campaign_complete": False, "completed_at": now()}
    write(output/"RAW_MANIFEST.json", {"files": manifests, "retention": "through verified final archival handoff"})
    write(output/"COMPLETION.json", result)
    return result
