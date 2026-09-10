"""Execute X04 against every current native consumer condition, with real profiling."""
from collections import Counter
import time
from .records import read, write, file_digest, digest, now
from .runtime import remaining_seconds
from .active_consumers import selected_units, requests, attack_consumers
from .recoded_interface import OPERATIONS
from .reader_process import ReaderProcess
from .resource_accounting import MeasuredReader, process_snapshot
from .fairness import information, audit_unit, cost_fields, amortization, READING, PREFIX

METER = PREFIX+"reading_cost_meter:public_reader"
DESIGN = {"card_id": "X04", "scope": "fixture", "selection": "index zero in every current required native consumer condition",
    "request_repetitions": 2, "cold_profile": "one first-seen actual frame per operation in a fresh process, including separately measured startup",
    "measurement": "real OS process CPU/RSS and IPC wall time; finite clock resolution, zero CPU increments retained",
    "cost_validation": "independent primitive interpreters, retained source-specific raw audits and exact inquiry scalar reference",
    "reading_counter": "supplement original counter with observed calls through a copied function binding; preserve original outputs and source",
    "comparison": "exact input equality groups plus explicit paid/different-access contracts; all-arm batch resources never attributed to individual arms",
    "amortization": "K01 1,8,32 repetitions of each declared paired target packet, initial costs explicit; no claim of newly generated workload",
    "claim_scope": "bounded cost calibration/profile, not inferential performance superiority; full scout curves keep constructor/maker uncertainty",
    "pending_consumers": ["B02"], "broken_controls": ["hide extra observation", "erase macro expansion cost", "erase learning cost", "reuse incomplete direct-table call counter"]}


def measured_request(reader, frame):
    measured = MeasuredReader(reader)
    actual = measured.request(frame["kind"], frame["public"], **frame["options"])
    if actual != frame["result"]:
        return {"instrument_state": "failed", "actual": actual, "expected": frame["result"], "resources": measured.samples[0]}
    return {"instrument_state": "valid", "resources": measured.samples[0], "result_sha256": digest(actual)}


def execute(root, output, heartbeat, packet):
    required = set(attack_consumers(root)["X04"])
    if required-{"B02"} != set(information_card for information_card in __import__(PREFIX+"fairness", fromlist=["ACCESS"]).ACCESS):
        raise ValueError("fairness consumer list differs from explicit contracts")
    selected = list(selected_units(root, required-{"B02"}))
    if {row["card_id"] for _, _, row in selected} != required-{"B02"}:
        raise ValueError("missing native fairness consumer")
    extensions = [kind for kind in OPERATIONS if ":" in kind]+[METER]
    manifests, counts, calls, metered_calls = {}, Counter(), 0, 0
    first_frames = {}
    with ReaderProcess(output/"public/warm-reader", extensions=extensions) as reader:
        for ordinal, (source_packet, source, row) in enumerate(selected):
            if remaining_seconds(root) <= 0:
                return {"execution_state": "checkpointed", "campaign_complete": False, "reason": "immutable ceiling"}
            frames = list(requests(source.parent.parent, row))
            for frame in frames:
                first_frames.setdefault(frame["kind"], frame)
            identity = {"packet_hash": packet["packet_hash"], "source_packet": source_packet,
                        "source_unit_sha256": file_digest(source), "source_requests_sha256": digest(frames)}
            relative = f'units/{row["unit_id"]}_points.json'
            target = output/relative
            if target.exists():
                record = read(target)
                if record["identity"] != identity or record["instrument_state"] != "valid":
                    raise ValueError("resumed fairness receipt changed or previously failed")
            else:
                try:
                    access = information(row["card_id"], frames)
                    audit = audit_unit(source.parent.parent, row, frames)
                    profiles = []
                    meters = []
                    for number, frame in enumerate(frames):
                        profile = [measured_request(reader, frame) for _ in range(DESIGN["request_repetitions"])]
                        profiles.append({"frame": number, "kind": frame["kind"], "options": frame["options"],
                                         "request_sha256": digest(frame["public"]), "repetitions": profile})
                        if any(item["instrument_state"] != "valid" for item in profile):
                            raise ValueError("resource-profile reader changed saved predictions")
                        if row["card_id"] in READING:
                            meter = reader.request(METER, frame["public"], **frame["options"])
                            if meter["result"] != frame["result"] or meter["invocations"]["observation_mass"] != audit[number]["actual_likelihood_call_invocations"]:
                                raise ValueError("observed call count disagrees with independent work accounting")
                            meters.append({"frame": number, **meter})
                    record = {"identity": identity, "card_id": row["card_id"], "condition": row["condition"],
                        "instrument_state": "valid", "information": access, "independent_cost_checks": audit,
                        "profiles": profiles, "reading_invocation_measurements": meters,
                        "named_cost_fields": [cost_fields(frame["result"]) for frame in frames],
                        "source_arm_outcomes": {name: arm["outcomes"] for name, arm in row["arms"].items()},
                        "outcome_scope": "one predeclared calibration unit per condition, not estimated population performance",
                        "amortization": amortization(frames[0]["public"], frames[0]["result"]) if row["card_id"] == "K01" else [],
                        "completed_at": now()}
                    write(target, record)
                except Exception as error:
                    write(output/"failures"/f'{row["unit_id"]}_points.json', {"identity": identity,
                        "card_id": row["card_id"], "condition": row["condition"], "error": repr(error), "recorded_at": now(),
                        "frames": frames, "profiles": locals().get("profiles", [])})
                    raise
            counts[row["card_id"]] += 1
            calls += len(frames)*DESIGN["request_repetitions"]
            metered_calls += len(record["reading_invocation_measurements"])
            manifests[relative] = file_digest(target)
            heartbeat(completed_units=ordinal+1, planned_units=len(selected), card_id=row["card_id"], reader_requests=calls)
    cold = []
    for ordinal, (kind, frame) in enumerate(sorted(first_frames.items())):
        if remaining_seconds(root) <= 0:
            return {"execution_state": "checkpointed", "campaign_complete": False, "reason": "immutable ceiling"}
        relative = f"private/cold-{ordinal:02d}_points.json"
        target = output/relative
        identity = {"packet_hash": packet["packet_hash"], "frame_sha256": digest(frame)}
        if target.exists():
            record = read(target)
            if record["identity"] != identity or any(p["instrument_state"] != "valid" for p in record["requests"]):
                raise ValueError("cold benchmark resume differs or failed")
        else:
            started, cpu = time.perf_counter(), time.process_time()
            with ReaderProcess(output/"public"/f"cold-{ordinal:02d}", extensions=extensions) as reader:
                startup = {"wall_seconds": time.perf_counter()-started, "parent_cpu_seconds": time.process_time()-cpu,
                           "reader_snapshot_at_ready": process_snapshot(reader.child.pid),
                           "scope": "process creation and all declared imports before ready; shared once, not charged afresh to every scientific arm"}
                profiles = [measured_request(reader, frame) for _ in range(2)]
                record = {"identity": identity, "kind": kind, "public_frame": frame, "startup": startup, "requests": profiles}
                write(target, record)
                if any(p["instrument_state"] != "valid" for p in profiles):
                    raise ValueError("cold reader baseline differs; evidence retained")
        cold.append({"kind": kind, "receipt": relative})
        manifests[relative] = file_digest(target)
        heartbeat(card_id="X04", phase="cold-profile", completed_operations=ordinal+1, planned_operations=len(first_frames))
    result = {"card_id": "X04", "execution_state": "completed", "instrument_state": "valid",
        "native_consumer_condition_profiles": dict(sorted(counts.items())), "profiled_reader_requests": calls,
        "reading_meter_requests": metered_calls, "fresh_process_operations": len(cold),
        "pending_consumers": ["B02"], "consumer_completeness": "native consumers complete; B02 requires a later explicit receipt",
        "historical_resources": "not reconstructed; these are new bounded measurements",
        "reading_cost_qualification": "original direct-table likelihood counter omits repeated evidence calls. Exact observed invocation supplements retained; predictions and primary scientific scores unchanged.",
        "campaign_complete": False, "completed_at": now()}
    write(output/"RAW_MANIFEST.json", {"files": manifests, "retention": "through verified final archival handoff"})
    write(output/"COMPLETION.json", result)
    return result
