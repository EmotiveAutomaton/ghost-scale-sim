"""X03 actual consumer baselines plus source-specific known collision controls."""
from collections import Counter
import uuid
from .records import read, write, file_digest, digest, now
from .active_consumers import attack_consumers, selected_units, requests
from .collision_cases import CASES, Recorder
from .recoded_interface import OPERATIONS
from .reader_process import ReaderProcess
from .transfer_stable import Consumer, envelope
from .runtime import remaining_seconds

FAMILIES = {**{card: "reading" for card in ("K02", "P01", "P02", "P03")},
    **{f"R{i:02d}": "inquiry" for i in range(1, 6)},
    **{f"O{i:02d}": "opportunity" for i in range(1, 5)},
    **{card: "self" for card in ("S01", "S02", "S04")}, "S05": "trajectory",
    "M01": "recognition", "M02": "selection", "M03": "audience", "M04": "multi-actor",
    "V01": "tradeoffs", "V02": "tradeoffs", "V03": "preference-probe", "P04": "mechanism"}
TRANSFER_FAMILIES = ("reading", "inquiry", "opportunity", "self", "trajectory")
DESIGN = {"card_id": "X03", "scope": "fixture", "selection": "index zero in each current declared consumer condition",
    "actual_consumer": "reexecute every saved arm/phase, then run the matching known-answer collision and evidence controls",
    "control_families": sorted(CASES), "targets": "program, repertoire, cause, identity, old goal and future prediction separately",
    "broken_controls": ["force a unique posterior from ambiguous input", "ignore genuinely separating evidence"],
    "scope_limit": "known constructed ambiguity patterns in the declared readers; no general identifiability theorem",
    "sampling": "calibration executions share fixed witnesses; neither repeated controls nor nested requests are independent maker samples",
    "standalone": "repeat applicable controls through the actual corrected B01 process; evaluator witness stays outside its payload"}


class StandaloneAdapter:
    def __init__(self, consumer):
        self.consumer, self.sent = consumer, []
        self.lineage = uuid.uuid4().hex

    def request(self, kind, public, **options):
        observed = envelope({"kind": kind, "public": public, "options": options}, uuid.uuid4().hex, self.lineage)
        response = self.consumer.request(observed)
        if not response.get("ok") or response.get("observation_sha256") != digest(observed):
            raise ValueError("standalone collision control request failed or changed")
        self.sent.append({"public": observed, "response": response})
        return response["result"]


def check_case(reader, family):
    recorded = Recorder(reader)
    checks, witness = CASES[family](recorded)
    return {"family": family, "checks": checks, "witness": witness, "requests": recorded.frames,
            "instrument_state": "valid" if checks and all(checks.values()) else "failed"}


def execute(root, output, heartbeat, packet):
    required = set(attack_consumers(root)["X03"])
    selected = list(selected_units(root, required-{"B01"}))
    if {row["card_id"] for _, _, row in selected} != required-{"B01"}:
        raise ValueError("missing collision consumer source")
    write(output/"CONSUMERS.json", {"required": sorted(required), "calibration_families": FAMILIES,
        "assembly": "P04 W2 conditions use the independent assembly collision case", "source_units": len(selected)})
    native_requests, calibration_requests, counts, manifests = 0, 0, Counter(), {}
    with ReaderProcess(output/"public/reader", extensions=[kind for kind in OPERATIONS if ":" in kind]) as reader:
        for ordinal, (source_packet, source, row) in enumerate(selected):
            if remaining_seconds(root) <= 0:
                return {"execution_state": "checkpointed", "reason": "immutable ceiling", "campaign_complete": False,
                        "completed_units": ordinal, "planned_units": len(selected)+len(TRANSFER_FAMILIES)}
            family = "assembly-mechanism" if row["card_id"] == "P04" and row["condition"].startswith("W2-") else FAMILIES[row["card_id"]]
            frames = list(requests(source.parent.parent, row))
            identity = {"source_unit_sha256": file_digest(source), "source_requests_sha256": digest(frames),
                        "packet_hash": packet["packet_hash"], "source_packet": source_packet, "family": family}
            relative = f'units/{row["unit_id"]}_points.json'
            path = output/relative
            if path.exists():
                record = read(path)
                if record["identity"] != identity or record["control"]["instrument_state"] != "valid":
                    raise ValueError("resumed collision receipt differs or is failed")
            else:
                for frame in frames:
                    actual = reader.request(frame["kind"], frame["public"], **frame["options"])
                    if actual != frame["result"]:
                        write(output/"private/BASELINE_FAILURE_points.json", {"source": str(source.relative_to(root)), "request": frame, "actual": actual})
                        raise ValueError("actual collision consumer baseline differs from saved prediction")
                control = check_case(reader, family)
                record = {"card_id": row["card_id"], "condition": row["condition"], "identity": identity,
                          "control": control, "source_reader_requests": len(frames), "completed_at": now()}
                write(path, record)
                if control["instrument_state"] != "valid":
                    raise ValueError("collision control failed; complete failing evidence retained")
            counts[row["card_id"]] += 1
            native_requests += len(frames)
            calibration_requests += len(record["control"]["requests"])
            manifests[relative] = file_digest(path)
            heartbeat(completed_units=ordinal+1, planned_units=len(selected)+len(TRANSFER_FAMILIES), card_id=row["card_id"], family=family)
    transfer = root/"transfer-fixture-2"
    source_manifest = read(transfer/"PUBLIC_MANIFEST.json")
    with Consumer(transfer/"public/consumer") as consumer:
        if consumer.identity["sources"] != source_manifest["consumer_sources"]:
            raise ValueError("standalone collision source differs from corrected export")
        for ordinal, family in enumerate(TRANSFER_FAMILIES):
            if remaining_seconds(root) <= 0:
                return {"execution_state": "checkpointed", "reason": "immutable ceiling", "campaign_complete": False,
                        "completed_units": len(selected)+ordinal, "planned_units": len(selected)+len(TRANSFER_FAMILIES)}
            relative = f"private/B01-{family}_points.json"
            target = output/relative
            if target.exists():
                record = read(target)
                if record["packet_hash"] != packet["packet_hash"] or record["control"]["instrument_state"] != "valid" or record["source_manifest_sha256"] != file_digest(transfer/"PUBLIC_MANIFEST.json"):
                    raise ValueError("resumed standalone collision record differs or failed")
            else:
                adapter = StandaloneAdapter(consumer)
                control = check_case(adapter, family)
                record = {"card_id": "B01", "packet_hash": packet["packet_hash"], "control": control,
                          "exact_transport": adapter.sent, "source_manifest_sha256": file_digest(transfer/"PUBLIC_MANIFEST.json")}
                write(target, record)
            control = record["control"]
            manifests[relative] = file_digest(target)
            if control["instrument_state"] != "valid":
                raise ValueError("standalone collision control failed")
            calibration_requests += len(control["requests"])
            counts["B01"] += 1
            heartbeat(completed_units=len(selected)+ordinal+1, planned_units=len(selected)+len(TRANSFER_FAMILIES), card_id="B01", family=family)
    result = {"card_id": "X03", "execution_state": "completed", "instrument_state": "valid",
        "source_condition_calibrations": dict(sorted(counts.items())), "actual_source_reader_requests": native_requests,
        "actual_calibration_reader_requests": calibration_requests, "completed_at": now(), "campaign_complete": False,
        "sampling_scope": DESIGN["sampling"], "target_scope": DESIGN["targets"], "scope_limit": DESIGN["scope_limit"]}
    write(output/"RAW_MANIFEST.json", {"files": manifests, "retention": "through final verified archival handoff"})
    write(output/"COMPLETION.json", result)
    return result
