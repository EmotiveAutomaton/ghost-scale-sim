"""Serialization-aware recount; original failed verifier remains frozen.

Canonical JSON is the persisted witness format: tuple/list normalization preserves
its contents. All source hashes, requests, check results and scope remain checked.
"""
from .records import read, file_digest, digest
from .completion_guard import bound_receipt
from .consumer_frames import requests
from .attack_collisions import check_case as collision, FAMILIES as COLLISIONS
from .attack_misspecification import check_case as misspecification, FAMILIES as MISSPECIFICATIONS


class SavedReader:
    def __init__(self, frames):
        self.frames, self.index = frames, 0

    def request(self, kind, public, **options):
        if self.index >= len(self.frames):
            raise ValueError("retained control omitted a request")
        frame = self.frames[self.index]
        self.index += 1
        if (frame["kind"], frame["public"], frame["options"]) != (kind, public, options):
            raise ValueError("retained control request changed")
        if "request_error" in frame:
            raise RuntimeError(frame["request_error"])
        return frame["result"]


def verify_records(root, spec, refs):
    if file_digest(root/"COMMISSION_MANIFEST.json") != spec["commission_sha256"]:
        raise ValueError("supplement commission changed")
    manifest = {row["card_id"]: row for row in read(root/"COMMISSION_MANIFEST.json")["cards"]}
    expected = {}
    for gap in spec["gaps"]:
        original = bound_receipt(root, gap["original_control"]["path"], gap["original_control"]["sha256"])
        missing = set(manifest[gap["card_id"]]["dependency_ids"])-set(original["required_adversaries"])
        if missing != set(gap["missing_adversaries"]) or not missing:
            raise ValueError("supplement omits or substitutes a commissioned dependency")
        original_conditions = {row["condition"]: row for row in original["conditions"]}
        if len(original_conditions) != len(original["conditions"]) or set(original_conditions) != {row["condition"] for row in gap["conditions"]}:
            raise ValueError("supplement omitted an original condition")
        for item in gap["conditions"]:
            old = original_conditions[item["condition"]]
            if item["original_condition"] != {"path": old["receipt"], "sha256": old["sha256"]}:
                raise ValueError("supplement changed the original condition binding")
            expected[(gap["card_id"], item["condition"])] = (gap, item)
    seen, total_baselines, total_calls = set(), 0, 0
    for ref in refs:
        record = bound_receipt(root, ref["path"], ref["sha256"])
        identity = record["identity"]
        key = (identity["card_id"], identity["condition"])
        if key in seen or key not in expected:
            raise ValueError("duplicate or unexpected supplemental condition")
        seen.add(key)
        gap, item = expected[key]
        if any(identity[name] != value for name, value in item.items()) or record["instrument_state"] != "valid":
            raise ValueError("supplement source identity changed or control failed")
        unit = bound_receipt(root, item["unit"]["path"], item["unit"]["sha256"])
        old = bound_receipt(root, item["original_condition"]["path"], item["original_condition"]["sha256"])
        if old["source_unit_sha256"] != item["unit"]["sha256"] or old["source_unit_id"] != unit["unit_id"] or unit["seed_components"]["index"] != 0:
            raise ValueError("supplement source selection changed")
        frames = list(requests(root/gap["source"], unit))
        if record["baselines"] != [{**frame, "actual": frame["result"]} for frame in frames] or not frames:
            raise ValueError("supplement baseline prediction/count differs")
        total_baselines += len(frames)
        if set(record["controls"]) != set(gap["missing_adversaries"]):
            raise ValueError("supplement missing a commissioned check")
        for attack, control in record["controls"].items():
            producer, families = {"X03": (collision, COLLISIONS), "X06": (misspecification, MISSPECIFICATIONS)}[attack]
            reader = SavedReader(control["requests"])
            reproduced = producer(reader, families[key[0]])
            if digest(reproduced) != digest(control) or control["instrument_state"] != "valid" or reader.index != len(reader.frames):
                raise ValueError("supplement retained witness differs or failed")
            total_calls += reader.index
    if seen != set(expected):
        raise ValueError("supplement omitted a required condition")
    return {"execution_state": "completed", "instrument_state": "valid", "condition_controls": len(seen),
        "source_baselines": total_baselines, "known_answer_requests": total_calls,
        "commission_dependencies_verified": True,
        "scope": "Separate raw source/frame/condition recount and retained-request witness replay; shared known-answer definitions are not an independent physical model"}


def verify(root, ref):
    receipt = bound_receipt(root, ref["path"], ref["sha256"])
    if receipt.get("execution_state") != "completed" or receipt.get("instrument_state") != "valid" or receipt.get("commission_dependencies_verified") is not True:
        raise ValueError("supplement has no valid completed execution")
    packet = read(root/"packets/commission-controls-1.json")
    if receipt["packet_hash"] != packet["packet_hash"] or receipt["specification"] != packet["identity"]["design"]:
        raise ValueError("supplement frozen design changed")
    for item in receipt["records"]:
        if bound_receipt(root, item["path"], item["sha256"])["identity"]["packet_hash"] != packet["packet_hash"]:
            raise ValueError("supplement condition belongs to another packet")
    finalizer = read(root/"packets/commission-controls-finalization-1.json")
    if receipt.get("finalizer_packet_hash") != finalizer["packet_hash"] or finalizer["identity"]["design"]["original_packet_hash"] != packet["packet_hash"]:
        raise ValueError("supplement serialization finalizer binding changed")
    checked = verify_records(root, receipt["specification"], receipt["records"])
    audit = receipt["independent_audit"]
    if checked != bound_receipt(root, audit["path"], audit["sha256"]):
        raise ValueError("supplement independent recount differs")
    return receipt
