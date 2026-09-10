"""Independent source-control denominators, frame joins and observed cost arithmetic."""
from collections import Counter
import math
from .runtime import campaign
from .records import read, file_digest, digest
from .consumer_frames import requests
from .fairness import audit_unit

SOURCES = {"constructor-controls-1": "constructor-expansion-1", "boundary-controls-1": "boundary-expansion-1",
    "confirmation-controls": "confirmation-1"}


def resource_totals(records):
    result = {"requests": 0, "wall_seconds": 0., "parent_cpu_seconds": 0., "reader_cpu_seconds": 0.,
        "unavailable_reader_cpu_requests": 0, "maximum_observed_resident_bytes": 0, "maximum_observed_peak_resident_bytes": 0}
    for row in records:
        result["requests"] += 1
        for key in ["wall_seconds", "parent_cpu_seconds", "reader_cpu_seconds"]:
            value = row[key]
            if key == "reader_cpu_seconds" and value is None:
                result["unavailable_reader_cpu_requests"] += 1
            elif not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid retained platform measurement")
            else:
                result[key] += value
        for key in ["resident_bytes", "peak_resident_bytes"]:
            value = row[key]
            if value is not None:
                if not isinstance(value, int) or value < 0:
                    raise ValueError("invalid retained memory measurement")
                field = "maximum_observed_"+key
                result[field] = max(result[field], value)
    return result


def condition(source, row, receipt):
    frames = list(requests(source, row))
    if receipt["source_unit_id"] != row["unit_id"] or receipt["source_packet_hash"] != row["packet_hash"] or receipt["source_unit_sha256"] != file_digest(source/"units"/(row["unit_id"]+"_points.json")):
        raise ValueError("independent actual control source differs")
    if receipt["source_frames_sha256"] != digest(frames) or receipt["instrument_state"] != "valid":
        raise ValueError("independent actual control frame identity differs")
    evidence = receipt["evidence"]
    if len(evidence["private_reads_denied"]) != 2 or not all(evidence["private_reads_denied"]) or len(evidence["private_variants"]) != 2 or evidence["private_variants"][0] == evidence["private_variants"][1]:
        raise ValueError("independent private-access witness differs")
    baselines, aliases = evidence["actual_baselines"], evidence["alias_requests"]
    if len(baselines) != len(frames) or len(aliases) != len(frames):
        raise ValueError("independent actual control request count differs")
    for frame, baseline, alias in zip(frames, baselines, aliases):
        if baseline["public_sha256"] != digest(frame["public"]) or baseline["result_sha256"] != digest(frame["result"]) or alias["result_sha256"] != baseline["result_sha256"]:
            raise ValueError("independent warm/alias prediction comparison differs")
        if baseline.get("invocations") != alias.get("invocations"):
            raise ValueError("independent alias invocation count differs")
    if evidence["costs"] != audit_unit(source, row, frames):
        raise ValueError("independent actual source physical costs differ")
    required = set(receipt["required_adversaries"])-{"X08"}
    if not required <= {key for key, passed in receipt["completed_local_checks"].items() if passed is True}:
        raise ValueError("independent actual control omitted a declared adversary")
    for key in ["X03", "X05", "X06"]:
        if key in required and (not evidence[key]["checks"] or not all(evidence[key]["checks"].values())):
            raise ValueError("retained known-answer adversary witness failed")
    if "X02" in required and (len(evidence["recoding"]) != len(frames) or not all(item["physical"]["passed"] for item in evidence["recoding"])):
        raise ValueError("independent physical recoding control count differs")
    if "X07" in required and evidence["duplicates"]["instrument_state"] != "valid":
        raise ValueError("independent duplicate-control witness failed")
    return {"source_frames": len(frames), "recorded_reader_calls": len(receipt["requests"]),
        "platform_measurements": resource_totals(receipt["resources"]), "adversaries": sorted(required),
        "scope": "Independent raw frame/hash/denominator/cost recount; known adversary witness labels retain their separately validated original meaning"}


def run(root, name, report):
    if name not in SOURCES:
        raise ValueError("unknown explicit source-control packet")
    source_name = SOURCES[name]
    packet = read(root/"packets"/(source_name+".json"))
    source_root = root/("confirmation" if source_name == "confirmation-1" else source_name)
    science = read(source_root/"COMPLETION.json")
    if science.get("execution_state") != "completed" or science.get("instrument_state") != "valid":
        raise ValueError("source-control recount requires completed valid science")
    top = science if name == "confirmation-controls" else read(root/name/"COMPLETION.json")
    if top["execution_state"] != "completed" or top["instrument_state"] != "valid":
        raise ValueError("source-control recount requires completed actual controls")
    control_hash = packet["packet_hash"] if name == "confirmation-controls" else read(root/"packets"/(name+".json"))["packet_hash"]
    results = []
    for entry in packet["identity"]["design"]["cards"]:
        card = entry["card_id"]
        source, destination = source_root/card, root/name/card
        saved = read(destination/"COMPLETION.json")
        inventory_path = destination/"source_inventory_points.json"
        if saved["source_inventory_sha256"] != file_digest(inventory_path):
            raise ValueError("source-control inventory checksum differs")
        inventory = read(inventory_path)
        rows = []
        for path in sorted((source/"units").glob("*_points.json")):
            relative = path.relative_to(source).as_posix()
            if inventory["unit_hashes"].get(relative) != file_digest(path):
                raise ValueError("independent source-control unit checksum differs")
            rows.append(read(path))
        expected = {item["id"]: entry["n_per_condition"] for item in entry["design"]["conditions"]}
        if Counter(row["condition"] for row in rows) != expected or len(rows) != len(inventory["unit_hashes"]) or len(rows) != saved["source_condition_records"]:
            raise ValueError("independent source-control condition denominator differs")
        if len({row["unit_id"] for row in rows}) != len(rows) or any(row["packet_hash"] != packet["packet_hash"] or row["lineage"] != entry["namespace"] or row["card_id"] != card for row in rows):
            raise ValueError("independent source-control identity/grouping differs")
        grouping = Counter((row["lineage"], card, row["condition"], row["constructor_id"]) for row in rows)
        expected_groups = [{"lineage": lineage, "card": label, "condition": condition_id, "constructor": constructor, "makers": n}
            for (lineage, label, condition_id, constructor), n in sorted(grouping.items())]
        if inventory["grouping"]["constructor_groups"] != expected_groups or inventory["grouping"]["n_retained_unit_records"] != len(rows):
            raise ValueError("independent constructor-group recount differs")
        selected, supplements, missing_supplements = {}, [], []
        for condition_id, count in expected.items():
            subset = [row for row in rows if row["condition"] == condition_id]
            if {row["seed_components"]["index"] for row in subset} != set(range(count)) or len({row["constructor_id"] for row in subset}) != min(entry["constructors"], count):
                raise ValueError("independent source seed/constructor coverage differs")
            selected[condition_id] = next(row for row in subset if row["seed_components"]["index"] == 0)
        condition_reports, cold_results = [], []
        for original in entry["design"]["conditions"]:
            row = selected[original["id"]]
            folder = destination/"conditions"/digest(original["id"])[:16]
            path = folder/"RECEIPT.json"
            receipt = read(path)
            if receipt["control_packet_hash"] != control_hash or set(receipt["required_adversaries"]) != set(entry["design"]["adversaries"]):
                raise ValueError("actual source control changed its packet or attack allocation")
            for relative, expected_hash in receipt["retained_control_files"].items():
                if file_digest(folder/relative) != expected_hash:
                    raise ValueError("retained source-control evidence differs")
            checked = condition(source, row, receipt)
            condition_reports.append({"condition": original["id"], "receipt_sha256": file_digest(path), **checked})
            cold_results.extend(digest(frame["result"]) for frame in requests(source, row))
        cold_path = destination/"COLD.json"
        cold = read(cold_path)
        if saved["cold_sha256"] != file_digest(cold_path) or not cold["fresh_process"] or cold["warm_pid"] == cold["cold_pid"] or [row["result_sha256"] for row in cold["requests"]] != cold_results:
            raise ValueError("independent cold-reader prediction/count differs")
        for row in rows:
            path = source/"private"/(row["unit_id"]+"-expansion-resources.json")
            if path.exists():
                record = read(path)
                if record["unit_id"] != row["unit_id"]:
                    raise ValueError("observed platform supplement belongs to another unit")
                supplements.extend(record["requests"])
            else:
                missing_supplements.append(row["unit_id"])
        result = {"card_id": card, "instrument_state": "valid", "source_condition_records": len(rows),
            "condition_controls": len(condition_reports), "conditions": condition_reports,
            "source_platform_measurements": resource_totals(supplements), "units_without_optional_platform_supplement": missing_supplements,
            "platform_scope": "Original observed request measurements; missing supplementary OS measurements remain explicit and never count as scientific failure or zero cost",
            "source_inventory_sha256": file_digest(inventory_path), "control_completion_sha256": file_digest(destination/"COMPLETION.json")}
        report(card, result)
        results.append(result)
    count = sum(row["source_condition_records"] for row in results)
    conditions = sum(row["condition_controls"] for row in results)
    if name != "confirmation-controls" and (top["source_condition_records"] != count or top["condition_controls"] != conditions or top["source_packet_hash"] != packet["packet_hash"]):
        raise ValueError("independent full source-control totals differ")
    return {"instrument_state": "valid", "control_packet": name, "cards": results, "source_condition_records": count,
        "condition_controls": conditions, "runtime_scope": "Original actual CLI interruption proofs remain separately bound; no continuous-occupancy claim"}
