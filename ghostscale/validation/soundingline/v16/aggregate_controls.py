"""Independent recounts of X01-X08 records and saved cost arithmetic.

These are fixture calculations, never extra maker sample sizes. Recounting a
prescribed call schedule is not retroactive proof of continuous CPU occupancy.
"""
from collections import Counter, defaultdict
from pathlib import Path
import math
import random
from .records import read, file_digest, digest
from .consumer_frames import requests


def equal(actual, expected, label):
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        okay = math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10)
    else:
        okay = actual == expected
    if not okay:
        raise ValueError("independent control aggregate differs: "+label)


def rows(base):
    return [(path, read(path)) for path in sorted((base/"units").glob("*_points.json"))]


def source_frames(root, path, record):
    identity = record["identity"]
    uid = path.stem.removesuffix("_points")
    base = root/identity["source_packet"]
    candidates = [base/"units"/(uid+"_points.json"), base/record["card_id"]/"units"/(uid+"_points.json")]
    located = [p for p in candidates if p.exists()]
    if len(located) != 1:
        raise ValueError("control source unit missing or ambiguous")
    target = located[0]
    equal(file_digest(target), identity.get("source_unit_sha256", identity.get("source_sha256")), "control source hash")
    original = read(target)
    frames = list(requests(target.parent.parent, original))
    equal(digest(frames), identity.get("source_requests_sha256", identity.get("request_sha256")), "control source frames")
    return target.parent.parent, original, frames


def access(root, name):
    base = root/name
    complete = read(base/"COMPLETION.json")
    counts, frame_list, units = Counter(), [], 0
    stable = name.endswith("-2")
    pattern = "*scout-2.json" if stable else "*scout-1.json"
    for packet in sorted((root/"packets").glob(pattern)):
        if stable and packet.stem != "inquiry-scout-2":
            continue
        for path in sorted((root/packet.stem).glob("**/units/*_points.json")):
            row = read(path)
            if row["seed_components"]["index"] != 0:
                continue
            units += 1
            for i, frame in enumerate(requests(path.parent.parent, row)):
                frame_list.append({"packet": packet.stem, "card_id": row["card_id"], "condition": row["condition"],
                                   "unit_id": row["unit_id"], "source_sha256": file_digest(path),
                                   "request_index": i, "request": frame})
                counts[row["card_id"]] += 1
    selection = read(base/"SELECTION.json")
    equal(digest(frame_list), selection["requests_sha256"], name+" request identity")
    equal(units, selection["n_units"], name+" source denominator")
    equal(len(frame_list), selection["n_requests"], name+" source requests")
    export = root/("transfer-fixture-2" if stable else "transfer-fixture-1")
    transfer_count = len(read(export/"PREDICTIONS_COMMITTED.json")["prediction_files"])
    counts["B01"] = transfer_count
    expected = {"consumer_cards": dict(counts), "n_units": units, "n_source_requests": len(frame_list),
                "actual_reader_requests": 4*len(frame_list)+2+3*transfer_count}
    for key, value in expected.items():
        equal(value, complete[key], name+"/"+key)
    return {**expected, "scope": "Retained source frames and frozen call schedule; private-read probes are two declared requests."}


def recoding(root):
    base = root/"recoding-attack-fixture-1"
    data = rows(base)
    counts, calls = Counter(), 0
    for _, record in data:
        counts[record["card_id"]] += len(record["checks"])
        for check in record["checks"]:
            calls += len(check["encodings"])+int(check["physical"].get("passed", False))
    bridge = read(base/"private/TRANSFER_points.json")
    counts["B01"] = len(bridge)
    for check in bridge:
        calls += len(check["representation"])+int(check["physical"].get("passed", False))
    expected = {"source_units": len(data), "consumer_requests_checked": dict(counts), "actual_reader_requests": calls}
    complete = read(base/"COMPLETION.json")
    for key, value in expected.items():
        equal(value, complete[key], "X02/"+key)
    return expected


def calibrated(root, name):
    base = root/name
    counts, source_count, calibration_count = Counter(), 0, 0
    for path, record in rows(base):
        _, _, frames = source_frames(root, path, record)
        equal(len(frames), record["source_reader_requests"], name+" source record")
        counts[record["card_id"]] += 1
        source_count += len(frames)
        calibration_count += len(record["control"]["requests"])
    for path in sorted((base/"private").glob("B01-*_points.json")):
        record = read(path)
        counts["B01"] += 1
        calibration_count += len(record["control"]["requests"])
    expected = {"source_condition_calibrations": dict(counts), "actual_source_reader_requests": source_count,
                "actual_calibration_reader_requests": calibration_count}
    complete = read(base/"COMPLETION.json")
    for key, value in expected.items():
        equal(value, complete[key], name+"/"+key)
    return expected


def fairness(root):
    base = root/"fairness-attack-fixture-1"
    counts, profiled, metered = Counter(), 0, 0
    for path, record in rows(base):
        _, original, frames = source_frames(root, path, record)
        counts[record["card_id"]] += 1
        equal(len(record["profiles"]), len(frames), "X04 frame denominator")
        for profile in record["profiles"]:
            frame = frames[profile["frame"]]
            equal(profile["request_sha256"], digest(frame["public"]), "X04 profile request")
            for repetition in profile["repetitions"]:
                equal(repetition["result_sha256"], digest(frame["result"]), "X04 measured prediction")
                resources = repetition["resources"]
                for key in ["wall_seconds", "parent_cpu_seconds"]:
                    if not math.isfinite(resources[key]) or resources[key] < 0:
                        raise ValueError("invalid retained platform measurement")
            profiled += len(profile["repetitions"])
        metered += len(record["reading_invocation_measurements"])
        for item in record["reading_invocation_measurements"]:
            equal(item["result"], frames[item["frame"]]["result"], "X04 metered output")
        equal(record["source_arm_outcomes"], {name: arm["outcomes"] for name, arm in original["arms"].items()}, "X04 source outcomes")
        for item in record["amortization"]:
            arm = frames[0]["result"][item["arm"]]
            search = sum(p["search_primitives"] for p in arm["submissions"])
            training, definition = arm["costs"]["training_primitives"], arm["costs"]["library_definition"]
            workload = item["future_task_packets"]
            for key, value in {"training_primitives_once": training, "definition_tokens_once": definition,
                               "search_primitives": workload*search,
                               "training_plus_search_primitives_per_packet": training/workload+search,
                               "definition_tokens_per_packet": definition/workload}.items():
                equal(item[key], value, "X04 amortization "+key)
    cold = list((base/"private").glob("cold-*_points.json"))
    expected = {"native_consumer_condition_profiles": dict(counts), "profiled_reader_requests": profiled,
                "reading_meter_requests": metered, "fresh_process_operations": len(cold)}
    complete = read(base/"COMPLETION.json")
    for key, value in expected.items():
        equal(value, complete[key], "X04/"+key)
    return {**expected, "platform_scope": "Saved measured values and arithmetic; no claim of identical CPU/wall clocks on replay."}


def hierarchy(base):
    record = read(base/"private/HIERARCHICAL_CONTROL_points.json")
    valid = record["valid_summary"]
    rng = random.Random(991)
    blocks = [[1.0 if block < 4 else -1.0]*8 for block in range(8)]
    estimates = []
    for _ in range(999):
        total = 0.0
        for _ in range(8):
            block = blocks[rng.randrange(8)]
            for _ in range(8):
                total += block[rng.randrange(8)]
        estimates.append(total/64)
    estimates.sort()
    expected = {"mean": 0.0, "interval_95": [estimates[int(.025*998)], estimates[int(.975*998)]],
                "paired_standard_deviation": math.sqrt(64/63), "n_makers": 64, "n_constructors": 8}
    for key, value in expected.items():
        equal(value, valid[key], "X07 independent hierarchy "+key)
    bad = record["intentionally_broken_summary"]
    equal(bad["n"], 64*32, "X07 intentionally broken denominator")
    equal(bad["interval_95"], [-1.96/math.sqrt(64*32), 1.96/math.sqrt(64*32)], "X07 intentionally broken interval")
    return expected


def dependence(root):
    base = root/"dependence-attack-fixture-1"
    ledger = read(base/"private/SOURCE_IDENTITIES_points.json")
    groups = defaultdict(list)
    for uid, identity in ledger["sources"].items():
        path = root/identity["path"]
        equal(file_digest(path), identity["raw_sha256"], "X07 retained source")
        row = read(path)
        equal(digest(row), identity["canonical_sha256"], "X07 canonical source")
        equal(uid, row["unit_id"], "X07 source identity")
        groups[(row["lineage"], row["card_id"], row["condition"])].append(row)
    for reported in ledger["grouping"]["conditions"]:
        group = groups[(reported["lineage"], reported["card_id"], reported["condition"])]
        # JSON object keys are strings even when the frozen constructor ID is an integer.
        constructors = Counter(str(row["constructor_id"]) for row in group)
        expected = {"makers": len(group), "constructors": len(constructors), "makers_per_constructor": dict(constructors),
                    "seed_indices": sorted(row["seed_components"]["index"] for row in group),
                    "unit_ids_sha256": digest(sorted(row["unit_id"] for row in group))}
        for key, value in expected.items():
            equal(value, reported[key], "X07 grouping "+key)
    equal(len(groups), len(ledger["grouping"]["conditions"]), "X07 missing groups")
    counts, calls, batches = Counter(), 0, Counter()
    for path, record in rows(base):
        source, row, frames = source_frames(root, path, record)
        equal(record["reader_requests"], len(frames), "X07 source call count")
        counts[record["card_id"]] += 1
        calls += len(frames)
        for batch in record["production_batches"]:
            parts = batch["path"].split("/")
            # The path starts with the separately saved private JSON filename.
            payload = read(source/"/".join(parts[:2]))
            for key in parts[2:]:
                payload = payload[int(key)] if isinstance(payload, list) else payload[key]
            candidates = payload["candidates"]
            kept = payload.get("retained_indices", [payload["retained_index"]] if "retained_index" in payload else [])
            rejected = [i for i in range(len(candidates)) if i not in kept]
            for key, value in {"produced":len(candidates), "retained":len(kept), "rejected":len(rejected),
                               "retained_indices":kept, "rejected_indices":rejected,
                               "candidate_digests":[digest(x) for x in candidates]}.items():
                equal(value, batch[key], "X07 partition "+key)
            batches.update({"produced":len(candidates), "retained":len(kept), "rejected":len(rejected)})
    expected = {"native_consumer_condition_calibrations":dict(counts), "actual_reader_requests":calls,
                "source_unit_records_checked":sum(len(group) for group in groups.values()),
                "registered_conditions":len(groups), "nested_production_inventory":dict(batches)}
    complete = read(base/"COMPLETION.json")
    for key, value in expected.items():
        equal(value, complete[key], "X07/"+key)
    return {**expected, "known_hierarchy":hierarchy(base)}


def noise(root):
    base = root/"noise-runtime-attack-fixture-1"
    counts, calls, noise_calls = Counter(), 0, 0
    for path, record in rows(base):
        _, _, frames = source_frames(root, path, record)
        equal(len(frames), record["source_reader_requests"], "X08 source requests")
        calls += len(frames)
        counts[record["card_id"]] += 1
        noise_calls += sum(len(case["requests"]) for case in record["cases"])
    expected = {"native_consumer_condition_calibrations":dict(counts),
                "actual_source_reader_requests":calls, "known_noise_reader_requests":noise_calls}
    complete = read(base/"COMPLETION.json")
    for key, value in expected.items():
        equal(value, complete[key], "X08/"+key)
    return expected


def run(root, report=lambda label, value: None):
    output = {}
    jobs = [(name, lambda n=name: access(root,n)) for name in ["access-attack-fixture-1","access-attack-fixture-2"]]
    jobs += [("recoding-attack-fixture-1",lambda:recoding(root))]
    jobs += [(name,lambda n=name:calibrated(root,n)) for name in ["collision-attack-fixture-1","context-attack-fixture-1","misspecification-attack-fixture-1"]]
    jobs += [("fairness-attack-fixture-1",lambda:fairness(root)),
             ("dependence-attack-fixture-1",lambda:dependence(root)),
             ("noise-runtime-attack-fixture-1",lambda:noise(root))]
    for name, function in jobs:
        output[name] = function()
        report(name, output[name])
    return output
