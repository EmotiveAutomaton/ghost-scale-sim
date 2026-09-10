"""X07 actual native source joins; bridge/confirmation consumers remain explicit."""
from collections import Counter
from .records import read, write, file_digest, digest, now
from .runtime import remaining_seconds
from .active_consumers import packets, selected_units, attack_consumers, requests
from .reader_process import ReaderProcess
from .recoded_interface import OPERATIONS
from .fairness import audit_unit
from .dependence import duplicate_attacks, source_bound_sample, sampling_groups, unit_batch_inventory, hierarchical_control

SNAPSHOT = "results/v16/checkpoint-0510/INTEGRITY.json"
DESIGN = {"card_id": "X07", "scope": "fixture", "native_selection": "every retained current scout sampling identity, plus index-zero source replay in each required condition",
    "source_snapshot": SNAPSHOT, "source_snapshot_scope": "frozen checksums of the previously verified raw manifests; original inquiry remains preserved but superseded",
    "attacks": ["renamed row", "renamed history", "renamed seed", "entire identity forged", "nested episodes treated as independent", "discard rejected production"],
    "unit_rule": "maker task packet; M01 reference-maker pair and M04 producer/editor pair remain one paired packet, not two independent scored trials",
    "constructor_rule": "constructors then makers within each registered condition; cross-condition interactions must preserve shared constructors",
    "same_artifact_rule": "coincident artifacts from fresh independently generated histories are permitted",
    "pending_consumers": ["B02", "B03", "B04"],
    "interpretation": "validity of declared source identities, nested grouping and retained production, not a new scientific sample or population effect"}


def admitted_rows(root, required):
    snapshot = read(root/"checkpoint-0510/INTEGRITY.json")
    expected_manifests = {item["manifest"]: item["sha256"] for item in snapshot["scientific_retention"]["manifests"]}
    rows, sources = [], {}
    for packet_path, packet in packets(root):
        base = root/packet_path.stem
        for manifest_path in sorted(base.glob("**/RAW_MANIFEST.json")):
            relative = str(manifest_path.relative_to(root)).replace("\\", "/")
            if expected_manifests.get(relative) != file_digest(manifest_path):
                raise ValueError("raw manifest differs from frozen integrity snapshot")
            mapping = read(manifest_path)["files"]
            unit_paths = sorted((manifest_path.parent/"units").glob("*_points.json"))
            expected = {name for name in mapping if name.startswith("units/")}
            if {"units/"+path.name for path in unit_paths} != expected:
                raise ValueError("unit archive has missing or additional source rows")
            for path in unit_paths:
                actual = file_digest(path)
                if actual != mapping["units/"+path.name]:
                    raise ValueError("unit differs from previously admitted raw source")
                row = read(path)
                if row["packet_hash"] != packet["packet_hash"]:
                    raise ValueError("unit source packet differs")
                if row["card_id"] not in required:
                    continue
                rows.append(row)
                sources[row["unit_id"]] = {"path": str(path.relative_to(root)).replace("\\", "/"),
                    "raw_sha256": actual, "canonical_sha256": digest(row), "raw_manifest": relative,
                    "raw_manifest_sha256": expected_manifests[relative]}
    return rows, sources


def execute(root, output, heartbeat, packet):
    required = set(attack_consumers(root)["X07"])
    native = required-set(DESIGN["pending_consumers"])
    rows, sources = admitted_rows(root, native)
    if {row["card_id"] for row in rows} != native:
        raise ValueError("missing required native sampling consumer")
    joined = source_bound_sample(rows, {uid: item["canonical_sha256"] for uid, item in sources.items()})
    grouped = sampling_groups(rows)
    # This receipt covers the frozen scouts. Future expansion/confirmation sources
    # require another explicit admission join; they do not inherit these counts.
    if any(item["makers"] != 64 or item["constructors"] != 8 or item["seed_indices"] != list(range(64)) for item in grouped["conditions"]):
        raise ValueError("current scout allocation differs from its 64/8 contract")
    ledger_path = output/"private/SOURCE_IDENTITIES_points.json"
    ledger = {"packet_hash": packet["packet_hash"], "sources": sources, "grouping": grouped,
              "snapshot_sha256": file_digest(root/"checkpoint-0510/INTEGRITY.json")}
    write(ledger_path, ledger)
    control = hierarchical_control()
    if control["instrument_state"] != "valid":
        write(output/"private/HIERARCHY_FAILURE_points.json", control)
        raise ValueError("known dependence ruler failed")
    control_path = output/"private/HIERARCHICAL_CONTROL_points.json"
    write(control_path, control)
    manifests = {str(path.relative_to(output)).replace("\\", "/"): file_digest(path) for path in [ledger_path, control_path]}
    counts, reader_requests, batch_counts = Counter(), 0, Counter()
    selected = list(selected_units(root, native))
    with ReaderProcess(output/"public/reader", extensions=[kind for kind in OPERATIONS if ":" in kind]) as reader:
        for ordinal, (source_packet, source, row) in enumerate(selected):
            if remaining_seconds(root) <= 0:
                return {"execution_state": "checkpointed", "reason": "immutable ceiling", "campaign_complete": False}
            frames = list(requests(source.parent.parent, row))
            identity = {"packet_hash": packet["packet_hash"], "source_packet": source_packet,
                        "source_sha256": file_digest(source), "request_sha256": digest(frames)}
            relative = f'units/{row["unit_id"]}_points.json'
            target = output/relative
            if target.exists():
                record = read(target)
                if record["identity"] != identity or record["instrument_state"] != "valid":
                    raise ValueError("dependence resume record changed or failed")
            else:
                for frame in frames:
                    actual = reader.request(frame["kind"], frame["public"], **frame["options"])
                    if actual != frame["result"]:
                        write(output/"failures"/f'{row["unit_id"]}_points.json', {"identity": identity, "request": frame, "actual": actual})
                        raise ValueError("dependence consumer baseline differs")
                audit_unit(source.parent.parent, row, frames)
                duplicates = duplicate_attacks(row)
                batches = unit_batch_inventory(source.parent.parent, row)
                if row["card_id"] in {"M02", "M03", "M04", "V01", "V02"} and not batches:
                    raise ValueError("selection consumer has no retained/rejected production inventory")
                record = {"identity": identity, "card_id": row["card_id"], "condition": row["condition"],
                    "instrument_state": duplicates["instrument_state"], "duplicate_controls": duplicates,
                    "production_batches": batches, "reader_requests": len(frames),
                    "actual_reader_request_count_is_not_sample_size": True,
                    "batch_counts_are_nested": True, "completed_at": now()}
                write(target, record)
                if record["instrument_state"] != "valid":
                    raise ValueError("actual source duplicate controls failed; retained")
            counts[row["card_id"]] += 1
            reader_requests += len(frames)
            for batch in record["production_batches"]:
                batch_counts.update({key: batch[key] for key in ["produced", "retained", "rejected"]})
            manifests[relative] = file_digest(target)
            heartbeat(completed_units=ordinal+1, planned_units=len(selected), card_id=row["card_id"], source_units_checked=joined["n_retained_unit_records"])
    result = {"card_id": "X07", "execution_state": "completed", "instrument_state": "valid",
        "native_consumer_condition_calibrations": dict(sorted(counts.items())),
        "source_unit_records_checked": joined["n_retained_unit_records"], "registered_conditions": len(grouped["conditions"]),
        "actual_reader_requests": reader_requests, "nested_production_inventory": dict(batch_counts),
        "production_inventory_scope": "file/path-addressed archived batch entries, not new independent makers; full original and later private records included; source-specific audits verify physical production counts",
        "source_identity_qualification": "frozen archive provenance; not arbitrary forgery detection outside that boundary",
        "sampling_scope": "independent sample counts remain per condition; reference/actor pairs and all their episodes stay paired packets",
        "pending_consumers": DESIGN["pending_consumers"], "campaign_complete": False, "completed_at": now()}
    write(output/"RAW_MANIFEST.json", {"files": manifests, "retention": "through verified final archive handoff"})
    write(output/"COMPLETION.json", result)
    return result
