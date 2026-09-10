"""B02 actual source-bound access, collision, cost, misspecification and nesting joins."""
from .records import read, write, file_digest, digest
from .selection import EXTENSION, METHODS
from .reader_process import ReaderProcess
from .collision_cases import Recorder, selection_case as collision_case
from .misspecification_cases import selection_case as misspecification_case
from .dependence import duplicate_attacks, source_bound_sample


def validate_consumers(output, packet_hash, *, fixture=False):
    receipt = output/"CONSUMER_ATTACKS.json"
    if receipt.exists():
        result = read(receipt)
        if result["packet_hash"] != packet_hash:
            raise ValueError("archive attack source changed")
        return result
    source_rows, admitted = [], {}
    for folder in ["search", "followup"]:
        for path in sorted((output/"private"/folder).glob("**/units/*_points.json")):
            row = read(path)
            bound = read(path.parent.parent/"private"/(row["unit_id"]+"-archive-case.json"))
            if bound["unit_sha256"] != file_digest(path) or row["packet_hash"] != packet_hash:
                raise ValueError("new archive source lost its admitted identity")
            source_rows.append(row); admitted[row["unit_id"]] = digest(row)
    sampling = source_bound_sample(source_rows, admitted)
    base = output/"private/followup"
    selected = [row for row in source_rows if row["lineage"].startswith("v16-explanatory-archive-followup-1") and row["seed_components"]["index"] == 0]
    if len(selected) != 12:
        raise ValueError("archive attacks lack one source for every frozen follow-up condition")
    records = []
    with ReaderProcess(output/"public/attack-reader", extensions=[EXTENSION]) as reader:
        for row in selected:
            relative = "private/consumer-attacks/"+row["unit_id"]+"_points.json"
            destination = output/relative
            if destination.exists():
                record = read(destination)
                if record["source_unit_sha256"] != file_digest(base/"units"/(row["unit_id"]+"_points.json")):
                    raise ValueError("archive attack resume source changed")
                records.append(record)
                continue
            public = read(base/"public"/(row["unit_id"]+".json"))
            predictions = read(base/"predictions"/(row["unit_id"]+".json"))["arms"]
            # Explicit evaluator-only poison in a scratch file, not a mutation of
            # the immutable source. Both variants are retained in the control.
            shadow = output/"private/consumer-attacks"/(row["unit_id"]+"-shadow.json")
            variants = [{"maker": 0, "seed": "private-first", "future": 0},
                        {"maker": 1, "seed": "private-second", "future": 1}]
            observations = []
            denied = []
            for variant in variants:
                write(shadow, variant, immutable=False)
                observations.append({method: reader.request(EXTENSION, public, method=method) for method in METHODS})
                error = None
                try:
                    reader.request("_probe_forbidden_read", {"path": str(shadow.resolve())})
                except RuntimeError as caught:
                    error = str(caught)
                denied.append(error is not None and "PermissionError" in error)
            collision = Recorder(reader)
            collision_checks, collision_witness = collision_case(collision)
            changed_world = Recorder(reader)
            misspecification_checks, misspecification_witness = misspecification_case(changed_world)
            duplicate = duplicate_attacks(row)
            checks = {"X01": observations[0] == observations[1] == predictions and all(denied),
                "X03": all(collision_checks.values()), "X04": set(predictions) == set(METHODS),
                "X06": all(misspecification_checks.values()), "X07": duplicate["instrument_state"] == "valid"}
            record = {"source_unit_sha256": file_digest(base/"units"/(row["unit_id"]+"_points.json")),
                "source_packet": packet_hash, "condition": row["condition"], "checks": checks,
                "source_public_sha256": row["public_hash"], "private_poison_variants": variants,
                "fixed_bytes_reader_outputs": observations, "private_reads_denied": denied,
                "collision": {"checks": collision_checks, "witness": collision_witness, "requests": collision.frames},
                "misspecification": {"checks": misspecification_checks, "witness": misspecification_witness, "requests": changed_world.frames},
                "duplicates": duplicate, "scope": "actual archived reader bytes plus source-specific known controls; no additional independent makers"}
            write(destination, record)
            if not all(checks.values()):
                raise ValueError("archive consumer attack failed; evidence retained")
            records.append(record)
    runtime = None if fixture else read(output/"private/runtime-control/RECEIPT.json")
    if not fixture and (runtime["source_identity"] != packet_hash or runtime["instrument_state"] != "valid"):
        raise ValueError("archive runtime source-specific join missing")
    result = {"card_id": "B02", "packet_hash": packet_hash, "instrument_state": "valid", "conditions": len(records),
        "attack_checks": {attack: all(record["checks"][attack] for record in records) for attack in ["X01", "X03", "X04", "X06", "X07"]},
        "X08": "fixture scope only" if fixture else "actual archive child interruption, preserved units and clocks, finite follow-up completed",
        "sampling": sampling, "scope": "new archive source checks; source M02 physics/scoring independently checked for every candidate",
        "same_input": "all four native reader arms receive the same public bytes; the separately declared rejected-work ablation receives less evidence",
        "compute": "equal produced-primitive budget; likelihood terms, actual resources and verification work kept separate"}
    write(receipt, result)
    return result
