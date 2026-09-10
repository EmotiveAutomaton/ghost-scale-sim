"""X02 consumer-specific syntax and physical-output representation checks."""
from collections import Counter
import copy
from pathlib import Path
from .records import read, write, file_digest, digest, now
from .active_consumers import attack_consumers, selected_units, requests
from .reader_process import ReaderProcess
from .recoded_interface import encode, decode, OPERATIONS
from .recoding import (close, reading_observation, recognition_observation, inquiry_observation,
                       inquiry_result_equivalent, pushed_probabilities, SWAP_MOTIFS)
from .transfer_stable import Consumer, envelope

EXTENSION = "ghostscale.validation.soundingline.v16.recoded_interface:public_reader"
INQUIRY_KINDS = {"choose": "inquiry", "learn_public": "inquiry-learning",
                 "construct_public": "inquiry-construction", "read_maker_public": "inquiry-reading"}
DESIGN = {"card_id": "X02", "scope": "fixture", "selection": "seed index zero, every condition, all reader arms and phases",
    "syntax": "two explicit encodings of recorded primitive sequences; whole-program versus one-primitive macro boundaries, renamed definitions",
    "physical_recodings": {"reading": "exchange finite motif families", "inquiry": "output-cell permutation (2,0,3,1)",
                            "recognition": "reverse sixteen physical cell names and transform public world law"},
    "comparison_tolerance": 1e-10,
    "scope_limit": "declared macro interface and stated world isomorphisms; no unknown-grammar or changed-search-order invariance claim",
    "historical_source": "original inquiry remains a preserved failing diagnostic; current consumers use the bounded amendment"}


class RecodingFailure(ValueError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def compare_physical(kind, public, expected, options, call):
    """Compare relevant predictions and uncertainty, not a unique program string."""
    if kind == "reading":
        transformed = reading_observation(public)
        actual = call(kind, transformed, options)
        valid = actual["model_mismatch"] == expected["model_mismatch"]
        if valid and not expected["model_mismatch"]:
            valid = close(pushed_probabilities(expected["future_probabilities"], SWAP_MOTIFS), actual["future_probabilities"])
            history = expected["history_probabilities"]
            valid &= close(None if history is None else [history[index] for index in (0, 2, 1, 3)], actual["history_probabilities"])
            valid &= close(expected["costs"], actual["costs"])
        method = "motif-family physical output recoding"
    elif kind.endswith("recognition:public_reader"):
        transformed = recognition_observation(public, tuple(reversed(range(16))))
        actual = call(kind, transformed, options)
        valid = close(expected, actual)
        method = "sixteen-cell world-law recoding"
    elif "inquiry_stable:" in kind or kind in ("inquiry", "inquiry-learning", "inquiry-construction", "inquiry-reading"):
        logical_kind = INQUIRY_KINDS[kind.split(":")[-1]] if ":" in kind else kind
        permutation = (2, 0, 3, 1)
        transformed = inquiry_observation(public, permutation)
        actual = call(kind, transformed, options)
        valid = inquiry_result_equivalent(logical_kind, expected, actual, public, transformed, permutation)
        method = "four-cell map/noise belief and future-task recoding"
    else:
        return {"scope": "macro representation only; no physical isomorphism declared for this algorithm"}
    if not valid:
        raise RecodingFailure("relevant prediction changed under declared physical recoding",
                              {"kind": kind, "public": public, "expected": expected, "options": options,
                               "transformed_public": transformed, "actual": actual, "method": method})
    return {"method": method, "original_public_sha256": digest(public),
            "recoded_public_sha256": digest(transformed), "result_sha256": digest(actual), "passed": True}


def execute(root, output, heartbeat):
    required = attack_consumers(root)["X02"]
    selected = list(selected_units(root, set(required)-{"B01"}))
    observed = {row["card_id"] for _, _, row in selected}
    if observed != set(required)-{"B01"}:
        raise ValueError("recoding battery has missing actual consumers")
    write(output/"CONSUMERS.json", {"required": required, "source_units": len(selected),
        "rule": "union of immutable commission and current packet adversary declarations"})
    counts, requests_count = Counter(), 0
    receipts = {}
    with ReaderProcess(output/"public/reader", extensions=[EXTENSION, *[kind for kind in OPERATIONS if ":" in kind]]) as worker:
        def call(kind, public, options):
            return worker.request(kind, public, **options)
        for ordinal, (packet_id, path, row) in enumerate(selected):
            checks = []
            for frame in requests(path.parent.parent, row):
                expected = frame["result"]
                encodings = []
                for spelling, split in (("first", False), ("renamed", True)):
                    encoded = encode(frame["public"], spelling, split)
                    actual = worker.request(EXTENSION, encoded, operation=frame["kind"], reader_options=frame["options"])
                    if actual["result"] != expected:
                        write(output/"private/FAILURE_points.json", {"frame": frame, "encoded": encoded, "actual": actual})
                        raise ValueError("actual consumer changed under macro syntax")
                    encodings.append({"encoded_sha256": digest(encoded), **actual["representation"]})
                    requests_count += 1
                physical = compare_physical(frame["kind"], frame["public"], expected, frame["options"], call)
                requests_count += int(physical.get("passed", False))
                checks.append({"kind": frame["kind"], "public_sha256": digest(frame["public"]),
                               "result_sha256": digest(expected), "encodings": encodings, "physical": physical})
                counts[row["card_id"]] += 1
            relative = f'units/{row["unit_id"]}_points.json'
            receipts[relative] = write(output/relative, {"card_id": row["card_id"], "packet": packet_id,
                "source_unit_sha256": file_digest(path), "checks": checks, "instrument_state": "valid"})
            heartbeat(completed_units=ordinal+1, planned_units=len(selected)+1, card_id=row["card_id"],
                      attack_phase="macro interface and declared physical recoding")
    # Test the separately installed corrected export through its actual process.
    transfer = root/"transfer-fixture-2"
    manifest = read(transfer/"PUBLIC_MANIFEST.json")
    transfer_checks = []
    with Consumer(transfer/"public/consumer") as consumer:
        for relative in manifest["observations"]:
            public = read(transfer/relative)
            kind = public["declared_context"]["operation"]
            observation = public["declared_context"]["observation"]
            options = public["declared_context"]["reader_options"]
            expected = read(transfer/"predictions"/f'{public["task_id"]}.json')["result"]
            def call_transfer(operation, observed, reader_options):
                framed = envelope({"kind": operation, "public": observed, "options": reader_options},
                                  public["task_id"], public["lineage_id"])
                response = consumer.request(framed)
                if not response.get("ok"):
                    raise ValueError("standalone recoding consumer error")
                return response["result"]
            representations = []
            for spelling, split in (("first", False), ("renamed", True)):
                recoded, represented = decode(encode(observation, spelling, split))
                actual = call_transfer(kind, recoded, options)
                if actual != expected:
                    raise RecodingFailure("standalone consumer changed under compiled macro syntax",
                                          {"public": public, "recoded": recoded, "actual": actual, "expected": expected})
                representations.append(represented)
                requests_count += 1
            physical = compare_physical(kind, observation, expected, options, call_transfer)
            requests_count += int(physical.get("passed", False))
            transfer_checks.append({"public_sha256": file_digest(transfer/relative), "operation": kind,
                                    "representation": representations, "physical": physical})
    counts["B01"] = len(transfer_checks)
    receipts["private/TRANSFER_points.json"] = write(output/"private/TRANSFER_points.json", transfer_checks)
    report = {"card_id": "X02", "execution_state": "completed", "instrument_state": "valid",
        "consumer_requests_checked": dict(sorted(counts.items())), "actual_reader_requests": requests_count,
        "source_units": len(selected), "scope": DESIGN["scope_limit"], "campaign_complete": False,
        "original_inquiry_failure": "preserved in inquiry-ties-1; no inherited passing credit", "completed_at": now()}
    write(output/"RAW_MANIFEST.json", {"files": receipts, "retention": "through verified archival handoff"})
    write(output/"COMPLETION.json", report)
    heartbeat(completed_units=len(selected)+1, planned_units=len(selected)+1, execution_state="completed", result=report)
    return report
