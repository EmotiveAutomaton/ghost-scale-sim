"""X01 actual consumer checks under private-state, filename and cache changes."""
from collections import Counter
import copy
from pathlib import Path
import uuid
from .consumer_frames import EXTENSIONS as ORIGINAL_EXTENSIONS
from .active_consumers import requests, selected_units
EXTENSIONS = [*ORIGINAL_EXTENSIONS, *["ghostscale.validation.soundingline.v16.inquiry_stable:"+name
    for name in ("choose", "learn_public", "construct_public", "read_maker_public")]]
def scout_units(root):
    yield from selected_units(root, {"R01", "R02", "R03", "R04", "R05"})
from .reader_process import ReaderProcess
from .records import read, write, file_digest, digest, now

DESIGN = {"card_id": "X01", "scope": "fixture",
          "selection": "seed index zero in each amended R01-R05 condition; every recorded reader arm and phase; corrected standalone B01 export",
          "attacks": ["private state and seed replacement", "private file rename",
                      "public opaque alias change", "warm cache after interleaved other requests",
                      "fresh second process"],
          "comparison": "exact JSON prediction equality with original; runtime metrics excluded because not part of scientific result",
          "private_scope": "fixed trusted Python functions under post-import file/process/network guard",
          "not_claimed": "hostile native extension or operating-system sandbox",
          "eligibility": "each consumer receives its own demonstrated receipt; no gate-name-only inheritance",
          "broken_consumer": "a data-leaking fixture must change under private-state replacement",
          "repair_budget": 1}


def alias(value):
    if isinstance(value, dict):
        return {key: uuid.uuid4().hex if key in {"task_id", "lineage_id"} else alias(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [alias(item) for item in value]
    return value


def comparison(actual, expected):
    if actual != expected:
        raise ValueError("public-only reader differs under access/cache attack")


def one(reader, request):
    return reader.request(request["kind"], request["public"], **request["options"])


def execute(root, output, heartbeat):
    selected = list(scout_units(root))
    if not selected:
        raise ValueError("empty access battery cannot pass")
    frames = []
    for packet, path, row in selected:
        for index, request in enumerate(requests(path.parent.parent, row)):
            frames.append({"packet": packet, "card_id": row["card_id"], "condition": row["condition"],
                           "unit_id": row["unit_id"], "source_sha256": file_digest(path),
                           "request_index": index, "request": request})
    write(output/"SELECTION.json", {"rule": DESIGN["selection"], "n_units": len(selected),
                                    "n_requests": len(frames), "requests_sha256": digest(frames)})
    # Private fixture files are intentionally present next to the reader's cwd.
    # Neither their content nor their filenames enter a scientific request.
    first_dir = output/"public/reader-workspace"
    canary = first_dir/"evaluator-seed-001.json"
    write(canary, {"secret_library": [0, 1], "seed": 1, "condition": "first"})
    counts = Counter()
    with ReaderProcess(first_dir, extensions=EXTENSIONS) as reader:
        leaked_before = read(canary)["secret_library"]  # intentionally broken external fixture
        for ordinal, item in enumerate(frames):
            expected = item["request"]["result"]
            actual = one(reader, item["request"])
            comparison(actual, expected)
            renamed = copy.deepcopy(item["request"])
            renamed["public"] = alias(renamed["public"])
            comparison(one(reader, renamed), expected)
            counts[item["card_id"]] += 1
            if ordinal % 50 == 0:
                heartbeat(completed_units=ordinal+1, planned_units=3*len(frames), attack_phase="baseline and aliases")
        replaced = first_dir/"evaluator-seed-999.json"
        write(replaced, {"secret_library": [2, 3], "seed": 999, "condition": "second"})
        write(output/"private/CANARY_BEFORE_points.json", read(canary))
        write(canary, read(replaced), immutable=False)
        # The same filename now has different private state; a second filename
        # carries the replacement too. Preserve the original fixture separately.
        if read(canary)["secret_library"] == leaked_before:
            raise ValueError("broken leakage fixture failed to expose its known defect")
        renamed_canary = first_dir/"renamed-evaluator-seed-999.json"
        canary.rename(renamed_canary)
        for forbidden in (canary, renamed_canary):
            try:
                reader.request("_probe_forbidden_read", {"path": str(forbidden.resolve())})
            except RuntimeError as error:
                if "PermissionError" not in str(error):
                    raise
            else:
                raise ValueError("scientific consumer read private canary")
        # Reverse order after all other public calls have populated the caches.
        for ordinal, item in enumerate(reversed(frames)):
            comparison(one(reader, item["request"]), item["request"]["result"])
            if ordinal % 50 == 0:
                heartbeat(completed_units=len(frames)+ordinal+1, planned_units=3*len(frames),
                          attack_phase="private replacement and warm interleaved caches")
    # A fresh process checks that cached results are not acting as hidden evidence.
    with ReaderProcess(output/"public/fresh-reader", extensions=EXTENSIONS) as reader:
        for ordinal, item in enumerate(frames):
            comparison(one(reader, item["request"]), item["request"]["result"])
            if ordinal % 50 == 0:
                heartbeat(completed_units=2*len(frames)+ordinal+1, planned_units=3*len(frames),
                          attack_phase="fresh reader process")
    # The actual standalone B01 interface is a separate consumer. Do not inherit
    # its X01 coverage merely because it copied the same scientific functions.
    from .transfer_stable import Consumer
    transfer = root/"transfer-fixture-2"
    manifest = read(transfer/"PUBLIC_MANIFEST.json")
    transfer_inputs = [(read(transfer/path), read(transfer/"predictions"/(Path(path).stem+".json"))["result"])
                       for path in manifest["observations"]]
    for cold in range(2):
        with Consumer(transfer/"public/consumer") as consumer:
            for public, expected in transfer_inputs if cold == 0 else reversed(transfer_inputs):
                comparison(consumer.request(public)["result"], expected)
                if cold == 0:
                    comparison(consumer.request(alias(public))["result"], expected)
    counts["B01"] = len(transfer_inputs)
    result = {"card_id": "X01", "execution_state": "completed", "instrument_state": "valid",
              "n_units": len(selected), "n_source_requests": len(frames),
              "actual_reader_requests": 4*len(frames)+2+3*len(transfer_inputs),
              "consumer_cards": dict(sorted(counts.items())),
              "broken_leakage_detected": True, "private_reads_denied": 2,
              "prediction_comparison": "exact; all original, alias, cache and fresh-process checks",
              "scope": DESIGN["private_scope"], "campaign_complete": False, "completed_at": now()}
    write(output/"COMPLETION.json", result)
    write(output/"RAW_MANIFEST.json", {"files": {
        str(path.relative_to(output)).replace("\\", "/"): file_digest(path)
        for directory in ["public", "private"] for path in sorted((output/directory).rglob("*.json"))},
        "record_type": "access fixture; no new independent maker sample",
        "source_requests_sha256": digest(frames)})
    heartbeat(completed_units=3*len(frames), planned_units=3*len(frames), execution_state="completed", result=result)
    return result
