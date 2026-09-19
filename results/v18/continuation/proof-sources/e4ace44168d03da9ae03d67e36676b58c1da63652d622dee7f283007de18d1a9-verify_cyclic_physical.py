"""Verify completed QUEUE-13 without restarting science or modifying its source."""
import gzip
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path


OUTER = Path(__file__).resolve().parents[3]
LOCAL = OUTER / ".local" / "v18-1"
REPO = OUTER / "ghost-scale-sim"
SOURCE = LOCAL / "source-cyclic-physical-1"
ROOT = LOCAL / "g2-cyclic-physical-1"
REPORT = LOCAL / "g2-cyclic-physical-report-1"
EXPECTED_EVENT_ID = "1422f442adad7966be9608aff697cfca5c71989191cdfe93220898c80999839c"

REPORT.mkdir(exist_ok=True)
sys.path.insert(0, str(SOURCE))

from ghostscale.validation.soundingline.v18_1 import runtime, verify
from ghostscale.validation.soundingline.v16.records import digest, file_digest, read, write


assert Path(verify.__file__).resolve().is_relative_to(SOURCE)
plan = read(ROOT / "PLAN.json")
complete = read(ROOT / "COMPLETE.json")
queue_status = read(LOCAL / "QUEUE-13-STATUS.json")
event = read(LOCAL / "QUEUE-13-EVENT.json")
event_id = hashlib.sha256(json.dumps(["QUEUE-13", event], sort_keys=True).encode()).hexdigest()
assert event_id == EXPECTED_EVENT_ID
assert event["kind"] == "queue_complete" and queue_status["state"] == "queue_complete"
assert event["jobs"] == queue_status["jobs"]
assert complete["plan_sha256"] == file_digest(ROOT / "PLAN.json")
assert event["jobs"][0]["complete_sha256"] == file_digest(ROOT / "COMPLETE.json")
assert len(complete["blocks"]) == 96
assert file_digest(ROOT / "INPUTS.json.gz") == plan["inputs_sha256"]

cases = json.loads(gzip.decompress((ROOT / "INPUTS.json.gz").read_bytes()))
by_id = {case["case_id"]: digest(case) for case in cases}
seen = set()
for relative, expected in plan["sources"].items():
    assert file_digest(SOURCE / relative) == expected
    assert file_digest(REPO / relative) == expected
for name in complete["blocks"]:
    block, receipt = runtime.load_block(ROOT, name)
    for unit in block["units"]:
        case_id = unit["case"]["case_id"]
        assert case_id not in seen
        assert digest(unit["case"]) == by_id[case_id]
        seen.add(case_id)
assert seen == set(by_id) and len(seen) == 768

cpu = time.process_time()
wall = time.monotonic()
report = verify.branch_report(ROOT, REPORT)
write(
    REPORT / "LINEAGE.json",
    {
        "passed": True,
        "event_id": event_id,
        "cases": 768,
        "blocks": 96,
        "rows": report["rows_checked"],
        "sources_checked": len(plan["sources"]),
        "frozen_inputs_sha256": plan["inputs_sha256"],
        "complete_sha256": file_digest(ROOT / "COMPLETE.json"),
        "scope": (
            "exact event receipt; every raw case matches the frozen input; all frozen "
            "source bytes match; full independent physics and aggregation pass"
        ),
        "verifier_cpu_seconds": time.process_time() - cpu,
        "verifier_wall_seconds": time.monotonic() - wall,
    },
)

archive = LOCAL / "source-cyclic-physical-1.zip"
with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as zipped:
    for relative, expected in sorted(plan["sources"].items()):
        assert file_digest(SOURCE / relative) == expected
        zipped.write(SOURCE / relative, relative)

print(
    json.dumps(
        {
            "passed": True,
            "event_id": event_id,
            "rows": report["rows_checked"],
            "contexts": report["distinct_context_units"],
            "archive_sha256": file_digest(archive),
        },
        sort_keys=True,
    )
)
