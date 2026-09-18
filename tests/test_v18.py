"""V18 known answers, information limits, budgets, resume and source-to-report."""
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ghostscale.validation.soundingline.v16.records import canonical, digest, read
from ghostscale.validation.soundingline.v16.runtime import local_owner
from ghostscale.validation.soundingline.v16.world import execute
from ghostscale.validation.soundingline.v16.craft import construct
from ghostscale.validation.soundingline.v18.study import (
    SCHEMA, offered_view, select_offers, learn_public, make_case, evaluate, use_library,
)
from ghostscale.validation.soundingline.v18.verify import execution, search, verify_case, verify_row, reconstruct

NAMESPACE = "v18-discarded-development-1"


def case(c=0, h=0):
    return make_case(NAMESPACE, c, h, 8)


def test_offered_projection_and_private_swap_do_not_reach_selection_or_learning():
    c = case()
    assert all(set(x) == {"offer", "topic"} for x in c["offered"]["offers"])
    original_selection = select_offers(canonical(c["offered"]))
    original = learn_public(canonical(c["acquisitions"]["broad"]["request"]))
    altered = deepcopy(c)
    altered["private"] = {"purpose": "wrong purpose", "future": "unrelated"}
    altered["transfer_targets"] = {"forbidden_future": [15]}
    assert select_offers(canonical(altered["offered"])) == original_selection
    assert learn_public(canonical(altered["acquisitions"]["broad"]["request"])) == original
    invalid = deepcopy(c["offered"]); invalid["offers"][0]["success"] = True
    with pytest.raises(ValueError, match="private"):
        select_offers(canonical(invalid))
    invalid = deepcopy(c["acquisitions"]["broad"]["request"]); invalid["target"] = 3
    with pytest.raises(ValueError, match="schema"):
        learn_public(canonical(invalid))


def test_all_eight_development_constructors_keep_pairing_and_targets():
    accidental_exposures = 0
    for c in range(8):
        for h in range(2):
            sample = case(c, h)
            verify_case(sample)
            for name, expected in (("broad", [8, 8]), ("focused", [12, 4])):
                topics = [x["topic"] for x in sample["acquisitions"][name]["request"]["processed"]]
                assert [topics.count(0), topics.count(1)] == expected
                processed = sample["acquisitions"][name]["request"]["processed"]
                for trial in processed:
                    if trial["artifact"] in sample["transfer_targets"]["changed"]:
                        accidental_exposures += 1
                        assert trial["feedback"] is False
                        assert trial["program"] not in sample["acquisitions"][name]["result"]["library"]
    assert accidental_exposures > 0  # Retained, not silently filtered or resampled.


def record(offer, program, feedback):
    return {"offer": offer, "topic": 0, "instruction": program, "proposed": program, "program": program,
            "artifact": execute(program).artifact, "feedback": feedback}


def test_learning_empty_failures_threshold_and_deterministic_tie():
    records = [record(i, [0, 1], False) for i in range(16)]
    assert learn_public(canonical({"schema": SCHEMA, "processed": records}))["library"] == []
    for i in range(2): records[i]["feedback"] = True
    assert learn_public(canonical({"schema": SCHEMA, "processed": records}))["library"] == []
    records[2]["feedback"] = True
    for i in range(3, 6): records[i] = record(i, [2, 3], True)
    learned = learn_public(canonical({"schema": SCHEMA, "processed": records}))
    assert learned["library"] == [[0, 1]]
    assert learned["top_ties"] == [[0, 1], [2, 3]]
    assert learn_public(canonical({"schema": SCHEMA, "processed": list(reversed(records))})) == learned


@pytest.mark.parametrize("actions,start,expected", [
    ([0, 1], 0, (3, True, 2)), ([0, 4], 0, (0, True, 2)),
    ([8], 0, (0, False, 1)), ([0, 1, 2, 3], 0, (7, False, 3)), ([], 9, (9, True, 0)),
])
def test_hand_calculated_executor(actions, start, expected):
    native, independent = execute(actions, start=start), execution(actions, start)
    assert (native.artifact, native.legal, native.primitive_cost) == expected
    assert (independent["artifact"], independent["legal"], independent["primitive_cost"]) == expected


def test_known_search_and_checking_costs():
    assert construct(1, [], primitive_budget=1)["program"] == [0]
    assert construct(2, [], primitive_budget=1)["search_timeout"]
    assert construct(3, [[0, 1]], primitive_budget=2)["search_primitives"] == 2
    for target in range(16):
        for library in ([], [[0, 1]], [[2, 3]], [[8, 0]]):
            for budget in (0, 1, 2, 8, 32, 128):
                assert construct(target, library, primitive_budget=budget) == search(target, library, budget)
    base = {"schema": SCHEMA, "library": [[0, 1]], "target": 5, "budget": 32, "checking": True, "extra_search": False}
    checked = use_library(canonical(base))
    assert checked["active_library"] == []
    assert checked["costs"]["checking_primitives"] == 2
    assert checked["costs"]["search_envelope"] == 30
    assert checked["checks"][1]["goal_error_after"] > checked["checks"][1]["goal_error_before"]
    extra = use_library(canonical({**base, "extra_search": True}))
    assert extra["costs"]["search_envelope"] == 32
    empty = use_library(canonical({**base, "library": []}))
    assert empty["costs"]["checking_primitives"] == 0 and empty["checks"] == []


def test_independent_trace_verifier_rejects_corrupt_outcome_and_cost():
    sample = case()
    rows = evaluate(sample, 32)
    assert rows == evaluate(json.loads(canonical(sample)), 32)
    assert len(rows) == 42
    for row in rows: verify_row(sample, row)
    bad = deepcopy(rows[0]); bad["success"] = not bad["success"]
    with pytest.raises(ValueError, match="outcome"): verify_row(sample, bad)
    bad = deepcopy(rows[0]); bad["costs"]["search_primitives"] += 1
    with pytest.raises(ValueError, match="cost"): verify_row(sample, bad)
    bad = deepcopy(rows[0]); bad["submission"]["attempted_programs"] = []
    with pytest.raises(ValueError, match="trace"): verify_row(sample, bad)


def command(*args, expected=0):
    result = subprocess.run([sys.executable, "-B", "-m", "runners.run_v18", *map(str, args)],
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == expected, result.stdout + result.stderr
    return result


def freeze_fixture(root, start):
    command("freeze", "--root", root, "--development", "--constructors", "8", "--histories", "2",
            "--namespace", NAMESPACE, "--started-at", start)


def test_literal_resume_no_new_draws_and_source_to_report(tmp_path):
    start = datetime.now(timezone.utc).isoformat()
    resumed, whole = tmp_path/"resumed", tmp_path/"whole"
    freeze_fixture(resumed, start); freeze_fixture(whole, start)
    command("run", "--root", resumed, "--stop-after-blocks", "1")
    retained = {p.name: p.read_bytes() for p in (resumed/"raw").glob("*.gz")}
    assert len(retained) == 1
    command("run", "--root", resumed); command("run", "--root", whole)
    assert all((resumed/"raw"/name).read_bytes() == data for name, data in retained.items())
    assert {p.name: p.read_bytes() for p in (resumed/"raw").glob("*.gz")} == {p.name: p.read_bytes() for p in (whole/"raw").glob("*.gz")}
    summary, units = reconstruct(resumed)
    assert summary["counts"]["transfer_executions"] == 1344
    assert summary["counts"]["acquisition_histories"] == 16
    assert sum(a["accidental_changed_endpoint_exposures"] for a in summary["acquisition"]) == sum(
        trial["artifact"] in sample["transfer_targets"]["changed"]
        for sample in [case(c, h) for c in range(8) for h in range(2)]
        for a in sample["acquisitions"].values() for trial in a["request"]["processed"])
    before = {p.name: p.read_bytes() for p in (resumed/"blocks").glob("*.json")}
    command("run", "--root", resumed)
    assert before == {p.name: p.read_bytes() for p in (resumed/"blocks").glob("*.json")}
    output = tmp_path/"report"
    report = subprocess.run([sys.executable, "-B", "-m", "runners.report_v18", "--root", str(resumed),
                             "--output", str(output), "--replay", "--archive"], text=True, capture_output=True, timeout=90)
    assert report.returncode == 0, report.stdout + report.stderr
    assert read(output/"VERIFICATION.json")["full_execution_replay_rows"] == 1344
    import zipfile
    extracted = tmp_path/"extracted"
    with zipfile.ZipFile(output/"replay.zip") as archive: archive.extractall(extracted)
    portable = subprocess.run([sys.executable, "-B", "-m", "runners.report_v18", "--root", "run", "--output", "reproduced", "--replay"],
                              cwd=extracted, text=True, capture_output=True, timeout=90)
    assert portable.returncode == 0, portable.stdout + portable.stderr
    assert read(extracted/"reproduced/COMPARISONS.json") == summary
    # A changed retained block is never accepted even after a completion receipt.
    raw = next((resumed/"raw").glob("*.gz")); raw.write_bytes(raw.read_bytes() + b"corrupt")
    command("run", "--root", resumed, expected=1)
    with pytest.raises(ValueError, match="checksum"): reconstruct(resumed)
    plan = read(whole/"PLAN.json"); plan["sources"]["runners/run_v18.py"] = "changed"
    (whole/"PLAN.json").write_bytes(canonical(plan))
    command("run", "--root", whole, expected=1)


def test_owner_lock_rejects_second_literal_worker(tmp_path):
    freeze_fixture(tmp_path, datetime.now(timezone.utc).isoformat())
    with local_owner(tmp_path):
        result = command("run", "--root", tmp_path, expected=1)
        assert "another supervisor" in result.stderr
    assert not (tmp_path/"STATUS.json").exists()
