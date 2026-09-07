"""Closure checks must catch corruption and must not invent runtime success."""
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from runners.audit_v15_closure import check_block, waiting_receipt
from ghostscale.validation.soundingline.v15 import coverage as CV


def fixture():
    block = CV.block(0)
    block.update(n_ok=75, digest=CV.block_digest(0))
    for cell in block["cells"]:
        cell["secondary"] = block["secondary"]
        cell["by_architecture"] = {name: {"log_score": -1.0, "accuracy": 0.5,
                                          "likelihood_evaluations": 1.0, "n": 2}
                                    for name in CV.COVERAGE_ARCHITECTURES}
        cell["n_rows"] = 12
        cell["joint_minus_independent"] = 0.0
    return block


def test_complete_block_is_valid():
    assert check_block(fixture(), 0) == []


@pytest.mark.parametrize("corruption", ["duplicate", "incomplete", "settings", "arithmetic", "nonfinite"])
def test_actual_coverage_corruption_is_detected(corruption):
    block = fixture()
    if corruption == "duplicate":
        block["cells"][1] = deepcopy(block["cells"][0])
    elif corruption == "incomplete":
        block["cells"].pop()
    elif corruption == "settings":
        block["cells"][0]["dose"] = 999
    elif corruption == "arithmetic":
        block["cells"][0]["joint_minus_independent"] = 0.5
    else:
        block["cells"][0]["by_architecture"]["particle"]["log_score"] = float("nan")
    assert check_block(block, 0)


def test_waiting_is_capped_at_the_scientific_window_end():
    rows = [{"kind": "confirmation", "t": "2026-09-07T01:00:06Z"},
            {"kind": "integrity", "t": "2026-09-07T15:43:22Z"}]
    registry = {"packet": [1], "results": [1], "amendments": []}
    deadline = {"opened": "2026-08-31T17:43:05Z", "confirmation_end_hour": 166}
    result = waiting_receipt(rows, registry, deadline, datetime(2026, 9, 7, 18, tzinfo=timezone.utc))
    assert result["RUNTIME_FAILED"] is True
    assert result["waiting_hours"] == pytest.approx((14*3600+42*60+59)/3600)
    assert result["to_utc"] == "2026-09-07T15:43:05+00:00"


def test_a_live_or_amended_confirmation_is_not_declared_idle():
    rows = [{"kind": "confirmation", "t": "2026-09-07T01:00:06Z"}]
    deadline = {"opened": "2026-08-31T17:43:05Z", "confirmation_end_hour": 166}
    now = datetime(2026, 9, 7, 18, tzinfo=timezone.utc)
    registry = {"packet": [1, 2], "results": [1], "amendments": []}
    assert waiting_receipt(rows, registry, deadline, now)["reconciled"] is False
    rows.append({"kind": "integrity", "t": "2026-09-07T15:43:22Z"})
    registry.update(results=[1, 2], amendments=[{"added": 2}])
    assert waiting_receipt(rows, registry, deadline, now)["reconciled"] is False


def test_report_cannot_present_failed_runtime_as_success(tmp_path, monkeypatch):
    import json
    from runners import report_v15 as report
    monkeypatch.setattr(report, "v15_dir", lambda: tmp_path)
    monkeypatch.setattr(report, "_verdicts", lambda lane: {})
    monkeypatch.setattr(report.RC, "window", lambda: {})
    monkeypatch.setattr(report.RC, "window_closed", lambda: False)
    (tmp_path / "WORKER_OCCUPANCY.json").write_text(json.dumps({
        "RUNTIME_FAILED": True, "occupancy_ratio": 1.0, "occupancy_ratio_verified": False,
        "waited_for_deadline_hours": 14.7}), encoding="utf-8")
    text = report.build(pass_b=True)
    assert "Runtime contract failed" in text.split("## Pass A")[0]
    assert "not verified as whole-window utilization" in text
    assert "pre-deadline scratch draft" in text
    assert "after the 168-hour window closed" not in text
