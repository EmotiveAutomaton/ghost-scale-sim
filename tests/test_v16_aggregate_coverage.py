import pytest
from ghostscale.validation.soundingline.v16.aggregate_coverage import PHASES, require_phases, native_reports
from ghostscale.validation.soundingline.v16.records import write, file_digest


def test_full_calculation_coverage_requires_all_phases():
    require_phases(dict.fromkeys(PHASES))
    with pytest.raises(ValueError, match="eleven"):
        require_phases(dict.fromkeys(PHASES-{"confirmation_power"}))


def test_native_join_rejects_missing_summary_bad_denominator_and_erased_invalidity(tmp_path):
    root = tmp_path/"results/v16"
    summary = root/"inquiry-scout-1/R01/SUMMARY.json"
    write(summary, {"card_id": "R01"})
    write(summary.parent/"COMPLETION.json", {"execution_state": "failed"})
    write(summary.parent/"RAW_MANIFEST.json", {"files": {"units/a_points.json": "a", "units/b_points.json": "b"}})
    auditor = tmp_path/"ghostscale/validation/soundingline/v16/audit_inquiry.py"
    auditor.parent.mkdir(parents=True)
    auditor.write_text("known independent auditor", encoding="utf-8")
    design = {"card_id": "R01", "conditions": [{"id": "one"}]}
    report = {"summary_sha256": file_digest(summary), "card_id": "R01", "registered_conditions": 1,
        "scientific_instrument_state": "invalid preserved original", "qualification": "Invalidity retained",
        "auditor_source_sha256": file_digest(auditor), "calculation": {"execution_state": "completed", "instrument_state": "valid",
            "all_reported_aggregates_reproduced": True, "n_raw_units": 2}}
    reports = {summary.relative_to(root).as_posix(): report}
    work = [(summary, design, 2, True)]
    assert native_reports(root, tmp_path, reports, work)["condition_records"] == 2
    with pytest.raises(ValueError, match="omitted"):
        native_reports(root, tmp_path, {}, work)
    report["calculation"]["n_raw_units"] = 1
    with pytest.raises(ValueError, match="denominator"):
        native_reports(root, tmp_path, reports, work)
    report["calculation"]["n_raw_units"] = 2
    report["scientific_instrument_state"] = "valid"
    with pytest.raises(ValueError, match="invalid"):
        native_reports(root, tmp_path, reports, work)
