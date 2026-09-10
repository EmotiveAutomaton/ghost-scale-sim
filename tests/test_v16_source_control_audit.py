from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v16.source_control_audit import resource_totals


def test_missing_platform_measurement_is_not_fabricated_as_zero_or_full_availability():
    rows = [{"wall_seconds": .2, "parent_cpu_seconds": .1, "reader_cpu_seconds": None,
        "resident_bytes": None, "peak_resident_bytes": None},
        {"wall_seconds": .3, "parent_cpu_seconds": .2, "reader_cpu_seconds": .05,
        "resident_bytes": 4096, "peak_resident_bytes": 8192}]
    result = resource_totals(rows)
    assert result["requests"] == 2 and result["wall_seconds"] == .5
    assert result["unavailable_reader_cpu_requests"] == 1
    assert result["reader_cpu_seconds"] == .05
    assert result["maximum_observed_peak_resident_bytes"] == 8192
    for key, invalid in [("wall_seconds", -1), ("reader_cpu_seconds", float("nan")), ("resident_bytes", -1)]:
        bad = deepcopy(rows)
        bad[1][key] = invalid
        with pytest.raises(ValueError):
            resource_totals(bad)
