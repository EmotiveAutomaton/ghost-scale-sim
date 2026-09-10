import errno
import json
import os
import pytest
from ghostscale.validation.soundingline.v16 import runtime_status_read as status


def test_status_reader_reads_a_complete_record_and_leaves_replacement_available(tmp_path):
    path = tmp_path/"RUNNER_STATUS.json"
    path.write_text('{"completed_units": 3}', encoding="utf-8")
    assert status.read_status(path) == {"completed_units": 3}
    replacement = tmp_path/"replacement.json"
    replacement.write_text('{"completed_units": 4}', encoding="utf-8")
    os.replace(replacement, path)
    assert status.read_status(path) == {"completed_units": 4}


def test_status_reader_recovers_transient_access_but_exhausts_a_permanent_denial(tmp_path, monkeypatch):
    path = tmp_path/"RUNNER_STATUS.json"
    remaining = 2
    def denied_then_ready(path):
        nonlocal remaining
        if remaining:
            remaining -= 1
            raise PermissionError(errno.EACCES, "known transient sharing denial")
        return b'{"completed_units": 1}'
    monkeypatch.setattr(status, "_bytes", denied_then_ready)
    assert status.read_status(path, attempts=3, delay=0)["completed_units"] == 1
    remaining = 3
    with pytest.raises(PermissionError):
        status.read_status(path, attempts=3, delay=0)
    assert remaining == 0


def test_status_reader_does_not_retry_missing_corrupt_or_scientific_records(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="mutable heartbeat"):
        status.read_status(tmp_path/"unit_points.json")
    with pytest.raises(FileNotFoundError):
        status.read_status(tmp_path/"RUNNER_STATUS.json")
    path = tmp_path/"RUNNER_STATUS.json"
    path.write_bytes(b"corrupt known fixture")
    with pytest.raises(json.JSONDecodeError):
        status.read_status(path)
