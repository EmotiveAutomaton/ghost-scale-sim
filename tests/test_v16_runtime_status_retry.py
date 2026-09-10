import time
import pytest
from ghostscale.validation.soundingline.v16 import runtime_status_retry as status
from ghostscale.validation.soundingline.v16.records import read
from ghostscale.validation.soundingline.v16.expansion_interruption import fixture_campaign


def denied():
    error = PermissionError("known Windows status sharing failure")
    error.winerror = 5
    return error


def test_transient_status_failure_retries_but_scientific_records_never_do(tmp_path, monkeypatch):
    real = status._write
    calls = []
    def writer(path, value, **options):
        calls.append(path.name)
        if len(calls) <= 2:
            raise denied()
        return real(path, value, **options)
    monkeypatch.setattr(status, "_write", writer)
    status.retry_status_write(tmp_path/"RUNNER_STATUS.json", {"completed_units": 7}, immutable=False)
    assert len(calls) == 3
    assert read(tmp_path/"RUNNER_STATUS.json")["completed_units"] == 7
    assert len((tmp_path/"operations/status-publication-retries.jsonl").read_text().splitlines()) == 2
    calls.clear()
    with pytest.raises(PermissionError):
        status.retry_status_write(tmp_path/"unit_points.json", {"value": 1})
    assert len(calls) == 1


def test_persistent_failure_stops_at_its_bound_and_timer_uses_retry(tmp_path, monkeypatch):
    real = status._write
    monkeypatch.setattr(status, "MAX_ATTEMPTS", 2)
    monkeypatch.setattr(status, "_write", lambda *args, **kwargs: (_ for _ in ()).throw(denied()))
    with pytest.raises(PermissionError):
        status.retry_status_write(tmp_path/"failed/RUNNER_STATUS.json", {}, immutable=False)
    assert len((tmp_path/"failed/operations/status-publication-retries.jsonl").read_text().splitlines()) == 2
    root = tmp_path/"campaign"
    fixture_campaign(root)
    calls = []
    def occasionally_denied(path, value, **options):
        calls.append(value.get("heartbeat_source"))
        if len(calls) in {1, 3}:
            raise denied()
        return real(path, value, **options)
    monkeypatch.setattr(status, "_write", occasionally_denied)
    with status.supervisor(root, "pilot", heartbeat_interval=.01) as heartbeat:
        time.sleep(.13)
        heartbeat(completed_units=2, planned_units=4)
    saved = read(root/"RUNNER_STATUS.json")
    assert saved["completed_units"] == 2 and not saved["unit_loop_complete"]
    assert any(name and name.startswith("liveness timer") for name in calls)


@pytest.mark.parametrize("card", ["K01", "R02"])
def test_actual_resilient_cli_interrupts_and_resumes_unchanged_known_packet(card, tmp_path):
    from ghostscale.validation.soundingline.v16.status_retry_interruption import control
    assert control(tmp_path, card)["instrument_state"] == "valid"
