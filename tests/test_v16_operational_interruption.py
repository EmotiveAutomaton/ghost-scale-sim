import pytest
from ghostscale.validation.soundingline.v16.operational_interruption import control, stable_bytes


def test_general_cli_really_interrupts_and_resumes_its_frozen_active_job(tmp_path):
    result = control(tmp_path)
    assert result["instrument_state"] == "valid"
    assert all(result["checks"].values())


def test_post_termination_read_retries_only_bounded_permission_sharing():
    class SharedFile:
        def __init__(self, failures):
            self.remaining = failures
        def read_bytes(self):
            self.remaining -= 1
            if self.remaining >= 0:
                raise PermissionError("known transient sharing fixture")
            return b"preserved status"
    assert stable_bytes(SharedFile(2)) == b"preserved status"
    with pytest.raises(PermissionError):
        stable_bytes(SharedFile(21))
