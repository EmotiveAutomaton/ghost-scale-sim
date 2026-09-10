import pytest
from ghostscale.validation.soundingline.v16.records import write
from runners.verify_v16_suite import fixture_integrity


def test_fixture_archive_checks_non_json_source_and_rejects_missing_or_extra_bytes(tmp_path):
    base = tmp_path/"access-attack-fixture-1"
    path = base/"public/source.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x = 1\n")
    from hashlib import sha256
    mapping = {"public/source.py": sha256(path.read_bytes()).hexdigest()}
    write(base/"RAW_MANIFEST.json", {"files": mapping})
    write(base/"COMPLETION.json", {"execution_state": "completed", "instrument_state": "valid"})
    result = fixture_integrity(tmp_path, ["access-attack-fixture-1"])
    assert result[0]["files"] == 1
    assert result[0]["scientific_maker_sample_size"] == "not applicable"
    path.write_bytes(b"x = 2\n")
    with pytest.raises(ValueError, match="raw bytes changed"):
        fixture_integrity(tmp_path, ["access-attack-fixture-1"])
    path.write_bytes(b"x = 1\n")
    write(base/"private/unmanifested.json", {"hidden": True})
    with pytest.raises(ValueError, match="unmanifested"):
        fixture_integrity(tmp_path, ["access-attack-fixture-1"])
