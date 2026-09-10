from pathlib import Path
from zipfile import BadZipFile
import pytest
from ghostscale.validation.soundingline.v16.raw_archive import manifest, create, identity
from ghostscale.validation.soundingline.v16.records import write, digest
from ghostscale.validation.soundingline.v16.archive_coverage import coverage, snapshot


def batch(repo, archive, paths, name):
    plan = manifest(repo, paths, scope={"known_fixture": True})
    target = archive/name
    receipt = create(repo, target, plan)
    record = archive/(name+".json")
    write(record, {"execution_state": "completed", "instrument_state": "valid", "chunks": [{
        "relative_archive": name+"/raw.zip", "archive_identity": receipt["archive_identity"], "plan_sha256": digest(plan)}]})
    return record


def test_missing_and_obsolete_members_cannot_pass_current_raw_coverage(tmp_path):
    repo, archive = tmp_path/"repo", tmp_path/"archive"
    repo.mkdir()
    first, second = repo/"one_points.json", repo/"two_points.json"
    write(first, {"value": 1})
    write(second, {"value": 2})
    initial = batch(repo, archive, [first], "first")
    required = {path.name: identity(path) for path in [first, second]}
    result = coverage(archive, [initial], required)
    assert not result["verified_complete_accessible"] and result["missing_or_different_files"] == [second.name]
    write(first, {"value": 3}, immutable=False)
    required[first.name] = identity(first)
    result = coverage(archive, [initial], required)
    assert result["covered_files"] == 0
    supplement = batch(repo, archive, [first, second], "supplement")
    result = coverage(archive, [initial, supplement], required)
    assert result["verified_complete_accessible"] and result["covered_files"] == 2
    assert not coverage(archive, [initial, supplement], required, verify_bytes=False)["verified_complete_accessible"]


def test_corrupted_zip_and_empty_required_inventory_are_rejected(tmp_path):
    repo, archive = tmp_path/"repo", tmp_path/"archive"
    repo.mkdir()
    source = repo/"known_points.json"
    write(source, {"value": 1})
    receipt = batch(repo, archive, [source], "first")
    with pytest.raises(ValueError, match="empty"):
        coverage(archive, [receipt], {})
    (archive/"first/raw.zip").write_bytes(b"corrupt retained fixture")
    with pytest.raises(BadZipFile):
        coverage(archive, [receipt], {source.name: identity(source)})


def test_raw_snapshot_keeps_failures_and_requires_reproduction_inputs(tmp_path):
    root = tmp_path/"results/v16"
    for name in ["pyproject.toml", "uv.lock", "docs/METHODS.md", "docs/versions/v16-acquired-craft/CODING_PACKAGE.md",
                 "results/v16/science/units/one_points.json", "results/v16/failure/FAILED_STATUS.json",
                 "results/v16/RUNNER_STATUS.json", "results/v16/closeout/admin.json"]:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("known retained evidence", encoding="utf-8")
    result = snapshot(tmp_path, root)
    assert "results/v16/science/units/one_points.json" in result
    assert "results/v16/failure/FAILED_STATUS.json" in result
    assert "results/v16/RUNNER_STATUS.json" not in result
    assert "results/v16/closeout/admin.json" not in result
    (tmp_path/"uv.lock").unlink()
    with pytest.raises(ValueError, match="environment record"):
        snapshot(tmp_path, root)
