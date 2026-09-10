from pathlib import Path
import pytest
from runners import finalize_v16_raw_archive as finalizer
from ghostscale.validation.soundingline.v16.records import write, read, file_digest, digest
from ghostscale.validation.soundingline.v16.raw_archive import manifest, create
from ghostscale.validation.soundingline.v16 import operational_queue


def fixture(tmp_path, monkeypatch):
    repo, archive = tmp_path/"repo", tmp_path/"archive"
    root = repo/"results/v16"
    paths = ["pyproject.toml", "uv.lock", "docs/METHODS.md", "docs/versions/v16-acquired-craft/CODING_PACKAGE.md",
             "runners/finalize_v16_raw_archive.py", "ghostscale/validation/soundingline/v16/archive_coverage.py"]
    for name in paths:
        target = repo/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("known archival fixture", encoding="utf-8")
    monkeypatch.setattr(finalizer, "REPO", repo)
    monkeypatch.setattr(finalizer, "__file__", str(repo/paths[-2]))
    monkeypatch.setattr(finalizer, "remaining_seconds", lambda root: 3600)
    monkeypatch.setattr(operational_queue, "remaining_seconds", lambda root: 3600)
    write(root/"packets/known.json", {"packet_hash": "known-fixture", "identity": {
        "packet_id": "known", "files": {paths[-1]: file_digest(repo/paths[-1])}}})
    inputs = {"proofs": {}}
    baseline_path = root/"known-setup/baseline_points.json"
    write(baseline_path, {"files": {paths[-1]: file_digest(repo/paths[-1])}})
    for name, flag in [("independent_aggregates", "full_aggregate_regeneration"), ("scientific_replay", "whole_unit_replay")]:
        path = root/"closeout"/(name+".json")
        write(path, {"execution_state": "completed", "instrument_state": "valid", flag: True,
            "raw_input_baseline": {"path": baseline_path.relative_to(root).as_posix(), "sha256": file_digest(baseline_path)}})
        inputs["proofs"][name] = {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)}
    for name in ["boundary-expansion-1", "boundary-controls-1", "expansion-closure", "confirmation"]:
        write(root/name/"COMPLETION.json", {"execution_state": "completed", "instrument_state": "valid"})
    measured = root/"known-setup/measurement.json"
    write(measured, {"wall_seconds": 1})
    forecast = root/"closeout/FORECAST.json"
    write(forecast, {"kind": "measured forecast", "measured_seconds": 1, "source_path": measured.relative_to(root).as_posix(),
        "source_sha256": file_digest(measured), "conservative_seconds": 10, "closeout_reserve_seconds": 60})
    inputs["forecast"] = {"path": forecast.relative_to(root).as_posix(), "sha256": file_digest(forecast)}
    input_path = root/"closeout/INPUTS.json"
    write(input_path, inputs)
    plan = manifest(repo, [measured], scope={"known_fixture": True})
    receipt = create(repo, archive/"prior", plan)
    write(archive/"RECEIPTS/prior.json", {"execution_state": "completed", "instrument_state": "valid", "chunks": [{
        "relative_archive": "prior/raw.zip", "archive_identity": receipt["archive_identity"], "plan_sha256": digest(plan)}]})
    return root, archive, input_path


def test_final_raw_archive_covers_residuals_and_preserves_completed_attempt(tmp_path, monkeypatch):
    root, archive, inputs = fixture(tmp_path, monkeypatch)
    output = root/"closeout/known-final"
    result = finalizer.run(root, archive, output, inputs)
    assert result["verified_complete_accessible"] and result["current_source_snapshot_unchanged"]
    assert result["verified_chunks"] > 1
    assert not read(output/"coverage_points.json")["missing_or_different_files"]
    before = {str(path): file_digest(path) for path in output.rglob("*.json")}
    with pytest.raises(ValueError, match="new retained attempt"):
        finalizer.run(root, archive, output, inputs)
    assert before == {str(path): file_digest(path) for path in output.rglob("*.json")}


def test_final_raw_archive_refuses_incomplete_proof_and_source_drift(tmp_path, monkeypatch):
    root, archive, inputs = fixture(tmp_path, monkeypatch)
    evidence = read(inputs)["proofs"]["scientific_replay"]
    write(root/evidence["path"], {"execution_state": "completed", "instrument_state": "valid", "whole_unit_replay": False}, immutable=False)
    changed = read(inputs)
    changed["proofs"]["scientific_replay"]["sha256"] = file_digest(root/evidence["path"])
    write(inputs, changed, immutable=False)
    with pytest.raises(ValueError, match="completed independent"):
        finalizer.run(root, archive, root/"closeout/partial", inputs)
    write(root/evidence["path"], {"execution_state": "completed", "instrument_state": "valid", "whole_unit_replay": True}, immutable=False)
    changed["proofs"]["scientific_replay"]["sha256"] = file_digest(root/evidence["path"])
    write(inputs, changed, immutable=False)
    original = finalizer.snapshot
    calls = 0
    def changing(repo, results):
        nonlocal calls
        calls += 1
        if calls == 2:
            write(root/"confirmation/new_points.json", {"unarchived": True})
        return original(repo, results)
    monkeypatch.setattr(finalizer, "snapshot", changing)
    with pytest.raises(ValueError, match="source changed"):
        finalizer.run(root, archive, root/"closeout/drift", inputs)
    assert not (root/"closeout/drift/RECEIPT.json").exists()
    assert list((archive/"final-residuals").rglob("raw.zip"))


def test_raw_archive_cannot_substitute_changed_inputs_for_reproduced_arithmetic():
    finalizer.baseline_matches({"raw": {"sha256": "original", "bytes": 8}}, {"raw": "original"})
    for current in [{}, {"raw": {"sha256": "changed", "bytes": 8}}]:
        with pytest.raises(ValueError, match="regenerated input baseline"):
            finalizer.baseline_matches(current, {"raw": "original"})
