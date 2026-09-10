"""A portable archive must pass its own frozen-source check after extraction."""
import zipfile
import pytest
from ghostscale.validation.soundingline.v16.records import write, file_digest
from ghostscale.validation.soundingline.v16.record_integrity import source_locks
from ghostscale.validation.soundingline.v16.raw_archive import manifest, create
from runners.finalize_v16_replay import implementation_files


def test_extracted_inventory_keeps_locked_tests_and_provenance_and_rejects_drift(tmp_path):
    repo = tmp_path/"source"
    root = repo/"results/v16"
    names = ["ghostscale/example.py", "runners/replay_v16_example.py",
        "tests/test_packet_admission.py", "results/v16/provenance/INTEGRITY.json",
        "pyproject.toml", "uv.lock", "docs/versions/v16-acquired-craft/CODING_PACKAGE.md"]
    for name in names:
        path = repo/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("retained known source\n", encoding="utf-8")
    locked = {name: file_digest(repo/name) for name in names[:4]}
    packet_path = root/"packets/known.json"
    write(packet_path, {"packet_hash": "known-source-closure", "identity": {
        "packet_id": "known", "files": locked}})
    paths = implementation_files(root, repo)
    plan = manifest(repo, paths, scope={"known_portable_dependency_closure": True})
    created = create(repo, tmp_path/"archive", plan)
    assert created["verified_members"] == len(names)+1
    extracted = tmp_path/"extracted"
    with zipfile.ZipFile(tmp_path/"archive/raw.zip") as archive:
        archive.extractall(extracted)
    checked = source_locks(extracted/"results/v16", extracted)
    assert checked["working_source"] == "valid" and checked["unique_scientific_files"] == 4
    for name, expected in locked.items():
        assert file_digest(extracted/name) == expected
    marker = repo/names[3]
    original = marker.read_bytes()
    marker.write_text("changed provenance\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dependency is missing, escaping or changed"):
        implementation_files(root, repo)
    marker.write_bytes(original)
    locked["../outside.json"] = file_digest(marker)
    write(packet_path, {"packet_hash": "escaping-source", "identity": {
        "packet_id": "known", "files": locked}}, immutable=False)
    with pytest.raises(ValueError, match="dependency is missing, escaping or changed"):
        implementation_files(root, repo)
