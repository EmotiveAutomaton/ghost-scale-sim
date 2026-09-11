import pytest
from ghostscale.validation.soundingline.v16 import archive_coverage as archive


def fixture(repo):
    for name in ["pyproject.toml", "uv.lock", "docs/METHODS.md", "docs/versions/v16-acquired-craft/CODING_PACKAGE.md"]:
        path = repo/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"known immutable contract")
    root = repo/"results/v16"
    root.mkdir(parents=True)
    for index, payload in enumerate([b"", b"one", b"large"*300000]):
        (root/(str(index)+"_points.json")).write_bytes(payload)
    return root


def test_parallel_snapshot_matches_serial_bytes_and_sees_new_or_changed_files(tmp_path):
    root = fixture(tmp_path)
    serial = archive.snapshot(tmp_path, root)
    progress = []
    assert archive.snapshot(tmp_path, root, workers=4, report=lambda n,total: progress.append((n,total))) == serial
    assert progress[0][0] == 0 and progress[-1][0] == progress[-1][1]
    (root/"1_points.json").write_bytes(b"changed")
    (root/"new_points.json").write_bytes(b"new")
    changed = archive.snapshot(tmp_path, root, workers=4)
    assert changed != serial
    assert changed == archive.snapshot(tmp_path, root)


def test_parallel_snapshot_propagates_read_failure_and_refuses_unbounded_workers(tmp_path, monkeypatch):
    root = fixture(tmp_path)
    original = archive.identity
    def broken(path):
        if path.name == "1_points.json":
            raise OSError("injected unreadable source")
        return original(path)
    monkeypatch.setattr(archive, "identity", broken)
    with pytest.raises(OSError, match="unreadable"):
        archive.snapshot(tmp_path, root, workers=4)
    for workers in [0, 9, True]:
        with pytest.raises(ValueError, match="workers"):
            archive.snapshot(tmp_path, root, workers=workers)


def test_parallel_snapshot_refuses_escaping_link(tmp_path):
    repo = tmp_path/"repo"
    root = fixture(repo)
    outside = tmp_path/"outside.json"
    outside.write_bytes(b"outside")
    try:
        (root/"escape_points.json").symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation unavailable on this platform")
    with pytest.raises(ValueError, match="escaping link"):
        archive.snapshot(repo, root, workers=4)
