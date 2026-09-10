import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from ghostscale.validation.soundingline.v16.runtime import REPO
from ghostscale.validation.soundingline.v16.study import unit
from ghostscale.validation.soundingline.v16.craft import DESIGN
from ghostscale.validation.soundingline.v16.records import write, read, file_digest
from ghostscale.validation.soundingline.v16.replay_plan import item
from ghostscale.validation.soundingline.v16.raw_archive import manifest, create


def test_extracted_known_bundle_runs_actual_cli_and_refuses_changed_source(tmp_path):
    source, extracted = tmp_path/"source", tmp_path/"extracted"
    for path in [*REPO.joinpath("ghostscale").rglob("*.py"), *REPO.joinpath("runners").glob("*v16*.py")]:
        destination = source/path.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    root = source/"results/v16"
    condition = DESIGN["conditions"][1]
    row = unit(root/"native", condition, 3, packet_hash="portable-known-fixture",
               namespace="portable-known-world", constructors=8, evidence_scope="fixture")
    path = root/"native/units"/(row["unit_id"]+"_points.json")
    code = "ghostscale/validation/soundingline/v16/study.py"
    write(root/"packets/known.json", {"packet_hash": "portable-known-fixture", "identity": {
        "packet_id": "known", "files": {code: file_digest(source/code)}}})
    paths = [path for path in source.rglob("*") if path.is_file()]
    dependencies = {path.relative_to(source).as_posix(): file_digest(path) for path in paths}
    dep_path = root/"bundle/dependency_points.json"
    write(dep_path, {"files": dependencies})
    plan_path = root/"bundle/PLAN.json"
    write(plan_path, {"selected": [item(root, path, "known", "native-card", condition, 8)],
        "dependencies": dep_path.relative_to(source).as_posix(), "dependencies_sha256": file_digest(dep_path)})
    plan = manifest(source, [*paths, dep_path, plan_path], scope={"known_portable_fixture": True})
    create(source, tmp_path/"archive", plan)
    with zipfile.ZipFile(tmp_path/"archive/raw.zip") as archive:
        # All names are emitted by the checked repo-relative manifest above.
        archive.extractall(extracted)
    environment = dict(os.environ, PYTHONPATH=str(extracted), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-B", "-m", "runners.replay_v16_bundle", "--root", "results/v16",
        "--plan", "results/v16/bundle/PLAN.json", "--output", "results/v16/known-replay-1"]
    completed = subprocess.run(command, cwd=extracted, env=environment, text=True, capture_output=True, timeout=60)
    assert completed.returncode == 0, completed.stdout+completed.stderr
    receipt = read(extracted/"results/v16/known-replay-1/RECEIPT.json")
    assert receipt["selected_units_wholly_replayed"] and receipt["distinct_units"] == 1
    assert receipt["checks"][0]["world_reader_and_score_match"]
    original = extracted/"results/v16/native/units"/(row["unit_id"]+"_points.json")
    write(original, {"changed_score": True}, immutable=False)
    command[-1] = "results/v16/known-replay-2"
    rejected = subprocess.run(command, cwd=extracted, env=environment, text=True, capture_output=True, timeout=60)
    assert rejected.returncode != 0 and "dependency is missing, escaping or changed" in rejected.stderr
    assert not (extracted/"results/v16/known-replay-2/private").exists()
