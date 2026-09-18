"""Package and execute the bounded V17 continuation replay from an extracted ZIP."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import zipfile
import zlib


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--replay-proof", type=Path, required=True)
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--stitch", type=Path, required=True)
    args = parser.parse_args()
    root, work, products = args.run_root.resolve(), args.work.resolve(), args.products.resolve()
    work.mkdir(parents=True, exist_ok=False)
    proof = read(args.replay_proof)
    assert proof["passed"] and proof["units"] == 102
    db = sqlite3.connect((root / "records.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True)
    subset = work / "records.sqlite"
    selected = sqlite3.connect(subset)
    selected.execute("CREATE TABLE units(stage TEXT, branch TEXT, ci INTEGER, digest TEXT, body BLOB, PRIMARY KEY(stage,branch,ci))")
    for check in proof["checks"]:
        key = (check["stage"], check["branch"], check["constructor_index"])
        digest, body = db.execute("SELECT digest,body FROM units WHERE stage=? AND branch=? AND ci=?", key).fetchone()
        assert hashlib.sha256(zlib.decompress(body)).hexdigest() == digest == check["unit_sha256"]
        selected.execute("INSERT INTO units VALUES(?,?,?,?,?)", (*key, digest, body))
    selected.commit()
    selected.close()
    db.close()
    files = {"records.sqlite": subset}
    plan = read(root / "PLAN.json")
    for name, expected in plan["source_files"].items():
        path = root / "source" / name
        assert sha(path) == expected
        files["source/" + name] = path
    files["source/runners/replay_v17_continuation.py"] = Path(__file__).with_name("replay_v17_continuation.py")
    assert sha(files["source/runners/replay_v17_continuation.py"]) == proof["replay_script_sha256"]
    for name in ("PLAN.json", "CONTROLLER.json", "SOURCE_SNAPSHOT.json"):
        files[name] = root / name
    for name in ("reader.zip", "evaluator.zip"):
        files["closeout/transfer/" + name] = root / "closeout/transfer" / name
    files["ORIGINAL_REPLAY.json"] = args.replay_proof.resolve()
    manifest = {name: dict(bytes=path.stat().st_size, sha256=sha(path)) for name, path in files.items()}
    instructions = b"""# V17 bounded scientific replay (evaluator material)

This archive contains private constructed truth and must not be supplied as blind
reader evidence. The database is deliberately a 102-unit subset, not the full
continuation database. It reproduces the first/last committed units in all 51
phase/branch surfaces: 792 cases and 11,480 rows. It cannot reconstruct full-run
aggregate statistics. ORIGINAL_REPLAY.json identifies the selection and hashes.

Extract to a new directory. Use Python 3.13 and the exact admitted native Stitch
executable plus libunwind.dll (external hashes are in PLAN.json; binaries are not
included). From source/, run with an explicit interpreter and absolute paths:

python -B -m runners.replay_v17_continuation --run-root <extracted-root> --out <new-output-directory> --stitch <compress.exe>

The command checks all frozen source/external hashes, retrains the development
controller, regenerates each full unit with fresh learner caches and compares
bytes, then executes the original extracted reader and private-read probe.
The original reader ZIP retains stale pilot wording; its scientific inputs are
unchanged. Use the separately delivered reader-final.zip for the final challenge.
This is deterministic bounded replay, not independent generative implementation,
architecture validation, human evidence or a complete public raw-data release.
"""
    archive_path = products / "replay-bundle.zip"
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.writestr("README.md", instructions)
        manifest["README.md"] = dict(bytes=len(instructions), sha256=hashlib.sha256(instructions).hexdigest())
        archive.writestr("MEMBER_MANIFEST.json", encoded(manifest))
    extracted = work / "extracted"
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == set(manifest) | {"MEMBER_MANIFEST.json"}
        for name, expected in manifest.items():
            data = archive.read(name)
            assert len(data) == expected["bytes"] and hashlib.sha256(data).hexdigest() == expected["sha256"]
            if extracted not in (extracted / name).resolve().parents:
                raise ValueError("unsafe member")
        archive.extractall(extracted)
    out = work / "verification"
    command = [sys.executable, "-B", "-m", "runners.replay_v17_continuation",
               "--run-root", str(extracted), "--out", str(out), "--stitch", str(args.stitch.resolve())]
    with (work / "verification.log").open("xb") as log:
        subprocess.run(command, cwd=extracted / "source", stdout=log, stderr=subprocess.STDOUT,
                       env=dict(os.environ, PYTHONPATH=""), check=True, timeout=600,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    repeated = read(out / "REPLAY.json")
    assert repeated["checks"] == proof["checks"] and repeated["passed"]
    assert repeated["replayed_points_sha256"] == proof["replayed_points_sha256"]
    receipt = dict(schema="v17.portable-bounded-replay.1", passed=True, archive="replay-bundle.zip",
                   archive_sha256=sha(archive_path), archive_bytes=archive_path.stat().st_size,
                   members=manifest, replay=repeated, reader_portability=read(out / "PORTABILITY.json"),
                   package_script_sha256=sha(Path(__file__)),
                   scope="Actual replay from extracted public subset archive; evaluator truth; not full raw-data availability or full regeneration.")
    with (products / "REPLAY_BUNDLE.json").open("xb") as stream:
        stream.write(encoded(receipt))
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("members", "replay", "reader_portability")}))


if __name__ == "__main__":
    main()
