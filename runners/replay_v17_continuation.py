"""Bounded full-unit continuation replay using extracted frozen scientific source."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import zipfile
import zlib


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stitch", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.run_root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan = read(root / "PLAN.json")
    source = Path(__file__).resolve().parents[1]
    for name, expected in plan["source_files"].items():
        if sha(source / name) != expected:
            raise ValueError("frozen extracted source mismatch: " + name)
    if sha(args.stitch) != plan["external_files"]["stitch_executable"]:
        raise ValueError("Stitch binary differs")
    if sha(args.stitch.with_name("libunwind.dll")) != plan["external_files"]["stitch_runtime"]:
        raise ValueError("Stitch runtime differs")
    os.environ["GS_V17_STITCH_EXE"] = str(args.stitch.resolve())
    from ghostscale.validation.soundingline.v17 import continuation_cases as cc
    from ghostscale.validation.soundingline.v16.records import canonical
    controller = read(root / "CONTROLLER.json")
    trained = cc.c.prepare(dict(namespace="v17-continuation-controller-development"))
    if canonical(trained) != canonical(controller):
        raise ValueError("development controller replay differs")
    start = time.time()
    db = sqlite3.connect((root / "records.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True)
    checks = []
    sample_file = out / "replayed_points.jsonl"
    with sample_file.open("xb") as samples:
        for stage, designs in (("expansion", plan["initial_expansions"]), ("confirmation", plan["confirmation"]), ("robustness", plan["initial_expansions"])):
            for original in designs:
                design = dict(original)
                if stage == "robustness":
                    design["namespace"] += "-robustness"
                if design["family"] == "C":
                    design["prepared"] = controller
                for order in ("ASC", "DESC"):
                    record = db.execute("SELECT ci,digest,body FROM units WHERE stage=? AND branch=? ORDER BY ci " + order + " LIMIT 1", (stage, design["id"])).fetchone()
                    if record is None:
                        raise ValueError("missing replay branch")
                    ci, expected, body = record
                    original_bytes = zlib.decompress(body)
                    if hashlib.sha256(original_bytes).hexdigest() != expected:
                        raise ValueError("raw replay source corrupt")
                    cache = out / "fresh-stitch-cache" / (stage + "-" + design["id"] + "-" + str(ci))
                    os.environ["GS_V17_STITCH_CACHE"] = str(cache)
                    items = []
                    for hi in range(design["histories_per_constructor"]):
                        case = cc.make_case(design, ci, hi)
                        items.append(dict(case=case, rows=cc.evaluate_case(case, design)))
                    if canonical(items) != original_bytes:
                        write(out / (stage + "-" + design["id"] + "-" + str(ci) + "-MISMATCH.json"), items)
                        raise ValueError("full-unit scientific replay differs: " + stage + "/" + design["id"] + "/" + str(ci))
                    check = dict(stage=stage, branch=design["id"], constructor_index=ci, unit_sha256=expected,
                                 cases=len(items), rows=sum(len(i["rows"]) for i in items))
                    samples.write(canonical(dict(**check, items=items)) + b"\n")
                    checks.append(check)
                    print(json.dumps(check), flush=True)
    db.close()
    for name, expected in plan["source_files"].items():
        if sha(source / name) != expected or sha(root / "source" / name) != expected:
            raise ValueError("source changed during replay")
    write(out / "REPLAY.json", dict(schema="v17.continuation-bounded-replay.1", passed=True,
        units=len(checks), cases=sum(c["cases"] for c in checks), rows=sum(c["rows"] for c in checks),
        elapsed_seconds=time.time() - start, controller_training_reproduced=True, checks=checks,
        selection="First and last committed unit in all 51 expansion/confirmation/robustness branches, including curtailed pooled assembly.",
        stitch="Exact admitted native binary, fresh isolated caches; no reuse of original learned cache results.",
        replayed_points_sha256=sha(sample_file), replay_script_sha256=sha(__file__),
        limitations="Bounded deterministic replay, not full regeneration or an independent generative implementation."))
    transfer = root / "closeout/transfer"
    extracted = out / "reader-extracted"
    with zipfile.ZipFile(transfer / "reader.zip") as archive:
        for name in archive.namelist():
            if extracted.resolve() not in (extracted / name).resolve().parents:
                raise ValueError("unsafe reader member")
        archive.extractall(extracted)
    with zipfile.ZipFile(transfer / "evaluator.zip") as archive:
        answers = [json.loads(line) for line in archive.read("answers.jsonl").splitlines()]
    expected = {a["task_id"]: next(r["probabilities"] for r in a["reference_rows"] if r["method"] == "inverse_maker") for a in answers}
    env = dict(os.environ, PYTHONPATH="")
    run = subprocess.run([sys.executable, "-B", "consumer.py"], cwd=extracted,
                         input=(extracted / "requests.jsonl").read_text(), capture_output=True, text=True,
                         env=env, timeout=180, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if run.returncode:
        raise ValueError("extracted reader failed: " + run.stderr[:200])
    predictions = [json.loads(line) for line in run.stdout.splitlines()]
    if len(predictions) != len(expected) or {p["task_id"] for p in predictions} != set(expected):
        raise ValueError("reader request coverage differs")
    for prediction in predictions:
        wanted = expected[prediction["task_id"]]
        if set(prediction["probabilities"]) != set(wanted) or any(not math.isclose(v, wanted[k], rel_tol=1e-11, abs_tol=1e-11) for k, v in prediction["probabilities"].items()):
            raise ValueError("reader forecast differs")
    (extracted / "PRIVATE-canary.txt").write_text("must not be read", encoding="utf-8")
    probe = subprocess.run([sys.executable, "-B", "consumer.py", "--guard-probe"], cwd=extracted,
                           capture_output=True, text=True, timeout=30, env=env,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if probe.returncode or not json.loads(probe.stdout).get("private_read_denied"):
        raise ValueError("reader private boundary failed")
    write(out / "PORTABILITY.json", dict(schema="v17.closeout-reader-portability.1", passed=True,
        requests=len(predictions), reader_sha256=sha(transfer / "reader.zip"), evaluator_sha256=sha(transfer / "evaluator.zip"),
        private_read_denied=True, scope="Actual fresh extracted trusted Python consumer; not a hostile-code sandbox."))


if __name__ == "__main__":
    main()
