"""Correct final V17 transfer metadata without changing requests or consumer code.

The frozen continuation exporter reused its early-pilot description. Keep its
original archive and receipts; publish this separately named, verified product.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--verification-dir", type=Path, required=True)
    args = parser.parse_args()
    products = args.products.resolve()
    extracted = args.verification_dir.resolve()
    extracted.mkdir(parents=True, exist_ok=False)
    original = products / "reader.zip"
    with zipfile.ZipFile(original) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(contents["MANIFEST.json"])
    assert set(contents) == set(manifest) | {"MANIFEST.json"}
    assert all(sha(contents[name]) == expected for name, expected in manifest.items())
    old_text = contents["README.md"].decode()
    old_selection = (
        "The evaluator ZIP is separate. Selection takes the first bounded number of\n"
        "complete cases per native family before inspecting their outcomes. This early\n"
        "pilot does not claim coverage of all five desired illustrative outcome types."
    )
    assert old_text.startswith("# V17 early observer challenge\n")
    assert old_selection in old_text
    text = old_text.replace("# V17 early observer challenge", "# V17 final illustrative observer challenge", 1)
    text = text.replace(old_selection, (
        "The evaluator ZIP is separate. These six cases (eighteen evidence-tier requests)\n"
        "were selected by lowest case hash within each observed outcome type and each\n"
        "of the four native observer families in the completed expansion cohort.\n"
        "Selection used outcomes: this is an illustrative challenge, not an unbiased\n"
        "evaluation population or a fresh confirmation. Wrong-recipient, wasted-work\n"
        "and surprising-edit examples are absent as blind reader types. Evaluator-only\n"
        "illustrations cover those types. A plausible-history-wrong tag is a proxy:\n"
        "multiple compatible histories plus a wrong future prediction, not an\n"
        "independently scored reconstruction of a particular historical path."
    ), 1)
    contents["README.md"] = text.encode()
    manifest["README.md"] = sha(contents["README.md"])
    contents["MANIFEST.json"] = canonical(manifest)
    final = products / "reader-final.zip"
    with zipfile.ZipFile(final, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in sorted(contents.items()):
            archive.writestr(name, body)
    with zipfile.ZipFile(final) as archive:
        for name in archive.namelist():
            if extracted not in (extracted / name).resolve().parents:
                raise ValueError("unsafe member")
            assert archive.read(name) == contents[name]
        archive.extractall(extracted)
    env = dict(os.environ, PYTHONPATH="")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    run = subprocess.run([sys.executable, "-B", "consumer.py"], cwd=extracted,
                         input=contents["requests.jsonl"], capture_output=True,
                         env=env, timeout=180, creationflags=flags, check=True)
    predictions = [json.loads(line) for line in run.stdout.splitlines()]
    with zipfile.ZipFile(products / "evaluator.zip") as archive:
        answers = [json.loads(line) for line in archive.read("answers.jsonl").splitlines()]
    expected = {a["task_id"]: next(r["probabilities"] for r in a["reference_rows"]
                                if r["method"] == "inverse_maker") for a in answers}
    assert len(predictions) == len(expected) == 18
    assert {p["task_id"] for p in predictions} == set(expected)
    for prediction in predictions:
        assert prediction["probabilities"] == expected[prediction["task_id"]]
    (extracted / "PRIVATE-canary.txt").write_text("must not be read", encoding="utf-8")
    probe = subprocess.run([sys.executable, "-B", "consumer.py", "--guard-probe"], cwd=extracted,
                           capture_output=True, env=env, timeout=30, creationflags=flags, check=True)
    assert json.loads(probe.stdout)["private_read_denied"] is True
    receipt = dict(schema="v17.final-transfer-metadata-repair.1", passed=True,
                   original_reader_sha256=sha(original.read_bytes()), reader="reader-final.zip",
                   reader_sha256=sha(final.read_bytes()), evaluator_sha256=sha((products / "evaluator.zip").read_bytes()),
                   changed_members=["README.md", "MANIFEST.json"], scientific_members_unchanged=True,
                   requests=18, exact_reference_forecasts=True, private_read_denied=True,
                   selection="Lowest case hash within observed outcome types and each E native family; completed expansion only; outcome-selected illustrations.",
                   script_sha256=sha(Path(__file__).read_bytes()),
                   scope="Documentation-only successor; actual extracted trusted consumer execution, not hostile-code sandboxing.")
    with (products / "TRANSFER_FINAL.json").open("xb") as stream:
        stream.write(canonical(receipt))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
