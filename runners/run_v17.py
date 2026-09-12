"""Thin V17 packet entry point. No old campaign stage is called."""
import argparse
import json
import os
from pathlib import Path

# Scope numeric thread limits to this new process only.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

from ghostscale.validation.soundingline.v16.records import read, write, file_digest
from ghostscale.validation.soundingline.v17.craft import select_budgets
from ghostscale.validation.soundingline.v17.packets import run_packet, specification, load_cases, source_identity
from ghostscale.validation.soundingline.v17.validity import checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("gates", "fixture", "pilot", "screen"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--stop-after-chunks", type=int)
    args = parser.parse_args()
    gates = checks()
    if args.stage == "gates":
        write(args.root/"GATES.json", gates)
        print(json.dumps(gates))
        return 0 if gates["passed"] else 1
    if not gates["passed"]:
        raise ValueError("local known-answer checks failed")
    budgets = None
    if args.stage == "screen":
        if args.admission is None:
            raise ValueError("screen requires a source-bound admission receipt")
        admission = read(args.admission)
        if admission.get("passed") is not True or admission.get("source_files") != source_identity():
            raise ValueError("admission absent, failed or source mismatch")
        if args.pilot is None:
            raise ValueError("screen requires a completed discarded pilot")
        completed = read(args.pilot/"COMPLETION.json")
        if completed["state"] != "completed":
            raise ValueError("pilot incomplete")
        pilot_lock = read(args.pilot/"LOCK.json")
        if pilot_lock["design"]["stage"] != "pilot":
            raise ValueError("budget input is not a pilot")
        rows = [row for item in load_cases(args.pilot, read(args.pilot/"INDEX.json")) for row in item["rows"]]
        budgets = select_budgets(rows)["budgets"]
    design = specification(args.stage, budgets)
    if args.stage == "screen":
        design["pilot_inputs"] = {name: file_digest(args.pilot/name) for name in ("LOCK.json", "INDEX.json", "COMPLETION.json")}
        design["admission_sha256"] = file_digest(args.admission)
    result = run_packet(args.root, design, args.stop_after_chunks)
    print(json.dumps(result or {"state": "paused_at_chunk_boundary"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
