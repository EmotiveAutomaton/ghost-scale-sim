"""Reconstruct V18 from retained traces and package its small complete replay."""
import argparse
import csv
import io
import json
from pathlib import Path
import statistics
import time
import zipfile

from ghostscale.validation.soundingline.v16.records import canonical, digest, file_digest, now, read, write
from ghostscale.validation.soundingline.v18.verify import reconstruct, METHODS


def examples(units):
    rules = {
        "focused acquisition benefit": lambda r: r["focused-unchecked"]["success"] > r["broad-unchecked"]["success"],
        "checking benefit": lambda r: any(r[a+"-checked"]["success"] > r[a+"-unchecked"]["success"] for a in ("broad", "focused")),
        "checking budget cost reversal": lambda r: any(r[a+"-checked-extra"]["success"] > r[a+"-checked"]["success"] for a in ("broad", "focused")),
        "no success gain with an acquired library": lambda r: not r["broad-unchecked"]["empty_library"] and len({x["success"] for x in r.values()}) == 1,
        "residual interference after checking": lambda r: any(r[a+"-checked"]["success"] < r["primitive"]["success"] for a in ("broad", "focused")),
        "unchecked interference": lambda r: any(r[a+"-unchecked"]["success"] < r["primitive"]["success"] for a in ("broad", "focused")),
    }
    candidates = {name: [] for name in rules}
    for unit in units:
        case = unit["case"]
        grouped = {}
        for row in unit["rows"]:
            grouped.setdefault((row["budget"], row["stratum"], row["target_index"]), {})[row["method"]] = row
        for key, rows in grouped.items():
            for name, condition in rules.items():
                if condition(rows):
                    candidates[name].append((digest([case["case_id"], key]), case, key, rows))
    selected, used = [], set()
    for name, items in candidates.items():
        available = [x for x in sorted(items, key=lambda x: x[0]) if x[0] not in used]
        if available:
            identity, case, key, rows = available[0]; used.add(identity)
            selected.append({"type": name, "illustration_id": identity, "case": case,
                             "budget": key[0], "stratum": key[1], "target_index": key[2], "rows": rows})
        else:
            selected.append({"type": name, "absent": True})
    return selected


def example_text(selected):
    lines = ["# V18 worked cases", "", "Outcome-selected illustrations, chosen by lowest case/target hash per observed type.",
             "Absent types are not fabricated. These contain evaluator truth; they are not blind reader requests.",
             "Actions 0–3 place cells 0–3; 4–7 remove cells 0–3. Endpoints below list occupied cells.", ""]
    cells = lambda mask: [c for c in range(4) if mask & (1 << c)]
    for item in selected:
        lines.extend(["## " + item["type"], ""])
        if item.get("absent"):
            lines.extend(["No distinct example of this type was observed.", ""]); continue
        case, rows = item["case"], item["rows"]
        target = rows["primitive"]["request"]["target"]
        lines.extend([f"Case `{case['case_id']}`; budget {item['budget']}; {item['stratum']} target {cells(target)}.",
                      "Before acquisition, every arm was instructed to prepare for topic A; this own target was disclosed afterward.",
                      f"Private source motifs: {case['private']['constructor']['permutation'][:2]} (A), {case['private']['constructor']['permutation'][2:]} (B).",
                      "These are evaluator annotations, not learner evidence.", "",
                      "Available offers (index: topic): " + ", ".join(f"{o['offer']}: {'A' if o['topic']==0 else 'B'}" for o in case["offered"]["offers"]) + ".", ""])
        for allocation in ("broad", "focused"):
            acq = case["acquisitions"][allocation]
            lines.extend([f"{allocation.title()} selected {acq['selected']}; learned {acq['result']['library']}; top ties {acq['result']['top_ties']}.", "",
                          "Each row shows a processed offer, topic, instruction, realized program, endpoint and admitted binary feedback.", "",
                          "| Offer | Topic | Instruction | Performed | Endpoint | Success feedback |", "|---:|---|---|---|---|---|"])
            for x in acq["request"]["processed"]:
                lines.append(f"| {x['offer']} | {'A' if x['topic']==0 else 'B'} | {x['instruction']} | {x['program']} | {cells(x['artifact'])} | {x['feedback']} |")
            lines.append("")
        lines.extend(["Each policy row separates its first hypothetical nonempty proposal from its final submitted execution. Costs are primitive executions for checking, search and final submission.", "",
                      "| Policy | First proposal | Submitted | Executed endpoint | Success | Check / search / final cost |",
                      "|---|---|---|---|---|---|"])
        for method in METHODS:
            row = rows[method]; cost = row["costs"]
            lines.append(f"| {method} | {row['first_nonempty_proposal']['program']} | {row['submission']['program']} | {cells(row['execution']['artifact'])} | {row['success']} | {cost['checking_primitives']} / {cost['search_primitives']} / {cost['submission_primitives']} |")
        lines.append("")
        for a in ("broad", "focused"):
            row = rows[a+"-checked"]
            text = "; ".join(f"action {x['action']}: {cells(x['before'])} -> {cells(x['after'])}, target distance {x['goal_error_before']} -> {x['goal_error_after']}" for x in row["checks"]) or "no learned fragment, no checks purchased"
            lines.extend([f"{a.title()} checking: {text}. Retained {row['active_library']}.", ""])
        lines.extend(["The dependency here is an indivisible two-action routine that may place an unwanted cell. The exact checker can prune that routine; the task has no necessary harmful detour. Search may time out without enacting the first proposal.", ""])
    return "\n".join(lines)


def report(root, output, *, replay=False, archive=False):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started, cpu = time.monotonic(), time.process_time()
    summary, units = reconstruct(root)
    plan = read(root / "PLAN.json")
    replay_count = 0
    if replay:
        from ghostscale.validation.soundingline.v18.study import make_case, evaluate
        replayed = {}
        for unit in units:
            case = unit["case"]
            if case["case_id"] not in replayed:
                replayed[case["case_id"]] = make_case(plan["namespace"], case["constructor_index"], case["history_index"], len(plan["constructors"]))
            if replayed[case["case_id"]] != case:
                raise ValueError("acquisition replay differs")
            rows = [{k: v for k, v in row.items() if k != "verified"} for row in unit["rows"]]
            if evaluate(case, rows[0]["budget"]) != rows:
                raise ValueError("transfer replay differs")
            replay_count += len(rows)
    write(output / "COMPARISONS.json", summary)
    stream = io.StringIO(newline="")
    if summary["cell_table"]:
        writer = csv.DictWriter(stream, fieldnames=list(summary["cell_table"][0])); writer.writeheader(); writer.writerows(summary["cell_table"])
    (output / "cells_summary.csv").write_text(stream.getvalue(), encoding="utf-8", newline="")
    selected = examples(units)
    write(output / "ILLUSTRATIONS.json", selected)
    (output / "EXAMPLES.md").write_text(example_text(selected), encoding="utf-8", newline="\n")
    complete = (root / "RUN_COMPLETE.json").exists()
    expected_core = summary["counts"]["planned_core_blocks"]
    core_count = len([n for n in summary["raw_bindings"] if n.startswith("core-")])
    extension_count = len([n for n in summary["raw_bindings"] if n.startswith("extension-")])
    extension = read(root / "EXTENSION.json") if (root / "EXTENSION.json").exists() else {"admitted": None}
    if complete:
        if core_count != expected_core or (extension["admitted"] and extension_count != expected_core):
            raise ValueError("completion receipt lacks registered population")
        if set(read(root / "RUN_COMPLETE.json")["blocks"]) != set(summary["raw_bindings"]):
            raise ValueError("completion inventory mismatch")
    proof = {"passed": True, "completed_at": now(), "plan_sha256": file_digest(root / "PLAN.json"),
             "summary_sha256": file_digest(output / "COMPARISONS.json"), "counts": summary["counts"],
             "full_execution_replay_rows": replay_count, "wall_seconds": time.monotonic() - started,
             "worker_cpu_seconds": time.process_time() - cpu, "child_cpu_seconds": 0,
             "scope": "independent selection, learning, physics, search, costs, cached-score checks and reaggregation; replay separately counted"}
    write(output / "VERIFICATION.json", proof)
    write(output / "SCOPE.json", {"execution_complete": complete, "core_completed_blocks": core_count,
                                   "core_planned_blocks": expected_core, "extension_completed_blocks": extension_count,
                                   "extension_admitted": extension["admitted"], "claim_status": plan["claim_status"],
                                   "unfinished_core_blocks": expected_core - core_count,
                                   "limits": ["capacity-one learner", "fixed action and tie ordering", "one four-cell architecture",
                                              "supplied focal instruction and exact own-target check", "no belief/value learning", "descriptive intervals"]})
    if archive:
        repo = Path(__file__).resolve().parents[1]
        payloads = {p: (repo / p).read_bytes() for p in plan["sources"]}
        for path in [root/"PLAN.json", root/"RUN_COMPLETE.json", root/"EXTENSION.json", *sorted((root/"blocks").glob("*.json")), *sorted((root/"raw").glob("*.gz"))]:
            if path.exists(): payloads["run/" + path.relative_to(root).as_posix()] = path.read_bytes()
        for name in ("COMPARISONS.json", "VERIFICATION.json", "SCOPE.json", "EXAMPLES.md", "cells_summary.csv"):
            payloads["report/" + name] = (output/name).read_bytes()
        payloads["REPLAY.md"] = ("# Complete V18 scientific replay\n\nContains evaluator truth. Extract, then run from this directory using the project's Python interpreter:\n\n"
                                "`python -B -m runners.report_v18 --root run --output reproduced --replay`\n\n"
                                "Uses only the standard library. No installation, old campaign stage or hidden raw archive is needed.\n").encode()
        import hashlib
        inventory = {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()}
        payloads["MANIFEST.json"] = canonical(inventory)
        archive_path = output / "replay.zip"
        with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name, payload in sorted(payloads.items()): bundle.writestr(name, payload)
        with zipfile.ZipFile(archive_path) as bundle:
            if set(bundle.namelist()) != set(payloads): raise ValueError("archive inventory differs")
            for name, payload in payloads.items():
                if bundle.read(name) != payload: raise ValueError("archive bytes differ")
        write(output / "ARCHIVE.json", {"passed": True, "sha256": file_digest(archive_path), "bytes": archive_path.stat().st_size,
                                       "members": len(payloads), "all_member_bytes_reread": True,
                                       "scope": "complete small-study traces and frozen source; contains evaluator truth"})
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    print(json.dumps(report(args.root, args.output, replay=args.replay, archive=args.archive)), flush=True)


if __name__ == "__main__":
    main()
