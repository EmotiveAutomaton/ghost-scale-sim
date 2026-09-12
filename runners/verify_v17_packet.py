"""Recalculate retained A scores independently and replay a fixed bounded subset."""
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from ghostscale.validation.soundingline.v16.records import write, now
from ghostscale.validation.soundingline.v17.craft import solve


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    root = Path(root)
    lock = json.loads((root/"LOCK.json").read_bytes())
    index = json.loads((root/"INDEX.json").read_bytes())
    completion = json.loads((root/"COMPLETION.json").read_bytes())
    summary = json.loads((root/"COMPARISONS.json").read_bytes())
    assert completion["lock_sha256"] == sha(root/"LOCK.json") == index["lock_sha256"]
    assert completion["index_sha256"] == sha(root/"INDEX.json")
    assert completion["summary_sha256"] == sha(root/"COMPARISONS.json")
    repo = Path(__file__).resolve().parents[1]
    for relative, expected in lock["source_files"].items():
        assert sha(repo/relative) == expected, relative
    groups, cases, retained = defaultdict(dict), set(), []
    rows_checked = 0
    for chunk in index["chunks"]:
        path = root/chunk["path"]
        assert path.resolve().is_relative_to(root.resolve())
        assert sha(path) == chunk["sha256"]
        items = [json.loads(line) for line in path.read_bytes().splitlines()]
        assert len(items) == chunk["cases"]
        assert [i["case"]["case_id"] for i in items] == chunk["case_ids"]
        assert sum(len(i["rows"]) for i in items) == chunk["rows"]
        for item in items:
            case = item["case"]
            assert case["case_id"] not in cases
            cases.add(case["case_id"])
            p = case["public"]
            keys = set()
            for row in item["rows"]:
                assert (row["method"],row["budget"]) not in keys
                keys.add((row["method"],row["budget"]))
                board, legal = p["initial"], True
                if row["program"] is None:
                    success = False
                    assert row["execution"] is None and row["missing_output"]
                else:
                    for action in row["program"]:
                        assert type(action) is int and action in range(32) and action not in p["forbidden"]
                        if action < 16:
                            board |= 1 << action
                        else:
                            board &= ~(1 << (action-16))
                    assert len(row["program"]) <= p["max_steps"]
                    assert board == row["execution"]["artifact"] and row["execution"]["legal"]
                    assert row["costs"]["actual_execution"] == len(row["program"])
                    success = board == p["target"]
                assert success == row["task_success"]
                assert not row["invalid_program"]
                costs=row["costs"]
                search=sum(costs[k] for k in ("retrieval","proposal_generation","argument_binding","hypothetical_execution","selection"))
                setup=sum(costs[k] for k in ("training_acquisition","learning","definition_storage"))
                assert search == costs["search_total"] <= row["budget"]
                assert search+costs["actual_execution"]+costs["feedback_query"] == costs["repeat_online"]
                assert setup+costs["repeat_online"] == costs["cold_total"]
                groups[(row["regime"],row["budget"],row["method"])][row["history_id"]] = row
                rows_checked += 1
            assert keys == {(m,b) for m in lock["design"]["methods"] for b in lock["design"]["budgets"]}
            retained.append(item)
    assert len(cases)==completion["cases"]==summary["attempted_cases"]
    assert rows_checked==completion["rows"]
    for table in summary["comparison_table"]:
        rows=groups[(table["regime"],table["budget"],table["method"])]
        clusters=defaultdict(list)
        for row in rows.values():
            clusters[row["constructor_id"]].append(int(row["task_success"]))
        exact=sum(Fraction(sum(xs),len(xs)) for xs in clusters.values())/len(clusters)
        assert float(exact)==table["solve_rate"]
        assert table["attempted"]==len(rows)
    for contrast in summary["contrasts"]:
        left=groups[(contrast["regime"],contrast["budget"],contrast["left"])]
        right=groups[(contrast["regime"],contrast["budget"],contrast["right"])]
        assert set(left)==set(right)
        clusters=defaultdict(list)
        for key, row in left.items():
            clusters[row["constructor_id"]].append(int(row["task_success"])-int(right[key]["task_success"]))
        exact=sum(Fraction(sum(xs),len(xs)) for xs in clusters.values())/len(clusters)
        assert float(exact)==contrast["paired_solve_rate_difference"]
    # First and last case in each regime, chosen without reading outcomes.
    chosen={}
    for regime in lock["design"]["regimes"]:
        matches=[item for item in retained if item["case"]["regime"]==regime]
        for item in (matches[0],matches[-1]):
            chosen[item["case"]["case_id"]]=item
    replay_rows=0
    for item in chosen.values():
        for expected in item["rows"]:
            actual=solve(item["case"]["public"],expected["method"],expected["budget"],expected["memory_cap"])
            for key, value in actual.items():
                assert value==expected[key], (key,item["case"]["case_id"])
            replay_rows+=1
    return {"schema":"v17.verification.1","verified_at":now(),"passed":True,
            "independent_physics_and_score_rows":rows_checked,"unique_case_ids":len(cases),
            "independent_arm_means":len(summary["comparison_table"]),
            "independent_paired_means":len(summary["contrasts"]),
            "bounded_same_implementation_replay_cases":len(chosen),"bounded_replay_rows":replay_rows,
            "source_lock_sha256":sha(root/"LOCK.json"),"index_sha256":sha(root/"INDEX.json"),
            "verifier_sha256":sha(Path(__file__)),
            "limits":["Bootstrap intervals are not independently reimplemented here.",
                      "Bounded replay does not mean every search was rerun.",
                      "One motif architecture; no confirmation or observer-inference claim."]}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args()
    report=verify(args.root)
    write(args.root/"VERIFICATION.json",report)
    print(json.dumps(report))
