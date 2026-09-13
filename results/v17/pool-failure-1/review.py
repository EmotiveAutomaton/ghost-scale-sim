"""Bounded read-only diagnosis of the retained V17 pooled-assembly failure.

Extracts the admitted archive into a separate checkout. Never opens the live
store with its writer class, runs a scientific stage, or invokes Stitch.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import zipfile
import zlib


EVENT = "f5813c3ce9ee38be8f30c8b615d591aeff29bf2c1cf67d237bdd044cb43f4f39"
FAILURE = "1e0d95c4d644f55bc7cb46771d0a372b652aada88ef40314ce3b0b61ac427bd2"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "source-archive", "source-manifest", "scratch", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    scratch = args.scratch.resolve()
    assert scratch != root and root not in scratch.parents
    assert not scratch.exists(), "use a fresh isolated review directory"
    manifest = read(args.source_manifest)
    assert sha(args.source_archive.read_bytes()) == manifest["archive_sha256"]
    source = scratch / "source"
    source.mkdir(parents=True)
    with zipfile.ZipFile(args.source_archive) as archive:
        for name, expected in manifest["members"].items():
            target = (source / name).resolve()
            assert source in target.parents
            data = archive.read(name)
            assert sha(data) == expected
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    live_hashes = {name: sha((root / "source" / name).read_bytes())
                   for name in manifest["members"]}
    assert live_hashes == manifest["members"]
    sys.path.insert(0, str(source))
    from ghostscale.validation.soundingline.v17 import continuation_cases as cc
    from ghostscale.validation.soundingline.v17.continuation_runtime import one_unit
    from ghostscale.validation.soundingline.v17.continuation_store import Store
    from ghostscale.validation.soundingline.v16.records import canonical

    with sqlite3.connect((root / "records.sqlite").as_uri() + "?mode=ro", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        event = dict(db.execute("SELECT * FROM events WHERE id=?", (EVENT,)).fetchone())
        event["payload"] = json.loads(event["payload"])
        failure = dict(db.execute("SELECT * FROM failures WHERE id=?", (FAILURE,)).fetchone())
        assert event["payload"]["failure_id"] == FAILURE
        assert sha(canonical([event["kind"], event["payload"]])) == EVENT
        assert sha(canonical([failure[k] for k in ("stage", "branch", "ci", "reason")])) == FAILURE
        assert failure["reason"] == "ValueError: pooled acquisition is empty"
        plan = json.loads(db.execute("SELECT value FROM metadata WHERE key='plan'").fetchone()[0])
        window = json.loads(db.execute("SELECT value FROM metadata WHERE key='window'").fetchone()[0])
        assert plan == read(root / "PLAN.json")
        assert window == read(root / "WINDOW.json")
        assert plan["source_files"] == live_hashes
        assert sha((root / "CONTROLLER.json").read_bytes()) == plan["controller_sha256"]
        failures = [dict(row) for row in db.execute("SELECT * FROM failures ORDER BY at,id")]
        branch_counts = [dict(row) for row in db.execute(
            "SELECT stage,branch,COUNT(*) AS units,SUM(cases) AS cases,SUM(rows) AS rows,MAX(ci) AS last_ci "
            "FROM units WHERE branch=? GROUP BY stage,branch", (failure["branch"],))]
        later_retained = db.execute(
            "SELECT COUNT(*) FROM units WHERE stage=? AND branch=? AND ci>=?",
            (failure["stage"], failure["branch"], failure["ci"])).fetchone()[0]
        assert later_retained == 0
        prior = db.execute(
            "SELECT ci,digest,body FROM units WHERE stage=? AND branch=? AND ci<? ORDER BY ci DESC LIMIT 1",
            (failure["stage"], failure["branch"], failure["ci"])).fetchone()
        prior_body = zlib.decompress(prior["body"])
        assert sha(prior_body) == prior["digest"]
        prior_items = json.loads(prior_body)
        db.rollback()

    design = next(d for d in plan["initial_expansions"] if d["id"] == failure["branch"])
    design = dict(design, namespace=design["namespace"] + "-robustness")
    histories = []
    for hi in range(design["histories_per_constructor"]):
        case = cc.a.make_case(design["namespace"], failure["ci"], hi, design["regime"])
        public = case["public"]
        counts = Counter()
        donors = []
        for donor in range(4):
            other = cc.a.make_case(design["namespace"] + "-pooled-training",
                                   failure["ci"] * 4 + donor, hi, design["regime"])
            outcomes = Counter()
            for row in other["public"]["training"]:
                if row["target"] == public["target"]:
                    outcome = "excluded_donor_target"
                else:
                    execution = cc.a.execute(public, row["program"], row["initial"])
                    if not execution["legal"]:
                        bad = next(t for t in execution["trace"] if not t["legal"])
                        outcome = "illegal_action_" + str(bad["action"])
                    elif execution["state"] == public["target"]:
                        outcome = "excluded_receiver_target"
                    else:
                        outcome = "eligible"
                outcomes[outcome] += 1
                counts[outcome] += 1
            donors.append(dict(donor=donor, world=other["public"]["world"], outcomes=dict(outcomes),
                               unique_programs=sorted({tuple(t["program"]) for t in other["public"]["training"]})))
        try:
            made = cc.make_case(design, failure["ci"], hi)
        except ValueError as exc:
            assert str(exc) == "pooled acquisition is empty" and not counts["eligible"]
            outcome = "reproduced_empty_pool"
        else:
            assert counts["eligible"] and len(made["public"]["pooled_training"]) == len(public["training"])
            outcome = "valid_pool"
        histories.append(dict(history_index=hi, outcome=outcome, receiver_world=public["world"],
                              receiver_target=public["target"], counts=dict(counts), donors=donors))
    assert any(h["outcome"] == "reproduced_empty_pool" for h in histories)
    for hi, item in enumerate(prior_items):
        assert cc.make_case(design, prior["ci"], hi) == item["case"]

    # Exercise the literal branch-failure guard against an isolated database.
    guard_root = scratch / "guard"
    guard = Store(guard_root / "records.sqlite")
    guard.failure(failure["stage"], failure["branch"], failure["ci"], failure["reason"])
    def must_not_emit(**unused):
        raise AssertionError("failed branch reached execution")
    one_unit(guard_root, guard, failure["stage"], design, failure["ci"] + 1, None, must_not_emit)
    assert guard.counts()["constructor_units"] == 0 and not (guard_root / "unit-cache").exists()
    guard.close()
    assert live_hashes == {name: sha((root / "source" / name).read_bytes()) for name in live_hashes}
    assert window == read(root / "WINDOW.json")
    report = dict(schema="v17.bounded-fault-review.1", reviewed_at=datetime.now(timezone.utc).isoformat(),
                  event=event, failure=failure, all_failures_at_snapshot=failures, design=design,
                  source_archive_sha256=manifest["archive_sha256"], source_member_count=len(live_hashes),
                  frozen_source_verified_unchanged=True, immutable_window=window,
                  plan_and_controller_bindings_verified=True, database_access="read-only; query_only; bounded snapshot",
                  branch_counts_at_snapshot=branch_counts, retained_failed_or_later_branch_units=later_retained,
                  reproduction=histories, preceding_valid_constructor=prior["ci"],
                  preceding_valid_case_replays=len(prior_items), isolated_branch_failure_guard_passed=True,
                  scientific_runner_launched=False, scientific_source_edited=False,
                  disposition="Retain apparatus failure and original branch halt; continue eligible branches under the frozen plan.",
                  closeout_requirements=["Report curtailed robustness branch and retained-prefix denominators.",
                                         "Do not treat empty acquisition as a scientific loss or null.",
                                         "Retain failure cache and this reproduction proof with the final archive.",
                                         "Any changed donor rule requires separate source, namespace and admission; no silent replacement."],
                  limits="Case generation and branch halt checked; no scientific scores recomputed, no full database or campaign verification.",
                  review_script_sha256=sha(Path(__file__).read_bytes()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(report, out, indent=2, sort_keys=True)
        out.write("\n")
    print(json.dumps({k: report[k] for k in ("reviewed_at", "failure", "branch_counts_at_snapshot",
          "preceding_valid_constructor", "preceding_valid_case_replays", "isolated_branch_failure_guard_passed")}))
    print(json.dumps([dict(history_index=h["history_index"], outcome=h["outcome"], counts=h["counts"])
                      for h in histories]))


if __name__ == "__main__":
    main()
