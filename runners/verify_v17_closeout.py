"""Read-only independent reconstruction of the completed V17 continuation.

Standard library only; imports no scientific scorer or runner. Original records
are never changed. Run from an isolated checkout with explicit output paths.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import sqlite3
import time
import zipfile
import zlib


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def near(a, b):
    if not math.isclose(a, b, rel_tol=2e-8, abs_tol=2e-9):
        raise ValueError(f"arithmetic mismatch: {a} != {b}")


def compare(actual, expected):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError("different keys")
        for key in expected:
            compare(actual[key], expected[key])
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError("different lengths")
        for a, b in zip(actual, expected):
            compare(a, b)
    elif type(expected) is float:
        near(actual, expected)
    elif actual != expected:
        raise ValueError(f"value mismatch: {actual!r} != {expected!r}")


def unit_stats(items):
    """Rebuild stored summaries from rows, independently of unit_statistics."""
    grouped = defaultdict(list)
    for item in items:
        for row in item["rows"]:
            grouped[row["method"], row["target"]].append(row)
    answer = []
    for (method, target), rows in sorted(grouped.items()):
        n = len(rows)
        avg = lambda values: math.fsum(values) / n
        answer.append(dict(
            method=method, target=target, rows=n,
            valid=sum(not r["missing_output"] and not r["invalid_program"] for r in rows),
            apparatus_failures=sum(bool(r.get("apparatus_failure")) for r in rows),
            score=avg(r["brier_score"] if "brier_score" in r else float(not r["task_success"]) for r in rows),
            success=avg(float(r["task_success"]) for r in rows),
            cold=avg(r["costs"]["cold_total"] for r in rows),
            online=avg(r["costs"]["repeat_online"] for r in rows),
            stop_regret=avg(r.get("stop_regret", 0) for r in rows),
            decision_regret=avg(r.get("decision_regret", 0) for r in rows),
            personal_goal_success=avg(float(r.get("personal_goal_success") or False) for r in rows),
            infinite=sum(r.get("log_loss_infinite", False) for r in rows),
            evidence_tiers=sorted({r["evidence_tier"] for r in rows})))
    return answer


def paired(items, contrast):
    gains = []
    for item in items:
        rows = {r["method"]: r for r in item["rows"] if r["target"] == contrast["target"]}
        a, b = rows[contrast["method"]], rows[contrast["rival"]]
        gains.append(float(a["task_success"]) - float(b["task_success"]) if contrast["metric"] == "success"
                     else b["brier_score"] - a["brier_score"])
    value = math.fsum(gains) / len(gains)
    if not contrast["low"] <= value <= contrast["high"]:
        raise ValueError("unbounded paired value")
    return value


def confirmed(values, design):
    if len(values) != design["constructors"]:
        raise ValueError("incomplete primary")
    n = len(values)
    mean = math.fsum(values) / n
    width = design["high"] - design["low"]
    p = math.exp(-2 * n * (max(0, mean - design["minimum_gain"]) / width) ** 2)
    return dict(id=design["id"], n=n, mean_gain=mean,
                variance=math.fsum((v - mean) ** 2 for v in values) / (n - 1),
                lower_bound=max(design["low"], mean - width * math.sqrt(math.log(60) / (2 * n))), p_value=p)


def holm(entries):
    adjusted = 0.0
    for rank, row in enumerate(sorted(entries, key=lambda r: (r["p_value"], r["id"]))):
        adjusted = max(adjusted, min(1, (len(entries) - rank) * row["p_value"]))
        row["holm_adjusted_p"] = adjusted
        row["holm_rejected"] = adjusted <= .05
    return entries


def verify(root, out, source_archive):
    started = time.time()
    plan = read(root / "PLAN.json")
    originals = read(root / "closeout/COMPARISONS.json")
    snapshot = read(root / "SOURCE_SNAPSHOT.json")
    if file_hash(source_archive) != snapshot["archive_sha256"]:
        raise ValueError("source archive changed")
    with zipfile.ZipFile(source_archive) as archive:
        if set(archive.namelist()) != set(snapshot["members"]) | {"SOURCE_MANIFEST.json"}:
            raise ValueError("source membership changed")
        compare(json.loads(archive.read("SOURCE_MANIFEST.json"))["members"], snapshot["members"])
        for name, expected in snapshot["members"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected or file_hash(root / "source" / name) != expected:
                raise ValueError("frozen source changed: " + name)
    if snapshot["members"] != plan["source_files"]:
        raise ValueError("admitted source differs")
    if file_hash(root / "CONTROLLER.json") != plan["controller_sha256"]:
        raise ValueError("controller changed")
    db = sqlite3.connect((root / "records.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True)
    db.execute("PRAGMA query_only=ON")
    for key, value in (("plan", plan), ("admission", read(root / "ADMISSION.json")), ("window", read(root / "WINDOW.json"))):
        compare(json.loads(db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()[0]), value)
    events = [dict(zip(("id", "kind", "payload", "at"), row)) for row in db.execute("SELECT id,kind,payload,at FROM events ORDER BY at")]
    for event in events:
        event["payload"] = json.loads(event["payload"])
        if digest([event["kind"], event["payload"]]) != event["id"]:
            raise ValueError("event hash mismatch")
    complete = next(e for e in events if e["kind"] == "run_complete")
    if digest(read(root / "RUN_COMPLETE.json")) != complete["payload"]["completion_sha256"]:
        raise ValueError("completion event mismatch")
    failures = [dict(zip(("id", "stage", "branch", "ci", "at", "reason"), row)) for row in db.execute("SELECT * FROM failures ORDER BY at")]
    for name, body, expected in db.execute("SELECT name,body,digest FROM snapshots"):
        raw = zlib.decompress(body)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("snapshot digest mismatch")
        if (root / (name + ".json")).exists():
            compare(json.loads(raw), read(root / (name + ".json")))
    designs = {d["id"]: d for d in plan["initial_expansions"] + plan["confirmation"]}
    primary_values = defaultdict(list)
    completed = []
    unit_count = case_count = row_count = issues = 0
    cpu = wall = 0.
    cursor = db.execute("SELECT stage,branch,ci,digest,body,stats,paired,cases,rows,cpu,wall FROM units ORDER BY stage,branch,ci")
    for (stage, branch), records in itertools.groupby(cursor, key=lambda r: r[:2]):
        print(json.dumps(dict(phase="branch", stage=stage, branch=branch, units_done=unit_count)), flush=True)
        surface = next(s for s in originals["surfaces"] if (s["stage"], s["id"]) == (stage, branch))
        groups = defaultdict(lambda: defaultdict(list))
        identities = {k: set() for k in ("public_problem_sha256", "private_construction_sha256", "structural_family")}
        constructor_ids = set()
        indices = []
        branch_cases = branch_rows = 0
        unit_chain = hashlib.sha256()
        for _, _, ci, expected, body, saved, pair, nc, nr, used_cpu, used_wall in records:
            raw = zlib.decompress(body)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("raw hash mismatch")
            unit_chain.update(canonical([ci, expected]))
            items = json.loads(raw)
            if len(items) != nc or sum(len(i["rows"]) for i in items) != nr:
                raise ValueError("raw denominator mismatch")
            if nc != designs[branch]["histories_per_constructor"]:
                raise ValueError("history count mismatch")
            stats = unit_stats(items)
            compare(stats, json.loads(zlib.decompress(saved)))
            ids = {i["case"]["constructor_id"] for i in items}
            if len(ids) != 1 or constructor_ids.intersection(ids):
                raise ValueError("constructor reuse")
            constructor_ids.update(ids)
            for item in items:
                for k in identities:
                    identities[k].add(item["case"][k])
            for row in stats:
                g = groups[row["method"], row["target"]]
                for k, value in row.items():
                    if k not in ("method", "target"):
                        g[k].append(value)
                issues += row["apparatus_failures"]
            if stage == "confirmation":
                value = paired(items, designs[branch])
                near(value, pair)
                primary_values[branch].append(value)
            elif pair is not None:
                raise ValueError("unexpected confirmatory pair")
            indices.append(ci)
            unit_count += 1
            case_count += nc
            row_count += nr
            branch_cases += nc
            branch_rows += nr
            cpu += used_cpu
            wall += used_wall
        if stage != "robustness" and indices != list(range(designs[branch]["constructors"])):
            raise ValueError("planned cohort incomplete")
        compare({k: len(v) for k, v in identities.items()}, surface["uniqueness"])
        saved_uniques = dict(db.execute("SELECT kind,count(*) FROM identities WHERE stage=? AND branch=? GROUP BY kind", (stage, branch)))
        compare(saved_uniques, surface["uniqueness"])
        if len(groups) != len(surface["comparisons"]):
            raise ValueError("missing method surface")
        for original in surface["comparisons"]:
            g = groups[original["method"], original["target"]]
            n = len(g["score"])
            mean = math.fsum(g["score"]) / n
            variance = math.fsum((x - mean) ** 2 for x in g["score"]) / (n - 1) if n > 1 else 0.
            near(mean, original["mean_brier_or_failure"])
            near(variance, original["between_constructor_variance"])
            compare([mean - 1.96 * math.sqrt(variance / n), mean + 1.96 * math.sqrt(variance / n)], original["descriptive_normal_95_interval"])
            for key, field in (("rows", "attempted"), ("valid", "valid"), ("apparatus_failures", "apparatus_failure_rows"), ("infinite", "infinite_log_loss_count")):
                compare(sum(g[key]), original[field])
            for key, field in (("success", "accuracy_or_success"), ("cold", "mean_cold_cost"), ("online", "mean_repeat_cost"), ("stop_regret", "mean_stop_regret"), ("decision_regret", "mean_decision_regret"), ("personal_goal_success", "mean_personal_goal_success")):
                near(math.fsum(g[key]) / n, original[field])
            compare(n, original["constructor_clusters"])
            compare(sorted({x for xs in g["evidence_tiers"] for x in xs}), original["evidence_tiers"])
        receipt = dict(stage=stage, branch=branch, units=len(indices), cases=branch_cases, rows=branch_rows,
                       first_constructor=indices[0], last_constructor=indices[-1], raw_unit_chain_sha256=unit_chain.hexdigest(),
                       uniqueness=surface["uniqueness"], method_targets_verified=len(groups))
        completed.append(receipt)
        write(out / "branches" / (stage + "-" + branch + ".json"), receipt)
    entries = holm([confirmed(primary_values[d["id"]], d) for d in plan["confirmation"]])
    for entry in entries:
        expected = next(e for e in originals["confirmation"]["entries"] if e["id"] == entry["id"])
        for key, field in (("n", "n_constructor_clusters"), ("mean_gain", "mean_gain"), ("variance", "between_constructor_variance"), ("lower_bound", "simultaneous_lower_bound"), ("p_value", "p_value"), ("holm_adjusted_p", "holm_adjusted_p"), ("holm_rejected", "holm_rejected")):
            # Tiny probability bounds need relative comparison, not an absolute floor.
            if key in ("p_value", "holm_adjusted_p"):
                if not math.isclose(entry[key], expected[field], rel_tol=1e-8, abs_tol=0):
                    raise ValueError("confirmation probability mismatch")
            else:
                compare(entry[key], expected[field])
    counts = dict(constructor_units=unit_count, cases=case_count, rows=row_count, worker_cpu_seconds=cpu,
                  unit_wall_seconds=wall, apparatus_issue_rows=issues, failed_units=len(failures))
    compare(counts, originals["counts"])
    compare(counts, read(root / "RUN_COMPLETE.json")["counts"])
    dependencies = dependency_files = dependency_bytes = 0
    for body, expected in db.execute("SELECT body,digest FROM dependencies"):
        raw = zlib.decompress(body)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("dependency archive payload mismatch")
        files = json.loads(raw)
        import base64
        for value in files.values():
            dependency_bytes += len(base64.b64decode(value, validate=True))
            dependency_files += 1
        dependencies += 1
    db.close()
    result = dict(schema="v17.independent-closeout.1", passed=True, written_at=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.time() - started, counts=counts, branches=completed, confirmation=entries,
                  failures=failures, events=events, source_members_verified=len(snapshot["members"]),
                  dependency_units_verified=dependencies, dependency_files_verified=dependency_files, dependency_bytes=dependency_bytes,
                  comparisons_sha256=file_hash(root / "closeout/COMPARISONS.json"), completion_payload_sha256=complete["payload"]["completion_sha256"],
                  verifier_sha256=file_hash(__file__),
                  scope="Every raw unit digest, raw-to-summary reconstruction, denominator, uniqueness, reported mean/variance/interval/cost, paired primary and independent Hoeffding/Holm arithmetic; all retained dependency payload hashes/base64.",
                  limitations="Uses retained row scores whose independent proper-score/physics checks precede this audit. Does not independently validate generative assumptions or regenerate every case. IID constructor draws with replacement do not imply unique architectures.")
    write(out / "REAGGREGATION.json", result)
    return result


def archive_run(root, out, source_archive, failure_archive):
    archive_path = out / "scientific-raw.zip"
    members = {"records.sqlite": root / "records.sqlite", "source.zip": source_archive,
               "failed-unit-cache.zip": failure_archive}
    for name in ("ADMISSION.json", "BRANCH_SELECTION.json", "CONFIRMATION.json", "CONTROLLER.json", "PLAN.json", "RUN_COMPLETE.json", "SOURCE_SNAPSHOT.json", "STATUS.json", "TEST_REPORT.json", "WINDOW.json"):
        members[name] = root / name
    for p in (root / "closeout").rglob("*"):
        if p.is_file() and "extracted" not in p.relative_to(root).parts:
            members[p.relative_to(root).as_posix()] = p
    manifest = {}
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, path in sorted(members.items()):
            print(json.dumps(dict(phase="archive", member=name)), flush=True)
            before = path.stat()
            checksum = hashlib.sha256()
            with path.open("rb") as src, archive.open(name, "w", force_zip64=True) as dst:
                for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
                    checksum.update(chunk)
                    dst.write(chunk)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("archive source changed")
            manifest[name] = dict(bytes=before.st_size, sha256=checksum.hexdigest())
        archive.writestr("MEMBER_MANIFEST.json", canonical(manifest))
    with zipfile.ZipFile(archive_path) as archive:
        for name, expected in manifest.items():
            with archive.open(name) as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected["sha256"] or file_hash(members[name]) != actual:
                raise ValueError("archive reread mismatch")
    receipt = dict(schema="v17.complete-raw-archive.1", passed=True, archive_location="machine-local " + out.parent.name + "/" + out.name + "/scientific-raw.zip",
                   archive_sha256=file_hash(archive_path), archive_bytes=archive_path.stat().st_size, members=manifest,
                   all_member_bytes_verified=True, original_source_bytes_rechecked=True,
                   dependency_scope="All successful-unit dependency receipts are embedded in records.sqlite; failed-unit cache retained separately inside this archive.",
                   limitations="Complete retained continuation science only; prior screens have separate archives. Machine-local archive, not public data availability or off-device backup.")
    write(out / "ARCHIVE.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    for flag in ("run-root", "out", "source-archive", "failure-archive"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        verify(args.run_root.resolve(), out, args.source_archive.resolve())
        archive_run(args.run_root.resolve(), out, args.source_archive.resolve(), args.failure_archive.resolve())
        write(out / "COMPLETE.json", dict(passed=True, reaggregation_sha256=file_hash(out / "REAGGREGATION.json"), archive_sha256=file_hash(out / "ARCHIVE.json")))
    except BaseException as error:
        write(out / "FAILED.json", dict(passed=False, kind=type(error).__name__, reason=str(error)))
        raise


if __name__ == "__main__":
    main()
