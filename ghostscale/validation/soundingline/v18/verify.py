"""Independent source-to-score reconstruction; no scientific scorer imports."""
from collections import Counter, defaultdict
import gzip
from itertools import product
import json
from pathlib import Path
import random
import statistics

from ..v16.records import canonical, digest, file_digest, read

METHODS = ("primitive", "broad-unchecked", "focused-unchecked", "broad-checked",
           "focused-checked", "broad-checked-extra", "focused-checked-extra")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def execution(actions, start=0):
    """Independent four-cell reference including attempted illegal-action charges."""
    state = start
    for index, action in enumerate(actions):
        if index == 3:
            return dict(artifact=state, legal=False, primitive_cost=3, stopped=False, error="timeout")
        if type(action) is not int or action not in range(8):
            return dict(artifact=state, legal=False, primitive_cost=index + 1, stopped=True, error="illegal primitive")
        bit = 2 ** (action % 4)
        state = state | bit if action < 4 else state & (15 ^ bit)
    return dict(artifact=state, legal=True, primitive_cost=len(actions), stopped=True, error=None)


def expand(tokens, library):
    actions = []
    for token in tokens:
        actions.extend(library[int(token[1:])] if isinstance(token, str) else [token])
    return actions


def search(target, library, budget):
    options = [f"m{i}" for i in range(len(library))] + list(range(8))
    attempts, cost = [], 0
    for length in range(4):
        for tokens in product(options, repeat=length):
            actions = expand(tokens, library)
            if cost + min(len(actions), 3) > budget:
                return dict(program=[], tokens=[], search_primitives=cost, considered=len(attempts),
                            search_timeout=True, attempted_programs=attempts)
            result = execution(actions)
            cost += result["primitive_cost"]
            attempts.append(dict(tokens=list(tokens), legal=result["legal"], artifact=result["artifact"], cost=result["primitive_cost"]))
            if result["legal"] and result["artifact"] == target:
                return dict(program=actions, tokens=list(tokens), search_primitives=cost,
                            considered=len(attempts), search_timeout=False, attempted_programs=attempts)
    return dict(program=[], tokens=[], search_primitives=cost, considered=len(attempts),
                search_timeout=True, attempted_programs=attempts)


def learn(records):
    counts = Counter(tuple(x["program"]) for x in records if x["feedback"])
    ordered = sorted((p for p, n in counts.items() if n >= 3), key=lambda p: (-counts[p], p))
    library = [list(ordered[0])] if ordered else []
    return library, counts, ordered


def verify_case(case):
    offers = case["offered"]["offers"]
    require(set(case["offered"]) == {"schema", "focal_topic", "offers"}, "offered schema")
    require(all(set(o) == {"offer", "topic"} for o in offers), "private offered field")
    require(len(offers) == 32 and [o["offer"] for o in offers] == list(range(32)), "offers")
    require(Counter(o["topic"] for o in offers) == {0: 16, 1: 16}, "offered balance")
    permutation = case["private"]["constructor"]["permutation"]
    require(sorted(permutation) == list(range(4)), "constructor permutation")
    motifs = [permutation[:2], permutation[2:]]
    source_targets = [sum(2 ** c for c in pair) for pair in motifs]
    expected_targets = {"compatible": [source_targets[0] | 2 ** y for y in motifs[1]],
                        "changed": [2 ** x | 2 ** y for x in motifs[0] for y in motifs[1]]}
    require(case["transfer_targets"] == expected_targets, "transfer target rule")
    require(not set(source_targets) & {t for ts in expected_targets.values() for t in ts}, "acquisition target overlap")
    require(case["constructor_sha256"] == digest(case["private"]["constructor"]), "constructor identity")
    require(case["structural_world_sha256"] == digest({"motifs": motifs, "primitive_order": list(range(8))}), "structural identity")
    acquired = {}
    for name, quotas in (("broad", (8, 8)), ("focused", (12, 4))):
        expected_selection = sorted(o["offer"] for t, q in enumerate(quotas) for o in [x for x in offers if x["topic"] == t][:q])
        acquisition = case["acquisitions"][name]
        require(acquisition["selected"] == case["allocations"][name] == expected_selection, "selection before outcomes")
        require(set(acquisition["request"]) == {"schema", "processed"}, "acquisition interface")
        records = acquisition["request"]["processed"]
        require([x["offer"] for x in records] == expected_selection, "processed selection")
        for record in records:
            require(set(record) == {"offer", "topic", "instruction", "proposed", "program", "artifact", "feedback"}, "private learner field")
            offer = case["private"]["offers"][record["offer"]]
            require(offer["topic"] == record["topic"] == offers[record["offer"]]["topic"], "topic correspondence")
            require(offer["intended_target"] == source_targets[record["topic"]], "source target")
            require(record["instruction"] == record["proposed"] == offer["instruction"], "instruction access")
            result = execution(record["program"])
            require(len(record["program"]) == 2 and record["artifact"] == result["artifact"], "acquisition execution")
            require(record["feedback"] is (result["legal"] and result["artifact"] == offer["intended_target"]), "feedback")
            identity = record["offer"]
            if identity in acquired:
                require(acquired[identity] == record, "unpaired realized opportunity")
            acquired[identity] = record
        library, counts, ordered = learn(records)
        learned = acquisition["result"]
        require(learned["library"] == library and learned["library_sha256"] == digest(library), "learned library")
        require(learned["eligible_counts"] == [{"program": list(p), "count": n} for p, n in sorted(counts.items())], "eligible counts")
        require(learned["top_ties"] == [list(p) for p in ordered if counts[p] == counts[ordered[0]]], "tie break")
        require(learned["costs"] == {"processed_trials": 16, "training_primitives": 32, "instruction_queries": 16,
                                   "feedback_queries": 16, "learning_records_scanned": 16,
                                   "eligible_programs_sorted": len(ordered), "storage_primitives": sum(map(len, library)),
                                   "offered_cues_scanned": 32}, "acquisition costs")


def verify_row(case, row):
    method, target = row["method"], row["request"]["target"]
    require(method in METHODS, "unregistered method")
    require(target == case["transfer_targets"][row["stratum"]][row["target_index"]], "unregistered target")
    library = [] if method == "primitive" else case["acquisitions"][method.split("-")[0]]["result"]["library"]
    checked, extra = "-checked" in method, method.endswith("-extra")
    require(row["request"] == {"schema": "v18.selective-acquisition.1", "library": library, "target": target,
                               "budget": row["budget"], "checking": checked, "extra_search": extra}, "transfer interface")
    retained, checks = [], []
    for fragment in library:
        state, reject = 0, False
        if checked:
            for action in fragment:
                result = execution([action], state)
                before, after = (state ^ target).bit_count(), (result["artifact"] ^ target).bit_count()
                checks.append(dict(before=state, action=action, after=result["artifact"], goal_error_before=before,
                                   goal_error_after=after, primitive_cost=result["primitive_cost"]))
                reject |= not result["legal"] or after > before
                state = result["artifact"]
        if not reject:
            retained.append(fragment)
    require(checks == row["checks"] and retained == row["active_library"], "paid local checking")
    cost = sum(c["primitive_cost"] for c in checks)
    budget = row["budget"] - (0 if extra else cost)
    submission = search(target, retained, budget)
    require(submission == row["submission"], "search trace or timeout differs from independent enumeration")
    result = execution(submission["program"])
    require(result == row["execution"], "submission execution")
    success = result["legal"] and result["artifact"] == target
    require(success == row["success"] and row["invalid"] == (not result["legal"]), "cached outcome")
    require(row["empty_library"] == (not library) and row["rejected_fragments"] == len(library) - len(retained), "library disposition")
    first = next((x for x in submission["attempted_programs"] if x["tokens"]), None)
    require(row["first_nonempty_proposal"] == {"program": expand(first["tokens"], retained) if first else [], "attempt": first}, "first proposal")
    costs = {"checking_primitives": cost, "search_primitives": submission["search_primitives"],
             "submission_primitives": result["primitive_cost"], "online_primitives": cost + submission["search_primitives"] + result["primitive_cost"],
             "search_envelope": budget, "attempts_considered": submission["considered"]}
    require(costs == row["costs"], "cost reconciliation")
    require(cost + costs["search_primitives"] <= row["budget"] + (cost if extra else 0), "primary envelope")
    # Returned outcomes are derived independently, not projected from cached scores.
    return {"success": int(success), "invalid": int(not result["legal"]), "timeout": int(submission["search_timeout"]),
            "empty_library": int(not library), "rejected": int(len(library) > len(retained)), **costs}


def interval(values, key):
    if not values:
        return None
    rng = random.Random(int(digest(["v18-descriptive-bootstrap", key])[:16], 16))
    boot = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(2000))
    return [boot[49], boot[1949]]


def reconstruct(root, *, check_source=True):
    root = Path(root)
    plan = read(root / "PLAN.json")
    repo = Path(__file__).resolve().parents[4]
    if check_source:
        require(all(file_digest(repo / p) == sha for p, sha in plan["sources"].items()), "source binding")
    plan_hash = file_digest(root / "PLAN.json")
    units, receipts, identities, cases_by_id = [], [], set(), {}
    for path in sorted((root / "blocks").glob("*.json")):
        receipt = read(path)
        raw = root / "raw" / (receipt["name"] + "_points.json.gz")
        require(file_digest(raw) == receipt["raw_sha256"], "raw block checksum")
        block = json.loads(gzip.decompress(raw.read_bytes()))
        require(digest(block) == receipt["content_sha256"] and block["plan_sha256"] == plan_hash, "content/plan binding")
        require(block["name"] == receipt["name"] == path.stem, "block identity")
        phase, block_index = block["name"].split("-")
        require(phase in {"core", "extension"} and block["budget"] == (32 if phase == "core" else 128), "budget phase")
        expected = [(c, h) for c in plan["constructors"][int(block_index)*8:(int(block_index)+1)*8] for h in plan["histories"]]
        require([(u["case"]["constructor_index"], u["case"]["history_index"]) for u in block["units"]] == expected, "registered block population")
        require(receipt["cases"] == len(block["units"]) and receipt["rows"] == sum(len(u["rows"]) for u in block["units"]), "receipt counts")
        for unit in block["units"]:
            case = unit["case"]
            require(case["namespace"] == plan["namespace"] and case["constructor_count"] == len(plan["constructors"]), "seed identity")
            require(case["source_index"] == case["history_index"] * len(plan["constructors"]) + case["constructor_index"], "local index")
            require(case["case_id"] == digest([plan["namespace"], case["constructor_index"], case["history_index"]]), "case identity")
            key = (block["budget"], case["case_id"])
            require(key not in identities, "duplicate acquisition history"); identities.add(key)
            if case["case_id"] in cases_by_id:
                require(cases_by_id[case["case_id"]] == case, "extension changed acquisition")
            else:
                verify_case(case); cases_by_id[case["case_id"]] = case
            expected_rows = {(s, t, m) for s, ts in case["transfer_targets"].items() for t in range(len(ts)) for m in METHODS}
            require(len(unit["rows"]) == 42 and {(r["stratum"], r["target_index"], r["method"]) for r in unit["rows"]} == expected_rows, "method/target coverage")
            for row in unit["rows"]:
                require(row["budget"] == block["budget"], "row budget")
                row["verified"] = verify_row(case, row)
            units.append(unit)
        receipts.append(receipt)
    # Target means first, then histories, then constructors. No target is an
    # independent replicate. Each completed block contains complete constructors.
    groups, constructor_vectors = defaultdict(list), {}
    for unit in units:
        for row in unit["rows"]:
            groups[(row["budget"], row["stratum"], row["method"])].append((unit["case"], row))
    table = []
    for key, entries in sorted(groups.items()):
        history_values = defaultdict(list)
        for case, row in entries:
            history_values[(case["constructor_index"], case["history_index"])].append(row["verified"]["success"])
        constructors = defaultdict(list)
        for (c, h), values in sorted(history_values.items()):
            constructors[c].append(statistics.fmean(values))
        vector = {c: statistics.fmean(xs) for c, xs in sorted(constructors.items())}
        constructor_vectors[key] = vector
        all_rows = [r for _, r in entries]
        verified = [r["verified"] for r in all_rows]
        good = [r for r in verified if r["success"]]
        table.append({"budget": key[0], "stratum": key[1], "method": key[2], "constructor_configurations": len(vector),
                      "histories": len(history_values), "target_executions": len(entries),
                      "success_fraction": statistics.fmean(vector.values()), "descriptive_95_interval": interval(list(vector.values()), key),
                      "successful_executions": len(good), "successful_online_primitives": statistics.fmean(r["online_primitives"] for r in good) if good else None,
                      **{"mean_" + field: statistics.fmean(r[field] for r in verified) for field in
                         ("search_primitives", "checking_primitives", "submission_primitives", "online_primitives", "attempts_considered")},
                      **{field + "_rate": statistics.fmean(r[field] for r in verified) for field in ("empty_library", "rejected", "invalid", "timeout")}})
    contrasts = []
    for budget, stratum in sorted({k[:2] for k in groups}):
        vectors = {m: constructor_vectors[(budget, stratum, m)] for m in METHODS}
        definitions = {
            "focus_unchecked": {"focused-unchecked": 1, "broad-unchecked": -1},
            "focus_checked": {"focused-checked": 1, "broad-checked": -1},
            "checking_broad": {"broad-checked": 1, "broad-unchecked": -1},
            "checking_focused": {"focused-checked": 1, "focused-unchecked": -1},
            "interaction": {"focused-checked": 1, "focused-unchecked": -1, "broad-checked": -1, "broad-unchecked": 1},
            **{m + "_minus_primitive": {m: 1, "primitive": -1} for m in METHODS if m != "primitive"},
            **{a + "_extra_minus_matched": {a + "-checked-extra": 1, a + "-checked": -1} for a in ("broad", "focused")}}
        for name, weights in definitions.items():
            values = [sum(w * vectors[m][c] for m, w in weights.items()) for c in sorted(vectors["primitive"])]
            contrasts.append({"budget": budget, "stratum": stratum, "contrast": name, "weights": weights,
                              "gain_fraction": statistics.fmean(values), "descriptive_95_interval": interval(values, [budget, stratum, name]),
                              "constructor_configurations": len(values), "constructor_gains": values})
    for cell in table:
        baseline = next(x for x in table if x["budget"] == cell["budget"] and x["stratum"] == cell["stratum"] and x["method"] == "primitive")
        cell["primitive_relative_success_fraction"] = cell["success_fraction"] - baseline["success_fraction"]
    cases = list(cases_by_id.values())
    acquisition = []
    for name in ("broad", "focused"):
        results = [c["acquisitions"][name]["result"] for c in cases]
        if results:
            exposures = [sum(x["artifact"] in c["transfer_targets"]["changed"] for x in
                             c["acquisitions"][name]["request"]["processed"]) for c in cases]
            acquisition.append({"allocation": name, "histories": len(results), "empty_library_count": sum(not r["library"] for r in results),
                                "top_tie_count": sum(len(r["top_ties"]) > 1 for r in results),
                                "accidental_changed_endpoint_exposures": sum(exposures),
                                "histories_with_accidental_endpoint_exposure": sum(x > 0 for x in exposures),
                                "mean_costs": {k: statistics.fmean(r["costs"][k] for r in results) for k in results[0]["costs"]},
                                "unique_libraries": len({r["library_sha256"] for r in results})})
    summary = {"schema": "v18.comparisons.1", "plan_sha256": plan_hash, "claim_status": plan["claim_status"],
               "interval_method": "paired constructor percentile bootstrap, 2000 draws; descriptive, no multiplicity-adjusted claims",
               "cell_table": table, "contrasts": contrasts, "acquisition": acquisition,
               "counts": {"constructor_configurations": len({c["constructor_index"] for c in cases}),
                          "unique_constructor_configurations": len({c["constructor_sha256"] for c in cases}),
                          "unique_structural_worlds": len({c["structural_world_sha256"] for c in cases}),
                          "architectural_families": 1 if cases else 0, "acquisition_histories": len(cases),
                          "distinct_transfer_endpoints": len({t for c in cases for ts in c["transfer_targets"].values() for t in ts}),
                          "transfer_executions": sum(len(u["rows"]) for u in units),
                          "completed_blocks": len(receipts), "planned_core_blocks": (len(plan["constructors"]) + 7) // 8},
               "raw_bindings": {r["name"]: r["raw_sha256"] for r in receipts}}
    return summary, units
