"""Independent B02 case interpretation and archive denominator checks.

No primary archive classifier or search policy import. Actual M02 records are
verified through the independent physical-production and likelihood enumerator.
"""
from .records import read
from .audit_selection import audit_unit, independent_prediction
from .recoding import close


def verify_case(base, row, reported):
    audit_unit(base, row)
    uid = row["unit_id"]
    public = read(base/"public"/(uid+".json"))
    private = read(base/"private"/(uid+".json"))
    # Read only the inputs; independently enumerate every output used to classify.
    aware = independent_prediction(public, "selection-aware")
    naive = independent_prediction(public, "release-naive")
    table = independent_prediction(public, "direct-table")
    ablation = reported["release_only_request"]
    if any(batch["full_candidates"] is not None for batch in ablation["public"]["history"]):
        raise ValueError("release-only ablation leaked rejected work")
    import copy
    expected = copy.deepcopy(public)
    for batch in expected["history"]:
        batch["full_candidates"] = None
    if expected != ablation["public"]:
        raise ValueError("archive ablation changed another observation")
    reduced = independent_prediction(expected, "selection-aware")
    if not close(reduced, ablation["result"]):
        raise ValueError("archive ablation disagrees with independent enumeration")
    true_style = private["maker"]["decoration"]["choice"]
    equal = all(abs(x-y) <= 1e-10 for key in ["acquired_core", "acquired_style", "audience", "future_raw_core", "future_raw_style", "future_release_style"]
                for x,y in zip(aware[key], table[key]))
    cases = {"direct-equivalence": equal,
        "artifact-core-ambiguity": abs(aware["acquired_core"][0]-0.5) <= 1e-10 and abs(aware["acquired_core"][1]-0.5) <= 1e-10,
        "process-narrows-core": any(p >= 0.75 for p in aware["acquired_core"]),
        "selection-changes-raw-prediction": abs(aware["future_raw_style"][0]-naive["future_raw_style"][0]) >= 0.05,
        "naive-confident-error": naive["acquired_style"][true_style] < 0.2 < 0.5 < aware["acquired_style"][true_style],
        "rejected-work-corrects-account": any(batch["full_candidates"] is not None for batch in public["history"])
            and aware["acquired_style"][true_style]-reduced["acquired_style"][true_style] >= 0.05}
    if cases != reported["cases"]:
        raise ValueError("semantic archive interpretation differs")
    collision = reported["physical_collision"]
    def execute(actions):
        state = set()
        for action in actions:
            if type(action) is not int or not 0 <= action < 32:
                raise ValueError("invalid archive witness program")
            state.add(action) if action < 16 else state.discard(action-16)
        return sum(2**cell for cell in state)
    if collision["alternative"] != [collision["original"][1], collision["original"][0], *collision["original"][2:]]:
        raise ValueError("archive collision is not the declared core-order edit")
    for side in ["original", "alternative"]:
        if execute(collision[side]) != collision[side+"_execution"]["artifact"]:
            raise ValueError("archive collision failed independent execution")
    if execute(collision["original"]) != execute(collision["alternative"]):
        raise ValueError("archive collision is not equifinal")
    work = next(iter(row["arms"].values()))["outcomes"]
    primitive_cost = sum(work[key] for key in ["training_primitives", "observed_production_primitives", "future_production_primitives"])
    if primitive_cost != 113 or reported["verification_primitive_steps"] != 10:
        raise ValueError("archive equal-budget physical work is incorrect")
    return {"instrument_state": "valid", "semantic_ids": sorted(key for key,value in cases.items() if value),
            "physical_primitives": primitive_cost, "separate_verification_primitives": 10,
            "produced_observed_candidates": 12, "fresh_future_candidates": 5}


def verify_summary(search, followup, reported, required_recurrences):
    counts, denominators = {}, {}
    for row in followup:
        condition = row["condition"]
        denominators[condition] = denominators.get(condition, 0)+1
        count = counts.setdefault(condition, {key: 0 for key in row["cases"]})
        for key, value in row["cases"].items():
            count[key] += int(value)
    if counts != reported["followup_frequencies"] or denominators != reported["followup_denominators"]:
        raise ValueError("independent archive follow-up denominator or frequency mismatch")
    for method, result in reported["methods"].items():
        expected_means, method_budget = [], 0
        for replicate in result["replicates"]:
            rows = sorted((row for row in search if row["method"] == method and row["replicate"] == replicate["replicate"]), key=lambda row: row["step"])
            first = {}
            for row in rows:
                for key in row["new_cases"]:
                    if key in first:
                        raise ValueError("repeated semantic novelty")
                    first[key] = row
            if set(first) != set(replicate["witnesses"]):
                raise ValueError("archive witness identities differ")
            validated = 0
            for key, first_row in first.items():
                item = replicate["witnesses"][key]
                condition = first_row["condition"]
                if item["candidate"] != first_row["candidate_receipt"] or item["condition"] != condition:
                    raise ValueError("first witness replaced using follow-up")
                expected = counts[condition][key] >= required_recurrences
                if item["validated"] != expected or item["followup_recurrences"] != counts[condition][key] or item["followup_makers"] != denominators[condition]:
                    raise ValueError("archive recurrence decision mismatch")
                validated += int(expected)
            budget = sum(row["physical_primitives"] for row in rows)
            verification = sum(row["separate_verification_primitives"] for row in rows)
            if (replicate["distinct_validated_cases"], replicate["distinct_provisional_cases"], replicate["candidates"], replicate["physical_primitives"], replicate["separate_verification_primitives"]) != (validated, len(first), len(rows), budget, verification):
                raise ValueError("archive case or work denominator mismatch")
            expected_means.append(validated); method_budget += budget
        if not expected_means or result["mean_distinct_validated_cases"] != sum(expected_means)/len(expected_means) or result["total_physical_primitives"] != method_budget:
            raise ValueError("archive method aggregate mismatch")
    return {"instrument_state": "valid", "followup_maker_condition_records": len(followup), "search_candidates": len(search)}
